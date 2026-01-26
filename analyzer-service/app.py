import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_exponential


class Settings(BaseSettings):
    llm_api_key: str = Field("", alias="LLM_API_KEY")
    llm_base_url: str = Field("https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_model: str = Field("gpt-4o-mini", alias="LLM_MODEL")
    llm_timeout: float = Field(45, alias="LLM_TIMEOUT")
    llm_max_text_chars: int = Field(50000, alias="LLM_MAX_TEXT_CHARS")
    llm_json_mode: bool = Field(True, alias="LLM_JSON_MODE")

    cache_ttl_seconds: int = Field(3600, alias="CACHE_TTL_SECONDS")
    cache_maxsize: int = Field(2048, alias="CACHE_MAXSIZE")

    # Optional OpenRouter headers:
    llm_http_referer: str = Field("", alias="LLM_HTTP_REFERER")
    llm_app_title: str = Field("", alias="LLM_APP_TITLE")

    class Config:
        populate_by_name = True


settings = Settings()
app = FastAPI(title="Article Analyzer Service", version="1.0.0")

_cache: TTLCache = TTLCache(maxsize=settings.cache_maxsize, ttl=settings.cache_ttl_seconds)


# ---------- API models ----------

class ArticleIn(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    full_text: Optional[str] = None
    categories: List[str] = Field(default_factory=list)


class CategoryOut(BaseModel):
    domain: str
    subcategory: str
    complexity: str  # Beginner|Intermediate|Advanced
    article_type: str  # Theory|Application|Survey|Tutorial


class SummaryOut(BaseModel):
    brief: str
    key_points: List[str]


class AnalysisOut(BaseModel):
    main_topic: str
    methodology: str
    key_findings: List[str]
    techniques: List[str]
    category: CategoryOut
    summary: SummaryOut


class AnalyzeResponse(BaseModel):
    arxiv_id: str
    analysis: AnalysisOut
    confidence: float
    analysis_timestamp: str


class BatchAnalyzeRequest(BaseModel):
    articles: List[ArticleIn]
    max_concurrent: int = 3


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- prompt + parsing ----------

def _clip_text(article: ArticleIn) -> str:
    text = (article.full_text or article.abstract or "").strip()
    if len(text) > settings.llm_max_text_chars:
        text = text[: settings.llm_max_text_chars]
    return text


def build_prompt(article: ArticleIn) -> str:
    text = _clip_text(article)
    cats = ", ".join(article.categories or [])

    # Strong JSON-only instruction (works even without JSON mode)
    return f"""
You are an expert scientific paper analyst.

Return ONLY valid JSON that matches this schema exactly:

{{
  "main_topic": "string",
  "methodology": "string (or 'N/A' if not applicable)",
  "key_findings": ["string", "..."],
  "techniques": ["string", "..."],
  "category": {{
    "domain": "Computer Science|Physics|Mathematics|Statistics|Biology|Other",
    "subcategory": "string (e.g. Machine Learning, NLP, Computer Vision, etc.)",
    "complexity": "Beginner|Intermediate|Advanced",
    "article_type": "Theory|Application|Survey|Tutorial"
  }},
  "summary": {{
    "brief": "2-3 sentences",
    "key_points": ["bullet", "..."]
  }}
}}

Paper metadata:
- arXiv ID: {article.arxiv_id}
- Title: {article.title}
- Categories: {cats}

Paper text (may be abstract or full text):
{text}
""".strip()


def _extract_json_object(s: str) -> str:
    """
    Try to find the first JSON object in a messy LLM response.
    """
    s = (s or "").strip()
    # Fast path
    if s.startswith("{") and s.endswith("}"):
        return s

    # Extract between first '{' and last '}'
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first:last + 1]

    # Try fenced block ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)

    raise ValueError("No JSON object found in LLM response")


def _cache_key(article: ArticleIn) -> str:
    text = _clip_text(article)
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{article.arxiv_id}:{h}:{settings.llm_model}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.8, min=1, max=8))
async def call_llm(prompt: str) -> Dict[str, Any]:
    if not settings.llm_api_key:
        raise HTTPException(status_code=500, detail="LLM_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    # OpenRouter optional headers
    if settings.llm_http_referer:
        headers["HTTP-Referer"] = settings.llm_http_referer
    if settings.llm_app_title:
        headers["X-Title"] = settings.llm_app_title

    payload: Dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "You MUST output only valid JSON. No commentary."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    # JSON mode where supported (OpenAI compatible)
    if settings.llm_json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
        r = await client.post(f"{settings.llm_base_url}/chat/completions", headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"LLM error {r.status_code}: {r.text[:300]}")

    data = r.json()
    content = data["choices"][0]["message"]["content"]
    try:
        json_str = _extract_json_object(content)
        obj = json.loads(json_str)
        if not isinstance(obj, dict):
            raise ValueError("LLM JSON is not an object")
        return obj
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")


async def analyze_one(article: ArticleIn) -> AnalyzeResponse:
    key = _cache_key(article)
    cached = _cache.get(key)
    if cached:
        return cached

    prompt = build_prompt(article)
    obj = await call_llm(prompt)

    # Validate with Pydantic (will raise if mismatched)
    analysis = AnalysisOut.model_validate(obj)

    # Simple confidence heuristic
    has_full_text = bool(article.full_text and len(article.full_text.strip()) > 500)
    confidence = 0.88 if has_full_text else 0.78

    resp = AnalyzeResponse(
        arxiv_id=article.arxiv_id,
        analysis=analysis,
        confidence=confidence,
        analysis_timestamp=_now_iso(),
    )
    _cache[key] = resp
    return resp


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(article: ArticleIn) -> AnalyzeResponse:
    return await analyze_one(article)


@app.post("/batch-analyze", response_model=List[AnalyzeResponse])
async def batch_analyze(req: BatchAnalyzeRequest) -> List[AnalyzeResponse]:
    max_conc = max(1, min(req.max_concurrent, 20))
    sem = asyncio.Semaphore(max_conc)

    async def worker(a: ArticleIn) -> AnalyzeResponse:
        async with sem:
            return await analyze_one(a)

    tasks = [asyncio.create_task(worker(a)) for a in req.articles]
    return await asyncio.gather(*tasks)
