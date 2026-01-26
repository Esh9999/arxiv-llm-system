import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


# -----------------------------
# Logging (structured-ish)
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("analyzer")


# -----------------------------
# Settings
# -----------------------------
class Settings(BaseSettings):
    # provider: "openai" | "yandex"
    llm_provider: str = Field("openai", alias="LLM_PROVIDER")

    # common
    llm_timeout: float = Field(45, alias="LLM_TIMEOUT")
    llm_max_text_chars: int = Field(50000, alias="LLM_MAX_TEXT_CHARS")

    # openai-compatible
    llm_api_key: str = Field("", alias="LLM_API_KEY")
    llm_base_url: str = Field("https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_model: str = Field("gpt-4o-mini", alias="LLM_MODEL")
    llm_json_mode: bool = Field(True, alias="LLM_JSON_MODE")

    # optional openrouter headers
    llm_http_referer: str = Field("", alias="LLM_HTTP_REFERER")
    llm_app_title: str = Field("", alias="LLM_APP_TITLE")

    # yandex cloud
    yandex_api_key: str = Field("", alias="YANDEX_API_KEY")
    yandex_folder_id: str = Field("", alias="YANDEX_FOLDER_ID")
    yandex_model: str = Field("yandexgpt/latest", alias="YANDEX_MODEL")

    # cache
    cache_ttl_seconds: int = Field(3600, alias="CACHE_TTL_SECONDS")
    cache_maxsize: int = Field(2048, alias="CACHE_MAXSIZE")

    class Config:
        populate_by_name = True


settings = Settings()
app = FastAPI(title="Article Analyzer Service", version="1.1.0")

_cache: TTLCache = TTLCache(maxsize=settings.cache_maxsize, ttl=settings.cache_ttl_seconds)


# -----------------------------
# Prometheus metrics
# -----------------------------
REQS = Counter("analyzer_requests_total", "Total requests", ["endpoint", "status"])
LAT = Histogram("analyzer_request_latency_seconds", "Latency", ["endpoint"])
LLM_ERR = Counter("analyzer_llm_errors_total", "LLM errors", ["provider", "status"])
CACHE_HIT = Counter("analyzer_cache_hits_total", "Cache hits")
CACHE_MISS = Counter("analyzer_cache_misses_total", "Cache misses")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -----------------------------
# API Models
# -----------------------------
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
    # ✅ Confidence from LLM (self-reported, validated)
    confidence: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    arxiv_id: str
    analysis: AnalysisOut
    confidence: float
    analysis_timestamp: str


class BatchAnalyzeRequest(BaseModel):
    articles: List[ArticleIn]
    max_concurrent: int = 3


# -----------------------------
# Helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clip_text(article: ArticleIn) -> str:
    text = (article.full_text or article.abstract or "").strip()
    if len(text) > settings.llm_max_text_chars:
        text = text[: settings.llm_max_text_chars]
    return text


def _cache_key(article: ArticleIn) -> str:
    text = _clip_text(article)
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    # include provider+model to avoid cross-provider collisions
    return f"{settings.llm_provider}:{settings.llm_model}:{article.arxiv_id}:{h}"


def build_prompt(article: ArticleIn) -> str:
    text = _clip_text(article)
    cats = ", ".join(article.categories or [])

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
  }},
  "confidence": 0.0
}}

Rules for confidence (0..1):
- 0.90-1.00: very certain, text clearly supports claims
- 0.70-0.89: reasonably certain, minor ambiguity
- 0.50-0.69: moderate uncertainty, missing details
- <0.50: highly uncertain / speculative

Paper metadata:
- arXiv ID: {article.arxiv_id}
- Title: {article.title}
- Categories: {cats}

Paper text (abstract or full text):
{text}
""".strip()


def _extract_json_object(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("{") and s.endswith("}"):
        return s

    # fenced ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)

    # first '{' ... last '}'
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first:last + 1]

    raise ValueError("No JSON object found in model response")


async def _call_with_retries(fn, attempts: int = 3) -> Any:
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            return await fn()
        except HTTPException as e:
            # Do not retry auth/access issues
            if e.status_code in (401, 403):
                raise
            last_exc = e
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
            last_exc = e

        await asyncio.sleep(min(2 ** i, 6))

    if isinstance(last_exc, HTTPException):
        raise last_exc
    raise HTTPException(status_code=502, detail=f"LLM call failed after retries: {last_exc}")


def heuristic_fallback(article: ArticleIn, reason: str) -> AnalyzeResponse:
    # Honest fallback: minimal extraction, low confidence.
    brief = (article.abstract or "").strip()
    if len(brief) > 400:
        brief = brief[:400]

    analysis = AnalysisOut(
        main_topic=(article.title or "Unknown").strip(),
        methodology="N/A",
        key_findings=[],
        techniques=[],
        category=CategoryOut(
            domain="Other",
            subcategory="Unknown",
            complexity="Beginner",
            article_type="Theory",
        ),
        summary=SummaryOut(
            brief=brief or "N/A",
            key_points=[],
        ),
        confidence=0.25,
    )

    logger.warning(
        "analyze.fallback",
        extra={"arxiv_id": article.arxiv_id, "reason": reason, "provider": settings.llm_provider},
    )

    return AnalyzeResponse(
        arxiv_id=article.arxiv_id,
        analysis=analysis,
        confidence=float(analysis.confidence),
        analysis_timestamp=_now_iso(),
    )


# -----------------------------
# LLM Providers
# -----------------------------
async def call_openai_compatible(prompt: str) -> Dict[str, Any]:
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
            {"role": "system", "content": "Return ONLY valid JSON. No markdown. No explanation."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    # JSON mode (if supported by provider)
    if settings.llm_json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:

        async def _do():
            r = await client.post(f"{settings.llm_base_url}/chat/completions", headers=headers, json=payload)
            if r.status_code != 200:
                LLM_ERR.labels(settings.llm_provider, str(r.status_code)).inc()
                logger.error(
                    "llm.error",
                    extra={"provider": settings.llm_provider, "status": r.status_code, "body": r.text[:200]},
                )
                raise HTTPException(status_code=502, detail=f"LLM error {r.status_code}: {r.text[:300]}")

            data = r.json()
            content = data["choices"][0]["message"]["content"]
            try:
                obj = json.loads(_extract_json_object(content))
                if not isinstance(obj, dict):
                    raise ValueError("JSON is not an object")
                return obj
            except Exception as e:
                LLM_ERR.labels(settings.llm_provider, "invalid_json").inc()
                logger.error("llm.invalid_json", extra={"provider": settings.llm_provider, "err": str(e)})
                raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

        return await _call_with_retries(_do, attempts=3)


async def call_yandex_gpt(prompt: str) -> Dict[str, Any]:
    if not settings.yandex_api_key or not settings.yandex_folder_id:
        raise HTTPException(status_code=500, detail="YANDEX_API_KEY or YANDEX_FOLDER_ID is not set")

    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Authorization": f"Api-Key {settings.yandex_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "modelUri": f"gpt://{settings.yandex_folder_id}/{settings.yandex_model}",
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 2000,
        },
        "messages": [
            {"role": "system", "text": "Return ONLY valid JSON. No markdown. No explanation."},
            {"role": "user", "text": prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:

        async def _do():
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                LLM_ERR.labels(settings.llm_provider, str(r.status_code)).inc()
                logger.error(
                    "llm.error",
                    extra={"provider": settings.llm_provider, "status": r.status_code, "body": r.text[:200]},
                )
                raise HTTPException(status_code=502, detail=f"YandexGPT error {r.status_code}: {r.text[:300]}")

            data = r.json()
            text = data["result"]["alternatives"][0]["message"]["text"]
            try:
                obj = json.loads(_extract_json_object(text))
                if not isinstance(obj, dict):
                    raise ValueError("JSON is not an object")
                return obj
            except Exception as e:
                LLM_ERR.labels(settings.llm_provider, "invalid_json").inc()
                logger.error("llm.invalid_json", extra={"provider": settings.llm_provider, "err": str(e)})
                raise HTTPException(status_code=502, detail=f"YandexGPT returned invalid JSON: {e}")

        return await _call_with_retries(_do, attempts=3)


async def call_llm(prompt: str) -> Dict[str, Any]:
    provider = (settings.llm_provider or "openai").strip().lower()
    if provider == "yandex":
        return await call_yandex_gpt(prompt)
    return await call_openai_compatible(prompt)


# -----------------------------
# Core analyze logic
# -----------------------------
async def analyze_one(article: ArticleIn) -> AnalyzeResponse:
    key = _cache_key(article)
    cached = _cache.get(key)
    if cached:
        CACHE_HIT.inc()
        return cached
    CACHE_MISS.inc()

    logger.info("analyze.request", extra={"arxiv_id": article.arxiv_id, "provider": settings.llm_provider})

    prompt = build_prompt(article)
    try:
        obj = await call_llm(prompt)
        analysis = AnalysisOut.model_validate(obj)  # strict validation
        resp = AnalyzeResponse(
            arxiv_id=article.arxiv_id,
            analysis=analysis,
            confidence=float(analysis.confidence),
            analysis_timestamp=_now_iso(),
        )
        _cache[key] = resp
        logger.info("analyze.success", extra={"arxiv_id": article.arxiv_id})
        return resp
    except HTTPException as e:
        # fallback for non-auth errors or temporary issues
        return heuristic_fallback(article, reason=f"HTTPException {e.status_code}: {e.detail}")
    except Exception as e:
        return heuristic_fallback(article, reason=f"Unexpected error: {e}")


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(article: ArticleIn) -> AnalyzeResponse:
    with LAT.labels("analyze").time():
        try:
            out = await analyze_one(article)
            REQS.labels("analyze", "200").inc()
            return out
        except HTTPException as e:
            REQS.labels("analyze", str(e.status_code)).inc()
            raise


@app.post("/batch-analyze", response_model=List[AnalyzeResponse])
async def batch_analyze(req: BatchAnalyzeRequest) -> List[AnalyzeResponse]:
    with LAT.labels("batch_analyze").time():
        max_conc = max(1, min(req.max_concurrent, 20))
        sem = asyncio.Semaphore(max_conc)

        async def worker(a: ArticleIn) -> AnalyzeResponse:
            async with sem:
                return await analyze_one(a)

        tasks = [asyncio.create_task(worker(a)) for a in req.articles]
        results = await asyncio.gather(*tasks)
        REQS.labels("batch_analyze", "200").inc()
        return results
