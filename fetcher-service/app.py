import io
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote_plus

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pypdf import PdfReader


ARXIV_API = "http://export.arxiv.org/api/query"


class Settings(BaseSettings):
    fetcher_http_timeout: float = Field(25, alias="FETCHER_HTTP_TIMEOUT")
    fetcher_max_pdf_pages: int = Field(10, alias="FETCHER_MAX_PDF_PAGES")
    fetcher_max_text_chars: int = Field(50000, alias="FETCHER_MAX_TEXT_CHARS")
    fetcher_user_agent: str = Field(
        "arxiv-llm-system/1.0",
        alias="FETCHER_USER_AGENT",
    )

    class Config:
        populate_by_name = True


settings = Settings()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("fetcher")

app = FastAPI(title="Article Fetcher Service", version="1.1.0")


# -----------------------------
# Prometheus metrics
# -----------------------------
REQS = Counter("fetcher_requests_total", "Total requests", ["endpoint", "status"])
LAT = Histogram("fetcher_request_latency_seconds", "Latency", ["endpoint"])
ARXIV_ERR = Counter("fetcher_arxiv_errors_total", "arXiv errors", ["status"])
PDF_ERR = Counter("fetcher_pdf_errors_total", "PDF errors", ["stage"])


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -----------------------------
# Models
# -----------------------------
class FetchRequest(BaseModel):
    arxiv_id: Optional[str] = None
    query: Optional[str] = None
    max_results: int = 5
    fetch_full_text: bool = False

    @model_validator(mode="after")
    def validate_mode(self):
        if bool(self.arxiv_id) == bool(self.query):
            raise ValueError("Provide exactly one of: arxiv_id OR query")
        if self.query and (self.max_results < 1 or self.max_results > 50):
            raise ValueError("max_results must be between 1 and 50")
        if self.arxiv_id and not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", self.arxiv_id.strip()):
            raise ValueError("Invalid arxiv_id format, expected like 2301.12345 or 2301.12345v2")
        return self


class ArticleOut(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    pdf_url: str
    full_text: Optional[str] = None
    text_length: int = 0


class FetchResponse(BaseModel):
    articles: List[ArticleOut]
    total: int


# -----------------------------
# arXiv parsing
# -----------------------------
def _strip(s: Optional[str]) -> str:
    return (s or "").strip().replace("\n", " ")


def _arxiv_id_from_entry_id(entry_id: str) -> str:
    # entry_id example: http://arxiv.org/abs/2301.12345v1
    m = re.search(r"/abs/([^/]+)$", entry_id.strip())
    if not m:
        return entry_id.strip()
    return m.group(1)


def parse_arxiv_atom(xml_text: str) -> List[ArticleOut]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML from arXiv: {e}")

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    articles: List[ArticleOut] = []
    for entry in root.findall("atom:entry", ns):
        entry_id = _strip(entry.findtext("atom:id", default="", namespaces=ns))
        arxiv_id = _arxiv_id_from_entry_id(entry_id)

        title = _strip(entry.findtext("atom:title", default="", namespaces=ns))
        abstract = _strip(entry.findtext("atom:summary", default="", namespaces=ns))
        published = _strip(entry.findtext("atom:published", default="", namespaces=ns))
        if published:
            # normalize to YYYY-MM-DD
            try:
                published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                published = published_dt.date().isoformat()
            except Exception:
                pass

        authors = []
        for a in entry.findall("atom:author", ns):
            name = _strip(a.findtext("atom:name", default="", namespaces=ns))
            if name:
                authors.append(name)

        categories = []
        for c in entry.findall("atom:category", ns):
            term = c.attrib.get("term", "").strip()
            if term:
                categories.append(term)

        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            href = link.attrib.get("href", "").strip()
            title_attr = link.attrib.get("title", "").strip().lower()
            type_attr = link.attrib.get("type", "").strip().lower()
            if title_attr == "pdf" or type_attr == "application/pdf":
                pdf_url = href
                break

        # fallback: if no pdf link found, build it from arxiv_id (remove version)
        if not pdf_url and arxiv_id:
            base_id = re.sub(r"v\d+$", "", arxiv_id)
            pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"

        articles.append(
            ArticleOut(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published=published,
                pdf_url=pdf_url,
                full_text=None,
                text_length=0,
            )
        )

    return articles


# -----------------------------
# Fetching
# -----------------------------
async def _get_arxiv_by_id(client: httpx.AsyncClient, arxiv_id: str) -> List[ArticleOut]:
    url = f"{ARXIV_API}?id_list={quote_plus(arxiv_id)}"
    r = await client.get(url)
    if r.status_code != 200:
        ARXIV_ERR.labels(str(r.status_code)).inc()
        raise HTTPException(status_code=502, detail=f"arXiv API error: {r.status_code}")
    try:
        return parse_arxiv_atom(r.text)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))


async def _search_arxiv(client: httpx.AsyncClient, query: str, max_results: int) -> List[ArticleOut]:
    q = quote_plus(f"all:{query}")
    url = f"{ARXIV_API}?search_query={q}&start=0&max_results={max_results}"
    r = await client.get(url)
    if r.status_code != 200:
        ARXIV_ERR.labels(str(r.status_code)).inc()
        raise HTTPException(status_code=502, detail=f"arXiv API error: {r.status_code}")
    try:
        return parse_arxiv_atom(r.text)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))


async def _extract_pdf_text(client: httpx.AsyncClient, pdf_url: str, max_pages: int) -> str:
    r = await client.get(pdf_url)
    if r.status_code != 200:
        PDF_ERR.labels("download").inc()
        raise HTTPException(status_code=502, detail=f"Failed to download PDF: {r.status_code}")

    try:
        reader = PdfReader(io.BytesIO(r.content))
        pages = reader.pages[:max_pages]
        parts = []
        for p in pages:
            t = p.extract_text() or ""
            parts.append(t)
        text = "\n".join(parts).strip()
        return text
    except Exception as e:
        PDF_ERR.labels("parse").inc()
        raise HTTPException(status_code=502, detail=f"PDF parse error: {e}")


def _limit_text(s: str) -> str:
    if len(s) > settings.fetcher_max_text_chars:
        return s[: settings.fetcher_max_text_chars]
    return s


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/fetch", response_model=FetchResponse)
async def fetch(req: FetchRequest) -> FetchResponse:
    with LAT.labels("fetch").time():
        headers = {"User-Agent": settings.fetcher_user_agent}
        timeout = httpx.Timeout(settings.fetcher_http_timeout)

        async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
            try:
                if req.arxiv_id:
                    articles = await _get_arxiv_by_id(client, req.arxiv_id.strip())
                else:
                    articles = await _search_arxiv(client, req.query.strip(), req.max_results)

                if not articles:
                    REQS.labels("fetch", "200").inc()
                    return FetchResponse(articles=[], total=0)

                if req.fetch_full_text:
                    for a in articles:
                        if a.pdf_url:
                            full = await _extract_pdf_text(client, a.pdf_url, settings.fetcher_max_pdf_pages)
                            full = _limit_text(full)
                            a.full_text = full
                            a.text_length = len(full)
                        else:
                            a.full_text = ""
                            a.text_length = 0

                else:
                    for a in articles:
                        a.text_length = len(a.abstract or "")

                REQS.labels("fetch", "200").inc()
                logger.info(
                    "fetch.success",
                    extra={"mode": "id" if req.arxiv_id else "query", "count": len(articles)},
                )
                return FetchResponse(articles=articles, total=len(articles))

            except HTTPException as e:
                REQS.labels("fetch", str(e.status_code)).inc()
                logger.error("fetch.http_error", extra={"status": e.status_code, "detail": str(e.detail)})
                raise
            except Exception as e:
                REQS.labels("fetch", "500").inc()
                logger.exception("fetch.unexpected_error")
                raise HTTPException(status_code=500, detail=str(e))
