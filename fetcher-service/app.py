import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class Settings(BaseSettings):
    fetcher_http_timeout: float = Field(25, alias="FETCHER_HTTP_TIMEOUT")
    fetcher_max_pdf_pages: int = Field(10, alias="FETCHER_MAX_PDF_PAGES")
    fetcher_max_text_chars: int = Field(50000, alias="FETCHER_MAX_TEXT_CHARS")
    fetcher_user_agent: str = Field(
        "arxiv-llm-system/1.0 (contact: you@example.com)",
        alias="FETCHER_USER_AGENT",
    )

    class Config:
        populate_by_name = True


settings = Settings()


class FetchRequest(BaseModel):
    arxiv_id: Optional[str] = None
    query: Optional[str] = None
    max_results: int = 5
    fetch_full_text: bool = False

    @model_validator(mode="after")
    def validate_input(self):
        if bool(self.arxiv_id) == bool(self.query):
            raise ValueError("Provide exactly one of: arxiv_id OR query")
        if self.query and not (1 <= self.max_results <= 50):
            raise ValueError("max_results must be within [1..50]")
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


app = FastAPI(title="Article Fetcher Service", version="1.0.0")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _extract_arxiv_id(entry_id: str) -> str:
    """
    From 'http://arxiv.org/abs/2301.12345v2' -> '2301.12345'
    """
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", entry_id or "")
    if not m:
        return (entry_id or "").strip()
    return m.group(1)


def _pdf_url_from_entry(entry: ET.Element, arxiv_id: str) -> str:
    for link in entry.findall("atom:link", ATOM_NS):
        href = link.attrib.get("href", "")
        ltype = link.attrib.get("type", "")
        title = (link.attrib.get("title", "") or "").lower()
        if ltype == "application/pdf" or title == "pdf" or href.endswith(".pdf"):
            return href
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def parse_arxiv_atom(xml_text: str) -> List[ArticleOut]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse arXiv XML: {e}")

    entries = root.findall("atom:entry", ATOM_NS)
    articles: List[ArticleOut] = []

    for entry in entries:
        entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        arxiv_id = _extract_arxiv_id(entry_id)

        title = _clean(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        abstract = _clean(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))

        authors = []
        for a in entry.findall("atom:author", ATOM_NS):
            name = _clean(a.findtext("atom:name", default="", namespaces=ATOM_NS))
            if name:
                authors.append(name)

        categories = []
        for c in entry.findall("atom:category", ATOM_NS):
            term = c.attrib.get("term")
            if term:
                categories.append(term)

        published_raw = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        published = ""
        if published_raw:
            try:
                published = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).date().isoformat()
            except Exception:
                published = published_raw[:10]

        pdf_url = _pdf_url_from_entry(entry, arxiv_id)

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


def _normalize_search_query(q: str) -> str:
    q = (q or "").strip()
    # If user already provides advanced query like 'cat:cs.LG AND ti:transformer'
    if ":" in q:
        return q
    return f"all:{q}"


async def _arxiv_get(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url)
    if r.status_code != 200:
        # arXiv sometimes returns HTML text on blocks; include a short snippet
        snippet = (r.text or "")[:200].replace("\n", " ")
        raise HTTPException(status_code=502, detail=f"arXiv API error {r.status_code}: {snippet}")
    return r.text


async def fetch_by_id(client: httpx.AsyncClient, arxiv_id: str) -> List[ArticleOut]:
    url = f"{ARXIV_API}?id_list={quote_plus(arxiv_id)}"
    xml = await _arxiv_get(client, url)
    return parse_arxiv_atom(xml)


async def search(client: httpx.AsyncClient, query: str, max_results: int) -> List[ArticleOut]:
    sq = quote_plus(_normalize_search_query(query))
    url = f"{ARXIV_API}?search_query={sq}&max_results={max_results}"
    xml = await _arxiv_get(client, url)
    return parse_arxiv_atom(xml)


async def extract_pdf_text(client: httpx.AsyncClient, pdf_url: str, max_pages: int) -> str:
    r = await client.get(pdf_url)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Failed to download PDF ({r.status_code})")

    try:
        doc = fitz.open(stream=r.content, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF open failed: {e}")

    pages = min(max_pages, doc.page_count)
    parts: List[str] = []

    for i in range(pages):
        try:
            page = doc.load_page(i)
            t = page.get_text("text")
            if t:
                parts.append(t)
        except Exception:
            continue

    return "\n".join(parts).strip()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/fetch", response_model=FetchResponse)
async def fetch(req: FetchRequest) -> FetchResponse:
    headers = {"User-Agent": settings.fetcher_user_agent}

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(settings.fetcher_http_timeout),
        follow_redirects=True,
        headers=headers,
    ) as client:
        if req.arxiv_id:
            articles = await fetch_by_id(client, req.arxiv_id)
        else:
            articles = await search(client, req.query or "", req.max_results)

        if not articles:
            raise HTTPException(status_code=404, detail="No articles found")

        if req.fetch_full_text:
            for a in articles:
                full_text = await extract_pdf_text(client, a.pdf_url, settings.fetcher_max_pdf_pages)
                # Hard limit for analysis
                if len(full_text) > settings.fetcher_max_text_chars:
                    full_text = full_text[: settings.fetcher_max_text_chars]
                a.full_text = full_text
                a.text_length = len(full_text)
        else:
            for a in articles:
                a.full_text = None
                a.text_length = len(a.abstract or "")

        return FetchResponse(articles=articles, total=len(articles))
