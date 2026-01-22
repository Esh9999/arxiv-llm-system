# arxiv-llm-system

Two-microservice system to fetch scientific papers from arXiv and analyze them with an LLM (structured JSON output).

## Architecture

- **fetcher-service** (FastAPI, port **8001**)
  - Fetch by arXiv ID or search by query using arXiv API (Atom/XML)
  - Optionally downloads PDF and extracts text (limited pages/size)
  - Returns normalized article payload

- **analyzer-service** (FastAPI, port **8002**)
  - Accepts article payload
  - Calls an LLM to extract key info + categorize + summarize
  - Returns strict JSON (validated by Pydantic)
  - Supports cache + batch processing

Flow: `Client -> fetcher (/fetch) -> analyzer (/analyze or /batch-analyze)`.

## Requirements

- Docker Desktop (Windows/macOS/Linux)
- (Optional) Python 3.10+ if you want to run locally without Docker

## Quick start (Docker)

### 1) Create `.env`

Copy example:
```bash
cp .env.example .env
Then edit .env and choose one provider:

Option A — OpenAI-compatible (OpenAI / OpenRouter / etc.)
LLM_PROVIDER=openai
LLM_API_KEY=YOUR_API_KEY_HERE
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=nvidia/nemotron-nano-9b-v2:free
LLM_JSON_MODE=1
Option B — YandexGPT (Yandex Cloud)
LLM_PROVIDER=yandex
YANDEX_API_KEY=YOUR_YANDEX_API_KEY_HERE
YANDEX_FOLDER_ID=YOUR_FOLDER_ID_HERE
YANDEX_MODEL=yandexgpt/latest
⚠️ Never commit .env (contains secrets). Use .env.example only.

2) Run services
docker compose up --build
3) Health checks
Fetcher: http://localhost:8001/health

Analyzer: http://localhost:8002/health

API
Fetch articles
Search
POST http://localhost:8001/fetch

{
  "query": "machine learning",
  "max_results": 2,
  "fetch_full_text": false
}
Fetch by arXiv ID
POST http://localhost:8001/fetch

{
  "arxiv_id": "2306.04338",
  "fetch_full_text": false
}
fetch_full_text=true downloads PDF and extracts text (limited).

Analyze one article
POST http://localhost:8002/analyze

{
  "arxiv_id": "2306.04338",
  "title": "Title",
  "abstract": "Abstract text...",
  "categories": ["stat.ML"]
}
Response (example shape):

{
  "arxiv_id": "2306.04338",
  "analysis": {
    "main_topic": "...",
    "methodology": "...",
    "key_findings": ["..."],
    "techniques": ["..."],
    "category": {
      "domain": "Statistics",
      "subcategory": "Machine Learning",
      "complexity": "Intermediate",
      "article_type": "Theory"
    },
    "summary": {
      "brief": "...",
      "key_points": ["..."]
    }
  },
  "confidence": 0.78,
  "analysis_timestamp": "2026-01-22T10:30:00Z"
}
Batch analyze
POST http://localhost:8002/batch-analyze

{
  "articles": [
    { "arxiv_id": "2306.04338", "title": "...", "abstract": "...", "categories": ["stat.ML"] },
    { "arxiv_id": "2301.12345", "title": "...", "abstract": "...", "categories": ["cs.LG"] }
  ],
  "max_concurrent": 3
}
PowerShell examples (Windows)
Fetch -> Analyze pipeline
$fetchBody = @{
  query = "machine learning"
  max_results = 1
  fetch_full_text = $false
} | ConvertTo-Json

$fetchResp = Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8001/fetch" `
  -ContentType "application/json" `
  -Body $fetchBody

$article = $fetchResp.articles[0]

$analyzeBody = @{
  arxiv_id = $article.arxiv_id
  title = $article.title
  abstract = $article.abstract
  categories = $article.categories
} | ConvertTo-Json -Depth 10

$analysis = Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8002/analyze" `
  -ContentType "application/json" `
  -Body $analyzeBody

$analysis | ConvertTo-Json -Depth 20
Tests
Unit tests are located in:

fetcher-service/tests

analyzer-service/tests

Run in containers:

docker compose exec fetcher pytest -q
docker compose exec analyzer pytest -q
Notes / Design choices
Async HTTP calls via httpx

Strict response validation via pydantic

LLM output is forced to JSON + validated

Text size is capped (LLM_MAX_TEXT_CHARS, default 50000)

Simple in-memory TTL cache avoids re-analyzing the same paper
