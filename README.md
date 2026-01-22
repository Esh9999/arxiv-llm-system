0) Что должно быть установлено

Docker Desktop (обязательно)

Открой Docker Desktop → убедись, что он запущен.

PowerShell (у тебя уже есть)

Проверка Docker:

docker version
docker compose version

1) Перейти в папку проекта
cd C:\Users\P2837\Downloads\arxiv-llm-system


Проверка что файлы на месте:

dir


Ты должен видеть: docker-compose.yml, fetcher-service, analyzer-service, .env.example.

2) Создать .env из примера
Copy-Item .env.example .env
notepad .env

3) Настроить LLM (выбери один вариант)
 YandexGPT (через Yandex Cloud)

В .env вставь:

LLM_PROVIDER=yandex
YANDEX_API_KEY=ТВОЙ_API_KEY_ИЗ_YC
YANDEX_FOLDER_ID=ТВОЙ_FOLDER_ID
YANDEX_MODEL=yandexgpt/latest


⚠️ Важно: YANDEX_API_KEY должен быть Api-Key, НЕ t1....

4) Запустить контейнеры

В корне проекта:

docker compose down
docker compose up --build -d
docker compose ps


Ожидаемо:

fetcher-service → Up → порт 8001

analyzer-service → Up → порт 8002

5) Проверка health
Invoke-RestMethod http://localhost:8001/health
Invoke-RestMethod http://localhost:8002/health


Должно вернуть:

{"status":"ok"}

6) Тест: получить статьи из arXiv (fetcher)
$fetchBody = @{
  query = "machine learning"
  max_results = 1
  fetch_full_text = $false
} | ConvertTo-Json

$fetchResp = Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8001/fetch" `
  -ContentType "application/json" `
  -Body $fetchBody

$fetchResp.total
$fetchResp.articles[0].arxiv_id
$fetchResp.articles[0].title

7) Тест: анализ статьи LLM (analyzer)
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

$analysis.analysis.main_topic
$analysis.analysis.summary.brief
$analysis.analysis.category
$analysis.confidence


Если это отработало — всё готово ✅

8) Если что-то сломалось — где смотреть

Логи:

docker compose logs fetcher --tail=200
docker compose logs analyzer --tail=200


Остановить всё:

docker compose down

9) Где открыть Swagger (удобно)

Fetcher docs: http://localhost:8001/docs

Analyzer docs: http://localhost:8002/docs
