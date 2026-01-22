import requests

FETCHER = "http://localhost:8001"
ANALYZER = "http://localhost:8002"

def main():
    fetch = requests.post(
        f"{FETCHER}/fetch",
        json={"query": "machine learning", "max_results": 2, "fetch_full_text": False},
        timeout=30,
    )
    fetch.raise_for_status()
    articles = fetch.json()["articles"]

    for a in articles:
        payload = {
            "arxiv_id": a["arxiv_id"],
            "title": a["title"],
            "abstract": a["abstract"],
            "full_text": a.get("full_text"),
            "categories": a.get("categories", []),
        }
        res = requests.post(f"{ANALYZER}/analyze", json=payload, timeout=60)
        res.raise_for_status()
        print(res.json())

if __name__ == "__main__":
    main()
