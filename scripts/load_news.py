# scripts/load_news.py
import os
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "prices.db"
PROVIDER = "finnhub"


def get_api_key() -> str:
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("Missing FINNHUB_API_KEY env var.")
    return key


def ensure_news_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        published_at TEXT NOT NULL,
        ticker TEXT NOT NULL,
        title TEXT NOT NULL,
        source TEXT,
        url TEXT,
        summary TEXT,
        provider TEXT NOT NULL,
        provider_id TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        UNIQUE(provider, provider_id)
    )
    """)
    conn.commit()


def get_holdings_tickers(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT ticker FROM holdings ORDER BY ticker").fetchall()
    return [str(r[0]).upper() for r in rows if r and r[0]]


def finnhub_company_news(ticker: str, from_date: str, to_date: str, api_key: str) -> list[dict]:
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": api_key,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def upsert_news(conn: sqlite3.Connection, ticker: str, items: list[dict]) -> int:
    if not items:
        return 0

    rows = []
    for it in items:
        provider_id = str(it.get("id") or "")
        if not provider_id:
            continue

        ts = it.get("datetime")
        if ts is None:
            continue

        published_at = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
        title = str(it.get("headline") or "").strip()
        if not title:
            continue

        rows.append((
            published_at,
            ticker,
            title,
            it.get("source"),
            it.get("url"),
            it.get("summary"),
            PROVIDER,
            provider_id
        ))

    if not rows:
        return 0

    conn.executemany("""
        INSERT INTO news (
            published_at, ticker, title, source, url, summary, provider, provider_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, provider_id) DO NOTHING
    """, rows)

    conn.commit()
    return len(rows)


def main():
    api_key = get_api_key()

    today = datetime.now(timezone.utc).date()
    from_date = (today - timedelta(days=7)).isoformat()
    to_date = today.isoformat()

    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_news_table(conn)

        tickers = get_holdings_tickers(conn)
        if not tickers:
            raise RuntimeError("No holdings found. Run load_holdings.py first.")

        total = 0
        for t in tickers:
            try:
                items = finnhub_company_news(t, from_date, to_date, api_key)
                inserted = upsert_news(conn, t, items)
                print(f"{t}: fetched {len(items)}, inserted {inserted}")
                total += inserted
                time.sleep(0.4)  # rate-limit safety
            except Exception as e:
                print(f"{t}: ERROR {e}")

        print(f"Done. Total inserted: {total}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
