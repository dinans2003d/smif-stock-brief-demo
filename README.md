# SMIF Daily Stock Brief — Equity Portfolio Intelligence Engine

A data-engineering + analytics project that ingests market prices, holdings, SEC filings, earnings context, and real-time company news into a normalized SQLite datastore, then serves an analyst-facing Streamlit brief for daily monitoring and research prioritization.

## What this does

### Portfolio Overview
- Loads holdings and latest prices from SQLite
- Computes market value and weights
- Visualizes sector allocation
- Displays top holdings by weight

### Single Stock Brief (per ticker)
- **Daily signal + explainability** (“What happened today?”)
  - Benchmark-relative return vs SPY
  - 1D price move
  - Volume anomaly vs rolling average
  - Rolling volatility percentile
  - News flow count (DB-backed)
- **Earnings**
  - Next earnings date (yfinance, anchored to as-of date)
  - EPS estimate vs reported EPS (beat/miss/inline)
- **SEC filings**
  - Latest 10-Q / 10-K / 8-K links via SEC EDGAR
- **Financial snapshot**
  - Market cap, P/E, margins, debt/cash metrics (best-effort via yfinance)
- **News snapshot**
  - DB-backed news pulled from Finnhub
  - Filter window: Today / Last 3 days / Last 7 days
  - Optional “Preferred sources only” filter (WSJ/FT/Bloomberg/CNBC when available)

---

## Architecture

**Data store**
- `data/prices.db` (SQLite) is the single source of truth.
- Tables:
  - `holdings` (loaded from `data/holdings.csv`)
  - `prices` (daily OHLC close + volume)
  - `news` (headline-level news items)

**ETL scripts**
- `scripts/load_holdings.py` — loads holdings CSV into SQLite
- `scripts/load_prices.py` — pulls historical daily prices from yfinance and upserts into SQLite
- `scripts/load_news.py` — pulls company news from Finnhub and upserts/deduplicates into SQLite

**App**
- `app.py` — Streamlit dashboard for Portfolio Overview + Single Stock Brief

---

## Setup

### 1) Create and activate a virtual environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

### 2) Install dependencies
```bash
pip install -r requirements.txt


### 2) Install dependencies
pip install -r requirements.txt

### 3) Set Finnhub API key (required for news)

Set a user-level environment variable (Windows / PowerShell):

setx FINNHUB_API_KEY "YOUR_FINNHUB_KEY"

Close and reopen your terminal (or restart VS Code) so the variable is available.

Verify:

echo $env:FINNHUB_API_KEY

---

## Run the pipeline (required order)

From the project root:

1) Load holdings  
python scripts/load_holdings.py

2) Load prices  
python scripts/load_prices.py

3) Load news (Finnhub)  
python scripts/load_news.py

Optional sanity check: confirm tables exist

python -c "import sqlite3; c=sqlite3.connect('data/prices.db'); print(c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()); c.close()"

---

## Run the Streamlit app

streamlit run app.py

Open the local URL shown in the terminal (typically http://localhost:8501).

---

## Notes / Limitations

- News source availability depends on your provider and licensing. The “Preferred sources only” toggle filters only if those sources exist in the feed.
- Some symbols (mutual funds / ETFs) may return 403 Forbidden from Finnhub company-news; the loader continues and logs the error.
- yfinance fields (market cap, margins, earnings dates) are best-effort and may be missing for certain tickers.
- SQLite is used for fast iteration and reproducibility; swapping to Postgres is straightforward if needed.

---

## Project highlights (for recruiters)

- Built an end-to-end data pipeline ingesting market and event data into a normalized analytical store (SQLite).
- Implemented fault-tolerant API ingestion with upserts and deduplication.
- Engineered analyst-oriented market signals (relative returns, volatility percentile, volume anomaly) with explainable “why today” output.
- Delivered a daily research brief UI to support portfolio monitoring workflows.

---

## Repo structure

smif-stock-brief-demo/
  app.py
  requirements.txt
  README.md
  data/
    holdings.csv
    prices.db
  scripts/
    load_holdings.py
    load_prices.py
    load_news.py
    check_db.py

---

## requirements.txt

pandas  
numpy  
requests  
streamlit  
yfinance  

If you want locked versions (recommended for sharing):

pip freeze > requirements.txt

---

## SEC User-Agent (recommended)

The SEC expects a real contact email in the User-Agent header.

Replace:

headers = {"User-Agent": "SMIF Dashboard (contact: your_email@domain.com)"}

With:

headers = {"User-Agent": "SMIF Dashboard (contact: dinan@email.com)"}

---

## License / Disclaimer

This project is for educational and research purposes. Market data and news are provided by third-party services and subject to their respective terms.


