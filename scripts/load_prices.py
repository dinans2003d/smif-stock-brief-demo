import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# Project root = parent of /scripts
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "prices.db"

LOOKBACK_DAYS = 120
BENCHMARK = "SPY"


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            close REAL,
            volume REAL,
            PRIMARY KEY (asof_date, ticker)
        )
        """
    )
    conn.commit()


def get_holdings_tickers(conn: sqlite3.Connection) -> list[str]:
    """
    Pulls tickers from the holdings table loaded by load_holdings.py.
    """
    rows = conn.execute("SELECT DISTINCT ticker FROM holdings ORDER BY ticker").fetchall()
    return [r[0].upper() for r in rows if r and r[0]]


def prune_prices_to_holdings(conn: sqlite3.Connection, allowed: list[str]) -> int:
    """
    Deletes any prices rows for tickers NOT in allowed.
    """
    if not allowed:
        return 0

    allowed = sorted(set(t.upper() for t in allowed))
    placeholders = ",".join(["?"] * len(allowed))
    cur = conn.execute(f"DELETE FROM prices WHERE UPPER(ticker) NOT IN ({placeholders})", tuple(allowed))
    conn.commit()
    return cur.rowcount


def fetch_prices(ticker: str, start_date: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date,
        progress=False,
        auto_adjust=False,
        actions=False,
        interval="1d",
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["asof_date", "ticker", "close", "volume"])

    df = df.reset_index()

    # yfinance sometimes returns multi-index columns; normalize if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={"Date": "asof_date", "Close": "close", "Volume": "volume"})
    df["ticker"] = ticker.upper()
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.strftime("%Y-%m-%d")

    return df[["asof_date", "ticker", "close", "volume"]]


def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    rows = list(df.itertuples(index=False, name=None))

    conn.executemany(
        """
        INSERT INTO prices (asof_date, ticker, close, volume)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(asof_date, ticker) DO UPDATE SET
            close=excluded.close,
            volume=excluded.volume
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def main() -> None:
    start_date = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_tables(conn)

        tickers = get_holdings_tickers(conn)
        if not tickers:
            raise RuntimeError("Holdings table is empty. Run: python scripts/load_holdings.py")

        # Always include SPY
        allowed = list(tickers)
        if BENCHMARK not in allowed:
            allowed.append(BENCHMARK)

        # ✅ One-time cleanup every run
        deleted = prune_prices_to_holdings(conn, allowed)
        print(f"Pruned prices rows (not in holdings): {deleted}")

        # ✅ Only load what holdings says (plus SPY)
        total_rows = 0
        for ticker in allowed:
            df = fetch_prices(ticker, start_date)
            count = upsert_prices(conn, df)
            print(f"{ticker}: loaded {count} rows")
            total_rows += count

        print(f"Total rows loaded: {total_rows}")
        print(f"Database saved at: {DB_PATH}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
