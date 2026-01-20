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
    """Pull tickers from the holdings table loaded by load_holdings.py."""
    try:
        rows = conn.execute("SELECT DISTINCT ticker FROM holdings ORDER BY ticker").fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def prune_prices_to_holdings(conn: sqlite3.Connection) -> int:
    """
    Delete any prices rows for tickers not present in holdings.
    Keeps BENCHMARK too.
    """
    allowed = set(get_holdings_tickers(conn))
    allowed.add(BENCHMARK)

    if not allowed:
        return 0

    placeholders = ",".join(["?"] * len(allowed))
    sql = f"DELETE FROM prices WHERE ticker NOT IN ({placeholders})"
    cur = conn.execute(sql, tuple(sorted(allowed)))
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
    df["ticker"] = ticker
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

    # Ensure /data exists (sqlite will not create folders)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_tables(conn)

        # Source of truth = holdings table
        tickers = get_holdings_tickers(conn)

        if not tickers:
            raise RuntimeError(
                "No holdings tickers found. Run: python scripts/load_holdings.py first."
            )

        # Ensure benchmark is present
        if BENCHMARK not in tickers:
            tickers.append(BENCHMARK)

        # ✅ Remove any old tickers not in holdings (plus benchmark)
        deleted = prune_prices_to_holdings(conn)
        print(f"Pruned prices rows (not in holdings): {deleted}")

        total_rows = 0
        for ticker in tickers:
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
