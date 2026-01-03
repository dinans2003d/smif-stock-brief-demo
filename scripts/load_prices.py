import sqlite3
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

DB_PATH = "data/prices.db"

# Demo tickers – you can change these later
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

LOOKBACK_DAYS = 120


def ensure_tables(conn):
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


def fetch_prices(ticker, start_date):
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
    df = df.rename(
        columns={
            "Date": "asof_date",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df["ticker"] = ticker
    df["asof_date"] = df["asof_date"].dt.strftime("%Y-%m-%d")

    return df[["asof_date", "ticker", "close", "volume"]]


def upsert_prices(conn, df):
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


def main():
    start_date = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_tables(conn)

        total_rows = 0
        for ticker in TICKERS:
            df = fetch_prices(ticker, start_date)
            count = upsert_prices(conn, df)
            print(f"{ticker}: loaded {count} rows")
            total_rows += count

        print(f"Total rows loaded: {total_rows}")
        print(f"Database saved at {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

