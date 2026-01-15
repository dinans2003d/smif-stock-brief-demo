import sqlite3
from pathlib import Path
from datetime import date
import pandas as pd

# Project root = parent of /scripts
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "prices.db"
HOLDINGS_CSV = BASE_DIR / "data" / "holdings.csv"

EXPECTED_COLS = [
    "Ticker", "Description", "Sector", "Allocation",
    "Quantity", "AvgCost", "MarketPrice", "MarketValue"
]

def ensure_holdings_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS holdings (
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            description TEXT,
            sector TEXT,
            allocation TEXT,
            quantity REAL,
            avg_cost REAL,
            market_price REAL,
            market_value REAL,
            PRIMARY KEY (asof_date, ticker)
        )
        """
    )
    conn.commit()

def load_and_clean_holdings() -> pd.DataFrame:
    if not HOLDINGS_CSV.exists():
        raise FileNotFoundError(f"Missing file: {HOLDINGS_CSV}")

    df = pd.read_csv(HOLDINGS_CSV)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"holdings.csv missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df[EXPECTED_COLS].copy()

    # Normalize ticker
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df[df["Ticker"].str.len() > 0]

    # Normalize text fields
    for col in ["Description", "Sector", "Allocation"]:
        df[col] = df[col].astype(str).str.strip()

    # Coerce numeric columns
    for col in ["Quantity", "AvgCost", "MarketPrice", "MarketValue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing required numeric fields
    df = df.dropna(subset=["Quantity", "MarketPrice", "MarketValue"])

    # Remove duplicate tickers (keep last)
    df = df.drop_duplicates(subset=["Ticker"], keep="last")

    # Sanity check: MarketValue ≈ Quantity * MarketPrice
    df["calc_mv"] = (df["Quantity"] * df["MarketPrice"]).round(2)
    df["mv"] = df["MarketValue"].round(2)
    bad = df[df["calc_mv"] != df["mv"]]
    if len(bad) > 0:
        sample = bad[["Ticker", "MarketValue", "calc_mv"]].head(10).to_string(index=False)
        raise ValueError(
            "MarketValue mismatch for some rows (MarketValue != Quantity*MarketPrice). Fix holdings.csv.\n"
            f"Sample:\n{sample}"
        )

    return df

def upsert_holdings(conn: sqlite3.Connection, df: pd.DataFrame, asof: str) -> int:
    rows = []
    for r in df.itertuples(index=False):
        rows.append((
            asof,
            r.Ticker,
            r.Description,
            r.Sector,
            r.Allocation,
            float(r.Quantity),
            float(r.AvgCost) if pd.notna(r.AvgCost) else None,
            float(r.MarketPrice) if pd.notna(r.MarketPrice) else None,
            float(r.MarketValue) if pd.notna(r.MarketValue) else None,
        ))

    conn.executemany(
        """
        INSERT INTO holdings (
            asof_date, ticker, description, sector, allocation,
            quantity, avg_cost, market_price, market_value
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(asof_date, ticker) DO UPDATE SET
            description=excluded.description,
            sector=excluded.sector,
            allocation=excluded.allocation,
            quantity=excluded.quantity,
            avg_cost=excluded.avg_cost,
            market_price=excluded.market_price,
            market_value=excluded.market_value
        """,
        rows,
    )
    conn.commit()
    return len(rows)

def main():
    df = load_and_clean_holdings()

    # Ensure /data exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        ensure_holdings_table(conn)
        asof = date.today().strftime("%Y-%m-%d")
        n = upsert_holdings(conn, df, asof)
        print(f"Loaded holdings rows: {n}")
        print(f"Tickers loaded: {df['Ticker'].tolist()}")
        print(f"DB saved at: {DB_PATH}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()

