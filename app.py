import requests
import sqlite3
import pandas as pd
import streamlit as st

# IMPORTANT: your loader uses data/prices.db
DB_PATH = "data/prices.db"

BENCHMARK = "SPY"  # benchmark ticker
VOL_LOOKBACK = 30
VOLATILITY_WINDOW = 20

st.set_page_config(page_title="SMIF Daily Stock Brief", layout="wide")

# ---------- View Switch ----------
mode = st.sidebar.radio("View", ["Portfolio Overview", "Single Stock Brief"])


# ---------- Data (shared helpers) ----------
@st.cache_data
def load_holdings() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM holdings", conn)
    conn.close()
    if df.empty:
        return df
    for col in ["quantity", "avg_cost", "market_price", "market_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


@st.cache_data
def load_latest_prices() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT p.ticker, p.asof_date, p.close, p.volume
        FROM prices p
        INNER JOIN (
            SELECT ticker, MAX(asof_date) AS max_date
            FROM prices
            GROUP BY ticker
        ) m
        ON p.ticker = m.ticker AND p.asof_date = m.max_date
        """,
        conn,
    )
    conn.close()
    if df.empty:
        return df
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


# ---------- Existing single-ticker loader ----------
@st.cache_data
def load_prices(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT asof_date, ticker, close, volume
        FROM prices
        WHERE ticker = ?
        ORDER BY asof_date
        """,
        conn,
        params=(ticker,),
    )
    conn.close()

    if df.empty:
        return df

    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce")
    df = df.dropna(subset=["asof_date"]).sort_values("asof_date")
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1d"] = df["close"].pct_change()
    return df


def rolling_volatility(df: pd.DataFrame, window: int = VOLATILITY_WINDOW) -> pd.Series:
    return df["ret_1d"].rolling(window).std() * (252 ** 0.5)


def safe_last(series: pd.Series):
    series = series.dropna()
    return series.iloc[-1] if len(series) else None


def calc_attention_score(
    rel_ret_1d: float | None,
    vol_anomaly: float | None,
    vol_pctile: float | None,
    news_count: int,
) -> tuple[int, str]:
    score = 0.0

    if vol_anomaly is not None:
        vol_component = min(max(vol_anomaly - 1.0, 0.0) / 2.0, 1.0)
        score += 30 * vol_component

    if vol_pctile is not None:
        score += 30 * vol_pctile

    if rel_ret_1d is not None:
        move = min(abs(rel_ret_1d) / 0.03, 1.0)
        score += 20 * move

    news_component = min(news_count / 5.0, 1.0)
    score += 20 * news_component

    score_int = int(round(min(max(score, 0), 100)))

    if score_int >= 75:
        label = "High Attention"
    elif score_int >= 50:
        label = "Elevated"
    else:
        label = "Normal"

    return score_int, label


@st.cache_data
def load_news_csv(path="data/news.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def get_news_for_ticker(news: pd.DataFrame, ticker: str, asof_date: pd.Timestamp) -> pd.DataFrame:
    if news.empty:
        return news

    df = news.copy()

    if "ticker" in df.columns:
        df = df[df["ticker"].astype(str).str.upper() == ticker.upper()]

    if "date" in df.columns and pd.notna(asof_date):
        day_start = asof_date.normalize()
        day_end = day_start + pd.Timedelta(days=1)
        df = df[(df["date"] >= day_start) & (df["date"] < day_end)]

    return df.head(5)


def fmt_pct(x: float | None) -> str:
    return "N/A" if x is None else f"{x:+.2%}"


def fmt_mult(x: float | None) -> str:
    return "N/A" if x is None else f"{x:.2f}×"


# ---------- SEC / Earnings / Financial helpers ----------
@st.cache_data
def get_cik_for_ticker(ticker: str) -> str | None:
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "SMIF Dashboard (contact: your_email@domain.com)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    t = ticker.upper()
    for _, row in data.items():
        if str(row.get("ticker", "")).upper() == t:
            return str(row.get("cik_str", "")).zfill(10)
    return None


@st.cache_data
def get_latest_filings(ticker: str, forms=("10-Q", "10-K", "8-K"), limit: int = 8) -> pd.DataFrame:
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return pd.DataFrame()

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "SMIF Dashboard (contact: your_email@domain.com)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    j = r.json()

    recent = j.get("filings", {}).get("recent", {})
    df = pd.DataFrame(recent)
    if df.empty:
        return df

    df = df[["form", "filingDate", "accessionNumber", "primaryDocument"]].copy()
    df = df[df["form"].isin(forms)].head(limit)

    df["accession_nodash"] = df["accessionNumber"].astype(str).str.replace("-", "", regex=False)
    df["filing_url"] = (
        "https://www.sec.gov/Archives/edgar/data/"
        + str(int(cik)) + "/"
        + df["accession_nodash"] + "/"
        + df["primaryDocument"].astype(str)
    )
    return df


def get_next_earnings_date_yf(ticker: str) -> str:
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None or cal.empty:
            return "N/A"
        for key in ["Earnings Date", "EarningsDate"]:
            if key in cal.index:
                val = cal.loc[key].values
                if len(val) > 0:
                    return str(val[0])
        return str(cal.iloc[0, 0])
    except Exception:
        return "N/A"


# =========================
# PORTFOLIO OVERVIEW
# =========================
if mode == "Portfolio Overview":
    st.title("SMIF Portfolio Overview")

    holdings = load_holdings()
    latest = load_latest_prices()

    if holdings.empty:
        st.error("No holdings found. Run python scripts/load_holdings.py first.")
        st.stop()

    if latest.empty:
        st.error("No price data found. Run python scripts/load_prices.py first.")
        st.stop()

    dfp = holdings.merge(
        latest[["ticker", "asof_date", "close"]],
        on="ticker",
        how="left",
        suffixes=("_hold", "_price"),
    )

    dfp["asof_date_price"] = pd.to_datetime(dfp["asof_date_price"], errors="coerce")
    dfp["quantity"] = pd.to_numeric(dfp["quantity"], errors="coerce")
    dfp["close"] = pd.to_numeric(dfp["close"], errors="coerce")
    dfp["market_value_calc"] = dfp["quantity"] * dfp["close"]

    dfp_ok = dfp.dropna(subset=["market_value_calc"]).copy()

    total_mv = float(dfp_ok["market_value_calc"].sum())
    dfp_ok["weight"] = dfp_ok["market_value_calc"] / total_mv

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Portfolio Value", f"${total_mv:,.0f}")
    c2.metric("Holdings", int(dfp_ok["ticker"].nunique()))
    largest = dfp_ok.loc[dfp_ok["weight"].idxmax(), "ticker"]
    c3.metric("Largest Position", largest)
    asof_latest = dfp_ok["asof_date_price"].max()
    c4.metric("Prices As Of", asof_latest.date().isoformat() if pd.notna(asof_latest) else "N/A")

    st.subheader("Sector Allocation (Market Value)")
    sector_mv = dfp_ok.groupby("sector")["market_value_calc"].sum().sort_values(ascending=False)
    st.bar_chart(sector_mv)

    st.subheader("Top Holdings by Weight")
    top = (
        dfp_ok[["ticker", "sector", "allocation", "quantity", "close", "market_value_calc", "weight"]]
        .sort_values("weight", ascending=False)
        .head(15)
        .copy()
    )
    top["weight"] = (top["weight"] * 100).round(2)
    top = top.rename(columns={"close": "latest_close", "market_value_calc": "market_value", "weight": "weight_%"})
    st.dataframe(top, use_container_width=True)

    with st.expander("Show full holdings"):
        full = dfp_ok[
            ["ticker", "description", "sector", "allocation", "quantity", "close", "market_value_calc", "weight"]
        ].copy()
        full["weight"] = (full["weight"] * 100).round(4)
        full = full.rename(columns={"close": "latest_close", "market_value_calc": "market_value", "weight": "weight_%"})
        st.dataframe(full.sort_values("market_value", ascending=False), use_container_width=True)

    st.stop()


# =========================
# SINGLE STOCK BRIEF
# =========================
st.title("SMIF Daily Stock Brief")

# Pull tickers from DB (prices table)
try:
    conn = sqlite3.connect(DB_PATH)
    tickers_db = pd.read_sql_query(
        "SELECT DISTINCT ticker FROM prices ORDER BY ticker", conn
    )["ticker"].tolist()
    conn.close()
    tickers = tickers_db
except Exception:
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"]

# keep SPY out of dropdown
tickers_dropdown = [t for t in tickers if t != BENCHMARK]
ticker = st.selectbox("Select Ticker", tickers_dropdown, index=min(0, len(tickers_dropdown) - 1))

df = load_prices(ticker)
if df.empty:
    st.error(f"No price data found for {ticker}. Run the loader script first.")
    st.stop()

df = add_returns(df)

# Benchmark
spy = load_prices(BENCHMARK)
if spy.empty:
    spy = pd.DataFrame(columns=["asof_date", "close", "ret_1d"])
else:
    spy = add_returns(spy)
    if "ret_1d" not in spy.columns:
        spy["ret_1d"] = pd.NA

# Merge on DATE key
df = df.copy()
spy = spy.copy()
df["asof_key"] = pd.to_datetime(df["asof_date"]).dt.date
spy["asof_key"] = pd.to_datetime(spy["asof_date"]).dt.date

merged = df[["asof_date", "asof_key", "close", "volume", "ret_1d"]].merge(
    spy[["asof_key", "close", "ret_1d"]].rename(columns={"close": "spy_close", "ret_1d": "spy_ret_1d"}),
    on="asof_key",
    how="left",
)

with st.expander("Data diagnostics"):
    st.caption(f"Benchmark overlap rows: {len(merged.dropna(subset=['spy_ret_1d']))} / {len(merged)}")

latest_row = merged.dropna(subset=["close"]).iloc[-1]
asof_date = pd.to_datetime(latest_row["asof_date"])

latest_close = float(latest_row["close"])
latest_volume = float(latest_row["volume"]) if pd.notna(latest_row["volume"]) else None

ret_1d = float(latest_row["ret_1d"]) if pd.notna(latest_row["ret_1d"]) else None
spy_ret_1d = float(latest_row["spy_ret_1d"]) if pd.notna(latest_row.get("spy_ret_1d")) else None
rel_ret_1d = (ret_1d - spy_ret_1d) if (ret_1d is not None and spy_ret_1d is not None) else None

# Volume anomaly vs 30-day average
vol_avg = merged["volume"].tail(VOL_LOOKBACK).mean() if "volume" in merged.columns else None
vol_anomaly = (latest_volume / vol_avg) if (latest_volume is not None and vol_avg and vol_avg > 0) else None

# Volatility + percentile
merged["volatility"] = rolling_volatility(merged, VOLATILITY_WINDOW)
vol_now = safe_last(merged["volatility"])

vol_pctile = None
vol_hist = merged["volatility"].dropna()
if vol_now is not None and len(vol_hist) >= 10:
    vol_pctile = float((vol_hist <= vol_now).mean())  # 0..1 percentile

# News (CSV)
news = load_news_csv("data/news.csv")
news_today = get_news_for_ticker(news, ticker, asof_date)
news_count = len(news_today) if not news_today.empty else 0

score, label = calc_attention_score(rel_ret_1d, vol_anomaly, vol_pctile, news_count)

# Why today bullets
why = []
if rel_ret_1d is not None:
    why.append(f"Relative move vs {BENCHMARK}: **{rel_ret_1d:+.2%}**")
if ret_1d is not None:
    why.append(f"1D price move: **{ret_1d:+.2%}**")

if vol_anomaly is not None:
    if vol_anomaly >= 1.5:
        why.append(f"Unusual volume: **{vol_anomaly:.2f}×** vs {VOL_LOOKBACK}D avg")
    elif vol_anomaly <= 0.7:
        why.append(f"Low volume: **{vol_anomaly:.2f}×** vs {VOL_LOOKBACK}D avg")
    else:
        why.append(f"Volume near normal: **{vol_anomaly:.2f}×** vs {VOL_LOOKBACK}D avg")

if vol_pctile is not None:
    if vol_pctile >= 0.8:
        why.append(f"Volatility elevated: **{vol_pctile*100:.0f}th percentile** (rolling {VOLATILITY_WINDOW}D)")
    elif vol_pctile <= 0.2:
        why.append(f"Volatility muted: **{vol_pctile*100:.0f}th percentile** (rolling {VOLATILITY_WINDOW}D)")
    else:
        why.append(f"Volatility mid-range: **{vol_pctile*100:.0f}th percentile** (rolling {VOLATILITY_WINDOW}D)")

if news_count > 0:
    why.append(f"News flow: **{news_count}** headline(s) in your CSV today")
else:
    why.append("News flow: **none found** (in your CSV today)")

why = why[:5]

# Header + banner (no "Decision")
st.subheader(f"{ticker} — Daily Brief ({asof_date.date()})")
st.success(f"Signal: **{label}**  |  Attention Score: **{score}/100**")

st.markdown("### Why today?")
st.markdown("\n".join([f"- {x}" for x in why]))

# KPI cards (no "Decision")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Attention", label, f"{score}/100")
c2.metric("Latest Close", f"${latest_close:,.2f}", fmt_pct(ret_1d) if ret_1d is not None else None)
c3.metric(f"Rel Return vs {BENCHMARK} (1D)", fmt_pct(rel_ret_1d) if rel_ret_1d is not None else "N/A")
c4.metric("Volume Anomaly", fmt_mult(vol_anomaly))

# ---- Replace charts with objective docs + earnings + financial snapshot ----
st.subheader("Company Docs (Objective) — SEC Filings")
filings = get_latest_filings(ticker)
if filings.empty:
    st.caption("No SEC filings found (CIK lookup failed or SEC limited access).")
else:
    for _, r in filings.iterrows():
        st.markdown(f"- **{r['form']}** ({r['filingDate']}) — [SEC Filing]({r['filing_url']})")

st.subheader("Earnings")
st.write(f"Next earnings (best-effort): **{get_next_earnings_date_yf(ticker)}**")

st.subheader("Financial Snapshot (best-effort)")
try:
    import yfinance as yf
    info = yf.Ticker(ticker).info or {}

    a, b, c, d = st.columns(4)
    a.metric("Market Cap", f"${info.get('marketCap', 0):,}" if info.get("marketCap") else "N/A")
    b.metric("Trailing P/E", f"{info.get('trailingPE'):.2f}" if info.get("trailingPE") else "N/A")
    c.metric("Forward P/E", f"{info.get('forwardPE'):.2f}" if info.get("forwardPE") else "N/A")
    d.metric("Dividend Yield", f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "N/A")

    fundamentals = {
        "Revenue (TTM)": info.get("totalRevenue"),
        "Gross Margin": info.get("grossMargins"),
        "Operating Margin": info.get("operatingMargins"),
        "Profit Margin": info.get("profitMargins"),
        "Free Cashflow": info.get("freeCashflow"),
        "EBITDA": info.get("ebitda"),
        "Total Cash": info.get("totalCash"),
        "Total Debt": info.get("totalDebt"),
    }
    fdf = pd.DataFrame([{"Metric": k, "Value": v} for k, v in fundamentals.items()])
    st.dataframe(fdf, use_container_width=True)

except Exception:
    st.caption("Financial snapshot unavailable for this ticker via yfinance.")

st.subheader("News Snapshot")
if news_today.empty:
    st.caption("No news rows found for this ticker/date in data/news.csv (or CSV missing expected columns).")
else:
    for _, r in news_today.iterrows():
        title = str(r.get("title", "Untitled"))
        source = str(r.get("source", ""))
        url = r.get("url", "")
        if isinstance(url, str) and url.strip():
            st.markdown(f"- [{title}]({url}) — *{source}*")
        else:
            st.markdown(f"- {title} — *{source}*")
