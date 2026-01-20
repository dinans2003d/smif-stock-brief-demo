import sqlite3
import requests
import pandas as pd
import streamlit as st
import yfinance as yf

# =========================
# CONFIG
# =========================
DB_PATH = "data/prices.db"
BENCHMARK = "SPY"
VOL_LOOKBACK = 30
VOLATILITY_WINDOW = 20

st.set_page_config(page_title="SMIF Daily Stock Brief", layout="wide")
mode = st.sidebar.radio("View", ["Portfolio Overview", "Single Stock Brief"])


# =========================
# DB HELPERS
# =========================
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


@st.cache_data
def get_ticker_meta() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT h.ticker, h.description, h.sector, h.allocation, h.asof_date
        FROM holdings h
        INNER JOIN (
            SELECT ticker, MAX(asof_date) AS max_date
            FROM holdings
            GROUP BY ticker
        ) m
        ON h.ticker = m.ticker AND h.asof_date = m.max_date
        """,
        conn,
    )
    conn.close()
    if df.empty:
        return df
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for col in ["description", "sector", "allocation"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


# =========================
# NEWS (DB-backed)
# =========================
@st.cache_data
def load_news_from_db(ticker: str, days: int = 7) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            """
            SELECT published_at, ticker, title, source, url, summary
            FROM news
            WHERE UPPER(ticker) = UPPER(?)
              AND published_at >= datetime('now', ?)
            ORDER BY published_at DESC
            LIMIT 50
            """,
            conn,
            params=(ticker, f"-{days} days"),
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    return df


# =========================
# MARKET CALCS
# =========================
def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1d"] = df["close"].pct_change()
    return df


def rolling_volatility(df: pd.DataFrame, window: int = VOLATILITY_WINDOW) -> pd.Series:
    return df["ret_1d"].rolling(window).std() * (252 ** 0.5)


def safe_last(series: pd.Series):
    s = series.dropna()
    return s.iloc[-1] if len(s) else None


def calc_attention_score(rel_ret_1d, vol_anomaly, vol_pctile, news_count) -> tuple[int, str]:
    score = 0.0
    if vol_anomaly is not None:
        score += 30 * min(max(vol_anomaly - 1.0, 0.0) / 2.0, 1.0)
    if vol_pctile is not None:
        score += 30 * vol_pctile
    if rel_ret_1d is not None:
        score += 20 * min(abs(rel_ret_1d) / 0.03, 1.0)
    score += 20 * min(news_count / 5.0, 1.0)

    score_int = int(round(min(max(score, 0), 100)))
    if score_int >= 75:
        return score_int, "High Attention"
    if score_int >= 50:
        return score_int, "Elevated"
    return score_int, "Normal"


def fmt_pct(x: float | None) -> str:
    return "N/A" if x is None else f"{x:+.2%}"


def fmt_mult(x: float | None) -> str:
    return "N/A" if x is None else f"{x:.2f}×"


# =========================
# SEC + YFINANCE HELPERS
# =========================
@st.cache_data(ttl=3600)
def yf_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def get_next_earnings_date_yf(ticker: str, anchor_date: str | None = None) -> str:
    """
    Returns next earnings date (estimated), anchored to the brief's as-of date.
    anchor_date: 'YYYY-MM-DD' (pass your asof_date from prices)
    """
    try:
        tk = yf.Ticker(ticker)

        anchor = pd.to_datetime(anchor_date, errors="coerce")
        if pd.isna(anchor):
            anchor = pd.Timestamp.today()
        anchor = anchor.normalize()

        # 1) Try earnings dates (if available)
        if hasattr(tk, "get_earnings_dates"):
            ed = tk.get_earnings_dates(limit=16)
            if isinstance(ed, pd.DataFrame) and not ed.empty:
                dates = pd.to_datetime(ed.index, errors="coerce").dropna().sort_values()
                future = dates[dates >= anchor]
                if len(future):
                    return future.iloc[0].strftime("%Y-%m-%d")

        # 2) Calendar fallback (dict or DataFrame)
        cal = tk.calendar

        if isinstance(cal, dict):
            val = cal.get("Earnings Date") or cal.get("EarningsDate")
            if val is None:
                return "N/A"
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            dt = pd.to_datetime(val, errors="coerce")
            return dt.strftime("%Y-%m-%d") if pd.notna(dt) else "N/A"

        if hasattr(cal, "empty") and not cal.empty:
            for key in ["Earnings Date", "EarningsDate"]:
                if key in cal.index:
                    vals = cal.loc[key].values
                    if len(vals) > 0:
                        dt = pd.to_datetime(vals[0], errors="coerce")
                        return dt.strftime("%Y-%m-%d") if pd.notna(dt) else "N/A"
            dt = pd.to_datetime(cal.iloc[0, 0], errors="coerce")
            return dt.strftime("%Y-%m-%d") if pd.notna(dt) else "N/A"

        return "N/A"
    except Exception:
        return "N/A"


@st.cache_data(ttl=3600)
def earnings_eps_series(ticker: str) -> pd.DataFrame:
    """
    Pull EPS estimate + reported EPS from yfinance earnings dates (if available).
    Returns last 5 rows.
    """
    try:
        tk = yf.Ticker(ticker)
        ed = (
            tk.get_earnings_dates(limit=12)
            if hasattr(tk, "get_earnings_dates")
            else getattr(tk, "earnings_dates", None)
        )
        if ed is None or not isinstance(ed, pd.DataFrame) or ed.empty:
            return pd.DataFrame()

        df = ed.reset_index()
        df.columns = [str(c).strip() for c in df.columns]

        date_col = next((c for c in df.columns if "Earnings Date" in c), df.columns[0])
        df = df.rename(columns={date_col: "earnings_date"})

        ren = {}
        for c in df.columns:
            lc = c.lower()
            if "eps estimate" in lc:
                ren[c] = "eps_estimate"
            if "reported eps" in lc:
                ren[c] = "reported_eps"
        df = df.rename(columns=ren)

        df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
        if "eps_estimate" in df.columns:
            df["eps_estimate"] = pd.to_numeric(df["eps_estimate"], errors="coerce")
        if "reported_eps" in df.columns:
            df["reported_eps"] = pd.to_numeric(df["reported_eps"], errors="coerce")

        df = df.dropna(subset=["earnings_date"]).sort_values("earnings_date").tail(5)
        df["period"] = df["earnings_date"].dt.strftime("%Y-%m-%d")

        cols = [c for c in ["period", "eps_estimate", "reported_eps"] if c in df.columns]
        return df[cols]
    except Exception:
        return pd.DataFrame()


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
        + str(int(cik))
        + "/"
        + df["accession_nodash"]
        + "/"
        + df["primaryDocument"].astype(str)
    )
    return df


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

try:
    conn = sqlite3.connect(DB_PATH)
    tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM holdings ORDER BY ticker", conn)["ticker"].tolist()
    conn.close()
except Exception:
    tickers = []

tickers_dropdown = [t for t in tickers if t != BENCHMARK]
ticker = st.selectbox("Select Ticker", tickers_dropdown, index=min(0, len(tickers_dropdown) - 1))

# meta lookup
meta = get_ticker_meta()
display_name = ticker
sector_name = None
allocation_name = None
if not meta.empty:
    row = meta.loc[meta["ticker"] == str(ticker).upper()]
    if not row.empty:
        desc = row.iloc[0].get("description")
        sec = row.iloc[0].get("sector")
        alloc = row.iloc[0].get("allocation")
        if isinstance(desc, str) and desc.strip() and desc.lower() != "nan":
            display_name = desc.strip()
        if isinstance(sec, str) and sec.strip() and sec.lower() != "nan":
            sector_name = sec.strip()
        if isinstance(alloc, str) and alloc.strip() and alloc.lower() != "nan":
            allocation_name = alloc.strip()

# prices + returns
df = load_prices(ticker)
if df.empty:
    st.error(f"No price data found for {ticker}. Run the loader script first.")
    st.stop()
df = add_returns(df)

spy = load_prices(BENCHMARK)
spy = add_returns(spy) if not spy.empty else pd.DataFrame(columns=["asof_date", "close", "ret_1d"])

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

vol_avg = merged["volume"].tail(VOL_LOOKBACK).mean() if "volume" in merged.columns else None
vol_anomaly = (latest_volume / vol_avg) if (latest_volume is not None and vol_avg and vol_avg > 0) else None

merged["volatility"] = rolling_volatility(merged, VOLATILITY_WINDOW)
vol_now = safe_last(merged["volatility"])

vol_pctile = None
vol_hist = merged["volatility"].dropna()
if vol_now is not None and len(vol_hist) >= 10:
    vol_pctile = float((vol_hist <= vol_now).mean())

# ✅ news count from DB (last 7 days)
news_df = load_news_from_db(ticker, days=7)
news_count = len(news_df) if not news_df.empty else 0

score, label = calc_attention_score(rel_ret_1d, vol_anomaly, vol_pctile, news_count)

# why bullets
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

why.append(
    f"News flow: **{min(news_count, 50)}** headline(s) in DB (last 7 days)"
    if news_count > 0
    else "News flow: **none found** (in DB, last 7 days)"
)
why = why[:5]

# header
subtitle_parts = [ticker]
if sector_name:
    subtitle_parts.append(sector_name)
if allocation_name:
    subtitle_parts.append(allocation_name)

st.markdown(f"## {display_name}")
st.caption(" • ".join(subtitle_parts) + f"  |  Daily Brief ({asof_date.date().isoformat()})")
st.success(f"Signal: **{label}**  |  Attention Score: **{score}/100**")

# =========================
# EARNINGS (optimized for analysts)
# =========================
st.subheader("Earnings")

SHOW_EARNINGS_DEBUG = False  # set True when you're debugging yfinance issues

if SHOW_EARNINGS_DEBUG:
    with st.expander("Earnings diagnostics (debug)"):
        try:
            tk_dbg = yf.Ticker(ticker)
            st.write("quoteType:", (tk_dbg.info or {}).get("quoteType", "N/A"))
            cal = tk_dbg.calendar
            st.write("calendar type:", type(cal).__name__)
            st.write("calendar raw:", cal)

            ed = tk_dbg.get_earnings_dates(limit=12) if hasattr(tk_dbg, "get_earnings_dates") else getattr(tk_dbg, "earnings_dates", None)
            st.write("earnings_dates type:", type(ed).__name__)
            if isinstance(ed, pd.DataFrame):
                st.write("earnings_dates shape:", ed.shape)
                st.dataframe(ed.head(10), use_container_width=True)
            else:
                st.write("earnings_dates raw:", ed)
        except Exception as e:
            st.error(f"Diagnostics error: {e}")

info = yf_info(ticker)
quote_type = str(info.get("quoteType", "")).upper()

if quote_type != "EQUITY":
    st.caption("Earnings not applicable for ETFs or funds.")
else:
    eps_df = earnings_eps_series(ticker)

    anchor_str = asof_date.strftime("%Y-%m-%d")
    next_date = get_next_earnings_date_yf(ticker, anchor_str)

    upcoming = pd.DataFrame()
    if not eps_df.empty:
        upcoming = eps_df[eps_df["reported_eps"].isna()].copy()

    if next_date == "N/A" and not upcoming.empty:
        next_date = str(upcoming.sort_values("period").iloc[0]["period"])

    st.markdown("#### Upcoming earnings")
    cA, cB, cC = st.columns(3)
    cA.metric("Next earnings (estimated)", next_date if next_date != "N/A" else "N/A")

    if not upcoming.empty and "eps_estimate" in upcoming.columns:
        est_val = upcoming.sort_values("period").iloc[0].get("eps_estimate")
        cB.metric("Consensus EPS estimate", f"{est_val:.2f}" if pd.notna(est_val) else "N/A")
    else:
        cB.metric("Consensus EPS estimate", "N/A")

    cC.metric("Earnings type", "Operating company")

    st.markdown("#### Historical earnings (Estimate vs Actual)")

    if eps_df.empty:
        st.caption("Earnings details unavailable via yfinance for this ticker.")
    else:
        hist = eps_df[eps_df["reported_eps"].notna()].copy()

        if not hist.empty and "eps_estimate" in hist.columns and "reported_eps" in hist.columns:
            tol = 1e-6
            diff = hist["reported_eps"] - hist["eps_estimate"]
            hist["result"] = "Inline"
            hist.loc[diff > tol, "result"] = "Beat"
            hist.loc[diff < -tol, "result"] = "Miss"
        else:
            hist["result"] = ""

        hist = hist.sort_values("period", ascending=False)

        show_cols = ["period", "eps_estimate", "reported_eps", "result"]
        show_cols = [c for c in show_cols if c in hist.columns]
        st.dataframe(hist[show_cols], use_container_width=True, hide_index=True)

        chart = hist.sort_values("period").set_index("period")
        chart = chart[[c for c in ["eps_estimate", "reported_eps"] if c in chart.columns]].copy()
        st.scatter_chart(chart)

# Why today
st.markdown("### What happened today?")
st.markdown("\n".join([f"- {x}" for x in why]))

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Attention", label, f"{score}/100")
c2.metric("Latest Close", f"${latest_close:,.2f}", fmt_pct(ret_1d) if ret_1d is not None else None)
c3.metric(f"Rel Return vs {BENCHMARK} (1D)", fmt_pct(rel_ret_1d) if rel_ret_1d is not None else "N/A")
c4.metric("Volume Anomaly", fmt_mult(vol_anomaly))

# SEC Filings
st.subheader("Company Docs (Objective) — SEC Filings")
filings = get_latest_filings(ticker)
if filings.empty:
    st.caption("No SEC filings found (CIK lookup failed or SEC limited access).")
else:
    for _, r in filings.iterrows():
        st.markdown(f"- **{r['form']}** ({r['filingDate']}) — [SEC Filing]({r['filing_url']})")

# Financial snapshot
st.subheader("Financial Snapshot (best-effort)")
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

# News (DB)
st.subheader("News Snapshot")

# 1) Clear date-range selector (human readable)
range_label = st.radio(
    "Show headlines from:",
    ["Today", "Last 3 days", "Last 7 days"],
    index=2,
    horizontal=True,
)
days_map = {"Today": 1, "Last 3 days": 3, "Last 7 days": 7}
days = days_map[range_label]

# 2) Optional preferred-source filter (Option B)
preferred_only = st.toggle(
    "Preferred sources only (WSJ / FT / Bloomberg / CNBC)",
    value=False
)

# NOTE: provider source strings vary; include common variants
preferred_sources = {
    "CNBC",
    "Financial Times",
    "FT",
    "Bloomberg",
    "Bloomberg News",
    "The Wall Street Journal",
    "Wall Street Journal",
    "WSJ",
}

# 3) Load from DB
news_df_ui = load_news_from_db(ticker, days=days)

# Price move badges
move_badge = f"{ticker} {fmt_pct(ret_1d)}"
rel_badge = f"vs {BENCHMARK} {fmt_pct(rel_ret_1d)}" if rel_ret_1d is not None else f"vs {BENCHMARK} N/A"
st.caption(f"{move_badge}  •  {rel_badge}")

if news_df_ui.empty:
    st.caption("No recent news found in DB for this ticker/window. Run: python scripts/load_news.py")
else:
    # Normalize sources for matching
    news_df_ui["source_norm"] = news_df_ui["source"].fillna("").astype(str).str.strip()

    total_before = len(news_df_ui)

    if preferred_only:
        news_df_ui = news_df_ui[news_df_ui["source_norm"].isin(preferred_sources)]

    total_after = len(news_df_ui)

    if preferred_only:
        st.caption(
            f"Preferred-source headlines: {total_after} / {total_before} "
            f"(if 0, your provider didn’t return those publishers)."
        )
    else:
        st.caption(f"Headlines: {total_before}")

    if news_df_ui.empty:
        st.warning("No headlines matched your preferred-source filter. Turn off the filter or change provider.")
    else:
        # De-dupe + limit
        news_df_ui = news_df_ui.drop_duplicates(subset=["title"]).head(10)

        for _, r in news_df_ui.iterrows():
            title = str(r.get("title", "Untitled"))
            source = str(r.get("source", "") or "")
            url = str(r.get("url", "") or "")
            ts = r.get("published_at")
            ts_str = ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(ts) else ""

            if url.strip():
                st.markdown(f"- [{title}]({url}) — *{source}*  \n  {ts_str}")
            else:
                st.markdown(f"- {title} — *{source}*  \n  {ts_str}")
