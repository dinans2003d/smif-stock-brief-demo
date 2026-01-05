import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime

# IMPORTANT: your loader uses data/prices.db
DB_PATH = "data/prices.db"

BENCHMARK = "SPY"  # benchmark ticker
VOL_LOOKBACK = 30
VOLATILITY_WINDOW = 20

st.set_page_config(page_title="SMIF Daily Stock Brief", layout="wide")


# ---------- Data ----------
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

    # 30% volume anomaly (cap at 3x)
    if vol_anomaly is not None:
        vol_component = min(max(vol_anomaly - 1.0, 0.0) / 2.0, 1.0)  # 1x->0, 3x->1
        score += 30 * vol_component

    # 30% volatility percentile (0..1)
    if vol_pctile is not None:
        score += 30 * vol_pctile

    # 20% absolute relative move (cap at 3%)
    if rel_ret_1d is not None:
        move = min(abs(rel_ret_1d) / 0.03, 1.0)
        score += 20 * move

    # 20% news activity (0 -> 0, 5+ -> max)
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


# ---------- Daily Brief helpers ----------
def decision_from_score(score: int) -> tuple[str, str]:
    """
    Returns (decision, color_bucket)
    """
    if score >= 65:
        return "ESCALATE", "error"
    elif score >= 40:
        return "MONITOR", "warning"
    else:
        return "NO ACTION", "success"


def fmt_pct(x: float | None) -> str:
    return "N/A" if x is None else f"{x:+.2%}"


def fmt_mult(x: float | None) -> str:
    return "N/A" if x is None else f"{x:.2f}×"


# ---------- UI ----------
st.title("SMIF Daily Stock Brief")

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"]

# Try to populate from DB
try:
    conn = sqlite3.connect(DB_PATH)
    tickers_db = pd.read_sql_query(
        "SELECT DISTINCT ticker FROM prices ORDER BY ticker", conn
    )["ticker"].tolist()
    conn.close()
    tickers = tickers_db or TICKERS
except Exception:
    tickers = TICKERS

# If SPY exists, keep it out of the dropdown (benchmark only)
tickers_dropdown = [t for t in tickers if t != BENCHMARK]
if not tickers_dropdown:
    tickers_dropdown = TICKERS

ticker = st.selectbox("Select Ticker", tickers_dropdown, index=min(2, len(tickers_dropdown) - 1))

df = load_prices(ticker)
if df.empty:
    st.error(f"No price data found for {ticker}. Run the loader script first.")
    st.stop()

df = add_returns(df)

# ----- Benchmark (SPY) -----
spy = load_prices(BENCHMARK)

# Bulletproof: never crash if SPY missing
if spy.empty:
    spy = pd.DataFrame(columns=["asof_date", "close", "ret_1d"])
else:
    spy = add_returns(spy)
    if "ret_1d" not in spy.columns:
        spy["ret_1d"] = pd.NA

# -------------------------------------------------------------------
# FIX: Merge on a DATE key so you don't get N/A from timestamp mismatch
# -------------------------------------------------------------------
df = df.copy()
spy = spy.copy()

df["asof_key"] = pd.to_datetime(df["asof_date"]).dt.date
spy["asof_key"] = pd.to_datetime(spy["asof_date"]).dt.date

merged = df[["asof_date", "asof_key", "close", "volume", "ret_1d"]].merge(
    spy[["asof_key", "close", "ret_1d"]].rename(
        columns={"close": "spy_close", "ret_1d": "spy_ret_1d"}
    ),
    on="asof_key",
    how="left",
)

# Quick proof to see if benchmark is actually matching
overlap = int(merged["spy_close"].notna().sum()) if "spy_close" in merged.columns else 0
with st.expander("Data diagnostics"):
    st.caption(f"Benchmark overlap rows: {len(merged.dropna(subset=['spy_ret_1d']))} / {len(merged)}")


latest_row = merged.dropna(subset=["close"]).iloc[-1]
asof_date = pd.to_datetime(latest_row["asof_date"])

# --- KPI calculations ---
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

# News
news = load_news_csv("data/news.csv")
news_today = get_news_for_ticker(news, ticker, asof_date)
news_count = len(news_today) if not news_today.empty else 0

score, label = calc_attention_score(rel_ret_1d, vol_anomaly, vol_pctile, news_count)
decision, bucket = decision_from_score(score)

# Build “Why today?” bullets (only include what exists)
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

# Header
st.subheader(f"{ticker} — Daily Brief ({asof_date.date()})")

# Banner
banner_text = f"**Decision: {decision}**  |  Signal: **{label}**  |  Attention Score: **{score}/100**"
if bucket == "error":
    st.error(banner_text)
elif bucket == "warning":
    st.warning(banner_text)
else:
    st.success(banner_text)

# Why today
st.markdown("### Why today?")
st.markdown("\n".join([f"- {x}" for x in why]))

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Decision", decision, f"{score}/100")
c2.metric("Latest Close", f"${latest_close:,.2f}", fmt_pct(ret_1d) if ret_1d is not None else None)
c3.metric(f"Rel Return vs {BENCHMARK} (1D)", fmt_pct(rel_ret_1d) if rel_ret_1d is not None else "N/A")
c4.metric("Volume Anomaly", fmt_mult(vol_anomaly))

# Charts
st.subheader("Stock vs Benchmark (Normalized)")
plot = merged.dropna(subset=["close"]).copy().tail(120)
plot["stock_norm"] = (plot["close"] / plot["close"].iloc[0]) * 100

if "spy_close" in plot.columns and plot["spy_close"].notna().any():
    plot_spy = plot.dropna(subset=["spy_close"]).copy()
    if not plot_spy.empty:
        first_spy = plot_spy["spy_close"].iloc[0]
        plot.loc[plot_spy.index, "spy_norm"] = (plot_spy["spy_close"] / first_spy) * 100
        st.line_chart(plot.set_index("asof_date")[["stock_norm", "spy_norm"]])
    else:
        st.line_chart(plot.set_index("asof_date")[["stock_norm"]])
else:
    st.line_chart(plot.set_index("asof_date")[["stock_norm"]])

st.subheader("Recent Prices")
st.line_chart(merged.set_index("asof_date")[["close"]].tail(120))

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
