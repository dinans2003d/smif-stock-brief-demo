# SMIF Daily Stock Brief
*A systematic daily monitoring and escalation tool for SMIF analysts*

---

## Overview
The **SMIF Daily Stock Brief** is a lightweight, rules-based analytics application designed to help Student Managed Investment Fund (SMIF) analysts **identify which portfolio names deserve attention on a given day**.

Rather than replacing fundamental research or investment judgment, this tool acts as a **daily signal scanner**—surfacing abnormal price behavior, volume shifts, volatility changes, and news activity in a concise, explainable format.

**Objective:**  
Reduce noise, focus analyst attention, and support faster, more disciplined monitoring decisions.

---

## Problem This Solves
SMIF analysts face a recurring operational challenge:
- Many portfolio holdings
- Limited analyst time
- No consistent process for deciding *which stocks matter today*

Common workflows rely on manual chart checks, scattered news searches, and subjective intuition.  
This initiative introduces a **repeatable, transparent daily process** to determine whether a stock warrants action, monitoring, or no follow-up.

---

## What the App Answers
For any selected ticker, the application answers:

1. Did the stock move unusually today?
2. Was trading volume abnormal relative to recent history?
3. Is volatility elevated or muted versus its own past?
4. Is there relevant same-day news in the dataset?
5. Based on these signals, should the stock be escalated, monitored, or ignored?

The system provides **attention signals**, not buy/sell recommendations.

---

## Core Features

### 1. Daily Decision Banner
Each stock is assigned a daily decision state:

- **NO ACTION** – Normal behavior, no follow-up required  
- **MONITOR** – Mild anomalies worth watching  
- **ESCALATE** – Significant signals justifying deeper analysis or discussion  

This decision is driven by a transparent scoring model.

---

### 2. “Why Today?” Explanation
Every decision includes a human-readable rationale based on:
- 1-day price movement
- Relative performance vs SPY (when available)
- Volume anomaly vs 30-day average
- Rolling volatility percentile
- Presence or absence of same-day news

This ensures all signals are **explainable, auditable, and discussion-ready**.

---

### 3. Attention Score (0–100)
A composite score summarizes the day’s signals:

| Component | Weight |
|---------|--------|
| Volume anomaly | 30% |
| Volatility percentile | 30% |
| Relative price move | 20% |
| News activity | 20% |

The score directly maps to the decision banner and allows analysts to rank stocks by urgency.

---

### 4. Benchmark Comparison
A normalized price chart compares each stock to **SPY**, providing immediate context on relative performance.

---

### 5. News Snapshot
Displays same-day news items from a local CSV source.  
The absence of news is explicitly surfaced and treated as informative.

---

## Data & Architecture

### Data Sources
- **Price & volume data:** SQLite database (`data/prices.db`)
- **News data:** CSV file (`data/news.csv`)

### Technology Stack
- Python
- Pandas
- SQLite
- Streamlit

### Design Principles
- Simple and interpretable logic
- Local-first data storage
- Fully transparent calculations
- No black-box models

This makes the system easy to audit, extend, and explain to faculty, sponsors, and future SMIF cohorts.

---

## Intended Use in SMIF
This tool is intended to be used as:
- A **daily or weekly monitoring dashboard**
- A **pre-meeting screening tool**
- A method to prioritize analyst attention
- A consistency layer across stock coverage

It complements, rather than replaces, fundamental valuation and thesis-driven research.

---

## Future Work
Planned extensions include:

### Portfolio-Level Analytics
- Rank all holdings by Attention Score
- Portfolio heatmaps of volatility and risk
- “Top names to review today” summary

### Enhanced News Integration
- API-based news ingestion
- Sentiment tagging
- Event classification (earnings, guidance, macro)

### Advanced Risk & Factor Signals
- Market vs idiosyncratic return decomposition
- Volatility regime detection
- Drawdown and momentum overlays

### Alerts & Automation
- Daily email or Slack summaries
- Automatic escalation flags
- Scheduled overnight data refresh

### Research Governance
- Analyst notes attached to daily signals
- Action tracking and follow-ups
- Historical evaluation of signal effectiveness

---

## What This Project Is Not
This system is **not**:
- A trading algorithm
- A price prediction model
- A buy/sell recommendation engine

It is a **decision-support and monitoring framework** designed to improve operational discipline within SMIF.

---

## Project Status
**Current status:** MVP / internal demo  
**Next phase:** Portfolio-wide scoring and expanded news data sources

---

## Summary
The SMIF Daily Stock Brief provides a systematic, explainable way to determine **which stocks deserve analyst attention today—and why**.
