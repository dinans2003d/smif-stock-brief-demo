# SMIF Daily Stock Intelligence Brief

## Overview
The **SMIF Daily Stock Intelligence Brief** is a single-screen monitoring and news-compression tool built to support Student Managed Investment Fund (SMIF) analysts in their daily stock coverage responsibilities.

Each SMIF analyst is assigned primary responsibility for one stock and is expected to review market activity and news each morning across multiple sources (CNBC, Wall Street Journal, Financial Times, Bloomberg, etc.). This process is time-intensive, repetitive, and inconsistent across analysts.

This application centralizes **price behavior, trading signals, and stock-specific news** into a concise, explainable daily brief—allowing analysts to quickly understand what happened, why it matters, and whether deeper follow-up is required.

The system provides **findings**, not recommendations.

---

## Objective
The goal of this initiative is to **reduce information overload** and **standardize daily monitoring** by giving each analyst a clear, structured view of their assigned stock—without requiring them to manually read dozens of articles each morning.

Specifically, the tool helps analysts:
- Quickly identify whether anything meaningful happened to their stock today
- Understand key drivers behind price or volume changes
- Detect important news events without scanning multiple publications
- Decide how to allocate their research time more effectively

---

## Problem This Solves
SMIF analysts face a recurring operational challenge:

- One stock per analyst  
- Limited time each morning  
- Fragmented news across multiple platforms  
- No consistent framework for determining which days require action  

Current workflows rely on manual chart checks, ad-hoc news searches, and subjective judgment. This initiative introduces a **repeatable, transparent daily process** to surface what matters—and explicitly show when nothing does.

---

## What the App Provides
For a selected stock and trading day, the application answers:

- Did the stock move unusually today?
- Did it meaningfully outperform or underperform the market?
- Was trading volume abnormal relative to recent history?
- Is volatility elevated or unusually muted?
- Was there any relevant same-day news coverage?
- Based on these signals, does the stock require attention today?

The output supports analyst judgment but does **not** replace fundamental research or thesis-driven analysis.

---

## Core Features

### 1. Daily Intelligence Banner
Each stock is assigned a daily attention state:

- **NO ACTION** – Normal behavior, no follow-up required  
- **MONITOR** – Mild anomalies or developing signals  
- **ESCALATE** – Significant activity or news warranting deeper review  

This state is driven by a transparent, rules-based scoring model.

---

### 2. “Why Today?” Explanation
Every attention state includes a concise, human-readable explanation based on:

- 1-day price movement  
- Relative performance vs SPY  
- Trading volume anomaly vs recent history  
- Rolling volatility percentile  
- Presence or absence of stock-specific news  

All signals are explainable, auditable, and discussion-ready.

---

### 3. Attention Score (0–100)
A composite score summarizes the day’s activity and urgency:

| Component | Weight |
|--------|--------|
| Volume anomaly | 30% |
| Volatility percentile | 30% |
| Relative price move vs SPY | 20% |
| News activity | 20% |

The score maps directly to the daily attention state and allows analysts to quantify urgency without generating buy/sell signals.

---

### 4. News Intelligence Snapshot
The application surfaces same-day, stock-specific news from a centralized dataset.

Rather than forcing analysts to read multiple full articles, the system is designed to highlight:
- Whether news exists
- How much news exists
- Where it originated

The **absence of news** is explicitly surfaced and treated as informative.

This component is intentionally designed to evolve into deeper news intelligence and summarization.

---

### 5. Benchmark Context
Normalized price charts compare the stock’s recent performance to SPY, providing immediate relative context for market vs idiosyncratic movement.

---

## Design Principles
- Analyst-first workflow  
- Minimal cognitive load  
- Transparent, explainable logic  
- No black-box models  
- Local-first, auditable data storage  

The system prioritizes interpretability and consistency over prediction.

---

## Intended Use in SMIF
This tool is intended to be used as:
- A morning analyst check-in  
- A pre-meeting preparation tool  
- A daily monitoring framework  
- A consistency layer across analyst coverage  

It complements—but does not replace—fundamental research, valuation work, or investment thesis updates.

---

## Future Work

### Enhanced News Intelligence
- API-based news ingestion  
- News categorization (earnings, guidance, regulatory, macro)  
- Sentiment tagging  
- Daily stock-specific news synthesis  
- “What changed since yesterday?” indicators  

### Analyst Workflow Extensions
- Analyst notes attached to daily signals  
- Follow-up tracking on escalated days  
- Historical review of signal effectiveness  

### Portfolio-Level Views
- Rank all holdings by attention score  
- “Top names to review today” summary  
- Portfolio volatility and risk heatmaps  

### Automation
- Daily email or Slack briefings  
- Scheduled overnight refresh  
- Automatic escalation alerts  

---

## What This Project Is Not
This system is **not**:
- A trading algorithm  
- A price prediction model  
- A buy/sell recommendation engine  

It is a **decision-support and monitoring framework** designed to improve analyst efficiency and operational discipline.

---

## Project Status
**Current status:** MVP / internal demo  
**Next phase:** News intelligence expansion and portfolio-level prioritization  

---

## Summary
The SMIF Daily Stock Intelligence Brief provides a systematic, explainable way for analysts to understand what matters about their assigned stock today—without reading everything—and to focus their time where it adds the most value.

