# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

台股 AI 全市場選股與量化分析平台 — a Taiwan-stock quant analysis and screening tool. No Gradio: a
FastAPI backend (`main.py` + `stock.py`) serves a single static frontend page (`index.html`, vanilla
HTML/CSS/JS + Plotly.js, no build step, no framework). This is not a git repository (no `.git`).

## Running locally

```
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

There is no test suite, linter, or build tooling configured in this repo — don't invent commands for
them.

## Deployment architecture (from README.md)

- Workflow: test locally with `uvicorn` → build/run as a Docker container on the user's own NAS
  (`Dockerfile` + `docker-compose.yml`). There is **no Render deployment** in this project anymore —
  don't reintroduce Render-specific config (e.g. relying on a `$PORT` env var Render injects).
- Data sources: yfinance (price/OHLCV), TWSE / TPEx open APIs (market snapshot), FinMind (news,
  institutional/margin/futures data), MiniMax Chat Completion API (AI commentary).
- Required env vars on the server: `MINIMAX_API_KEY` (+ optional `MINIMAX_MODEL`, default
  `MiniMax-M3`), `FINMIND_TOKEN`. Set them via `.env` (see `.env.example`) or the NAS
  container's environment settings — never hardcode them in `index.html`; they must stay server-side
  only, read via `os.environ` in `stock.py`.
- `Dockerfile` defaults to port 8000 (overridable via the `PORT` env var); `docker-compose.yml` maps
  `8000:8000` and reads `MINIMAX_API_KEY`/`MINIMAX_MODEL`/`FINMIND_TOKEN` from the shell/`.env`.

## Frontend/backend wiring

`index.html`'s `API_BASE` (`index.html:94`) is an **empty string** — the frontend calls same-origin
`/api/...` paths because `main.py` serves `index.html` itself at `/`. This works unchanged for both
`uvicorn` on localhost and the NAS Docker deployment. Only set `API_BASE` to an absolute URL if you
temporarily split frontend and backend across different ports/hosts during development, and revert
before committing.

## Backend structure (`main.py`)

FastAPI app with four JSON endpoints plus `/` (serves `index.html`) and `/api/health`:

- `GET /api/gainers` — top gainers list via `TaiwanMarketScanner.get_top_gainers`.
- `GET /api/scan` — full-market screener via `TaiwanMarketScanner.scan`.
- `POST /api/analyze` — single-ticker deep dive via `StockAnalyzer`; returns technical indicator
  summaries, MACD/leading-indicator diagnostics, FinMind news, MiniMax AI commentary, and a Plotly
  figure (as JSON via `plot.to_json()`, *not* FastAPI's default JSON encoder — required because the
  figure carries numpy/pandas/Timestamp values that Starlette's `JSONResponse` can't serialize).
- `POST /api/watchlist` — ranks a comma-separated ticker list via `StockAnalyzer.scan_watchlist`.

`df_records()` (`main.py:46`) is the shared DataFrame → JSON-safe-records converter used by every
endpoint that returns tabular data. It exists because raw NaN/±inf values from pandas/numpy break
FastAPI's JSON responses (they aren't valid JSON) — always route new DataFrame-returning endpoints
through it rather than calling `.to_dict()` directly.

## `stock.py` — the analysis engine

Two classes:

- **`StockAnalyzer`** — per-ticker pipeline. `fetch_data(period)` gets OHLCV via a **hybrid data
  source**: it tries FinMind's `TaiwanStockPrice` dataset first (via `_fetch_price_finmind()`), and
  only falls back to `yfinance` if FinMind has no token, returns too few rows for the requested
  `period` (thresholds in `_PERIOD_MIN_ROWS`), or errors. This exists because yfinance has real,
  observed gaps for Taiwan tickers (confirmed: several real, currently-listed stocks returned "no
  data / may be delisted" from yfinance while FinMind had complete data for the same tickers/dates),
  while FinMind's free tier caps at 600 requests/hour per token — fine for `/api/analyze` and
  `/api/watchlist`, but something to watch for `TaiwanMarketScanner.scan()` at high `candidate_count`
  since it already spends one FinMind call per candidate on the valuation bonus. `scan_watchlist()`
  uses the same hybrid fetch per ticker (via `_fetch_price_finmind(period, ticker=...)`, which takes an
  explicit ticker since watchlist tickers differ from `self.ticker`). Once fetched (from either
  source), `fetch_data()` computes the full indicator set in one pass (MA5/20/60/120/240, RSI, MACD,
  Bollinger Bands, ATR, KD, CCI, Williams %R, ROC, MFI, OBV, ADX/+DI/-DI, PSY, volume ratio, EMA50/200,
  momentum). `engineer_features()` derives ML-ready features (bias, daily return, normalized ATR) into
  `mega_df_cleaned_final`, though no model is currently trained/consumed from it in `main.py` (the
  sklearn/xgboost imports and `RandomForestClassifier` scaffolding in this file are currently unused
  by any endpoint). `get_ai_prediction_text()` is the only AI call site: it POSTs to MiniMax's
  OpenAI-compatible `https://api.minimax.cn/v1/chat/completions` endpoint (Bearer auth via
  `MINIMAX_API_KEY`, model from `MINIMAX_MODEL`, default `MiniMax-M3`; response parsed at
  `choices[0].message.content`, same shape as OpenAI's Chat Completions) using plain `requests` — no
  SDK dependency. No `GroupId` is needed for this endpoint.
- **`TaiwanMarketScanner`** — full-market snapshot and screener, backed by two public, key-less
  open-data endpoints: TWSE `STOCK_DAY_ALL` (`TWSE_URL`) for 上市 and TPEx
  `tpex_mainboard_daily_close_quotes` (`TPEx_URL`) for 上櫃. `get_market_snapshot()` fetches and
  normalizes both into a common schema, filtering to ordinary stocks only via `_STOCK_CODE_RE`
  (4-digit codes, excluding `00`-prefixed ETF codes like 0050/0056 — both markets happen to use
  4-digit codes for ETFs on TWSE, so this exclusion matters). `get_top_gainers()` filters that
  snapshot by trade value and sorts by % change. `scan()` takes the top `candidate_count` stocks by
  trade value as a candidate pool, then for each one instantiates a `StockAnalyzer` (full 1y history +
  indicators) and scores it via `_score_stock()` against the rubric documented in `index.html`'s
  「指標與規則」 tab (趨勢結構25/動能20/量價15/突破15/風險控制5 computed per stock, 相對強弱10 computed
  afterward via percentile rank of `ROC_12` across the candidate pool, 估值加分 up to 4 from FinMind's
  `TaiwanStockPER` dataset — silently 0 if no token or lookup fails). Because each candidate needs its
  own sequential `yfinance` download (same pattern as `scan_watchlist`), `scan()` is slow — roughly
  0.5–1s per candidate, so the default `candidate_count=60` takes well over half a minute; this is
  expected, not a bug. Individual candidate failures (delisted tickers, no yfinance data) are caught
  and skipped, never crash the whole scan.

`_fetch_finmind_api_data()` centralizes all FinMind calls (Bearer auth, per-dataset GET to
`api.finmindtrade.com/api/v4/data`); it deliberately returns an empty DataFrame on any failure or
missing token rather than fabricating mock data — preserve that behavior in any new FinMind-backed
method.

Tickers are normalized to bare numeric strings internally (`.TW`/`.TWO` stripped in
`StockAnalyzer.__init__` and in `main.py`'s `/api/analyze`), then re-suffixed with `.TW` only at the
yfinance call site and in the API response's `ticker` field.

## Frontend (`index.html`)

Single file, no build step: five tabs (全市場選股 / 漲幅前100 / 單檔深度分析 / 自選股排名 /
指標與規則) toggled by `data-tab` buttons, each backed by one `panel` section. Includes a small
hand-rolled Markdown-to-HTML renderer (`md()`/`inlineMd()`) used to display the text fields the
backend returns (`indicator_summary`, `macd_analysis`, `leading_analysis`, etc.) — headings, bold,
inline code, lists, and hr only, no tables/links. `table()` renders any array-of-objects API payload
generically by taking `Object.keys()` of the first row as columns.
