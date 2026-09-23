# 台股 AI 全市場選股與量化分析平台 (v6 GitHub + Render 部署版)

無 Gradio 版本：**GitHub 儲存庫 + Render FastAPI 後端 + 原生 HTML 前端**。
本架構完全退出 Hugging Face 流程。

## 部署架構

- 程式碼儲存庫：GitHub (例如 scratchinai01/taiwan-stock-ai)
- 雲端服務：Render Web Service (FastAPI + Uvicorn)
- 資料來源：yfinance + TWSE / TPEx + FinMind
- AI 模型：Google Gemini API
- 前端技術：原生 HTML + CSS + JavaScript + Plotly.js (由 FastAPI 託管 serving)

## Render Web Service 設定

- **Language**: Python 3
- **Branch**: main
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## 環境變數 (Render Environment Variables)

請至 Render Dashboard -> 你的 Service -> Environment -> Environment Variables 設定：

1. `GEMINI_API_KEY`: 填入你的 Google Gemini API Key
2. `FINMIND_TOKEN`: 填入你的 FinMind API Token

> ⚠️ 切勿將 API Key 寫死在 index.html 或 commit 到公開的 GitHub。

## 專案包含檔案

- `index.html`: 前端 SPA 介面 (全市場選股、漲幅排行、單檔 2330 深度分析、自選股排名)
- `main.py`: FastAPI 後端應用，提供 API 路由與靜態 HTML 服務
- `stock.py`: 台股量化分析引擎與 TWSE/TPEx 快照篩選器 (StockAnalyzer, TaiwanMarketScanner)
- `requirements.txt`: Python 相依套件清單
- `Dockerfile`: 容器化配置 (選用)
- `.dockerignore`: Docker 忽略設定檔
- `README.md`: 說明文件
