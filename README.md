# 台股 AI 全市場選股與量化分析平台 (v7 本機 / 自架 NAS 部署版)

無 Gradio 版本：**FastAPI 後端 + 原生 HTML 前端**，先在本機（localhost）測試，測試完成後以 Docker 容器部署到自己的 NAS。
本架構完全退出 Hugging Face 與 Render 流程。

## 部署架構

- 開發流程：本機 `uvicorn` 執行測試 → 測試通過後於 NAS 上以 Docker 容器部署
- 容器化：`Dockerfile` + `docker-compose.yml`（供 NAS 的 Docker/Container Manager 直接使用）
- 資料來源：yfinance + TWSE / TPEx + FinMind
- AI 模型：MiniMax API（Chat Completion）
- 前端技術：原生 HTML + CSS + JavaScript + Plotly.js（由 FastAPI 同源託管，`index.html` 內 `API_BASE` 為相對路徑）

## 本機測試

```
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

瀏覽 `http://localhost:8000` 即可測試完整前後端（前端與 API 同源，無需另外設定 CORS 目標）。

## 部署到自己的 NAS

1. 將專案（含 `Dockerfile`、`docker-compose.yml`）複製到 NAS。
2. 複製 `.env.example` 為 `.env`，填入下方環境變數。
3. 在 NAS 的 Docker / Container Manager 中：
   - 使用 `docker compose up -d --build`，或
   - 於圖形介面匯入 `docker-compose.yml` 建立專案。
4. 容器預設對外開放埠 `8000`（可用 `PORT` 環境變數或 `docker-compose.yml` 的 `ports` 對應改為其他埠）。

## 環境變數

1. `MINIMAX_API_KEY`：填入你的 MiniMax API Key
2. `MINIMAX_MODEL`：MiniMax 模型名稱，預設 `MiniMax-M3`（選填）
3. `FINMIND_TOKEN`：填入你的 FinMind API Token

> ⚠️ 切勿將 API Key 寫死在 index.html 或 commit 到公開的儲存庫；一律透過環境變數（`.env` / NAS 容器設定）提供。

## 專案包含檔案

- `index.html`: 前端 SPA 介面 (全市場選股、漲幅排行、單檔 2330 深度分析、自選股排名)
- `main.py`: FastAPI 後端應用，提供 API 路由與靜態 HTML 服務
- `stock.py`: 台股量化分析引擎與 TWSE/TPEx 快照篩選器 (StockAnalyzer, TaiwanMarketScanner)
- `requirements.txt`: Python 相依套件清單
- `Dockerfile`: 容器化配置，供 NAS 部署使用
- `docker-compose.yml`: NAS 上一鍵建置/啟動用的 Compose 設定
- `.env.example`: 環境變數範本
- `.dockerignore`: Docker 忽略設定檔
- `README.md`: 說明文件
