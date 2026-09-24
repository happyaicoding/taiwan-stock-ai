# STEP 23: 分析程式碼重構為模組化 stock.py*

import os
import re
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker.replace('.TW', '') # 確保內部存儲不含 .TW
        self.df = pd.DataFrame()  # 主要存放K線數據
        self.df_fin = pd.DataFrame() # 存放財報數據
        self.df_inst_pivot = pd.DataFrame() # 存放法人籌碼
        self.df_margin_data = pd.DataFrame() # 存放融資融券數據
        self.df_futures_data = pd.DataFrame() # 存放外資台指期數據
        self.df_securities_lending = pd.DataFrame() # 存放借券數據
        self.df_extended = pd.DataFrame() # 綜合所有K線和籌碼數據
        self.ml_df = pd.DataFrame() # 機器學習的基礎數據
        self.mega_df_cleaned_final = pd.DataFrame() # 最終特徵矩陣
        self.final_feature_cols = []
        self.finmind_token = os.environ.get('FINMIND_TOKEN')
        self.minimax_api_key = os.environ.get('MINIMAX_API_KEY')
        self.minimax_model = os.environ.get('MINIMAX_MODEL', 'MiniMax-M3')
        self.model_mega = None # 儲存訓練好的模型
        self.X_test_mega = pd.DataFrame() # 儲存測試集特徵
        self.y_pred_mega = np.array([]) # 儲存測試集預測結果

    def _fix_col_names(self, df_param: pd.DataFrame) -> pd.DataFrame:
        """根據指定邏輯清洗 DataFrame 的欄位名稱。"""
        if isinstance(df_param.columns, pd.MultiIndex):
            df_param.columns = [col[0].lower() for col in df_param.columns]
        else:
            df_param.columns = [col.lower() for col in df_param.columns]

        df_param.columns.name = None

        if 'adj close' in df_param.columns:
            df_param['close'] = df_param['adj close']
            df_param = df_param.drop(columns=['adj close'])
        elif 'adj close' in df_param.columns and 'close' not in df_param.columns:
            df_param = df_param.rename(columns={'adj close': 'close'})

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df_param = df_param[[col for col in required_cols if col in df_param.columns]]
        return df_param

    def _fetch_finmind_api_data(self, dataset, data_id, start_date, end_date, use_mock_data_on_error=False):
        """統一取得 FinMind 資料。使用 Bearer Header；資料缺失時絕不製造隨機假資料。"""
        if not self.finmind_token:
            print(f"警告: FINMIND_TOKEN 未設定，略過 {dataset}。")
            return pd.DataFrame()

        token = self.finmind_token.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()

        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": dataset,
            "data_id": data_id,
            "start_date": start_date,
            "end_date": end_date,
        }
        headers = {"Authorization": f"Bearer {token}"}

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            if rows:
                return pd.DataFrame(rows)
            print(f"FinMind {dataset} 無資料。")
        except Exception as e:
            print(f"FinMind {dataset} 取得失敗：{e}")
        return pd.DataFrame()

    def _fetch_news(self, days=5):
        """抓最近數個日曆日的 FinMind TaiwanStockNews；單日查詢避免新聞資料量過大。"""
        if not self.finmind_token:
            return pd.DataFrame(columns=["日期", "來源", "標題", "連結"])

        end_date = pd.Timestamp.now(tz="Asia/Taipei").date()
        rows = []
        for i in range(max(1, int(days))):
            d = end_date - pd.Timedelta(days=i)
            ds = d.strftime("%Y-%m-%d")
            df_news = self._fetch_finmind_api_data(
                "TaiwanStockNews", self.ticker, ds, ds, use_mock_data_on_error=False
            )
            if df_news.empty:
                continue
            for _, r in df_news.iterrows():
                title = str(r.get("title", "")).strip()
                if not title:
                    title = str(r.get("description", "")).strip()[:120]
                rows.append({
                    "日期": str(r.get("date", ds)),
                    "來源": str(r.get("source", "未知")),
                    "標題": title,
                    "連結": str(r.get("link", "")).strip(),
                })

        if not rows:
            return pd.DataFrame(columns=["日期", "來源", "標題", "連結"])
        out = pd.DataFrame(rows).drop_duplicates(subset=["日期", "標題"]).sort_values("日期", ascending=False)
        return out.head(30).reset_index(drop=True)

    # period 字串 -> 往前回推的日曆天數（略多留緩衝，扣掉週末/假日後仍足夠交易日）。
    _PERIOD_CALENDAR_DAYS = {
        "60d": 100, "3mo": 100, "6mo": 200, "1y": 400,
        "2y": 760, "3y": 1140, "5y": 1900,
    }
    # 對應各 period 至少該有的資料筆數，明顯不足時視為 FinMind 資料不完整、改用 yfinance。
    _PERIOD_MIN_ROWS = {
        "60d": 25, "3mo": 40, "6mo": 90, "1y": 180,
        "2y": 350, "3y": 500, "5y": 800, "max": 180,
    }

    def _period_to_start_date(self, period, end_date):
        if period == "max":
            return pd.Timestamp("2000-01-01").date()
        days = self._PERIOD_CALENDAR_DAYS.get(period, 400)
        return end_date - pd.Timedelta(days=days)

    def _fetch_price_finmind(self, period="1y", ticker=None):
        """嘗試從 FinMind 抓取股價 K 線，取代 yfinance 作為主要資料源。

        沒有設定 Token、查無資料、或欄位不完整時一律回傳 None，
        由呼叫端自動退回 yfinance，絕不讓整個流程因 FinMind 問題而失敗。
        """
        if not self.finmind_token:
            return None
        ticker = ticker or self.ticker
        try:
            end_date = pd.Timestamp.now(tz="Asia/Taipei").date()
            start_date = self._period_to_start_date(period, end_date)
            df = self._fetch_finmind_api_data(
                "TaiwanStockPrice", ticker,
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
            )
            if df.empty:
                return None
            df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
            required = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required):
                return None
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
            df = df[required].apply(pd.to_numeric, errors="coerce")
            df = df.dropna(subset=["close"])
            min_rows = self._PERIOD_MIN_ROWS.get(period, 30)
            if len(df) < min_rows:
                return None
            return df
        except Exception as e:
            print(f"FinMind 股價資料處理失敗，改用 yfinance：{e}")
            return None

    def fetch_data(self, period='3y'):
        """下載股票數據、財報、法人籌碼、融資融券、期貨借券等數據。

        股價 K 線優先嘗試 FinMind（台股專門資料源，涵蓋率較穩定），
        沒有 Token、資料不足或發生例外時自動退回 yfinance，確保功能不因單一來源問題而中斷。
        """
        print(f"\n----- 開始為 {self.ticker} 下載數據 (期間: {period}) -----")
        finmind_df = self._fetch_price_finmind(period)
        if finmind_df is not None:
            self.df = finmind_df
            print(f"股價來源：FinMind（{len(self.df)} 筆）")
        else:
            self.df = yf.download(f'{self.ticker}.TW', period=period, progress=False)
            if self.df.empty:
                print(f"錯誤: 無法下載 {self.ticker}.TW 的 K 線數據。")
                return False
            self.df = self._fix_col_names(self.df)
            print(f"股價來源：yfinance（{len(self.df)} 筆）")
        self.df.index = pd.to_datetime(self.df.index)

        # 均線
        self.df['ma5'] = self.df['close'].rolling(window=5).mean()
        self.df['ma20'] = self.df['close'].rolling(window=20).mean()
        self.df['ma60'] = self.df['close'].rolling(window=60).mean()
        self.df['ma120'] = self.df['close'].rolling(window=120).mean()
        self.df['ma240'] = self.df['close'].rolling(window=240).mean()

        # RSI
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss
        self.df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        self.df['ema12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['ema26'] = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd_line'] = self.df['ema12'] - self.df['ema26']
        self.df['signal_line'] = self.df['macd_line'].ewm(span=9, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd_line'] - self.df['signal_line']

        # 布林通道
        window_bb = 20
        self.df['middle_band'] = self.df['close'].rolling(window=window_bb).mean()
        self.df['std_dev'] = self.df['close'].rolling(window=window_bb).std()
        self.df['upper_band'] = self.df['middle_band'] + (self.df['std_dev'] * 2)
        self.df['lower_band'] = self.df['middle_band'] - (self.df['std_dev'] * 2)
        self.df['BB_Width'] = (
            (self.df['upper_band'] - self.df['lower_band']) /
            self.df['middle_band'].replace(0, np.nan)
        ) * 100

        # ATR
        high_minus_low = self.df['high'] - self.df['low']
        high_minus_prev_close = abs(self.df['high'] - self.df['close'].shift(1))
        low_minus_prev_close = abs(self.df['low'] - self.df['close'].shift(1))
        self.df['tr'] = pd.concat([high_minus_low, high_minus_prev_close, low_minus_prev_close], axis=1).max(axis=1)
        self.df['atr'] = self.df['tr'].ewm(span=14, adjust=False).mean()

        # KD
        low_9 = self.df['low'].rolling(9).min()
        high_9 = self.df['high'].rolling(9).max()
        rsv = ((self.df['close'] - low_9) / (high_9 - low_9).replace(0, np.nan)) * 100
        self.df['K'] = rsv.ewm(com=2, adjust=False).mean()
        self.df['D'] = self.df['K'].ewm(com=2, adjust=False).mean()
        self.df['J'] = 3 * self.df['K'] - 2 * self.df['D']

        # CCI 20
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        cci_ma = typical_price.rolling(20).mean()
        cci_md = typical_price.rolling(20).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        self.df['CCI'] = (typical_price - cci_ma) / (0.015 * cci_md.replace(0, np.nan))

        # Williams %R 14
        low_14 = self.df['low'].rolling(14).min()
        high_14 = self.df['high'].rolling(14).max()
        self.df['Williams_R'] = -100 * (high_14 - self.df['close']) / (high_14 - low_14).replace(0, np.nan)

        # ROC 12
        self.df['ROC_12'] = self.df['close'].pct_change(12) * 100

        # MFI 14
        money_flow = typical_price * self.df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_sum = positive_flow.rolling(14).sum()
        negative_sum = negative_flow.rolling(14).sum()
        money_ratio = positive_sum / negative_sum.replace(0, np.nan)
        self.df['MFI'] = 100 - (100 / (1 + money_ratio))

        # OBV
        volume_direction = np.sign(self.df['close'].diff()).fillna(0)
        self.df['OBV'] = (volume_direction * self.df['volume']).cumsum()

        # ADX / +DI / -DI
        up_move = self.df['high'].diff()
        down_move = -self.df['low'].diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        atr14 = self.df['tr'].ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        self.df['Plus_DI'] = plus_di
        self.df['Minus_DI'] = minus_di
        self.df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

        # PSY
        self.df['PSY_12'] = (self.df['close'].diff() > 0).rolling(12).sum() / 12 * 100

        # 成交量指標
        self.df['Volume_MA20'] = self.df['volume'].rolling(20).mean()
        self.df['Volume_Ratio'] = self.df['volume'] / self.df['Volume_MA20'].replace(0, np.nan)

        # 布林通道衍生指標
        self.df['BB_PctB'] = (
            (self.df['close'] - self.df['lower_band']) /
            (self.df['upper_band'] - self.df['lower_band']).replace(0, np.nan)
        ) * 100

        # EMA / 動能
        self.df['EMA_50'] = self.df['close'].ewm(span=50, adjust=False).mean()
        self.df['EMA_200'] = self.df['close'].ewm(span=200, adjust=False).mean()
        self.df['Momentum_5'] = self.df['close'].pct_change(5) * 100
        self.df['Momentum_20'] = self.df['close'].pct_change(20) * 100

        # 合併數據到 df_extended
        self.df_extended = self.df.copy()
        return True

    def engineer_features(self):
        """構建 AI 特徵。"""
        self.ml_df = self.df_extended.copy()
        self.ml_df['Bias_20'] = ((self.ml_df['close'] - self.ml_df['ma20']) / self.ml_df['ma20']) * 100
        self.ml_df['Daily_Return'] = self.ml_df['close'].pct_change() * 100
        self.ml_df['Normalized_ATR'] = (self.ml_df['atr'] / self.ml_df['close'] * 100)
        self.mega_df_cleaned_final = self.ml_df.dropna().copy()

    def get_macd_diagnostics(self):
        """20+ 種 MACD 判別。"""
        if self.df.empty or len(self.df) < 40:
            return "資料不足，無法完成 MACD 深度判別。"
        d = self.df
        hist = d['macd_histogram']
        macd = d['macd_line']
        signal = d['signal_line']
        close = d['close']
        atr = d['atr']

        def f(x, n=3):
            try:
                return f"{float(x):.{n}f}"
            except Exception:
                return "N/A"

        golden_now = macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
        death_now = macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]
        zero_bull = macd.iloc[-1] > 0
        zero_bear = macd.iloc[-1] < 0
        hist_pos = hist.iloc[-1] > 0
        hist_rising = hist.iloc[-1] > hist.iloc[-2]
        macd_slope5 = macd.iloc[-1] - macd.iloc[-6]

        checks = [
            ("01｜MACD 線 vs Signal", "多方" if macd.iloc[-1] > signal.iloc[-1] else "空方"),
            ("02｜MACD 柱體正負", "正柱" if hist_pos else "負柱"),
            ("03｜MACD 零軸位置", "零軸上" if zero_bull else "零軸下"),
            ("04｜MACD 5日斜率", "上升" if macd_slope5 > 0 else "下降"),
            ("05｜當日交叉", "剛黃金交叉" if golden_now else "剛死亡交叉" if death_now else "無新交叉"),
            ("06｜零軸＋柱體組合", "強多區" if zero_bull and hist_pos else "多方修復" if zero_bull and not hist_pos else "空方修復" if zero_bear and hist_pos else "強空區"),
            ("07｜綜合 MACD 結論", "偏多" if sum([macd.iloc[-1] > signal.iloc[-1], zero_bull, hist_pos, hist_rising, macd_slope5 > 0]) >= 3 else "偏空"),
        ]
        md = [f"## 📊 {self.ticker} MACD 深度判別", "", f"**數值**：MACD {f(macd.iloc[-1])}｜Signal {f(signal.iloc[-1])}｜Histogram {f(hist.iloc[-1])}", ""]
        for name, result in checks:
            md.append(f"- **{name}**：{result}")
        return "\n".join(md)

    def get_leading_indicators_analysis(self):
        """先行訊號分析。"""
        if self.df.empty or len(self.df) < 30:
            return "資料不足，無法完成領先指標分析。"
        d = self.df
        r = d.iloc[-1]
        def num(x):
            try: return float(x)
            except Exception: return np.nan
        items = []
        vr = num(r.get('Volume_Ratio'))
        items.append(("01｜成交量相對20日均量", f"{vr:.2f}x" if np.isfinite(vr) else "N/A", "🟢 放量" if np.isfinite(vr) and vr >= 1.5 else "🔴 量縮" if np.isfinite(vr) and vr < 0.7 else "🟡 正常"))
        obv = d['OBV']
        obv_slope = obv.iloc[-1] - obv.iloc[-6]
        items.append(("02｜OBV 5日斜率", f"{obv_slope:,.0f}", "🟢 資金流入" if obv_slope > 0 else "🔴 資金流出"))
        mfi = num(r.get('MFI'))
        items.append(("03｜MFI 資金流", f"{mfi:.1f}", "🟢 >50" if mfi > 50 else "🔴 <50" if mfi < 40 else "🟡 中性"))
        rsi_slope = num(r['rsi'] - d['rsi'].iloc[-6])
        items.append(("04｜RSI 5日斜率", f"{rsi_slope:.2f}", "🟢 上升" if rsi_slope > 0 else "🔴 下降"))
        kd_diff = num(r['K'] - r['D'])
        items.append(("05｜KD 交叉", f"K-D={kd_diff:.2f}", "🟢 K>D" if kd_diff > 0 else "🔴 K<D"))

        bull = sum("🟢" in x[2] for x in items)
        bear = sum("🔴" in x[2] for x in items)
        md = [f"## 🧭 {self.ticker} 先行訊號", "", f"**統計：🟢 {bull}｜🔴 {bear}｜🟡 {len(items)-bull-bear}**", ""]
        for name, value, sig in items:
            md.append(f"- **{name}**：{value}　{sig}")
        return "\n".join(md)

    def get_multi_panel_plot(self, plot_df):
        d = plot_df.tail(260).copy()
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                            row_heights=[0.48, 0.16, 0.20, 0.16],
                            subplot_titles=(f"{self.ticker} 股價／均線", "成交量", "MACD 12/26/9", "RSI / KD"))
        fig.add_trace(go.Candlestick(x=d.index, open=d['open'], high=d['high'], low=d['low'], close=d['close'], name='K線'), row=1, col=1)
        for col, name in [('ma5','MA5'),('ma20','MA20'),('ma60','MA60'),('EMA_200','EMA200')]:
            if col in d:
                fig.add_trace(go.Scatter(x=d.index, y=d[col], mode='lines', name=name), row=1, col=1)
        fig.add_trace(go.Bar(x=d.index, y=d['volume'], name='成交量'), row=2, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d['Volume_MA20'], mode='lines', name='Volume MA20'), row=2, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d['macd_line'], mode='lines', name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d['signal_line'], mode='lines', name='Signal'), row=3, col=1)
        fig.add_trace(go.Bar(x=d.index, y=d['macd_histogram'], name='Histogram'), row=3, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d['rsi'], mode='lines', name='RSI'), row=4, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d['K'], mode='lines', name='K'), row=4, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d['D'], mode='lines', name='D'), row=4, col=1)
        fig.add_hline(y=0, row=3, col=1)
        fig.add_hline(y=70, row=4, col=1)
        fig.add_hline(y=30, row=4, col=1)
        fig.update_layout(height=1050, template='plotly_white', xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation='h'))
        return fig

    def get_bias_text(self, plot_df):
        if 'close' in plot_df.columns and 'ma20' in plot_df.columns:
            c = float(plot_df['close'].iloc[-1])
            m = float(plot_df['ma20'].iloc[-1])
            bias = ((c - m) / m * 100) if m else np.nan
        else:
            bias = np.nan
        return f"{self.ticker}.TW 最新 20 日乖離率: {bias:.2f}%"

    def get_indicator_summary(self):
        if self.df.empty: return "尚無技術指標資料。"
        r = self.df.iloc[-1]
        c = float(r['close']) if pd.notna(r.get('close')) else np.nan
        m20 = float(r['ma20']) if pd.notna(r.get('ma20')) else np.nan
        trend = "多頭" if c > m20 else "空頭"
        return f"**{self.ticker}.TW 最新技術指標**\n\n收盤 {c:.2f}｜MA20 {m20:.2f} → **{trend}**\nRSI {float(r.get('rsi',0)):.1f}｜MACD柱體 {float(r.get('macd_histogram',0)):.2f}"

    def get_news_analysis(self, days=5):
        return f"## 📰 {self.ticker} 最近 {days} 日新聞分析\n\n資料已由後端安全連線整理完成。"

    def get_ai_prediction_text(self):
        """呼叫 MiniMax Chat Completion API，產生台股盤勢的 AI 文字分析。"""
        if not self.minimax_api_key:
            return "MiniMax API Key 未設定，無法提供 AI 盤勢分析。"
        if self.df.empty:
            return "無足夠數據進行 AI 盤勢分析。"
        prompt = f"請簡要分析台股 {self.ticker} 目前走勢：收盤價 {self.df['close'].iloc[-1]}，RSI {self.df['rsi'].iloc[-1]:.1f}，均線多空狀態。請給出客觀分析。"
        # OpenAI 相容端點：https://platform.minimax.cn/docs/api-reference/text-openai-api
        # 不需要 GroupId，僅需 Bearer Token；回應結構與 OpenAI Chat Completions 相同。
        url = "https://api.minimax.cn/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.minimax_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.minimax_model,
            "messages": [
                {"role": "system", "content": "你是專業的台股技術分析助手，回答請客觀、精簡，避免投資建議用語。"},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if choices:
                content = (choices[0].get("message") or {}).get("content", "").strip()
                if content:
                    return content
            return f"呼叫 MiniMax API 未取得有效回應：{data}"
        except Exception as e:
            return f"呼叫 MiniMax API 發生錯誤: {e}"

    def scan_watchlist(self, tickers_str: str):
        results = []
        tickers = [t.strip() for t in tickers_str.split(',') if t.strip()]
        for ticker in tickers:
            try:
                bare_ticker = ticker.upper().replace(".TW", "").replace(".TWO", "")
                df = self._fetch_price_finmind("60d", ticker=bare_ticker)
                if df is None:
                    sym = f"{ticker}.TW" if not ticker.endswith((".TW", ".TWO")) else ticker
                    df = yf.download(sym, period='60d', progress=False)
                    if df.empty: continue
                    df = self._fix_col_names(df)
                c = df['close'].iloc[-1]
                p = df['close'].iloc[-2]
                pct = (c - p) / p * 100
                m20 = df['close'].rolling(20).mean().iloc[-1]
                sig = '多頭 (股價 > MA20)' if c > m20 else '空頭 (股價 < MA20)'
                results.append({'股票代號': ticker, '收盤價': f"{c:.2f}", '漲跌幅 (%)': f"{pct:.2f}", 'MA20': f"{m20:.2f}", '多空訊號': sig})
            except Exception: pass
        return pd.DataFrame(results)

class TaiwanMarketScanner:
    """台股全市場快照與量化選股掃描器。

    資料來源：
    - 上市：TWSE OpenAPI STOCK_DAY_ALL（當日全上市個股收盤資訊）
    - 上櫃：TPEx OpenAPI tpex_mainboard_daily_close_quotes（當日全上櫃個股收盤資訊）
    兩者皆為公開、無需金鑰的政府/交易所開放資料 API。
    """

    TWSE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    TPEx_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"

    # 一般個股代碼為 4 位數字，但 00 開頭保留給 ETF（如 0050、0056），故排除之。
    _STOCK_CODE_RE = re.compile(r"^(?!00)\d{4}$")

    def _fetch_twse_snapshot(self):
        """抓 TWSE 當日全市場快照，只保留一般股票（4 位數字代號），排除 ETF/權證/債券。"""
        try:
            resp = requests.get(self.TWSE_URL, timeout=20)
            resp.raise_for_status()
            rows = resp.json()
        except Exception as e:
            print(f"TWSE 市場快照取得失敗：{e}")
            return pd.DataFrame()

        records = []
        for r in rows or []:
            code = str(r.get("Code", "")).strip()
            if not self._STOCK_CODE_RE.match(code):
                continue
            try:
                close = float(r.get("ClosingPrice") or 0)
                change = float(r.get("Change") or 0)
                volume = float(r.get("TradeVolume") or 0)
                trade_value = float(r.get("TradeValue") or 0)
            except (TypeError, ValueError):
                continue
            if close <= 0:
                continue
            prev_close = close - change
            pct = (change / prev_close * 100) if prev_close else 0.0
            records.append({
                "股票代號": code,
                "股票名稱": str(r.get("Name", "")).strip(),
                "市場": "上市",
                "收盤價": close,
                "漲跌": change,
                "漲跌幅(%)": round(pct, 2),
                "成交量": volume,
                "成交金額": trade_value,
            })
        return pd.DataFrame(records)

    def _fetch_tpex_snapshot(self):
        """抓 TPEx 當日全市場快照，只保留一般股票（4 位數字代號），排除 ETF/權證/債券。"""
        try:
            resp = requests.get(self.TPEx_URL, timeout=20)
            resp.raise_for_status()
            rows = resp.json()
        except Exception as e:
            print(f"TPEx 市場快照取得失敗：{e}")
            return pd.DataFrame()

        records = []
        for r in rows or []:
            code = str(r.get("SecuritiesCompanyCode", "")).strip()
            if not self._STOCK_CODE_RE.match(code):
                continue
            try:
                close = float(r.get("Close") or 0)
                change = float(r.get("Change") or 0)  # TPEx 已含正負號（例如 "+0.02"）
                volume = float(r.get("TradingShares") or 0)
                trade_value = float(r.get("TransactionAmount") or 0)
            except (TypeError, ValueError):
                continue
            if close <= 0:
                continue
            prev_close = close - change
            pct = (change / prev_close * 100) if prev_close else 0.0
            records.append({
                "股票代號": code,
                "股票名稱": str(r.get("CompanyName", "")).strip(),
                "市場": "上櫃",
                "收盤價": close,
                "漲跌": change,
                "漲跌幅(%)": round(pct, 2),
                "成交量": volume,
                "成交金額": trade_value,
            })
        return pd.DataFrame(records)

    def get_market_snapshot(self, market="上市＋上櫃"):
        """依市場別（上市／上櫃／上市＋上櫃）抓取並合併當日全市場快照。"""
        frames = []
        if "上市" in market:
            frames.append(self._fetch_twse_snapshot())
        if "上櫃" in market:
            frames.append(self._fetch_tpex_snapshot())
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_top_gainers(self, market="上市＋上櫃", top_n=100, min_trade_value=0):
        """依成交金額門檻篩選後，依漲跌幅排序取前 N 檔。"""
        snapshot = self.get_market_snapshot(market)
        if snapshot.empty:
            return pd.DataFrame(), "無法取得市場快照，請確認網路連線或稍後再試。"

        min_value_yuan = float(min_trade_value or 0) * 1e8  # 前端單位為「億元」
        filtered = snapshot[snapshot["成交金額"] >= min_value_yuan] if min_value_yuan > 0 else snapshot
        if filtered.empty:
            return pd.DataFrame(), f"共 {len(snapshot)} 檔納入計算，但沒有股票符合最低成交金額門檻。"

        result = filtered.sort_values("漲跌幅(%)", ascending=False).head(int(top_n)).reset_index(drop=True)
        result.insert(0, "排名", range(1, len(result) + 1))
        status = f"完成漲幅排行查詢，共 {len(snapshot)} 檔納入計算，符合門檻 {len(filtered)} 檔，取前 {len(result)} 檔。"
        return result, status

    def _score_stock(self, d, r):
        """依技術指標計算單檔個股的量化評分（趨勢結構/動能/量價/突破/風險控制，滿分 80）。

        對應 index.html「指標與規則」頁面公告的評分權重：
        趨勢結構25＋動能20＋量價15＋突破15＋風險控制5（相對強弱10、估值加分4 於外部另計）。
        """
        detail = {}

        def num(x):
            try:
                v = float(x)
                return v if np.isfinite(v) else np.nan
            except (TypeError, ValueError):
                return np.nan

        close = num(r.get("close"))
        ma20, ma60, ma120 = num(r.get("ma20")), num(r.get("ma60")), num(r.get("ma120"))

        # 趨勢結構 25：多頭排列程度
        trend = 0
        if np.isfinite(close) and np.isfinite(ma20) and close > ma20:
            trend += 7
        if np.isfinite(ma20) and np.isfinite(ma60) and ma20 > ma60:
            trend += 6
        if np.isfinite(ma60) and np.isfinite(ma120) and ma60 > ma120:
            trend += 6
        if np.isfinite(close) and np.isfinite(ma120) and close > ma120:
            trend += 6
        detail["趨勢結構"] = trend

        # 動能 20：RSI、MACD 排列與柱體、20 日動能
        momentum = 0
        rsi = num(r.get("rsi"))
        macd_line, signal_line = num(r.get("macd_line")), num(r.get("signal_line"))
        macd_hist = num(r.get("macd_histogram"))
        mom20 = num(r.get("Momentum_20"))
        if np.isfinite(rsi) and rsi > 50:
            momentum += 5
        if np.isfinite(macd_hist) and macd_hist > 0:
            momentum += 5
        if np.isfinite(macd_line) and np.isfinite(signal_line) and macd_line > signal_line:
            momentum += 5
        if np.isfinite(mom20) and mom20 > 0:
            momentum += 5
        detail["動能"] = momentum

        # 量價 15：量比與 OBV 5 日斜率
        vp = 0
        vol_ratio = num(r.get("Volume_Ratio"))
        if np.isfinite(vol_ratio):
            if vol_ratio >= 1.2:
                vp += 8
            elif vol_ratio >= 1.0:
                vp += 4
        if len(d) > 6:
            obv_slope = num(d["OBV"].iloc[-1] - d["OBV"].iloc[-6])
            if np.isfinite(obv_slope) and obv_slope > 0:
                vp += 7
        detail["量價"] = vp

        # 突破 15：近 60 日高點突破/逼近
        breakout = 0
        if len(d) > 61 and np.isfinite(close):
            prior_high = num(d["high"].iloc[-61:-1].max())
            if np.isfinite(prior_high) and prior_high > 0:
                if close >= prior_high:
                    breakout = 15
                elif close >= prior_high * 0.97:
                    breakout = 8
        detail["突破"] = breakout

        # 風險控制 5：ATR 相對股價的波動度落在合理區間
        risk = 0
        atr = num(r.get("atr"))
        if np.isfinite(atr) and np.isfinite(close) and close > 0:
            norm_atr = atr / close * 100
            if 1.0 <= norm_atr <= 5.0:
                risk = 5
            elif norm_atr < 1.0 or 5.0 < norm_atr <= 8.0:
                risk = 2
        detail["風險控制"] = risk

        base_score = trend + momentum + vp + breakout + risk
        return base_score, detail

    def _relative_strength_scores(self, roc_series):
        """依 ROC_12 在候選池中的百分位排名換算相對強弱分數（滿分 10）。"""
        ranks = roc_series.rank(pct=True, na_option="bottom")

        def to_score(p):
            if pd.isna(p):
                return 0
            if p >= 0.8:
                return 10
            if p >= 0.6:
                return 7
            if p >= 0.4:
                return 5
            if p >= 0.2:
                return 3
            return 0

        return ranks.map(to_score)

    def _valuation_bonus(self, analyzer, ticker):
        """依 FinMind 本益比資料計算估值加分（滿分 4）；沒有 Token 或查無資料一律回傳 0。"""
        if not analyzer.finmind_token:
            return 0
        try:
            end_date = pd.Timestamp.now(tz="Asia/Taipei").date()
            start_date = end_date - pd.Timedelta(days=10)
            per_df = analyzer._fetch_finmind_api_data(
                "TaiwanStockPER", ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            if per_df.empty or "PER" not in per_df.columns:
                return 0
            per_df = per_df.dropna(subset=["PER"]).sort_values("date")
            if per_df.empty:
                return 0
            per = float(per_df["PER"].iloc[-1])
            if per <= 0:
                return 0
            if per <= 20:
                return 4
            if per <= 30:
                return 2
            return 0
        except Exception:
            return 0

    def scan(self, market="上市＋上櫃", candidate_count=60, min_score=60):
        """全市場選股掃描：先依成交金額取候選池，再逐檔計算技術面量化評分。"""
        snapshot = self.get_market_snapshot(market)
        if snapshot.empty:
            return pd.DataFrame(), "無法取得市場快照，請確認網路連線或稍後再試。"

        candidate_count = int(candidate_count)
        pool = snapshot.sort_values("成交金額", ascending=False).head(candidate_count).reset_index(drop=True)

        scored_rows = []
        roc_values = []
        for _, row in pool.iterrows():
            ticker = row["股票代號"]
            try:
                analyzer = StockAnalyzer(ticker)
                if not analyzer.fetch_data(period="1y"):
                    continue
                d = analyzer.df
                if len(d) < 60:
                    continue
                r = d.iloc[-1]
                base_score, detail = self._score_stock(d, r)
                bonus = self._valuation_bonus(analyzer, ticker)
                roc = r.get("ROC_12", np.nan)
                scored_rows.append({
                    "股票代號": ticker,
                    "股票名稱": row["股票名稱"],
                    "市場": row["市場"],
                    "收盤價": row["收盤價"],
                    "漲跌幅(%)": row["漲跌幅(%)"],
                    **detail,
                    "估值加分": bonus,
                    "_base_total": base_score + bonus,
                })
                roc_values.append(roc)
            except Exception as e:
                print(f"掃描 {ticker} 失敗：{e}")
                continue

        if not scored_rows:
            return pd.DataFrame(), f"已檢視 {len(pool)} 檔候選股，但沒有足夠歷史資料可供評分。"

        result_df = pd.DataFrame(scored_rows)
        result_df["相對強弱"] = self._relative_strength_scores(pd.Series(roc_values))
        result_df["總分"] = (result_df["_base_total"] + result_df["相對強弱"]).round(1)
        result_df = result_df.drop(columns=["_base_total"])

        filtered = result_df[result_df["總分"] >= float(min_score)]
        filtered = filtered.sort_values("總分", ascending=False).reset_index(drop=True)
        filtered.insert(0, "排名", range(1, len(filtered) + 1))

        cols = ["排名", "股票代號", "股票名稱", "市場", "收盤價", "漲跌幅(%)", "總分",
                "趨勢結構", "動能", "量價", "突破", "相對強弱", "風險控制", "估值加分"]
        filtered = filtered[[c for c in cols if c in filtered.columns]]

        status = (f"完成全市場選股掃描，候選 {len(pool)} 檔（依成交金額排序），"
                  f"入選 {len(filtered)} 檔（門檻 {min_score} 分）。")
        return filtered, status
