import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# ────── 基本參數 ────────────────────────────────────────────────────────────────
API_ENDPOINT = "https://fapi.binance.com/fapi/v1/klines"
OUTPUT_DIR   = "./../Data"              # 依需求修改
LIMIT        = 1500                     # Binance 最大值
SYMBOLS      = ["BTC", "ETH", "BNB", "LINK", "AVAX",
                "XRP", "DOGE", "SUI", "BCH", "SOL"]
INTERVALS    = ["1m", "1h"]             # 想加別的就放進來

START_TIME   = datetime(2020, 1, 1, tzinfo=timezone.utc)
END_TIME     = datetime(2025, 7, 7, tzinfo=timezone.utc)

# 間隔對應的 timedelta（LIMIT 根所跨的長度）
STEP = {
    "1m": timedelta(minutes=LIMIT),
    "3m": timedelta(minutes=3 * LIMIT),
    "5m": timedelta(minutes=5 * LIMIT),
    "15m": timedelta(minutes=15 * LIMIT),
    "30m": timedelta(minutes=30 * LIMIT),
    "1h": timedelta(hours=LIMIT),
    "2h": timedelta(hours=2 * LIMIT),
    "4h": timedelta(hours=4 * LIMIT),
    "1d": timedelta(days=LIMIT)
}

# K 線欄位
COLS = [
    "Open time", "Open", "High", "Low", "Close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
]

# ────── 工具函式 ────────────────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str,
                 start: datetime, end: datetime) -> pd.DataFrame:
    """連續抓到 end（不含）為止，回傳整合後的 DataFrame。"""
    all_rows = []
    cursor = start
    delta  = STEP[interval]

    while cursor < end:
        params = {
            "symbol"   : symbol,
            "interval" : interval,
            "limit"    : LIMIT,
            "startTime": int(cursor.timestamp() * 1000)
        }
        try:
            resp = requests.get(API_ENDPOINT, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logging.error(f"{symbol} {interval} @ {cursor}: {exc}")
            break

        # API 回 error dict（如頻率限制、合約下架…）
        if isinstance(data, dict) and data.get("code") is not None:
            logging.error(f"{symbol} {interval} @ {cursor}: {data}")
            break

        if not data:                       # 沒資料就代表到尾了
            break

        all_rows.extend(data)
        cursor += delta

    df = pd.DataFrame(all_rows, columns=COLS)
    if df.empty:
        return df

    df["Open time"]  = pd.to_datetime(df["Open time"],  unit="ms", utc=True)
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms", utc=True)
    df.set_index("Open time", inplace=True)

    # 去重（偶爾會因為 API 疊到）
    df = df[~df.index.duplicated(keep="first")]
    return df

# ────── 主程式 ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    for base in SYMBOLS:
        symbol = f"{base.upper()}USDT"
        for interval in INTERVALS:
            logging.info(f"Start {symbol} {interval}")
            df = fetch_klines(symbol, interval, START_TIME, END_TIME)
            if df.empty:
                logging.warning(f"No data retrieved for {symbol} {interval}")
                continue

            fname = (f"Binance_{interval}_{symbol}_"
                     f"{START_TIME.date()}_{END_TIME.date()}.csv")
            path  = os.path.join(OUTPUT_DIR, fname)
            df.to_csv(path)
            logging.info(f"Saved → {path}  ({len(df):,} rows)")

if __name__ == "__main__":
    main()