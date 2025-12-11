"""
download_yf.py
------------------------------------
从 yfinance 下载 ETF 历史数据，并存到 data_pipeline/raw/ 目录。

依赖:
    pip install yfinance pandas pyarrow

目录结构:
    data_pipeline/
      config/
        tickers.csv
      raw/
        SPY.parquet
        XLK.parquet
        ...


"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

# 允许你手动设定历史的最早日期，例如回测从 1990 年开始
DEFAULT_START_DATE = "1980-01-01"

# 获取项目根目录（scripts 向上两级）
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def load_tickers(config_path=None):
    """读取 tickers.csv 文件"""
    if config_path is None:
        config_path = PROJECT_ROOT / "data_pipeline" / "config" / "tickers.csv"
    tickers = pd.read_csv(config_path)
    if "ticker" not in tickers.columns:
        raise ValueError("tickers.csv 必须包含 'ticker' 列")
    return tickers["ticker"].tolist()


def download_single_ticker(ticker, start=DEFAULT_START_DATE, end=None):
    """从 yfinance 下载单一 ETF 数据"""
    print(f"[INFO] Downloading {ticker} ...")

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,   # 保留原 OHLCV + Adj Close
            progress=False,
            group_by="column",   # ✅ 关键：不要按 ticker 分组，避免 MultiIndex
        )

        if isinstance(df, pd.DataFrame) and df.empty:
            print(f"[WARNING] {ticker} 返回空数据！")
            return None

        # 万一某些版本还是给了 MultiIndex，做个兜底
        if isinstance(df.columns, pd.MultiIndex):
            # 尝试只取第一层（Open/High/Low/Close/...）
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        # 列名统一成小写 + 下划线，先转成字符串避免 'tuple' 问题
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        return df

    except Exception as e:
        print(f"[ERROR] 下载 {ticker} 时发生错误: {e}")
        return None


def save_raw(df, ticker, output_dir=None):
    """保存原始 parquet 文件"""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data_pipeline" / "raw"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{ticker}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"[OK] Saved → {output_path}")


def main():
    print("========== YF Downloader Start ==========")

    tickers = load_tickers()
    print(f"[INFO] Loaded tickers: {tickers}")

    for ticker in tickers:
        df = download_single_ticker(ticker)
        if df is not None:
            save_raw(df, ticker)
        else:
            print(f"[SKIP] 未保存 {ticker}。")

    print("========== Done ==========")


if __name__ == "__main__":
    main()