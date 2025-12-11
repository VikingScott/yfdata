"""
build_price_panel.py
------------------------------------
从 data_pipeline/raw/ 目录里的一票票 ETF parquet，
拼接出宽表价格矩阵 prices_wide.parquet。

输入（已存在）:
    data_pipeline/raw/{ticker}.parquet
        必须至少包含:
        - date
        - adj_close (优先)
        - close     (如果没有 adj_close，就退回用 close)

输出:
    data_pipeline/processed/prices_wide.parquet

结构:
    index: date (DatetimeIndex, 升序)
    columns: 每个 ticker 一列，值为 adj_close 或 close
"""

import os
import pandas as pd

# ------------ 路径设置 ------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

RAW_DIR = os.path.join(ROOT_DIR, "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "processed")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_tickers(config_path=None):
    """读取 config/tickers.csv，获取 ticker 列表"""
    if config_path is None:
        config_path = os.path.join(CONFIG_DIR, "tickers.csv")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到 tickers.csv: {config_path}")
    df = pd.read_csv(config_path)
    if "ticker" not in df.columns:
        raise ValueError("tickers.csv 必须包含 'ticker' 列")
    return df["ticker"].tolist()


def load_single_price_series(ticker: str) -> pd.Series:
    """
    从 raw/{ticker}.parquet 读取该资产的价格序列（优先 adj_close，没有则用 close）。
    返回: Series，index=date (DatetimeIndex)，name=ticker
    """
    path = os.path.join(RAW_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到原始数据文件: {path}")

    df = pd.read_parquet(path)

    # 列名统一一下，保险
    df.columns = [str(c).lower() for c in df.columns]

    if "date" not in df.columns:
        raise ValueError(f"{ticker} 数据中没有 'date' 列，当前列: {df.columns}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 价格列优先 adj_close，其次 close
    price_col = None
    if "adj_close" in df.columns:
        price_col = "adj_close"
    elif "close" in df.columns:
        price_col = "close"
    else:
        raise ValueError(f"{ticker} 数据中既没有 'adj_close' 也没有 'close' 列，当前列: {df.columns}")

    s = df.set_index("date")[price_col].copy()
    s.name = ticker

    return s


def build_prices_wide(tickers):
    """
    从多只资产的单独 price Series 拼成宽表 DataFrame
    index: date (全量 union)
    columns: tickers
    """
    series_list = []
    for ticker in tickers:
        print(f"[INFO] 处理 {ticker} ...")
        s = load_single_price_series(ticker)
        series_list.append(s)

    # 按列拼接，按日期 union，对齐
    prices_wide = pd.concat(series_list, axis=1).sort_index()
    return prices_wide


def main():
    print("========== Build Price Panel (prices_wide) ==========")

    tickers = load_tickers()
    print(f"[INFO] 从 tickers.csv 读取到标的: {tickers}")

    prices_wide = build_prices_wide(tickers)
    print(f"[INFO] prices_wide 形状: {prices_wide.shape}")
    print("[INFO] 列预览:", prices_wide.columns.tolist())
    print("[INFO] 日期范围:", prices_wide.index.min(), "→", prices_wide.index.max())

    # 简单缺失统计
    print("\n[INFO] 每个资产的缺失值数量:")
    print(prices_wide.isna().sum())

    # 保存
    output_path = os.path.join(PROCESSED_DIR, "prices_wide.parquet")
    prices_wide.to_parquet(output_path)
    print(f"\n[OK] 已保存 prices_wide → {output_path}")


if __name__ == "__main__":
    main()
