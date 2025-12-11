"""
build_features.py
------------------------------------
从 processed/prices_wide.parquet + macro/risk_free_irx.parquet
构建基础技术因子，输出到 data_pipeline/features/basic_tech_factors.parquet

因子包括：
- ret_1d      : 日收益
- mom_20d     : 20日动量
- mom_60d     : 60日动量
- mom_120d    : 120日动量
- vol_20d     : 20日滚动波动率（年化）
- ma_200d     : 200日移动平均价格
- trend_200d  : 相对200日均线的偏离度 (P/MA200 - 1)
- sharpe_60d  : 60日滚动Sharpe（超额收益）

依赖:
    pip install pandas pyarrow
"""

import os
import pandas as pd
import numpy as np

# ------------ 路径设置 ------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

PROCESSED_DIR = os.path.join(ROOT_DIR, "processed")
FEATURES_DIR = os.path.join(ROOT_DIR, "features")
MACRO_DIR = os.path.join(ROOT_DIR, "macro")

os.makedirs(FEATURES_DIR, exist_ok=True)


# ------------ 工具函数 ------------

def load_prices_wide(path=None):
    """读取 prices_wide.parquet，返回 DataFrame: index=date, columns=tickers"""
    if path is None:
        path = os.path.join(PROCESSED_DIR, "prices_wide.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"prices_wide.parquet 不存在: {path}")
    df = pd.read_parquet(path)
    # 确保 index 是 datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def build_returns_from_prices(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """从价格矩阵构建日收益矩阵（简单收益）"""
    rets = prices_wide.pct_change()
    return rets


def load_rf_daily(prices_index: pd.DatetimeIndex) -> pd.Series:
    """
    从 macro/risk_free_irx.parquet 读取 rf_daily，并对齐到 prices 的日期。
    如果文件不存在，则返回全0的 rf_daily。
    """
    rf_path = os.path.join(MACRO_DIR, "risk_free_irx.parquet")
    if not os.path.exists(rf_path):
        print("[WARNING] 未找到 risk_free_irx.parquet，rf_daily 将全为 0")
        rf_daily = pd.Series(0.0, index=prices_index, name="rf_daily")
        return rf_daily

    rf = pd.read_parquet(rf_path)
    if "date" not in rf.columns or "rf_daily" not in rf.columns:
        raise ValueError("risk_free_irx.parquet 需要包含 'date' 和 'rf_daily' 列")

    rf["date"] = pd.to_datetime(rf["date"])
    rf = rf.sort_values("date").set_index("date")

    rf_daily = rf["rf_daily"].reindex(prices_index).ffill()
    rf_daily.name = "rf_daily"
    return rf_daily


# ------------ 主逻辑：构建因子 ------------

def build_basic_tech_factors(prices_wide: pd.DataFrame,
                             returns_wide: pd.DataFrame,
                             rf_daily: pd.Series) -> pd.DataFrame:
    """
    从价格矩阵、收益矩阵、rf_daily 构建 long 格式的因子表
    返回 DataFrame: [date, ticker, ret_1d, mom_20d, mom_60d, mom_120d, vol_20d, ma_200d, trend_200d, sharpe_60d]
    """
    tickers = prices_wide.columns.tolist()
    dates = prices_wide.index

    # 为了方便计算，把 rf_daily 也变成 DataFrame，方便广播
    rf_df = rf_daily.to_frame().reindex(dates)

    all_rows = []

    for ticker in tickers:
        px = prices_wide[ticker]
        ret = returns_wide[ticker]

        # 基础 return
        df_feat = pd.DataFrame(index=dates)
        df_feat["ret_1d"] = ret

        # 多周期动量
        df_feat["mom_20d"] = px / px.shift(20) - 1
        df_feat["mom_60d"] = px / px.shift(60) - 1
        df_feat["mom_120d"] = px / px.shift(120) - 1

        # 波动率（20日滚动，年化）
        df_feat["vol_20d"] = (
            ret.rolling(window=20)
            .std()
            * np.sqrt(252)
        )

        # 长期均线 + 趋势
        df_feat["ma_200d"] = px.rolling(window=200).mean()
        df_feat["trend_200d"] = px / df_feat["ma_200d"] - 1

        # 60日滚动 Sharpe（超额收益）
        # 对齐 rf_daily
        excess = ret.sub(rf_df["rf_daily"], axis=0)
        rolling_mean = excess.rolling(window=60).mean()
        rolling_std = excess.rolling(window=60).std()
        df_feat["sharpe_60d"] = (rolling_mean / rolling_std) * np.sqrt(252)

        df_feat = df_feat.reset_index().rename(columns={"index": "date"})
        df_feat["ticker"] = ticker

        all_rows.append(df_feat)

    # 合并所有 ticker
    features_long = pd.concat(all_rows, axis=0, ignore_index=True)

    # 排序 & 列顺序
    col_order = [
        "date",
        "ticker",
        "ret_1d",
        "mom_20d",
        "mom_60d",
        "mom_120d",
        "vol_20d",
        "ma_200d",
        "trend_200d",
        "sharpe_60d",
    ]
    features_long = features_long[col_order].sort_values(["ticker", "date"]).reset_index(drop=True)
    return features_long


def main():
    print("========== Build Basic Tech Factors ==========")

    prices_wide = load_prices_wide()
    print(f"[INFO] prices_wide 形状: {prices_wide.shape}")

    returns_wide = build_returns_from_prices(prices_wide)
    print(f"[INFO] returns_wide 形状: {returns_wide.shape}")

    rf_daily = load_rf_daily(prices_wide.index)
    print(f"[INFO] rf_daily 长度: {len(rf_daily)}")

    features_long = build_basic_tech_factors(prices_wide, returns_wide, rf_daily)

    output_path = os.path.join(FEATURES_DIR, "basic_tech_factors.parquet")
    features_long.to_parquet(output_path, index=False)

    print(f"[OK] Saved basic tech factors → {output_path}")
    print(features_long.head())


if __name__ == "__main__":
    main()
