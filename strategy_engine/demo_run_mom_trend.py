"""
run_mom_trend_demo.py
------------------------------------
一个最小可用的策略回测 demo：

资产池：SPY, XLK, GLD, TLT
信号：
    - mom_120d > 0
    - trend_200d > 0
持仓规则：
    - 满足上述条件的资产等权分配
    - 剩余权重视为现金，收益为 rf_daily（来自 ^IRX）

输出：
    - strategy_engine/results/mom_trend_equity.parquet
        列: [date, eq_mom_trend, eq_spy_bh]
    - 控制台打印基本绩效指标
"""

import os
import numpy as np
import pandas as pd


# =============== 路径 ===============

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

DATA_PIPELINE_DIR = os.path.join(ROOT_DIR, "data_pipeline")
PROCESSED_DIR = os.path.join(DATA_PIPELINE_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_PIPELINE_DIR, "features")
MACRO_DIR = os.path.join(DATA_PIPELINE_DIR, "macro")

RESULTS_DIR = os.path.join(THIS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============== 工具函数 ===============

def load_prices_wide():
    path = os.path.join(PROCESSED_DIR, "prices_wide.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 prices_wide.parquet: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def load_features_long():
    path = os.path.join(FEATURES_DIR, "basic_tech_factors.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 basic_tech_factors.parquet: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_rf_daily():
    path = os.path.join(MACRO_DIR, "risk_free_irx.parquet")
    if not os.path.exists(path):
        print("[WARNING] 未找到 risk_free_irx.parquet，rf_daily 将为 0")
        return None
    rf = pd.read_parquet(path)
    if "date" not in rf.columns or "rf_daily" not in rf.columns:
        raise ValueError("risk_free_irx.parquet 需要包含 'date' 和 'rf_daily' 列")
    rf["date"] = pd.to_datetime(rf["date"])
    rf = rf.sort_values("date").set_index("date")
    return rf["rf_daily"]


def compute_returns_from_prices(prices_wide: pd.DataFrame) -> pd.DataFrame:
    rets = prices_wide.pct_change()
    return rets


def performance_stats(ret_series: pd.Series, rf_series: pd.Series = None) -> dict:
    """
    给定日度收益序列，计算年化收益、年化波动、Sharpe、最大回撤。
    ret_series / rf_series index 都是 date。
    """
    ret = ret_series.copy().dropna()
    n = len(ret)
    if n == 0:
        return {}

    # 总收益 & 年化收益
    total_return = (1 + ret).prod() - 1
    ann_return = (1 + total_return) ** (252.0 / n) - 1

    # 无风险 & 超额收益
    if rf_series is not None:
        rf_align = rf_series.reindex(ret.index).fillna(0.0)
        excess = ret - rf_align
    else:
        excess = ret.copy()

    ann_vol = excess.std() * np.sqrt(252)
    ann_excess_ret = excess.mean() * 252

    sharpe = ann_excess_ret / ann_vol if ann_vol > 0 else np.nan

    # 最大回撤
    equity = (1 + ret).cumprod()
    cummax = equity.cummax()
    dd = equity / cummax - 1
    max_dd = dd.min()

    return {
        "n_days": n,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol_excess": ann_vol,
        "ann_excess_ret": ann_excess_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# =============== 策略逻辑 ===============

def run_mom_trend_strategy():
    print("========== Run Mom + Trend Demo Strategy ==========")

    # ------- 1. 加载数据 -------
    prices_wide = load_prices_wide()
    features_long = load_features_long()
    rf_daily = load_rf_daily()

    risky_tickers = ["SPY", "XLK", "GLD", "TLT"]

    # 只保留风险资产的价格
    prices_wide = prices_wide[risky_tickers]
    # 确定共同起点：所有风险资产都有价格的最晚首日
    first_valid = prices_wide.apply(lambda s: s.first_valid_index())
    common_start = first_valid.max()
    prices_wide = prices_wide.loc[common_start:].copy()

    # 日收益
    returns_wide = compute_returns_from_prices(prices_wide).fillna(0.0)

    # rf_daily 对齐
    if rf_daily is not None:
        rf_daily = rf_daily.reindex(returns_wide.index).ffill().fillna(0.0)
    else:
        rf_daily = pd.Series(0.0, index=returns_wide.index, name="rf_daily")

    # ------- 2. 准备信号 (mom_120d, trend_200d) -------
    feat = features_long.copy()
    feat = feat[feat["ticker"].isin(risky_tickers)]

    # 只保留我们关心的列
    feat = feat[["date", "ticker", "mom_120d", "trend_200d"]]

    # pivot 成宽表：date × ticker
    mom120 = feat.pivot(index="date", columns="ticker", values="mom_120d")
    trend200 = feat.pivot(index="date", columns="ticker", values="trend_200d")

    # 对齐到 returns 的日期
    mom120 = mom120.reindex(returns_wide.index)
    trend200 = trend200.reindex(returns_wide.index)

    # 信号条件：mom_120d > 0 且 trend_200d > 0
    signals = (mom120 > 0) & (trend200 > 0)

    # ------- 3. 生成每日权重 -------
    # 信号为 True 的资产等权
    weights_risky = signals.astype(float)

    # 每天，把 True 的资产等权：除以行和
    row_sums = weights_risky.sum(axis=1)
    # 避免除以 0
    weights_risky = weights_risky.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)

    # 现金权重 = 1 - 风险资产总权重
    cash_weight = 1.0 - weights_risky.sum(axis=1)

    # ------- 4. 计算组合收益 -------
    # 风险资产部分收益
    port_ret_risky = (weights_risky * returns_wide).sum(axis=1)

    # 现金部分收益：用 rf_daily
    port_ret = port_ret_risky + cash_weight * rf_daily

    # ------- 5. 基准：SPY 买入持有 -------
    spy_ret = returns_wide["SPY"].fillna(0.0)

    # ------- 6. 生成 Equity Curve -------
    eq_mom_trend = (1.0 + port_ret).cumprod()
    eq_spy_bh = (1.0 + spy_ret).cumprod()

    equity_df = pd.DataFrame({
        "eq_mom_trend": eq_mom_trend,
        "eq_spy_bh": eq_spy_bh,
    })
    equity_df.index.name = "date"

    # ------- 7. 绩效指标 -------
    stats_mom = performance_stats(port_ret, rf_daily)
    stats_spy = performance_stats(spy_ret, rf_daily)

    print("\n=== Mom+Trend 策略绩效 ===")
    for k, v in stats_mom.items():
        print(f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}")

    print("\n=== SPY Buy&Hold 绩效 ===")
    for k, v in stats_spy.items():
        print(f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}")

    # ------- 8. 保存结果 -------
    output_path = os.path.join(RESULTS_DIR, "mom_trend_equity.parquet")
    equity_df.to_parquet(output_path)
    print(f"\n[OK] Equity curve 已保存 → {output_path}")

    return equity_df, stats_mom, stats_spy


if __name__ == "__main__":
    run_mom_trend_strategy()
