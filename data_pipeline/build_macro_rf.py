"""
build_risk_free_irx.py
------------------------------------
从 yfinance 下载 ^IRX (13 Week T-Bill Yield)，
转换为日度无风险收益率 rf_daily，并存到 data_pipeline/macro/risk_free_irx.parquet

依赖:
    pip install yfinance pandas pyarrow
"""

import os
import pandas as pd
import yfinance as yf

DEFAULT_START_DATE = "1980-01-01"

def download_irx(start=DEFAULT_START_DATE, end=None):
    print("[INFO] Downloading ^IRX (13W T-Bill Yield) ...")
    df = yf.download(
        "^IRX",
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df is None or df.empty:
        raise ValueError("^IRX 返回空数据")

    # 处理 MultiIndex 列的兜底
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # 我们只关心 adj_close
    if "adj_close" not in df.columns:
        raise ValueError("^IRX 数据中没有 adj_close 列，当前列: " + str(df.columns))

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df[["date", "adj_close"]]


def build_risk_free(df_irx: pd.DataFrame) -> pd.DataFrame:
    """
    将 ^IRX 的年化利率（百分比）转换为日度无风险收益率。
    假设 adj_close 是年化百分数，例如 5.0 表示 5%。
    """
    out = df_irx.copy()
    out = out.rename(columns={"adj_close": "irx_annual_pct"})

    # 年化百分比 -> 年化小数
    out["irx_annual"] = out["irx_annual_pct"] / 100.0

    # 粗略转换为日收益 (连续 252 个交易日)
    out["rf_daily"] = out["irx_annual"] / 252.0

    return out


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    macro_dir = os.path.join(script_dir, "macro")
    os.makedirs(macro_dir, exist_ok=True)

    df_irx_raw = download_irx()
    df_rf = build_risk_free(df_irx_raw)

    output_path = os.path.join(macro_dir, "risk_free_irx.parquet")
    df_rf.to_parquet(output_path, index=False)

    print(f"[OK] Saved risk-free series → {output_path}")
    print(df_rf.head())


if __name__ == "__main__":
    main()
