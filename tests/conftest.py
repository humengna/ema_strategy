# -*- coding: utf-8 -*-
"""测试夹具。

真实行情样本仅用于验证实现正确性(不变量、边界、一致性),
不是 A 股数据,不代表策略在 A 股的有效性。

数据来源:matplotlib/mplfinance 仓库的 examples/data(BSD 许可的公开示例数据)。
选它们是因为不依赖 QMT 即可跑通全部测试;A 股口径请用 xtdata 自行复跑。
"""
import os

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = {
    "INTC": "intc.csv",
    "GOOG": "yahoofinance-GOOG-20040819-20180120.csv",
    "AAPL": "yahoofinance-AAPL-20040819-20180120.csv",
}


def _load(filename: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(HERE, filename), parse_dates=["Date"]).set_index("Date")
    df = df.rename(columns=str.lower).sort_index()
    return df[["open", "high", "low", "close"]].dropna()


@pytest.fixture(scope="session")
def samples() -> dict:
    return {name: _load(fn) for name, fn in SAMPLES.items()}


@pytest.fixture(scope="session")
def intc(samples) -> pd.DataFrame:
    return samples["INTC"]


@pytest.fixture(scope="session")
def intc_with_volume() -> pd.DataFrame:
    """带真实成交量的样本,供筹码/放量相关测试使用。

    成交量必须用真实数据:设成常数会让「放量」条件永远不成立,
    测试看似通过实则什么都没验证。
    """
    df = pd.read_csv(os.path.join(HERE, SAMPLES["INTC"]), parse_dates=["Date"]).set_index("Date")
    df = df.rename(columns=str.lower).sort_index()
    out = df[["open", "high", "low", "close", "volume"]].dropna().copy()
    out["volume"] = out["volume"] / 100.0          # 换算成「手」,与 xtdata 口径一致
    out["amount"] = out["close"] * out["volume"] * 100
    return out
