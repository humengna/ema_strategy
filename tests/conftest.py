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
