# -*- coding: utf-8 -*-
"""回测口径:买卖时点与可成交判定。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from backtest import entry_tradable, forward_return  # noqa: E402

IDX = pd.bdate_range("2024-01-01", periods=10)


def _bars(open_, close):
    return pd.DataFrame({"open": open_, "high": np.array(close) * 1.01,
                         "low": np.array(close) * 0.99, "close": close}, index=IDX)


def test_forward_return_buys_next_open_sells_hold_close():
    """t 日信号 -> t+1 开盘买入 -> t+hold 收盘卖出。"""
    open_ = [10.0] * 10
    close = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    fwd = forward_return(_bars(open_, close), hold=5)
    # 第0日信号:第1日开盘 10.0 买入,第5日收盘 6.0 卖出
    assert fwd.iloc[0] == pytest_approx(6.0 / 10.0 - 1)


def pytest_approx(x, tol=1e-12):
    class _A:
        def __eq__(self, other):
            return abs(other - x) < tol
        def __repr__(self):
            return f"~{x}"
    return _A()


def test_forward_return_tail_is_nan():
    """末尾不足 hold 天的信号没有完整持仓期,必须是 NaN,不能用残缺区间充数。"""
    fwd = forward_return(_bars([10.0] * 10, list(range(1, 11))), hold=5)
    assert fwd.iloc[-5:].isna().all()
    assert fwd.iloc[:-5].notna().all()


def test_forward_return_hold_one():
    open_ = [10.0] * 10
    close = [float(i) for i in range(1, 11)]
    fwd = forward_return(_bars(open_, close), hold=1)
    # 第0日信号:第1日开盘买、第1日收盘卖
    assert fwd.iloc[0] == pytest_approx(close[1] / 10.0 - 1)


def test_entry_tradable_blocks_limit_up_open():
    """次日开盘一字涨停买不进。"""
    bars = pd.DataFrame({
        "open": [10.0, 11.0, 10.5], "high": [10.0, 11.0, 10.5],
        "low": [10.0, 11.0, 10.5], "close": [10.0, 11.0, 10.5],
        "preClose": [9.9, 10.0, 11.0],
    }, index=IDX[:3])
    ok = entry_tradable(bars, "600000.SH")
    # 第0日信号 -> 次日开盘 11.00,前收 10.00,正好涨停 -> 买不进
    assert not ok.iloc[0]
    # 第1日信号 -> 次日开盘 10.50,前收 11.00,未涨停 -> 可买
    assert ok.iloc[1]
