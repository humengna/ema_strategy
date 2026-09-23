# -*- coding: utf-8 -*-
"""筹码分布与获利比例。"""
import numpy as np
import pandas as pd
import pytest

from ema_strategy.chips import profit_ratio

N = 120
IDX = pd.bdate_range("2024-01-01", periods=N)
FLOAT = 1e9


def make(close: np.ndarray, amp: float = 0.01) -> pd.DataFrame:
    return pd.DataFrame({"high": close * (1 + amp), "low": close * (1 - amp),
                         "close": close, "volume": np.full(N, 2e6),
                         "amount": close * 2e6 * 100}, index=IDX)


def test_uptrend_mostly_profitable():
    assert profit_ratio(make(np.linspace(10, 30, N)), FLOAT).iloc[-1] > 0.85


def test_downtrend_mostly_losing():
    assert profit_ratio(make(np.linspace(30, 10, N)), FLOAT).iloc[-1] < 0.15


def test_flat_is_about_half():
    """横盘时成本集中在当前价附近,获利比例应接近 50%。"""
    assert profit_ratio(make(np.full(N, 20.0)), FLOAT).iloc[-1] == pytest.approx(0.5, abs=0.05)


def test_no_lookahead(intc):
    """同一天的获利比例不得因为后面多了K线而改变。

    回归测试:早期实现用整段数据的 min/max 划价格网格,
    截断到当天算得 63.4%、用全历史算得 100% —— 典型的未来函数。
    """
    bars = intc.copy()
    bars["volume"] = 2e6
    bars["amount"] = bars["close"] * 2e6 * 100
    full = profit_ratio(bars, 3.3e9)
    for cut in (200, 600, 1200):
        trunc = profit_ratio(bars.iloc[:cut], 3.3e9)
        assert np.allclose(full.iloc[:cut].to_numpy(), trunc.to_numpy(), equal_nan=True)


def test_bounded_0_1(intc):
    bars = intc.copy()
    bars["volume"] = 2e6
    ratio = profit_ratio(bars, 3.3e9).dropna()
    assert len(ratio) > 0
    assert ratio.between(0.0, 1.0).all()


def test_warmup_is_nan():
    ratio = profit_ratio(make(np.full(N, 20.0)), FLOAT, min_periods=30)
    assert ratio.iloc[:29].isna().all()
    assert ratio.iloc[29:].notna().all()


def test_missing_float_shares_gives_no_decay():
    """流通股本缺失时换手率无从计算,筹码不衰减,结果仍应有界。"""
    ratio = profit_ratio(make(np.linspace(10, 30, N)), float_shares=0)
    assert ratio.dropna().between(0.0, 1.0).all()


def test_requires_columns():
    with pytest.raises(ValueError, match="缺少列"):
        profit_ratio(pd.DataFrame({"close": [1.0]}, index=IDX[:1]), FLOAT)


def test_empty_input():
    empty = pd.DataFrame(columns=["high", "low", "close", "volume"],
                         index=pd.DatetimeIndex([]))
    assert len(profit_ratio(empty, FLOAT)) == 0


def test_limit_up_bar_does_not_crash():
    """一字板:high == low,分布退化为单个价位。"""
    close = np.full(N, 20.0)
    bars = pd.DataFrame({"high": close, "low": close, "close": close,
                         "volume": np.full(N, 2e6), "amount": close * 2e6 * 100}, index=IDX)
    assert profit_ratio(bars, FLOAT).dropna().between(0.0, 1.0).all()


def test_turnover_uses_hand_unit():
    """volume 以「手」计,换手率要乘 100;漏乘会高估获利比例。

    漏乘 -> 换手率小两个数量级 -> 筹码几乎不衰减 -> 早期低成本筹码一直留着
    -> 上涨行情的获利比例被系统性高估。对「获利筹码 > 90%」的阈值筛选,
    这是危险方向:一批本不该入选的票会假装达标。
    """
    bars = make(np.linspace(10, 30, N))
    correct = profit_ratio(bars, FLOAT, volume_unit=100).iloc[-1]
    wrong = profit_ratio(bars, FLOAT, volume_unit=1).iloc[-1]
    assert wrong > correct
    assert wrong > 0.99 and correct < 0.95
