# -*- coding: utf-8 -*-
"""定期调仓组合:换仓时点、持仓上限、期间不动、资金守恒。"""
import numpy as np
import pandas as pd
import pytest

from ema_strategy.rebalance import (RebalanceParams, period_returns,
                                    report_metrics, simulate)

IDX = pd.bdate_range("2024-01-01", periods=200)
CODES = [f"{i:06d}.SZ" for i in range(8)]


def make_bars(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for c in CODES:
        close = 20 * np.cumprod(1 + rng.normal(0.0008, 0.02, len(IDX)))
        out[c] = pd.DataFrame({"open": close * 0.998, "close": close}, index=IDX)
    return out


def all_true() -> dict:
    return {c: pd.Series(True, index=IDX) for c in CODES}


def all_false() -> dict:
    return {c: pd.Series(False, index=IDX) for c in CODES}


def test_no_signal_means_capital_untouched():
    p = RebalanceParams(init_capital=1_000_000.0)
    res = simulate(make_bars(), all_false(), p)
    assert np.allclose(res["equity"].to_numpy(), p.init_capital)


def test_rebalance_happens_on_schedule():
    p = RebalanceParams(freq_days=21, lookback=21, max_positions=5)
    res = simulate(make_bars(), all_true(), p)
    marks = pd.DatetimeIndex(res["rebalance_dates"])
    gaps = np.diff([IDX.get_loc(d) for d in marks])
    assert set(gaps) == {21}                       # 间隔严格等于 freq_days


def test_first_rebalance_waits_for_lookback():
    """回看窗口没填满之前不应建仓,否则用的是不完整的信号。"""
    p = RebalanceParams(freq_days=21, lookback=30, max_positions=5)
    res = simulate(make_bars(), all_true(), p)
    first = res["rebalance_dates"][0]
    assert IDX.get_loc(first) >= p.lookback


def test_max_positions_never_exceeded():
    p = RebalanceParams(freq_days=21, lookback=21, max_positions=3)
    res = simulate(make_bars(), all_true(), p)
    assert res["holdings"].max() <= 3


def test_holdings_constant_between_rebalances():
    """期间不动:两次调仓之间持仓只数不得变化。"""
    p = RebalanceParams(freq_days=21, lookback=21, max_positions=5)
    res = simulate(make_bars(), all_true(), p)
    marks = [IDX.get_loc(d) for d in res["rebalance_dates"]]
    h = res["holdings"].to_numpy()
    for a, b in zip(marks, marks[1:]):
        assert len(set(h[a:b])) == 1


def test_only_signalled_stocks_are_held():
    """只有回看窗口内触发过的股票才会被买入。"""
    p = RebalanceParams(freq_days=21, lookback=21, max_positions=8)
    sigs = all_false()
    sigs[CODES[0]] = pd.Series(True, index=IDX)
    res = simulate(make_bars(), sigs, p)
    for codes in res["picks"]["codes"]:
        assert codes in ("", CODES[0])


def test_score_decides_selection():
    p = RebalanceParams(freq_days=21, lookback=21, max_positions=1)
    scores = {c: pd.Series(0.0, index=IDX) for c in CODES}
    scores[CODES[3]] = pd.Series(99.0, index=IDX)
    res = simulate(make_bars(), all_true(), p, scores_by_code=scores)
    assert set(res["picks"]["codes"]) == {CODES[3]}


def test_cost_reduces_equity():
    free = simulate(make_bars(), all_true(),
                    RebalanceParams(freq_days=21, lookback=21, max_positions=5, cost=0.0))
    charged = simulate(make_bars(), all_true(),
                       RebalanceParams(freq_days=21, lookback=21, max_positions=5, cost=0.02))
    assert charged["equity"].iloc[-1] < free["equity"].iloc[-1]


def test_equity_stays_positive():
    res = simulate(make_bars(5), all_true(),
                   RebalanceParams(freq_days=21, lookback=21, max_positions=5))
    assert (res["equity"] > 0).all()


def test_period_returns_align_with_rebalances():
    p = RebalanceParams(freq_days=21, lookback=21, max_positions=5)
    res = simulate(make_bars(), all_true(), p)
    pr = period_returns(res["equity"], res["rebalance_dates"])
    assert len(pr) == len(res["rebalance_dates"]) - 1


def test_report_metrics_keys():
    p = RebalanceParams(freq_days=21, lookback=21, max_positions=5)
    m = report_metrics(simulate(make_bars(), all_true(), p))
    for key in ("总收益", "年化收益", "Sharpe", "最大回撤", "调仓期数", "期胜率", "夏普(按期)"):
        assert key in m


def test_rejects_too_short_freq():
    with pytest.raises(ValueError, match="过短"):
        RebalanceParams(freq_days=1)


def test_rejects_bad_lookback():
    with pytest.raises(ValueError, match="lookback"):
        RebalanceParams(lookback=0)


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="为空"):
        simulate({}, {}, RebalanceParams())
