# -*- coding: utf-8 -*-
"""组合回测:资金守恒、持有期、仓位约束、一字板过滤。"""
import numpy as np
import pandas as pd
import pytest

from ema_strategy.portfolio import PortfolioParams, metrics, simulate

IDX = pd.bdate_range("2024-01-01", periods=60)
CODES = ["A", "B", "C", "D", "E", "F"]


def make_bars(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for c in CODES:
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, len(IDX)))
        out[c] = pd.DataFrame({"open": close * 0.999, "close": close}, index=IDX)
    return out


def all_false() -> dict:
    return {c: pd.Series(False, index=IDX) for c in CODES}


def test_no_signal_means_capital_untouched():
    """没有任何信号时,权益必须恒等于初始资金 —— 任何漂移都是记账漏洞。"""
    p = PortfolioParams(init_capital=1_000_000.0)
    res = simulate(make_bars(), all_false(), p)
    assert np.allclose(res["equity"].to_numpy(), p.init_capital)
    assert len(res["trades"]) == 0


def test_hold_days_counts_trading_days():
    """持有天数应为交易日计数(含买入日与卖出日),不是索引差。"""
    p = PortfolioParams(hold_days=5, max_positions=6, max_new_per_day=6)
    sigs = all_false()
    sigs["A"].iloc[10] = True
    res = simulate(make_bars(), sigs, p)
    assert len(res["trades"]) == 1
    assert res["trades"]["hold_days"].iloc[0] == 5


def test_hold_days_param_respected():
    for hold in (2, 3, 10):
        p = PortfolioParams(hold_days=hold, max_positions=6, max_new_per_day=6)
        sigs = all_false()
        sigs["A"].iloc[10] = True
        res = simulate(make_bars(), sigs, p)
        assert res["trades"]["hold_days"].iloc[0] == hold


def test_max_positions_never_exceeded():
    p = PortfolioParams(hold_days=5, max_positions=3, max_new_per_day=3)
    sigs = {c: pd.Series(True, index=IDX) for c in CODES}   # 天天全体触发
    res = simulate(make_bars(), sigs, p)
    assert res["holdings"].max() <= p.max_positions


def test_new_per_day_limits_entries():
    p = PortfolioParams(hold_days=5, max_positions=6, max_new_per_day=2)
    sigs = {c: pd.Series(True, index=IDX) for c in CODES}
    res = simulate(make_bars(), sigs, p)
    entries = res["trades"].groupby("entry_date").size()
    assert entries.max() <= 2


def test_default_new_per_day_is_derived():
    p = PortfolioParams(hold_days=5, max_positions=20)
    assert p.new_per_day == 4          # ceil(20 / 5)


def test_untradable_signals_are_skipped():
    """次日开盘一字涨停的信号必须跳过并计数。"""
    p = PortfolioParams(hold_days=5, max_positions=6, max_new_per_day=6)
    sigs = all_false()
    sigs["A"].iloc[10] = True
    untradable = {c: pd.Series(True, index=IDX) for c in CODES}
    untradable["A"].iloc[10] = False
    res = simulate(make_bars(), sigs, p, tradable_by_code=untradable)
    assert len(res["trades"]) == 0
    assert res["blocked"] == 1


def test_score_decides_who_gets_the_slot():
    """空位不足时按 score 降序挑选。"""
    p = PortfolioParams(hold_days=5, max_positions=1, max_new_per_day=1)
    sigs = all_false()
    for c in ("A", "B", "C"):
        sigs[c].iloc[10] = True
    scores = {c: pd.Series(0.0, index=IDX) for c in CODES}
    scores["C"].iloc[10] = 99.0
    res = simulate(make_bars(), sigs, p, scores_by_code=scores)
    assert res["trades"]["code"].iloc[0] == "C"


def test_cost_reduces_return():
    sigs = all_false()
    sigs["A"].iloc[10] = True
    free = simulate(make_bars(), sigs, PortfolioParams(hold_days=5, max_positions=6,
                                                       max_new_per_day=6, cost=0.0))
    charged = simulate(make_bars(), sigs, PortfolioParams(hold_days=5, max_positions=6,
                                                          max_new_per_day=6, cost=0.01))
    assert charged["trades"]["ret"].iloc[0] < free["trades"]["ret"].iloc[0]
    assert charged["equity"].iloc[-1] < free["equity"].iloc[-1]


def test_equity_never_negative():
    p = PortfolioParams(hold_days=5, max_positions=3, max_new_per_day=3)
    sigs = {c: pd.Series(True, index=IDX) for c in CODES}
    res = simulate(make_bars(7), sigs, p)
    assert (res["equity"] > 0).all()


def test_metrics_shape():
    p = PortfolioParams(hold_days=5, max_positions=3, max_new_per_day=3)
    sigs = {c: pd.Series(True, index=IDX) for c in CODES}
    res = simulate(make_bars(), sigs, p)
    m = metrics(res["equity"], res["trades"], res["holdings"])
    for key in ("总收益", "年化收益", "最大回撤", "Sharpe", "单笔胜率", "平均持仓数"):
        assert key in m
    assert m["最大回撤"] <= 0


def test_rejects_t_plus_zero():
    """hold_days=1 等于当日买当日卖,A股 T+1 下不可执行,必须直接拒绝。"""
    with pytest.raises(ValueError, match="T\\+0"):
        PortfolioParams(hold_days=1)


def test_no_same_day_reentry():
    """一只票在卖出当日不得被重新买入 —— 先开盘买、后收盘卖的顺序保证了这一点。"""
    p = PortfolioParams(hold_days=5, max_positions=6, max_new_per_day=6)
    sigs = {c: pd.Series(True, index=IDX) for c in CODES}
    res = simulate(make_bars(), sigs, p)
    for code, grp in res["trades"].groupby("code"):
        grp = grp.sort_values("entry_date")
        assert (grp["entry_date"].shift(-1).dropna().to_numpy()
                > grp["exit_date"].iloc[:-1].to_numpy()).all()


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="为空"):
        simulate({}, {}, PortfolioParams())
