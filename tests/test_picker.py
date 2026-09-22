# -*- coding: utf-8 -*-
"""选股器:evaluate 与 explain 必须给出一致结论。"""
import pandas as pd
import pytest

from ema_strategy.feed import PRICE_COLS
from ema_strategy.picker import attach_tradability, evaluate, explain
from ema_strategy.sequence import Params, run

P = Params()


def _with_aux(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["volume"] = 1
    out["preClose"] = out["close"].shift(1)
    return out


def test_evaluate_hits_on_confirm_day(intc):
    seq = run(intc, P)["sequences"]
    hit = seq[seq["triggered"]].iloc[0]
    upto = intc.loc[:hit["confirm_date"]]
    res = evaluate("600000.SH", _with_aux(upto), P)
    assert res is not None
    assert res["confirm_date"] == hit["confirm_date"]
    assert res["start_bar"] == "贯穿"
    assert res["confirm_bar"] == "跳空"


def test_evaluate_misses_day_after(intc):
    """触发只在确认点当日有效,次日即失效。"""
    seq = run(intc, P)["sequences"]
    hit = seq[seq["triggered"]].iloc[0]
    i = intc.index.get_loc(hit["confirm_date"])
    assert evaluate("600000.SH", _with_aux(intc.iloc[:i + 2]), P) is None


def test_evaluate_returns_none_when_too_short(intc):
    assert evaluate("600000.SH", _with_aux(intc.iloc[:20]), P) is None


def test_explain_agrees_with_evaluate(samples):
    """逐日比对两条路径的结论,不允许任何分歧。"""
    checked = mismatch = 0
    for code, df in samples.items():
        bars = _with_aux(df)
        for i in range(40, len(bars), 11):
            sub = bars.iloc[:i]
            picked = evaluate(code, sub, P) is not None
            explained = "【入选】" in explain(code, sub, P)
            checked += 1
            mismatch += int(picked != explained)
    assert checked > 100
    assert mismatch == 0


def test_relaxing_start_bar_admits_more(samples):
    strict = loose = 0
    for code, df in samples.items():
        bars = _with_aux(df)
        for i in range(40, len(bars), 3):
            sub = bars.iloc[:i]
            strict += evaluate(code, sub, P) is not None
            loose += evaluate(code, sub, P, require_start_bar=False) is not None
    assert loose > strict


def test_explain_reports_missing_sequence():
    idx = pd.bdate_range("2024-01-01", periods=60)
    flat = pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0}, index=idx)
    text = explain("000001.SZ", _with_aux(flat), P)
    assert "未出现完整的" in text and "【不入选】" in text


def test_explain_short_input(intc):
    assert "数据不足" in explain("600000.SH", _with_aux(intc.iloc[:10]), P)


def test_attach_tradability_flags_limit_up():
    picks = pd.DataFrame({"code": ["600000.SH", "300750.SZ"]})
    out = attach_tradability(picks,
                             next_open={"600000.SH": 11.00, "300750.SZ": 11.50},
                             pre_close={"600000.SH": 10.00, "300750.SZ": 10.00})
    assert out.loc[0, "tradable"] is False or not out.loc[0, "tradable"]  # 一字涨停
    assert out.loc[1, "tradable"]                                          # 创业板未涨停


def test_attach_tradability_missing_open():
    picks = pd.DataFrame({"code": ["600000.SH"]})
    out = attach_tradability(picks, next_open={}, pre_close={"600000.SH": 10.0})
    assert not out.loc[0, "tradable"]


def test_attach_tradability_empty():
    assert len(attach_tradability(pd.DataFrame(), {}, {})) == 0


def test_picker_uses_only_ohlc(intc):
    """evaluate 不应依赖 volume/preClose 之外的附加列。"""
    seq = run(intc, P)["sequences"]
    hit = seq[seq["triggered"]].iloc[0]
    upto = intc.loc[:hit["confirm_date"]][PRICE_COLS]
    assert evaluate("600000.SH", upto, P) is not None
