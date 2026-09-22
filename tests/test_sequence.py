# -*- coding: utf-8 -*-
"""三金叉序列的不变量与边界。"""
import pandas as pd
import pytest

from ema_strategy.indicators import GAP, PIERCE, cross_up
from ema_strategy.sequence import Params, find_sequences, prepare, run

P = Params()


@pytest.fixture(scope="module")
def scanned(samples):
    return {name: run(df, P) for name, df in samples.items()}


def test_sequences_found(scanned):
    assert sum(len(r["sequences"]) for r in scanned.values()) > 0


def test_dates_strictly_ordered(scanned):
    """三个金叉必须依次出现:1号 <= 2号 <= 3号。"""
    for r in scanned.values():
        s = r["sequences"]
        assert (s["start_date"] <= s["c2_date"]).all()
        assert (s["c2_date"] <= s["confirm_date"]).all()


def test_each_date_is_the_right_cross(scanned):
    """三个日期必须分别对应 5上穿10、5上穿30、10上穿30。"""
    for r in scanned.values():
        d, s = r["daily"], r["sequences"]
        c1 = cross_up(d["ma_f"], d["ma_m"])
        c2 = cross_up(d["ma_f"], d["ma_s"])
        c3 = cross_up(d["ma_m"], d["ma_s"])
        assert c1.loc[s["start_date"]].all()
        assert c2.loc[s["c2_date"]].all()
        assert c3.loc[s["confirm_date"]].all()


def test_span_within_max(scanned):
    for r in scanned.values():
        assert (r["sequences"]["span_1_3"] <= P.max_span).all()


def test_spans_are_consistent(scanned):
    for r in scanned.values():
        s = r["sequences"]
        assert (s["span_1_2"] + s["span_2_3"] == s["span_1_3"]).all()


def test_triggered_definition(scanned):
    """triggered 等价于:启动点贯穿 且 确认点跳空。"""
    for r in scanned.values():
        s = r["sequences"]
        expected = (s["start_bar"] == PIERCE) & (s["confirm_bar"] == GAP)
        assert (s["triggered"] == expected).all()


def test_triggered_is_selective(intc):
    s = run(intc, P)["sequences"]
    assert 0 < s["triggered"].sum() < len(s)      # 既非全通过也非全拒绝


def test_triggered_bars_match(intc):
    s = run(intc, P)["sequences"]
    hit = s[s["triggered"]]
    assert set(hit["start_bar"]) == {PIERCE}
    assert set(hit["confirm_bar"]) == {GAP}


def test_max_span_shrinks_results(intc):
    loose = run(intc, Params(max_span=60))["sequences"]
    tight = run(intc, Params(max_span=3))["sequences"]
    assert len(tight) < len(loose)
    assert tight.empty or tight["span_1_3"].max() <= 3


def test_abort_on_dead_cross_is_stricter(intc):
    """关闭作废条件后,序列数不会减少。"""
    strict = run(intc, Params(abort_on_dead_cross=True))["sequences"]
    loose = run(intc, Params(abort_on_dead_cross=False))["sequences"]
    assert len(loose) >= len(strict)


def test_require_distinct_days(intc):
    default = run(intc, Params())["sequences"]
    distinct = run(intc, Params(require_distinct_days=True))["sequences"]
    assert len(distinct) <= len(default)
    if len(distinct):
        assert (distinct["span_1_2"] > 0).all()
        assert (distinct["span_2_3"] > 0).all()


def test_triggered_column_matches_daily_flag(scanned):
    for r in scanned.values():
        d, s = r["daily"], r["sequences"]
        expected = set(s.loc[s["triggered"], "confirm_date"])
        assert set(d.index[d["triggered"]]) == expected


def test_rejects_unsorted_index(intc):
    with pytest.raises(ValueError, match="升序"):
        run(intc.iloc[::-1], P)


def test_rejects_non_datetime_index(intc):
    df = intc.reset_index(drop=True)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        run(df, P)


def test_empty_input_returns_empty():
    df = pd.DataFrame(columns=["open", "high", "low", "close"],
                      index=pd.DatetimeIndex([], name="Date"))
    r = run(df, P)
    assert r["sequences"].empty


def test_short_input_has_no_sequences():
    idx = pd.bdate_range("2024-01-01", periods=10)
    df = pd.DataFrame({"open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0}, index=idx)
    assert run(df, P)["sequences"].empty        # 不足 MA30 预热期


def test_prepare_adds_columns(intc):
    d = prepare(intc, P)
    for col in ("ma_f", "ma_m", "ma_s", "bar_type", "regime"):
        assert col in d.columns


def test_find_sequences_columns_stable_when_empty():
    idx = pd.bdate_range("2024-01-01", periods=40)
    df = pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0}, index=idx)
    s = find_sequences(prepare(df, P), P)
    assert list(s.columns)[:3] == ["start_date", "c2_date", "confirm_date"]
