# -*- coding: utf-8 -*-
"""三金叉序列的不变量与边界。"""
import numpy as np
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


def test_distinct_days_is_the_default(intc):
    """三个金叉分属不同日是默认口径,不再是可选项。"""
    assert Params().require_distinct_days is True
    seq = run(intc, Params())["sequences"]
    assert len(seq)
    assert (seq["span_1_2"] > 0).all()
    assert (seq["span_2_3"] > 0).all()


def test_allowing_same_day_only_adds_sequences(intc):
    """放开同日只会多出序列,不会改变其余序列的内容。

    被剔除的那些序列在两种模式下走的是同一条状态机路径(命中 c3 时
    一样重置),所以严格结果就是宽松结果按「两段间隔都大于0」过滤,
    不存在因为剔除而连锁改变后续序列的情况。
    """
    loose = run(intc, Params(require_distinct_days=False))["sequences"]
    strict = run(intc, Params(require_distinct_days=True))["sequences"]
    expected = loose[(loose["span_1_2"] > 0) & (loose["span_2_3"] > 0)]
    assert len(strict) < len(loose)               # 真实数据上确实有同日的
    pd.testing.assert_frame_equal(strict.reset_index(drop=True),
                                  expected.reset_index(drop=True))


def _ramp(slope: float, jump: float, n: int = 45, tail: int = 20) -> pd.DataFrame:
    """先阴跌把 MA5 压到 MA10、MA30 下方,再一根大阳线跳上去。

    跳得越猛,MA5 越可能一天之内同时上穿 MA10 和 MA30(1号2号同日)。
    """
    close = list(20.0 + slope * np.arange(n))
    close += [close[-1] * jump] * tail
    idx = pd.bdate_range("2024-01-01", periods=len(close))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close},
                        index=idx)


def test_same_day_cross_is_rejected():
    """一次跳空里 MA5 同时上穿 MA10 与 MA30 —— 1号2号同日,正是要剔除的。"""
    df = _ramp(slope=-0.1, jump=1.5)
    loose = run(df, Params(require_distinct_days=False))["sequences"]
    assert len(loose) == 1
    assert loose.iloc[0]["span_1_2"] == 0          # 1号2号确实同日
    assert run(df, Params(require_distinct_days=True))["sequences"].empty


def test_genuinely_sequential_crosses_survive():
    """跌得缓一些,三个金叉自然分到三天,严格口径下应当保留。"""
    df = _ramp(slope=-0.2, jump=1.5)
    strict = run(df, Params(require_distinct_days=True))["sequences"]
    assert len(strict) == 1
    r = strict.iloc[0]
    assert r["span_1_2"] > 0 and r["span_2_3"] > 0
    assert r["start_date"] < r["c2_date"] < r["confirm_date"]


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
