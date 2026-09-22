# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from ema_strategy.indicators import (BELOW, BULL, GAP, HALF, OTHER, PIERCE, BEAR,
                                     add_ma, classify_bar, cross_down, cross_up, ma_regime)

IDX = pd.bdate_range("2024-01-01", periods=4)


def _bars(low, high, f, m, s):
    n = len(low)
    return pd.DataFrame({"low": low, "high": high, "ma_f": f, "ma_m": m, "ma_s": s},
                        index=pd.bdate_range("2024-01-01", periods=n))


def test_classify_gap():
    d = _bars([10.0], [12.0], [8.0], [7.0], [9.0])
    assert classify_bar(d).iat[0] == GAP          # low 10 >= max(MA) 9


def test_classify_pierce():
    d = _bars([5.0], [9.0], [6.0], [7.0], [8.0])
    assert classify_bar(d).iat[0] == PIERCE       # 三条均线都在 [5, 9] 内


def test_classify_below():
    d = _bars([1.0], [2.0], [5.0], [6.0], [7.0])
    assert classify_bar(d).iat[0] == BELOW        # high 2 < min(MA) 5


def test_classify_other():
    """整根K线夹在 MA10 与 MA5 之间,不触碰任何均线 —— 浅回踩。"""
    d = _bars([6.1], [6.9], [7.0], [6.0], [5.0])
    assert classify_bar(d).iat[0] == OTHER


def test_gap_wins_on_tie():
    """low 恰等于最高均线时,跳空与贯穿同时成立,按「相等归入 >=」判为跳空。"""
    d = _bars([9.0], [11.0], [8.0], [7.0], [9.0])
    assert classify_bar(d).iat[0] == GAP


def test_classify_nan_when_ma_missing():
    d = _bars([9.0], [11.0], [np.nan], [7.0], [9.0])
    assert pd.isna(classify_bar(d).iat[0])


@pytest.mark.parametrize("f,m,s,expected", [
    (3.0, 2.0, 1.0, BULL),
    (1.0, 2.0, 3.0, BEAR),
    (2.0, 2.0, 2.0, BULL),      # 三线相等归入全多头
    (2.0, 3.0, 1.0, HALF),
])
def test_ma_regime(f, m, s, expected):
    d = pd.DataFrame({"ma_f": [f], "ma_m": [m], "ma_s": [s]}, index=IDX[:1])
    assert ma_regime(d).iat[0] == expected


def test_cross_up_counts_equality_as_above():
    a = pd.Series([1.0, 2.0, 2.0, 3.0], index=IDX)
    b = pd.Series([2.0, 2.0, 2.0, 2.0], index=IDX)
    # 第2根 a 与 b 相等 -> 视为上穿;第3根维持,不重复触发
    assert cross_up(a, b).tolist() == [False, True, False, False]


def test_cross_down():
    a = pd.Series([3.0, 1.0, 1.0], index=IDX[:3])
    b = pd.Series([2.0, 2.0, 2.0], index=IDX[:3])
    assert cross_down(a, b).tolist() == [False, True, False]


def test_add_ma_requires_ohlc():
    with pytest.raises(ValueError, match="缺少列"):
        add_ma(pd.DataFrame({"close": [1.0, 2.0]}))


def test_add_ma_values(intc):
    d = add_ma(intc, 5, 10, 30)
    assert np.isclose(d["ma_f"].iat[4], intc["close"].iloc[:5].mean())
    assert d["ma_s"].iloc[:29].isna().all()       # MA30 预热期


def test_no_spurious_cross_at_warmup_boundary():
    """均线预热期结束的第一根K线不得被判成金叉。

    回归测试:曾因 `a >= b` 在 NaN 期为 False、转正后跳到 True,
    导致每只股票在 MA30 预热结束当天凭空多出 1/2/3 号金叉。
    """
    idx = pd.bdate_range("2024-01-01", periods=10)
    a = pd.Series([np.nan] * 5 + [2.0] * 5, index=idx)
    b = pd.Series([np.nan] * 5 + [1.0] * 5, index=idx)
    assert not cross_up(a, b).any()          # a 一直在 b 上方,从未穿越
    assert not cross_down(b, a).any()


def test_cross_still_detected_after_warmup():
    idx = pd.bdate_range("2024-01-01", periods=8)
    a = pd.Series([np.nan, np.nan, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0], index=idx)
    b = pd.Series([np.nan, np.nan, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], index=idx)
    assert cross_up(a, b).tolist() == [False] * 4 + [True] + [False] * 3
    assert cross_down(a, b).tolist() == [False] * 6 + [True, False]


def test_flat_series_produces_no_cross():
    """完全不动的价格不应产生任何金叉。"""
    idx = pd.bdate_range("2024-01-01", periods=40)
    flat = pd.Series(10.0, index=idx)
    ma5, ma30 = flat.rolling(5).mean(), flat.rolling(30).mean()
    assert not cross_up(ma5, ma30).any()
