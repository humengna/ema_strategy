# -*- coding: utf-8 -*-
"""xtdata 接入层的纯函数部分(不需要 QMT 环境)。"""
import numpy as np
import pandas as pd
import pytest

from ema_strategy.feed import (board_of, gap_flags, is_st_name, limit_prices,
                               limit_ratio, normalize_bars, tradable_at_open)


@pytest.mark.parametrize("code,board", [
    ("600000.SH", "主板"), ("000001.SZ", "主板"), ("002230.SZ", "主板"),
    ("300750.SZ", "创业板"), ("301236.SZ", "创业板"),
    ("688981.SH", "科创板"), ("689009.SH", "科创板"),
    ("430047.BJ", "北交所"), ("920047.BJ", "北交所"), ("831010.BJ", "北交所"),
])
def test_board_of(code, board):
    assert board_of(code) == board


@pytest.mark.parametrize("code,is_st,ratio", [
    ("600000.SH", False, 0.10),
    ("600000.SH", True, 0.05),
    ("300750.SZ", False, 0.20),
    ("300750.SZ", True, 0.20),      # ST 创业板仍是 20%,不是 5%
    ("920047.BJ", False, 0.30),
])
def test_limit_ratio(code, is_st, ratio):
    assert limit_ratio(code, is_st) == ratio


def test_limit_price_uses_half_up_rounding():
    """3.15 * 1.1 = 3.465 -> 交易所四舍五入为 3.47。

    Python 内建 round(3.465, 2) 因浮点表示会得到 3.46,差一分会导致一字板漏判。
    """
    assert limit_prices(3.15, 0.10)[0] == 3.47
    assert round(3.465, 2) == 3.46               # 反例:内建 round 不可用


@pytest.mark.parametrize("pre,ratio,up,down", [
    (10.00, 0.10, 11.00, 9.00),
    (2.005, 0.10, 2.21, 1.80),
    (100.0, 0.20, 120.0, 80.0),
])
def test_limit_prices(pre, ratio, up, down):
    assert limit_prices(pre, ratio) == (up, down)


def test_tradable_at_open():
    assert not tradable_at_open(11.00, 10.00, "600000.SH")   # 开盘即涨停
    assert tradable_at_open(10.99, 10.00, "600000.SH")
    assert tradable_at_open(11.50, 10.00, "300750.SZ")       # 创业板 20% 上限 12.00
    assert not tradable_at_open(12.00, 10.00, "300750.SZ")


@pytest.mark.parametrize("open_px,pre", [(np.nan, 10.0), (10.0, np.nan), (10.0, 0.0)])
def test_tradable_rejects_invalid(open_px, pre):
    assert not tradable_at_open(open_px, pre, "600000.SH")


def test_normalize_drops_fake_and_dedupes():
    raw = pd.DataFrame(
        {"open": [10, 0, 11, 11.5, 12], "high": [10.5, 0, 11.4, 11.9, 12.3],
         "low": [9.8, 0, 10.8, 11.2, 11.8], "close": [10.2, 0, 11.2, 11.6, 12.1],
         "volume": [100, 0, 0, 50, 80], "preClose": [9.9, 10.2, 10.2, 11.2, 11.6]},
        index=["20240102", "20240103", "20240104", "20240105", "20240105"])
    out = normalize_bars(raw)
    # 价格为0 与 volume=0 各剔除一根,重复日期保留最后一条 -> 剩 2 根
    assert len(out) == 2
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing
    assert not out.index.duplicated().any()
    assert out["close"].iloc[-1] == 12.1          # 重复日保留最后一条


def test_normalize_drops_inconsistent_bar():
    raw = pd.DataFrame({"open": [10], "high": [9], "low": [11], "close": [10],
                        "volume": [100], "preClose": [10]}, index=["20240108"])
    assert len(normalize_bars(raw)) == 0          # high < low


def test_normalize_empty():
    assert len(normalize_bars(pd.DataFrame())) == 0
    assert len(normalize_bars(None)) == 0


def test_gap_flags_marks_long_suspension():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-06-01", "2024-06-03"])
    flags = gap_flags(idx, max_gap_days=30)
    assert flags.tolist() == [False, False, True, False]


def test_gap_flags_empty():
    assert len(gap_flags(pd.DatetimeIndex([]))) == 0


@pytest.mark.parametrize("name,expected", [
    ("ST康美", True), ("*ST海航", True), ("退市海润", True),
    ("平安银行", False), (None, False), ("", False),
])
def test_is_st_name(name, expected):
    assert is_st_name(name) == expected


@pytest.mark.parametrize("value", [
    "99999999", 99999999,      # 未退市合约的哨兵值 —— 曾导致全市场扫描中断
    "9999999999", "00000000", "0", "", None, 0,
    "2099-12-31", "2024010", "abcdefgh",
])
def test_parse_ymd_rejects_invalid(value):
    """任何非法/哨兵日期一律返回 None,不得抛异常。

    回归测试:ExpireDate=99999999 会让 pd.to_datetime(format="%Y%m%d")
    抛 ValueError("unconverted data remains: 99"),中断整轮扫描。
    """
    from ema_strategy.feed import parse_ymd
    assert parse_ymd(value) is None


@pytest.mark.parametrize("value,expected", [
    ("20991231", "2099-12-31"),
    ("19910403", "1991-04-03"),
    (20240102, "2024-01-02"),
    ("  20240102  ", "2024-01-02"),
])
def test_parse_ymd_accepts_valid(value, expected):
    from ema_strategy.feed import parse_ymd
    assert parse_ymd(value) == pd.Timestamp(expected)


def test_normalize_winner_handles_percent_and_fraction():
    """获利比例可能以百分数或小数返回,统一归到 0~1。"""
    from ema_strategy.feed import _normalize_winner
    idx = ["20240102", "20240103"]
    assert _normalize_winner(pd.Series([95.0, 80.0], index=idx)).tolist() == [0.95, 0.80]
    assert _normalize_winner(pd.Series([0.95, 0.80], index=idx)).tolist() == [0.95, 0.80]


def test_normalize_winner_clips_and_sorts():
    from ema_strategy.feed import _normalize_winner
    s = _normalize_winner(pd.Series([1.5, -0.2], index=["20240103", "20240102"]))
    assert s.index.is_monotonic_increasing
    assert s.between(0.0, 1.0).all()


def test_normalize_winner_from_dataframe():
    from ema_strategy.feed import _normalize_winner
    df = pd.DataFrame({"winner": [0.9, 0.8]}, index=["20240102", "20240103"])
    assert _normalize_winner(df).tolist() == [0.9, 0.8]


def test_normalize_winner_empty():
    from ema_strategy.feed import _normalize_winner
    assert _normalize_winner(None) is None
    assert _normalize_winner(pd.Series(dtype=float)) is None


def test_fetch_winner_chips_reports_missing_function(monkeypatch):
    """xtquant 没有该接口时必须明确报错,不能悄悄退回自算。"""
    import types

    from ema_strategy import feed as feed_mod
    fake = types.SimpleNamespace()          # 不含 get_winner_chips
    monkeypatch.setitem(__import__("sys").modules, "xtquant",
                        types.SimpleNamespace(xtdata=fake))
    with pytest.raises(AttributeError, match="没有 get_winner_chips"):
        feed_mod.fetch_winner_chips(["000001.SZ"], "20240101", "20240201")
