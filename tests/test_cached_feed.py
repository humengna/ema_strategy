# -*- coding: utf-8 -*-
"""带缓存的取数层:只补缺口、结果与不走缓存时一致、局部跑不污染缓存。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ema_strategy import cached_feed, feed
from ema_strategy.bull import BullParams
from ema_strategy.chips import profit_ratio
from ema_strategy.store import ParquetStore

pytest.importorskip("pyarrow", reason="缓存需要 Parquet 引擎")


def mk_bars(n=60, start="2024-01-01", seed=0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    close = 10 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    vol = rng.integers(1e4, 1e5, n).astype(float)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": vol, "amount": close * vol * 100},
                        index=idx)


class FakeSource:
    """假行情源,记下每次被问到的 code 与区间 —— 断言「只补缺口」全靠它。"""

    def __init__(self, codes, n=60):
        self.bars = {c: mk_bars(n, seed=i) for i, c in enumerate(codes)}
        self.floats = {c: 1e9 + i for i, c in enumerate(codes)}
        self.calls = []

    def fetch_daily(self, codes, start, end, dividend_type="back", download=True):
        codes = list(codes)
        self.calls.append(("bars", tuple(sorted(codes)), start, end))
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        return {c: self.bars[c].loc[lo:hi] for c in codes if c in self.bars}

    def fetch_float_shares(self, codes, verbose=True):
        codes = list(codes)
        self.calls.append(("floats", tuple(sorted(codes)), None, None))
        return {c: self.floats[c] for c in codes if c in self.floats}


@pytest.fixture
def src(monkeypatch):
    s = FakeSource(["A", "B", "C"])
    monkeypatch.setattr(feed, "fetch_daily", s.fetch_daily)
    monkeypatch.setattr(feed, "fetch_float_shares", s.fetch_float_shares)
    return s


@pytest.fixture
def store(tmp_path):
    return ParquetStore(str(tmp_path / "cache"))


# ------------------------------------------------------------------ 行情
def test_first_run_fetches_then_second_run_does_not(store, src):
    codes = ["A", "B", "C"]
    first = cached_feed.fetch_daily(store, codes, "20240101", "20240325", verbose=False)
    assert len(src.calls) == 1
    assert set(first) == set(codes)

    src.calls.clear()
    second = cached_feed.fetch_daily(store, codes, "20240101", "20240325", verbose=False)
    assert src.calls == [], "第二次不该再问行情源要数据"
    for c in codes:
        assert np.allclose(second[c]["close"], first[c]["close"])


def test_only_the_tail_is_refetched(store, src):
    cached_feed.fetch_daily(store, ["A"], "20240101", "20240201", verbose=False)
    src.calls.clear()
    cached_feed.fetch_daily(store, ["A"], "20240101", "20240325", verbose=False)

    assert len(src.calls) == 1
    _, asked, lo, _ = src.calls[0]
    assert asked == ("A",)
    assert pd.Timestamp(lo) > pd.Timestamp("20240201"), "应当只补尾巴,不是整段重取"


def test_result_matches_uncached_exactly(store, src):
    """走不走缓存,拿到的数据必须逐格相同 —— 否则回测结论会悄悄变。"""
    codes = ["A", "B", "C"]
    cached_feed.fetch_daily(store, codes, "20240101", "20240201", verbose=False)
    merged = cached_feed.fetch_daily(store, codes, "20240101", "20240325", verbose=False)
    direct = src.fetch_daily(codes, "20240101", "20240325")
    for c in codes:
        pd.testing.assert_frame_equal(
            merged[c].sort_index(), direct[c][merged[c].columns].sort_index(),
            check_freq=False, check_dtype=False)


def test_returned_range_is_clipped_to_request(store, src):
    """缓存里存得更长,返回的也只能是请求的那一段。"""
    cached_feed.fetch_daily(store, ["A"], "20240101", "20240325", verbose=False)
    got = cached_feed.fetch_daily(store, ["A"], "20240201", "20240301", verbose=False)
    assert got["A"].index.min() >= pd.Timestamp("2024-02-01")
    assert got["A"].index.max() <= pd.Timestamp("2024-03-01")


def test_offline_never_calls_the_source(store, src, capsys):
    cached_feed.fetch_daily(store, ["A"], "20240101", "20240201", verbose=False)
    src.calls.clear()
    got = cached_feed.fetch_daily(store, ["A", "B"], "20240101", "20240201",
                                  verbose=False, offline=True)
    assert src.calls == []
    assert set(got) == {"A"}                      # B 不在缓存里,直接少一只
    assert "离线模式" in capsys.readouterr().err


def test_fetch_failure_does_not_poison_cache(store, src, monkeypatch, capsys):
    """补取抛异常时应跳过并留下缓存里已有的部分,不能整轮崩掉。"""
    cached_feed.fetch_daily(store, ["A"], "20240101", "20240201", verbose=False)

    def boom(*args, **kw):
        raise RuntimeError("行情源挂了")
    monkeypatch.setattr(feed, "fetch_daily", boom)

    got = cached_feed.fetch_daily(store, ["A"], "20240101", "20240325", verbose=False)
    assert "A" in got and len(got["A"])
    assert "行情补取失败" in capsys.readouterr().err


# ------------------------------------------------------------------ 流通股本
def test_float_shares_only_asks_for_missing(store, src):
    cached_feed.fetch_float_shares(store, ["A", "B"], verbose=False)
    src.calls.clear()
    got = cached_feed.fetch_float_shares(store, ["A", "B", "C"], verbose=False)
    assert len(src.calls) == 1
    assert src.calls[0][1] == ("C",), "只该问缓存里没有的那一只"
    assert set(got) == {"A", "B", "C"}


def test_float_shares_refresh_refetches_all(store, src):
    cached_feed.fetch_float_shares(store, ["A", "B"], verbose=False)
    src.calls.clear()
    cached_feed.fetch_float_shares(store, ["A", "B"], refresh=True, verbose=False)
    assert src.calls[0][1] == ("A", "B")


# ------------------------------------------------------------------ 筹码缓存
def _p() -> BullParams:
    return BullParams(chip_volume_unit=100)


def test_profit_resolver_matches_direct_computation(store, src):
    bars, fs = src.bars["A"], src.floats["A"]
    r = cached_feed.ProfitResolver(store, _p(), verbose=False)
    got = r.get("A", bars, fs)
    want = profit_ratio(bars, fs, decay=_p().chip_decay,
                        bin_pct=_p().chip_bin_pct, volume_unit=100)
    assert np.allclose(got.to_numpy(), want.to_numpy(), equal_nan=True)


def test_profit_resolver_hits_cache_on_second_run(store, src):
    bars, fs = src.bars["A"], src.floats["A"]
    r1 = cached_feed.ProfitResolver(store, _p(), verbose=False)
    first = r1.get("A", bars, fs)
    r1.flush()

    r2 = cached_feed.ProfitResolver(store, _p(), verbose=False)
    second = r2.get("A", bars, fs)
    assert (r2.hits, r2.misses) == (1, 0)
    assert np.allclose(first.to_numpy(), second.to_numpy(), equal_nan=True)


def test_profit_resolver_recomputes_when_bars_change(store, src):
    bars, fs = src.bars["A"], src.floats["A"]
    r1 = cached_feed.ProfitResolver(store, _p(), verbose=False)
    r1.get("A", bars, fs)
    r1.flush()

    r2 = cached_feed.ProfitResolver(store, _p(), verbose=False)
    r2.get("A", pd.concat([bars, mk_bars(5, start="2024-06-01", seed=7)]), fs)
    assert (r2.hits, r2.misses) == (0, 1), "多了几根K线就必须重算"


def test_profit_resolver_recomputes_when_float_shares_change(store, src):
    """换手率 = 成交量/流通股本。股本变了筹码就变,只看K线会读到旧值。"""
    bars, fs = src.bars["A"], src.floats["A"]
    r1 = cached_feed.ProfitResolver(store, _p(), verbose=False)
    r1.get("A", bars, fs)
    r1.flush()

    r2 = cached_feed.ProfitResolver(store, _p(), verbose=False)
    r2.get("A", bars, fs * 1.5)
    assert (r2.hits, r2.misses) == (0, 1)


def test_profit_resolver_params_key_ignores_threshold():
    """profit_min 只做阈值比较,不影响筹码计算,改它不该让缓存全废。"""
    k = cached_feed.ProfitResolver.params_key
    assert k(BullParams(profit_min=0.85)) == k(BullParams(profit_min=0.90))
    assert k(BullParams(chip_decay=1.0)) != k(BullParams(chip_decay=0.9))
    assert k(BullParams(chip_volume_unit=100)) != k(BullParams(chip_volume_unit=1))


def test_partial_run_does_not_wipe_other_codes(store, src):
    """只跑了一部分股票时,其余股票的缓存不能被整体重写抹掉。"""
    r1 = cached_feed.ProfitResolver(store, _p(), verbose=False)
    for c in ("A", "B", "C"):
        r1.get(c, src.bars[c], src.floats[c])
    r1.flush()

    # 只跑 A,且强制它重算(改一根K线),触发一次写盘
    r2 = cached_feed.ProfitResolver(store, _p(), verbose=False)
    changed = src.bars["A"].copy()
    changed.iloc[0, changed.columns.get_loc("close")] *= 1.01
    r2.get("A", changed, src.floats["A"])
    r2.flush()

    r3 = cached_feed.ProfitResolver(store, _p(), verbose=False)
    for c in ("B", "C"):
        r3.get(c, src.bars[c], src.floats[c])
    assert (r3.hits, r3.misses) == (2, 0), "B、C 的缓存被局部跑抹掉了"


def test_resolver_without_store_returns_none(src):
    r = cached_feed.ProfitResolver(None, _p(), verbose=False)
    assert r.get("A", src.bars["A"], src.floats["A"]) is None
    r.flush()                                     # 不该炸
