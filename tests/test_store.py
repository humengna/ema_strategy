# -*- coding: utf-8 -*-
"""本地 Parquet 缓存:读写、增量、失效、校验。

缓存最怕的不是慢,是**悄悄读到过期或错位的数据** —— 结果照样跑出来,
只是不对。所以这里的重点全在「什么时候必须作废」。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from ema_strategy.store import (ParquetStore, ProfitCache, digest_bars,
                                missing_ranges, verify)

pytest.importorskip("pyarrow", reason="缓存需要 Parquet 引擎")


def mk_bars(n=20, start="2024-01-01", seed=0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.02, n))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.integers(1e4, 1e5, n).astype(float),
                         "amount": close * 1e5}, index=idx)


@pytest.fixture
def store(tmp_path):
    return ParquetStore(str(tmp_path / "cache"))


# ------------------------------------------------------------------ 读写
def test_roundtrip_preserves_values(store):
    data = {"000001.SZ": mk_bars(seed=1), "600000.SH": mk_bars(seed=2)}
    store.save("bars", data, verbose=False)
    got = store.load("bars")
    assert set(got) == set(data)
    for code in data:
        pd.testing.assert_index_equal(got[code].index, data[code].index)
        assert np.allclose(got[code]["close"], data[code]["close"])


def test_load_filters_by_code_and_date(store):
    store.save("bars", {"A": mk_bars(30), "B": mk_bars(30, seed=9)}, verbose=False)
    got = store.load("bars", codes=["A"], start="20240108", end="20240112")
    assert set(got) == {"A"}
    assert got["A"].index.min() >= pd.Timestamp("2024-01-08")
    assert got["A"].index.max() <= pd.Timestamp("2024-01-12")


def test_save_overwrites_same_day(store):
    """行情会被数据源事后修订,同一 (code, date) 必须以新值为准。"""
    first = mk_bars(5)
    store.save("bars", {"A": first}, verbose=False)
    revised = first.copy()
    revised["close"] = revised["close"] * 2
    store.save("bars", {"A": revised}, verbose=False)
    got = store.load("bars")["A"]
    assert len(got) == 5                       # 不是追加成 10 行
    assert np.allclose(got["close"], revised["close"])


def test_append_spanning_years_keeps_order(store):
    """跨年追加后,同一只票的日期仍须升序 —— 切分靠的就是这个前提。"""
    store.save("bars", {"A": mk_bars(30, start="2024-12-01")}, verbose=False)
    store.save("bars", {"A": mk_bars(30, start="2025-02-01", seed=3)}, verbose=False)
    got = store.load("bars")["A"]
    assert got.index.is_monotonic_increasing
    assert len(os.listdir(store._dir("bars"))) >= 3   # 2024/2025 两个分片 + _meta


def test_load_empty_cache_returns_empty(store):
    assert store.load("bars") == {}
    assert store.coverage("bars").empty


def test_atomic_write_leaves_no_tmp(store):
    store.save("bars", {"A": mk_bars(5)}, verbose=False)
    assert not [f for f in os.listdir(store._dir("bars")) if f.endswith(".tmp")]


# ------------------------------------------------------------------ 复权口径
def test_dividend_mismatch_refuses_to_load(tmp_path):
    """后复权与前复权的价格不能混用,混了会凭空造出金叉。"""
    root = str(tmp_path / "c")
    ParquetStore(root, dividend_type="back").save("bars", {"A": mk_bars()}, verbose=False)
    other = ParquetStore(root, dividend_type="front")
    with pytest.raises(SystemExit):
        other.load("bars")


def test_dividend_recorded_in_meta(store):
    store.save("bars", {"A": mk_bars()}, verbose=False)
    assert store.read_meta("bars")["dividend_type"] == "back"


# ------------------------------------------------------------------ 增量
def test_missing_ranges_only_asks_for_the_tail(store):
    store.save("bars", {"A": mk_bars(20, start="2024-01-01")}, verbose=False)
    last = store.load("bars")["A"].index[-1]
    todo = missing_ranges(store, "bars", ["A"], "20240101", "20240301")
    assert "A" in todo
    lo, hi = todo["A"]
    assert pd.Timestamp(lo) == last + pd.Timedelta(days=1)
    assert hi == "20240301"


def test_missing_ranges_refetches_when_asked_for_earlier_data(store):
    """要的区间比缓存起点还早,中间挖空没法用一个区间表达,整段重取。"""
    store.save("bars", {"A": mk_bars(20, start="2024-06-01")}, verbose=False)
    todo = missing_ranges(store, "bars", ["A"], "20240101", "20240630")
    assert todo["A"] == ("20240101", "20240630")


def test_missing_ranges_skips_fully_covered(store):
    store.save("bars", {"A": mk_bars(40, start="2024-01-01")}, verbose=False)
    cov = store.coverage("bars")
    hi = pd.Timestamp(cov["last"].iloc[0]).strftime("%Y%m%d")
    assert missing_ranges(store, "bars", ["A"], "20240101", hi) == {}


def test_missing_ranges_flags_unknown_codes(store):
    store.save("bars", {"A": mk_bars(10)}, verbose=False)
    todo = missing_ranges(store, "bars", ["A", "B"], "20240101", "20240110")
    assert todo["B"] == ("20240101", "20240110")


# ------------------------------------------------------------------ 校验
def test_verify_detects_drift(store):
    bars = mk_bars(10)
    store.save("bars", {"A": bars}, verbose=False)
    assert verify(store, "bars", {"A": bars}, verbose=False) == []

    drifted = bars.copy()
    drifted.iloc[3, drifted.columns.get_loc("close")] *= 1.05
    assert verify(store, "bars", {"A": drifted}, verbose=False) == ["A"]


def test_verify_ignores_dates_not_in_cache(store):
    bars = mk_bars(10)
    store.save("bars", {"A": bars.iloc[:5]}, verbose=False)
    assert verify(store, "bars", {"A": bars}, verbose=False) == []


# ------------------------------------------------------------------ 流通股本
def test_floats_roundtrip_and_update(store):
    store.save_floats({"A": 1e9, "B": 2e9}, verbose=False)
    store.save_floats({"B": 3e9}, verbose=False)          # 增发后重取
    got = store.load_floats()
    assert got == {"A": 1e9, "B": 3e9}


# ------------------------------------------------------------------ 指纹
def test_digest_changes_with_close_and_dates():
    a = mk_bars(20)
    assert digest_bars(a) == digest_bars(a.copy())

    b = a.copy()
    b.iloc[5, b.columns.get_loc("close")] *= 1.001
    assert digest_bars(b) != digest_bars(a)

    assert digest_bars(a.iloc[:-1]) != digest_bars(a)     # 多一天就该重算


def test_digest_ignores_unrelated_columns():
    """派生结果只依赖收盘价与日期,别的字段微调不该让缓存失效。"""
    a = mk_bars(20)
    b = a.copy()
    b["high"] = b["high"] * 1.5
    assert digest_bars(b) == digest_bars(a)


# ------------------------------------------------------------------ 筹码缓存
def test_profit_cache_roundtrip(store):
    cache = ProfitCache(store, "k1")
    s = pd.Series([0.1, 0.2], index=pd.bdate_range("2024-01-01", periods=2))
    cache.save({"A": ("d1", s)}, verbose=False)
    got = ProfitCache(store, "k1").load()
    assert got["A"][0] == "d1"
    assert np.allclose(got["A"][1].to_numpy(), s.to_numpy())


def test_profit_cache_invalidated_by_params(store):
    """换了衰减系数/网格步长,旧值必须整体作废,不能混着用。"""
    s = pd.Series([0.5], index=pd.bdate_range("2024-01-01", periods=1))
    ProfitCache(store, "k1").save({"A": ("d1", s)}, verbose=False)
    assert ProfitCache(store, "k2").load() == {}
    assert ProfitCache(store, "k1").load() != {}


def test_profit_cache_meta_records_key(store):
    ProfitCache(store, "k1").save(
        {"A": ("d1", pd.Series([0.5], index=pd.bdate_range("2024-01-01", periods=1)))},
        verbose=False)
    with open(os.path.join(store.root, "profit", "_meta.json"), encoding="utf-8") as f:
        assert json.load(f)["params_key"] == "k1"


# ------------------------------------------------------------------ 取数水位线
def test_asked_watermark_prevents_refetching_empty_tail(store):
    """问到 0325、数据只到 0322(周末/停牌),不该再去问那段空尾巴。"""
    store.save("bars", {"A": mk_bars(20, start="2024-01-01")}, verbose=False)
    last = store.load("bars")["A"].index[-1]
    store.record_asked("bars", ["A"], "20240101", "20240325")
    assert last < pd.Timestamp("2024-03-25")          # 确实有一段空尾巴
    assert missing_ranges(store, "bars", ["A"], "20240101", "20240325") == {}


def test_asked_watermark_still_extends_for_new_dates(store):
    store.save("bars", {"A": mk_bars(20)}, verbose=False)
    store.record_asked("bars", ["A"], "20240101", "20240325")
    todo = missing_ranges(store, "bars", ["A"], "20240101", "20240501")
    assert pd.Timestamp(todo["A"][0]) == pd.Timestamp("2024-03-26")


def test_asked_watermark_covers_codes_with_no_data(store):
    """整段停牌、一根K线都没有的票,问过一次就别再问了。"""
    store.record_asked("bars", ["DEAD"], "20240101", "20240325")
    assert missing_ranges(store, "bars", ["DEAD"], "20240101", "20240325") == {}


def test_record_asked_merges_ranges(store):
    store.record_asked("bars", ["A"], "20240201", "20240301")
    store.record_asked("bars", ["A"], "20240101", "20240401")
    lo, hi = store.load_asked("bars")["A"]
    assert lo == pd.Timestamp("2024-01-01") and hi == pd.Timestamp("2024-04-01")


def test_missing_ranges_falls_back_to_coverage_without_watermark(store):
    """老缓存没有水位线文件时,行为退回按数据覆盖判断,不能直接崩。"""
    store.save("bars", {"A": mk_bars(20, start="2024-01-01")}, verbose=False)
    assert not os.path.exists(store._asked_path("bars"))
    todo = missing_ranges(store, "bars", ["A"], "20240101", "20240501")
    assert "A" in todo
