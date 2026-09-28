# -*- coding: utf-8 -*-
"""带本地缓存的取数层:先读 Parquet,只把缺的那一段问 QMT 要。

没有缓存时全市场 4992 只一轮的取数耗时(实测):
    行情 42s + 资金流/股本 265s = 307s,占整轮 8.6min 的 59%。
这些数据每次跑都一模一样,重复取纯属浪费。

命中缓存后:整段读盘约 6s,增量只补最后几天。

三个函数与 feed 里的同名函数签名一致,调用方只需换个入口。
"""
from __future__ import annotations

import sys
import time
from typing import Iterable, Optional

import pandas as pd

from . import feed
from .bull import BullParams
from .chips import profit_ratio
from .store import ParquetStore, ProfitCache, digest_bars, missing_ranges


class ProfitResolver:
    """按需提供获利筹码:缓存命中直接给,没命中就算一次并记下来。

    筹码分布占「计算」那一半的大头(全市场一轮约 3.6min)。它完全由
    (收盘价序列, 流通股本, 参数) 决定,所以可以整只缓存、按指纹校验。

    指纹里必须带上流通股本:换手率 = 成交量/流通股本,股本变了(增发、解禁,
    或者 --refresh-floats 重取到新值)筹码分布就跟着变,只看K线会读到旧值。
    """

    def __init__(self, store: Optional[ParquetStore], p: BullParams,
                 verbose: bool = True):
        self.cache = (ProfitCache(store, self.params_key(p))
                      if store is not None else None)
        self.p = p
        self.verbose = verbose
        self._loaded = self.cache.load() if self.cache is not None else {}
        self._out = {}
        self.hits = self.misses = 0

    @staticmethod
    def params_key(p: BullParams) -> str:
        """只放真正影响筹码计算的参数。

        放多了(比如 profit_min 这种只做阈值比较的)会让改个阈值就全量重算,
        放少了则会读到用别的参数算出来的值 —— 后者是静默错值,更要命。
        """
        return (f"decay={p.chip_decay}|bin={p.chip_bin_pct}"
                f"|unit={p.chip_volume_unit}")

    def _digest(self, bars: pd.DataFrame, float_shares: float) -> str:
        return f"{digest_bars(bars)}:{float(float_shares):.6g}"

    def get(self, code: str, bars: pd.DataFrame, float_shares: float):
        """返回该股的获利筹码序列;缓存关掉时返回 None,由 bull.run 自己算。"""
        if self.cache is None or not float_shares or float_shares <= 0:
            return None
        digest = self._digest(bars, float_shares)
        got = self._loaded.get(code)
        if got is not None and got[0] == digest:
            self.hits += 1
            self._out[code] = got
            return got[1]

        self.misses += 1
        series = profit_ratio(bars, float_shares, decay=self.p.chip_decay,
                              bin_pct=self.p.chip_bin_pct,
                              volume_unit=self.p.chip_volume_unit)
        self._out[code] = (digest, series)
        return series

    def flush(self) -> None:
        if self.cache is None or not self._out:
            return
        if self.verbose:
            total = self.hits + self.misses
            print(f"  获利筹码缓存:命中 {self.hits}/{total},重算 {self.misses} 只")
        if not self.misses:
            return          # 全部命中,盘上的内容与手里的一模一样,不必重写

        # 必须把这轮没碰到的条目一起写回去。ProfitCache.save 是整体重写,
        # 而这轮可能只跑了一部分(--limit、某批取数失败),
        # 直接写 _out 会把其余股票的缓存抹掉 —— 下次就得全量重算。
        merged = dict(self._loaded)
        merged.update(self._out)
        self.cache.save(merged, verbose=self.verbose)


def _group_by_range(todo: dict) -> dict:
    """把 {code: (start, end)} 反转成 {(start, end): [codes]}。

    QMT 一次调用只能给一个起止区间,而缺口通常只有两种:
    全新的票要整段,老票只差尾巴 —— 归并后一般只剩两三次调用。
    """
    out = {}
    for code, rng in todo.items():
        out.setdefault(rng, []).append(code)
    return out


def fetch_daily(store: ParquetStore, codes: Iterable[str], start: str, end: str,
                dividend_type: str = "back", download: bool = True,
                batch: int = 300, verbose: bool = True,
                offline: bool = False) -> dict:
    """日线:缓存优先。offline=True 时完全不碰 QMT,缺什么就少什么。"""
    codes = list(codes)
    store.check_dividend("bars")
    t0 = time.perf_counter()
    cached = store.load("bars", codes=codes, start=start, end=end)
    t_cache = time.perf_counter() - t0

    todo = {} if offline else missing_ranges(store, "bars", codes, start, end)
    if verbose:
        print(f"  行情:缓存命中 {len(cached)}/{len(codes)} 只({t_cache:.1f}s)"
              + (f",需补 {len(todo)} 只" if todo else ",无需补取"))
    if offline and len(cached) < len(codes):
        print(f"  [!] 离线模式:{len(codes) - len(cached)} 只不在缓存里,本轮跳过。"
              f"先跑 scripts/build_cache.py 把它们补上。", file=sys.stderr)

    fresh_all = {}
    for (lo, hi), group in _group_by_range(todo).items():
        for i in range(0, len(group), batch):
            chunk = group[i:i + batch]
            try:
                got = feed.fetch_daily(chunk, lo, hi, dividend_type=dividend_type,
                                       download=download)
            except Exception as exc:
                print(f"    行情补取失败({lo}~{hi},{len(chunk)} 只):"
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            fresh_all.update({c: v for c, v in got.items() if v is not None and len(v)})
            # 取到空也要记:停牌、周末、尚未收盘都会「问到了但没数据」,
            # 不记就会每跑一次都重新问一遍这段空尾巴。只有抛异常的批次不记。
            store.record_asked("bars", chunk, lo, hi)

    if fresh_all:
        store.save("bars", fresh_all, verbose=verbose)
        # 补取的那一段与缓存拼起来,再按请求区间裁一次
        lo_ts, hi_ts = pd.Timestamp(start), pd.Timestamp(end)
        for code, df in fresh_all.items():
            old = cached.get(code)
            merged = df if old is None else pd.concat([old, df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            cached[code] = merged.loc[lo_ts:hi_ts]
    return cached


def fetch_money_flow(store: ParquetStore, codes: Iterable[str], start: str, end: str,
                     download: bool = True, batch: int = 300, verbose: bool = True,
                     offline: bool = False) -> dict:
    """资金流:缓存优先。取不到仍然返回空,由调用方拒绝该条件。"""
    codes = list(codes)
    t0 = time.perf_counter()
    cached = store.load("flow", codes=codes, start=start, end=end)
    t_cache = time.perf_counter() - t0

    todo = {} if offline else missing_ranges(store, "flow", codes, start, end)
    if verbose:
        print(f"  资金流:缓存命中 {len(cached)}/{len(codes)} 只({t_cache:.1f}s)"
              + (f",需补 {len(todo)} 只" if todo else ",无需补取"))

    fresh_all = {}
    for (lo, hi), group in _group_by_range(todo).items():
        for i in range(0, len(group), batch):
            chunk = group[i:i + batch]
            try:
                got = feed.fetch_money_flow(chunk, lo, hi, download=download,
                                            verbose=verbose and i == 0)
            except Exception as exc:
                print(f"    资金流补取失败:{type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            fresh_all.update({c: v for c, v in got.items() if v is not None and len(v)})
            store.record_asked("flow", chunk, lo, hi)

    if fresh_all:
        store.save("flow", fresh_all, verbose=verbose)
        lo_ts, hi_ts = pd.Timestamp(start), pd.Timestamp(end)
        for code, df in fresh_all.items():
            old = cached.get(code)
            merged = df if old is None else pd.concat([old, df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            cached[code] = merged.loc[lo_ts:hi_ts]
    return cached


def fetch_float_shares(store: ParquetStore, codes: Iterable[str],
                       refresh: bool = False, verbose: bool = True,
                       offline: bool = False) -> dict:
    """流通股本:缓存优先。

    这一项没有日期维度,却是原先最慢的一环之一 —— get_instrument_detail
    要一只一只调,4992 只就是 4992 次进程间调用。存一次就不用再问了。

    代价:流通股本只能取到**当前**值,增发/解禁后会变。所以 refresh=True
    可以强制重取,建议隔一段时间刷一次。
    """
    codes = list(codes)
    cached = {} if refresh else store.load_floats()
    missing = [c for c in codes if c not in cached]
    if verbose:
        print(f"  流通股本:缓存命中 {len(codes) - len(missing)}/{len(codes)} 只"
              + (f",需取 {len(missing)} 只" if missing else ""))

    if missing and not offline:
        t0 = time.perf_counter()
        got = feed.fetch_float_shares(missing, verbose=False)
        if got:
            store.save_floats(got, verbose=verbose)
            cached.update(got)
        if verbose:
            print(f"    取到 {len(got)}/{len(missing)} 只({time.perf_counter() - t0:.1f}s)")

    return {c: cached[c] for c in codes if c in cached}


def fetch_universe(store: ParquetStore, sector: str = "沪深A股",
                   asof: Optional[str] = None, offline: bool = False,
                   verbose: bool = True, **kw) -> list:
    """股票池。离线时退回「缓存里有行情的所有 code」。

    这不等于真实股票池(没有 ST / 次新 / 退市过滤,也停在上次建缓存那天),
    所以只在明确离线时才用,并且会说清楚。
    """
    if not offline:
        return feed.fetch_universe(sector, asof=asof, verbose=verbose, **kw)
    cov = store.coverage("bars")
    codes = sorted(str(c) for c in cov["code"]) if len(cov) else []
    if verbose:
        print(f"  离线股票池:取缓存里有行情的 {len(codes)} 只"
              f"(不含 ST/次新/退市过滤,口径停在上次建缓存时)")
    return codes
