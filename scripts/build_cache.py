# -*- coding: utf-8 -*-
"""把全市场行情/资金流/流通股本一次性抓进本地 Parquet 缓存。

    # 首次建缓存(慢的那一次,跑完就不用再等了)
    python scripts/build_cache.py --start 20230101 --end 20260928

    # 之后每天只补新增的那几天(秒级)
    python scripts/build_cache.py --end 20260929

    # 怀疑缓存过期/错位:抽 50 只重取,与缓存逐格比对
    python scripts/build_cache.py --verify 50

    # 看看缓存里现在有什么
    python scripts/build_cache.py --status

为什么值得做:全市场 4992 只跑一轮,取数 307s 占总耗时的 59%,
而这些数据每次都一模一样。落盘之后整段读回约 6s。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ema_strategy import cached_feed                            # noqa: E402
from ema_strategy import feed                                    # noqa: E402
from ema_strategy.bull import BullParams                         # noqa: E402
from ema_strategy.store import ParquetStore, verify              # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="build_cache.py", description="把行情/资金流/流通股本抓进本地 Parquet 缓存")
    ap.add_argument("--cache", default="cache", help="缓存目录,默认 ./cache")
    ap.add_argument("--start", default="", help="起始日 YYYYMMDD;增量时可省略")
    ap.add_argument("--end", default="", help="截止日 YYYYMMDD,默认今天")
    ap.add_argument("--sector", default="沪深A股")
    ap.add_argument("--codes", default="", help="逗号分隔;留空则取整个板块")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 只,先小规模试")
    ap.add_argument("--batch", type=int, default=300, help="每批多少只")
    ap.add_argument("--dividend", default="back", choices=["back", "front", "none"],
                    help="复权方式。信号判定必须 back;换口径请换缓存目录")
    ap.add_argument("--no-download", action="store_true", dest="no_download",
                    help="不向服务器下载,只读 QMT 本地已有的数据")
    ap.add_argument("--skip-flow", action="store_true", dest="skip_flow",
                    help="不抓资金流(没有投研版/Level2 权限时)")
    ap.add_argument("--refresh-floats", action="store_true", dest="refresh_floats",
                    help="强制重取流通股本(增发/解禁后应刷一次)")
    ap.add_argument("--skip-profit", action="store_true", dest="skip_profit",
                    help="不预算获利筹码(默认会算并存好,免得第一次回测再等一次)")
    ap.add_argument("--chip-decay", type=float, default=BullParams().chip_decay,
                    dest="chip_decay", help="筹码换手衰减系数")
    ap.add_argument("--chip-bin-pct", type=float, default=BullParams().chip_bin_pct,
                    dest="chip_bin_pct", help="筹码价格网格步长,别调大")
    ap.add_argument("--volume-unit", type=int, default=BullParams().chip_volume_unit,
                    dest="volume_unit", help="成交量单位:A股按手计=100")
    ap.add_argument("--status", action="store_true", help="只打印缓存现状,不取数")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="抽 N 只重新取数,与缓存逐格比对")
    return ap


def show_status(store: ParquetStore) -> int:
    print(f"缓存目录: {store.root}")
    if not os.path.isdir(store.root):
        print("  (还不存在)")
        return 1
    total = 0
    for root, _, files in os.walk(store.root):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    print(f"  占用: {total / 1e6:.0f} MB")

    for name in ("bars", "flow"):
        cov = store.coverage(name)
        meta = store.read_meta(name)
        if not len(cov):
            print(f"  {name:6s} 空")
            continue
        print(f"  {name:6s} {len(cov)} 只 | {pd.Timestamp(cov['first'].min()).date()}"
              f" ~ {pd.Timestamp(cov['last'].max()).date()}"
              f" | {int(cov['rows'].sum()):,} 行"
              + (f" | 复权 {meta['dividend_type']}" if meta.get("dividend_type") else "")
              + (f" | 更新于 {meta['updated']}" if meta.get("updated") else ""))
        # 最后一天不齐通常意味着上次是中途停的,或者有票停牌
        last = pd.Timestamp(cov["last"].max())
        stale = int((cov["last"] < last).sum())
        if stale:
            print(f"         其中 {stale} 只没到最新日 {last.date()}(停牌或上次未取全)")

    floats = store.load_floats()
    print(f"  floats {len(floats)} 只")

    prof = os.path.join(store.root, "profit", "profit.parquet")
    ckpt = os.path.join(store.root, "profit", "checkpoint.parquet")
    if os.path.exists(prof):
        import pandas as _pd
        d = _pd.read_parquet(prof, columns=["code", "date"])
        n_ck = len(_pd.read_parquet(ckpt, columns=["code"])) if os.path.exists(ckpt) else 0
        print(f"  profit {d['code'].nunique()} 只 | {int(len(d)):,} 行"
              f" | 至 {_pd.Timestamp(d['date'].max()).date()}"
              f" | 续算检查点 {n_ck} 只")
        if not n_ck:
            print("         没有检查点:下次补数据会整只重算,"
                  "跑一次 build_cache.py 即可补上")
    else:
        print("  profit 空(回测时会自己算;"
              "或现在跑 build_cache.py 预先算好)")
    return 0


def do_verify(store: ParquetStore, codes: list, start: str, end: str,
              n: int, dividend: str, skip_flow: bool) -> int:
    """抽样重取并与缓存比对 —— 缓存最怕的不是慢,是悄悄读到过期数据。"""
    import random
    cov = store.coverage("bars")
    pool = [str(c) for c in cov["code"]] if len(cov) else []
    pool = [c for c in pool if c in set(codes)] or pool
    if not pool:
        print("缓存是空的,没什么可校验的", file=sys.stderr)
        return 1
    sample = random.sample(pool, min(n, len(pool)))
    print(f"抽样校验 {len(sample)} 只({start}~{end})...")

    rc = 0
    fresh = feed.fetch_daily(sample, start, end, dividend_type=dividend)
    bad = verify(store, "bars", fresh)
    print(f"  行情: {len(fresh)} 只重取,{len(bad)} 只与缓存不一致"
          + (f" -> {bad[:10]}" if bad else " ✓"))
    rc |= 1 if bad else 0

    if not skip_flow:
        fresh_f = feed.fetch_money_flow(sample, start, end, verbose=False)
        if fresh_f:
            bad_f = verify(store, "flow", fresh_f)
            print(f"  资金流: {len(fresh_f)} 只重取,{len(bad_f)} 只不一致"
                  + (f" -> {bad_f[:10]}" if bad_f else " ✓"))
            rc |= 1 if bad_f else 0
        else:
            print("  资金流: 重取不到数据,跳过校验")

    if rc:
        print("\n[!] 缓存与行情源对不上。数据源事后修订过,或缓存写坏了。\n"
              "    处理:删掉缓存目录重建,别拿对不上的数据出结论。", file=sys.stderr)
    return rc


def build_profit(store: ParquetStore, codes: list, floats: dict, a) -> None:
    """预算获利筹码并存好。

    不做这一步缓存也能用 —— 第一次回测会自己算一遍再存。放在这里只是为了
    把「要等的那一次」都集中在建缓存阶段,回测什么时候跑都是快的。

    走的是与回测完全相同的 ProfitResolver,所以能续算的就只算新增那几天:
    每天补一根K线时,这一步是秒级而不是把 5000 只从头滚 400 根。
    """
    p = BullParams(chip_decay=a.chip_decay, chip_bin_pct=a.chip_bin_pct,
                   chip_volume_unit=a.volume_unit)
    resolver = cached_feed.ProfitResolver(store, p, verbose=False)
    have = [c for c in codes if floats.get(c, 0) > 0]
    print(f"预算获利筹码 {len(have)} 只 ...", flush=True)

    t0 = time.perf_counter()
    done = 0
    for i in range(0, len(have), a.batch):
        chunk = have[i:i + a.batch]
        bars = store.load("bars", codes=chunk)
        for code in chunk:
            b = bars.get(code)
            if b is None or not len(b):
                continue
            try:
                resolver.get(code, b, floats[code])
                done += 1
            except Exception as exc:
                print(f"    {code} 筹码计算失败:{type(exc).__name__}: {exc}",
                      file=sys.stderr)
        used = time.perf_counter() - t0
        n = min(i + a.batch, len(have))
        print(f"  [{n}/{len(have)}] 已用{used:.0f}s"
              f" 预计还需{used / max(n, 1) * (len(have) - n):.0f}s", flush=True)

    total = resolver.hits + resolver.resumed + resolver.misses
    print(f"获利筹码:整段命中 {resolver.hits}/{total}、"
          f"续算 {resolver.resumed} 只、整只重算 {resolver.misses} 只"
          f"({time.perf_counter() - t0:.0f}s)")
    resolver.flush()


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    store = ParquetStore(a.cache, dividend_type=a.dividend)
    if a.status:
        return show_status(store)
    store.check_dividend("bars")

    end = a.end or pd.Timestamp.today().strftime("%Y%m%d")
    codes = ([c.strip() for c in a.codes.split(",") if c.strip()] if a.codes
             else feed.fetch_universe(a.sector, asof=end))
    if a.limit:
        codes = codes[:a.limit]

    # 增量时 --start 可省略:从缓存里最早的一天接着往后补
    start = a.start
    if not start:
        cov = store.coverage("bars")
        if not len(cov):
            print("缓存是空的,首次建缓存必须给 --start", file=sys.stderr)
            return 2
        start = pd.Timestamp(cov["first"].min()).strftime("%Y%m%d")
        print(f"未给 --start,沿用缓存起点 {start}")

    if a.verify:
        return do_verify(store, codes, start, end, a.verify, a.dividend, a.skip_flow)

    print(f"标的 {len(codes)} 只 | {start} ~ {end} | 复权 {a.dividend} | 缓存 {store.root}")
    t_start = time.perf_counter()
    n_bars = n_flow = 0

    for i in range(0, len(codes), a.batch):
        chunk = codes[i:i + a.batch]
        t0 = time.perf_counter()
        try:
            bars = feed.fetch_daily(chunk, start, end, dividend_type=a.dividend,
                                    download=not a.no_download)
        except Exception as exc:
            print(f"    行情批次失败跳过:{type(exc).__name__}: {exc}", file=sys.stderr)
            bars = {}
        t_b = time.perf_counter() - t0

        t0 = time.perf_counter()
        flows = {}
        if not a.skip_flow:
            try:
                flows = feed.fetch_money_flow(chunk, start, end,
                                              download=not a.no_download,
                                              verbose=(i == 0))
            except Exception as exc:
                print(f"    资金流批次失败跳过:{type(exc).__name__}: {exc}",
                      file=sys.stderr)
        t_f = time.perf_counter() - t0

        t0 = time.perf_counter()
        n_bars += store.save("bars", bars, verbose=False)
        n_flow += store.save("flow", flows, verbose=False)
        # 记下「这批问到哪天」。停牌的票取不到数据,但也算问过了,
        # 不记的话之后每跑一次回测都会重新去问这段空尾巴。
        store.record_asked("bars", chunk, start, end)
        if not a.skip_flow:
            store.record_asked("flow", chunk, start, end)
        t_w = time.perf_counter() - t0

        done = min(i + a.batch, len(codes))
        used = time.perf_counter() - t_start
        eta = used / done * (len(codes) - done)
        print(f"[{done}/{len(codes)}] 行情{t_b:.1f}s 资金流{t_f:.1f}s 落盘{t_w:.1f}s"
              f" | 已用{used / 60:.1f}min 预计还需{eta / 60:.1f}min", flush=True)

    t0 = time.perf_counter()
    floats = feed.fetch_float_shares(codes, verbose=False)
    store.save_floats(floats, verbose=False)
    print(f"流通股本 {len(floats)}/{len(codes)} 只({time.perf_counter() - t0:.1f}s)")

    if not a.skip_profit:
        build_profit(store, codes, floats, a)

    print(f"\n完成,用时 {(time.perf_counter() - t_start) / 60:.1f}min。"
          f"行情 {n_bars:,} 行、资金流 {n_flow:,} 行")
    show_status(store)
    print("\n之后回测加 --cache %s 即可走缓存;每天只需再跑一次本脚本补最新几天。"
          % a.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
