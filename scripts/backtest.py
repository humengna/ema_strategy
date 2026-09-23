# -*- coding: utf-8 -*-
"""选股效果回测:输出每日选股明细,并计算多个持有期的收益。

    python scripts/backtest.py --strategy bull --start 20240101 --end 20260904
    python scripts/backtest.py --strategy bull --sector 沪深A股 --holds 1,3,5,7,10,15,30
    python scripts/backtest.py --strategy bull --csv-dir data    # 本地CSV,不依赖 QMT

输出两份文件:
    <out>              每条信号一行:代码、选股日期、各持有期收益、是否可成交
    <out>_by_date.csv  按选股日汇总:当日选出几只、各持有期平均收益

判断标准
    绝对收益不足以说明问题 —— 牛市里 +2% 是跑输。本脚本以「同期全市场等权平均」
    为基准:对每个买入日,取当日全部标的按同样口径计算的 N 日收益均值,
    两者之差即超额收益。这是选股器最直接的对照,且不需要额外的指数数据。

口径
    买入  触发日的次日开盘(A股 T+1)。次日开盘一字涨停视为买不进,剔除。
    卖出  买入后第 N 个交易日收盘。
    成本  默认往返 0.3%(印花税+佣金+冲击),--cost 可调。

注意 持有1天 = 次日开盘买入、当日收盘卖出,即 T+0,A股不可执行。
    仍会计算(便于观察日内动量),但报告中标注为不可执行,不应据此下单。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ema_strategy import feed                                   # noqa: E402
from ema_strategy.io_utils import safe_to_csv                   # noqa: E402
from ema_strategy.bull import BullParams                        # noqa: E402
from ema_strategy.bull import run as bull_run                   # noqa: E402
from ema_strategy.sequence import Params as SeqParams           # noqa: E402
from ema_strategy.sequence import run as cross_run              # noqa: E402


def forward_return(bars: pd.DataFrame, hold: int) -> pd.Series:
    """信号日 t 的前瞻收益:t+1 开盘买入,t+hold 收盘卖出。"""
    entry = bars["open"].shift(-1)
    exit_ = bars["close"].shift(-hold)
    return (exit_ / entry - 1).rename("ret")


def entry_tradable(bars: pd.DataFrame, code: str) -> pd.Series:
    """次日开盘能否买入(一字涨停买不进)。"""
    nxt_open = bars["open"].shift(-1)
    pre = bars["preClose"].shift(-1) if "preClose" in bars.columns else bars["close"]
    return pd.Series(
        [feed.tradable_at_open(o, p, code) if pd.notna(o) and pd.notna(p) else False
         for o, p in zip(nxt_open, pre)], index=bars.index)


def signals_for(code: str, bars: pd.DataFrame, strategy: str,
                float_shares: float, flow, p_bull: BullParams,
                p_seq: SeqParams) -> pd.Series:
    """返回逐日布尔触发序列。"""
    if strategy == "bull":
        if not float_shares or float_shares <= 0:
            return pd.Series(False, index=bars.index)
        return bull_run(bars, float_shares, flow, p_bull)["daily"]["triggered"]

    res = cross_run(bars[feed.PRICE_COLS], p_seq)
    return res["daily"]["triggered"]


def collect(code: str, bars: pd.DataFrame, strategy: str, holds: list,
            float_shares: float, flow, p_bull: BullParams,
            p_seq: SeqParams) -> tuple[pd.DataFrame, dict]:
    """返回(该股触发事件含各持有期收益, {持有期: 逐日前瞻收益})。

    后者用于构造全市场等权基准。
    """
    warmup = p_bull.warmup if strategy == "bull" else p_seq.warmup
    if len(bars) < warmup + max(holds) + 2:
        return pd.DataFrame(), {}

    fwds = {h: forward_return(bars, h) for h in holds}
    trig = signals_for(code, bars, strategy, float_shares, flow, p_bull, p_seq)
    ok = entry_tradable(bars, code)

    hit = trig.reindex(bars.index).fillna(False).astype(bool)
    if not hit.any():
        return pd.DataFrame(), fwds

    events = pd.DataFrame({"code": code, "signal_date": bars.index[hit],
                           "tradable": ok[hit].to_numpy()})
    for h in holds:
        events[f"ret_{h}d"] = fwds[h][hit].to_numpy()
    return events, fwds


MIN_UNIVERSE = 30      # 少于这个数量,「全市场等权平均」不构成有效基准


def summarize(events: pd.DataFrame, bench: dict, holds: list, cost: float,
              universe_size: int = 0) -> None:
    if not len(events):
        print("没有产生任何信号")
        return

    blocked = int((~events["tradable"]).sum())
    ev = events[events["tradable"]].copy()
    print(f"\n{'=' * 78}")
    print(f"触发 {len(events)} 次 | 次日一字涨停买不进 {blocked} 次 | 可成交 {len(ev)} 次")
    if not len(ev):
        print("可成交样本为 0")
        return

    print(f"涉及 {ev['code'].nunique()} 只股票、{ev['signal_date'].nunique()} 个交易日 "
          f"({ev['signal_date'].min().date()} ~ {ev['signal_date'].max().date()})")

    if 0 < universe_size < MIN_UNIVERSE:
        print(f"\n  [警告] 股票池只有 {universe_size} 只,「全市场等权平均」基准无意义 —— "
              f"等于拿这几只自己跟自己比。\n"
              f"         超额一栏不可据此判断效果,请至少用 {MIN_UNIVERSE} 只以上重跑。")

    print(f"\n--- 各持有期收益(次日开盘买入,第N日收盘卖出;成本 {cost:.2%})---")
    head = (f"{'持有':>6}{'样本':>7}{'平均':>9}{'中位':>9}{'胜率':>8}"
            f"{'扣成本后':>10}{'同期基准':>10}{'超额':>9}{'跑赢':>8}{'t值':>7}")
    print(head)
    print("-" * len(head))
    for h in holds:
        col = f"ret_{h}d"
        r = ev[col].dropna()
        if not len(r):
            continue
        b = ev["signal_date"].map(bench[h])
        x = (ev[col] - b).dropna()
        se = x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan
        t = x.mean() / se if se and se > 0 else np.nan
        tag = " *" if h == 1 else ""
        print(f"{str(h) + '日' + tag:>6}{len(r):>7}{r.mean():>9.2%}{r.median():>9.2%}"
              f"{(r > 0).mean():>8.1%}{r.mean() - cost:>10.2%}"
              f"{b.mean():>10.2%}{x.mean():>9.2%}{(x > 0).mean():>8.1%}{t:>7.2f}")
    if 1 in holds:
        print("  * 持有1日 = 次日开盘买、当日收盘卖 = T+0,A股不可执行,仅供观察日内动量")
    print("  |t| < 2 表示超额与 0 无法区分,无论平均收益多好看")

    main_h = 5 if 5 in holds else holds[0]
    col = f"ret_{main_h}d"
    r = ev[col].dropna()
    if len(r):
        q = r.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"\n--- 持有{main_h}日的收益分布 ---")
        print(f"  5% {q[0.05]:+.2%} | 25% {q[0.25]:+.2%} | 50% {q[0.5]:+.2%} "
              f"| 75% {q[0.75]:+.2%} | 95% {q[0.95]:+.2%}")
        print(f"  最好 {r.max():+.2%}  最差 {r.min():+.2%}")
        if r.sum() != 0:
            top = r.nlargest(max(1, len(r) // 20))
            print(f"  最好 5% 的 {len(top)} 笔贡献总收益的 {top.sum() / r.sum():.0%}")

        ev = ev.copy()
        ev["month"] = ev["signal_date"].dt.to_period("M")
        by_month = ev.groupby("month").agg(
            选出只数=("code", "size"), 平均收益=(col, "mean"),
            胜率=(col, lambda s: (s > 0).mean())).round(4)
        if len(by_month) > 1:
            print(f"\n--- 按月(持有{main_h}日)---\n{by_month.to_string()}")


def daily_table(events: pd.DataFrame, holds: list) -> pd.DataFrame:
    """按选股日汇总:当日选出几只、各持有期平均收益。"""
    ev = events[events["tradable"]]
    if not len(ev):
        return pd.DataFrame()
    agg = {"选出只数": ("code", "size")}
    for h in holds:
        agg[f"ret_{h}d"] = (f"ret_{h}d", "mean")
    out = ev.groupby("signal_date").agg(**agg)
    out.insert(1, "股票", ev.groupby("signal_date")["code"].apply(lambda s: ",".join(sorted(s))))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="bull", choices=["bull", "cross"])
    ap.add_argument("--holds", default="1,3,5,7,10,15,30",
                    help="持有交易日数,逗号分隔。1日为T+0不可执行,仅供观察")
    ap.add_argument("--start", default="20240101")
    ap.add_argument("--end", default="")
    ap.add_argument("--codes", default="")
    ap.add_argument("--sector", default="沪深A股")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-download", action="store_true", dest="no_download",
                    help="跳过下载,直接读 QMT 本地缓存(数据已下过时用,快很多)")
    ap.add_argument("--csv-dir", default="", dest="csv_dir",
                    help="用本地CSV代替 xtdata(文件名即代码,需含 Date/OHLC/volume)")
    ap.add_argument("--cost", type=float, default=0.003, help="往返成本,默认 0.3%%")
    ap.add_argument("--pullback-window", type=int, default=6, dest="pullback_window")
    ap.add_argument("--no-pullback", action="store_true", dest="no_pullback")
    ap.add_argument("--profit-min", type=float, default=0.90, dest="profit_min")
    ap.add_argument("--volume-ratio", type=float, default=1.5, dest="volume_ratio")
    ap.add_argument("--volume-window", type=int, default=5, dest="volume_window")
    ap.add_argument("--no-flow", action="store_true",
                    help="[bull] 忽略资金流条件(无投研版/L2权限时用,结果不代表完整策略)")
    ap.add_argument("--out", default="backtest_events.csv")
    a = ap.parse_args(argv)

    holds = sorted({int(x) for x in a.holds.split(",") if x.strip()})
    if not holds:
        print("--holds 不能为空", file=sys.stderr)
        return 2

    p_seq = SeqParams()
    p_bull = BullParams(seq=p_seq, profit_min=a.profit_min,
                        volume_ratio=a.volume_ratio, volume_window=a.volume_window,
                        require_pullback=not a.no_pullback,
                        pullback_window=a.pullback_window)

    all_events, fwd_by_code = [], {h: {} for h in holds}
    no_flow_count = 0

    def handle(code: str, bars: pd.DataFrame, float_shares: float, flow) -> None:
        nonlocal no_flow_count
        if a.strategy == "bull" and flow is None and not a.no_flow:
            no_flow_count += 1
            return
        if a.no_flow and flow is None and a.strategy == "bull":
            # 用恒正的占位资金流,等价于跳过该条件
            flow = pd.DataFrame({"bidMostAmount": 1.0, "offMostAmount": 0.0},
                                index=bars.index)
        ev, fwds = collect(code, bars, a.strategy, holds, float_shares, flow, p_bull, p_seq)
        if len(ev):
            all_events.append(ev)
        for h, series in fwds.items():
            if len(series):
                fwd_by_code[h][code] = series

    if a.csv_dir:
        for path in sorted(glob.glob(os.path.join(a.csv_dir, "*.csv"))):
            code = os.path.splitext(os.path.basename(path))[0]
            d = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
            d = d.rename(columns=str.lower).sort_index()
            cols = feed.PRICE_COLS + (["volume"] if "volume" in d.columns else [])
            d = d[cols].dropna()
            if "volume" in d.columns and "amount" not in d.columns:
                d["amount"] = d["close"] * d["volume"] * 100
            handle(code, d, 1e9, None)
    else:
        codes = ([c.strip() for c in a.codes.split(",") if c.strip()] if a.codes
                 else feed.fetch_universe(a.sector, asof=a.end or None))
        if a.limit:
            codes = codes[:a.limit]
        print(f"标的 {len(codes)} 只 | 策略 {a.strategy} | "
              f"持有 {'/'.join(str(h) for h in holds)} 日")
        print("跳过下载,直接读 QMT 本地缓存" if a.no_download
              else "将下载缺失的历史数据(首次较慢;数据已下过可加 --no-download)")
        t_start = time.perf_counter()
        for i in range(0, len(codes), 200):
            chunk = codes[i:i + 200]
            try:
                data = feed.fetch_daily(chunk, a.start, a.end,
                                        download=not a.no_download)
            except Exception as exc:
                print(f"    行情批次失败:{type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            flows = (feed.fetch_money_flow(chunk, a.start, a.end, verbose=(i == 0),
                                           download=not a.no_download)
                     if a.strategy == "bull" and not a.no_flow else {})
            floats = feed.fetch_float_shares(chunk) if a.strategy == "bull" else {}
            for code, bars in data.items():
                handle(code, bars, floats.get(code, 0.0), flows.get(code))
            done = i + len(chunk)
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (len(codes) - done)
            print(f"  [{done}/{len(codes)}] 已用 {elapsed / 60:.1f}min,"
                  f"预计还需 {eta / 60:.1f}min", flush=True)

    if a.strategy == "bull" and no_flow_count:
        print(f"\n跳过 {no_flow_count} 只:无资金流数据。"
              f"若无投研版/Level2权限,可加 --no-flow 先看其余条件的效果"
              f"(但那不代表完整策略)。", file=sys.stderr)

    if not all_events:
        print("没有产生任何信号")
        return 1

    events = pd.concat(all_events, ignore_index=True).sort_values(
        ["signal_date", "code"]).reset_index(drop=True)
    # 全市场等权基准:每个买入日,全部标的同口径 N 日收益的均值
    bench = {h: pd.concat(d.values(), axis=1).mean(axis=1) if d else pd.Series(dtype=float)
             for h, d in fwd_by_code.items()}

    by_date = daily_table(events, holds)

    if a.no_flow:
        print("\n注意:--no-flow 已忽略资金流条件,以下结果不代表完整策略。")
    universe = len(fwd_by_code[holds[0]])
    summarize(events, bench, holds, a.cost, universe_size=universe)

    if len(by_date):
        print(f"\n--- 最近 10 个选股日 ---")
        show = by_date.tail(10).copy()
        show["股票"] = show["股票"].str.slice(0, 60)
        cols = ["选出只数", "股票"] + [f"ret_{h}d" for h in holds if f"ret_{h}d" in show.columns]
        print(show[cols].to_string())

    # 报告已完整输出,最后才落盘 —— 文件被占用也不至于丢掉全部计算
    written = safe_to_csv(events, a.out)
    if written:
        print(f"\n已写出 {written}({len(events)} 行,每条信号一行)")
    if len(by_date):
        path = safe_to_csv(by_date, a.out.replace(".csv", "_by_date.csv"), index=True)
        if path:
            print(f"已写出 {path}({len(by_date)} 个选股日)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
