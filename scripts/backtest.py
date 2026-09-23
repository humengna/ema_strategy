# -*- coding: utf-8 -*-
"""选股效果回测:触发日次日开盘买入,持有 N 个交易日后收盘卖出。

    python scripts/backtest.py --strategy bull --start 20240101 --end 20260904 --hold 5
    python scripts/backtest.py --strategy cross --hold 5 --sector 沪深300 --limit 200
    python scripts/backtest.py --strategy bull --csv-dir data --hold 5    # 本地CSV,不依赖 QMT

判断标准
    绝对收益不足以说明问题 —— 牛市里 +2% 是跑输。本脚本以「同期全市场等权平均」
    为基准:对每个买入日,取当日全部标的按同样口径计算的 N 日收益均值,
    两者之差即超额收益。这是选股器最直接的对照,且不需要额外的指数数据。

口径
    买入  触发日的次日开盘(A股 T+1)。次日开盘一字涨停视为买不进,剔除。
    卖出  买入后第 N 个交易日收盘。
    成本  默认往返 0.3%(印花税+佣金+冲击),--cost 可调。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ema_strategy import feed                                   # noqa: E402
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


def collect(code: str, bars: pd.DataFrame, strategy: str, hold: int,
            float_shares: float, flow, p_bull: BullParams,
            p_seq: SeqParams) -> tuple[pd.DataFrame, pd.Series]:
    """返回(该股触发事件, 该股逐日前瞻收益)。后者用于构造全市场基准。"""
    warmup = p_bull.warmup if strategy == "bull" else p_seq.warmup
    if len(bars) < warmup + hold + 2:
        return pd.DataFrame(), pd.Series(dtype=float)

    fwd = forward_return(bars, hold)
    trig = signals_for(code, bars, strategy, float_shares, flow, p_bull, p_seq)
    ok = entry_tradable(bars, code)

    hit = trig & fwd.notna()
    events = pd.DataFrame({
        "code": code, "signal_date": bars.index[hit],
        "ret": fwd[hit].to_numpy(), "tradable": ok[hit].to_numpy(),
    })
    return events, fwd


MIN_UNIVERSE = 30      # 少于这个数量,「全市场等权平均」不构成有效基准


def summarize(events: pd.DataFrame, bench: pd.Series, hold: int, cost: float,
              universe_size: int = 0) -> None:
    if not len(events):
        print("没有产生任何信号")
        return

    blocked = int((~events["tradable"]).sum())
    ev = events[events["tradable"]].copy()
    print(f"\n{'=' * 64}")
    print(f"触发 {len(events)} 次 | 次日一字涨停买不进 {blocked} 次 | 可成交 {len(ev)} 次")
    if not len(ev):
        print("可成交样本为 0")
        return

    ev["bench"] = ev["signal_date"].map(bench)
    ev["excess"] = ev["ret"] - ev["bench"]
    ev["net"] = ev["ret"] - cost

    n_codes = ev["code"].nunique()
    n_days = ev["signal_date"].nunique()
    print(f"涉及 {n_codes} 只股票、{n_days} 个交易日 "
          f"({ev['signal_date'].min().date()} ~ {ev['signal_date'].max().date()})")

    r, x, net = ev["ret"], ev["excess"], ev["net"]
    print(f"\n--- 持有{hold}日(次日开盘买入,第{hold}日收盘卖出)---")
    print(f"  毛收益   平均 {r.mean():+.2%}  中位 {r.median():+.2%}  "
          f"胜率 {(r > 0).mean():.1%}  标准差 {r.std():.2%}")
    print(f"  扣成本后 平均 {net.mean():+.2%}  中位 {net.median():+.2%}  "
          f"胜率 {(net > 0).mean():.1%}   (往返成本 {cost:.2%})")
    if 0 < universe_size < MIN_UNIVERSE:
        print(f"\n  [警告] 股票池只有 {universe_size} 只,「全市场等权平均」基准无意义 —— "
              f"等于拿这几只自己跟自己比。\n"
              f"         超额收益一栏不可据此判断策略效果,请至少用 {MIN_UNIVERSE} 只以上"
              f"(如 --sector 沪深300)重跑。")

    valid = x.notna()
    if valid.any():
        xv = x[valid]
        se = xv.std() / np.sqrt(len(xv)) if len(xv) > 1 else np.nan
        t = xv.mean() / se if se and se > 0 else np.nan
        print(f"\n--- 对比同期全市场等权平均 ---")
        print(f"  同期基准 平均 {ev['bench'][valid].mean():+.2%}")
        print(f"  超额收益 平均 {xv.mean():+.2%}  中位 {xv.median():+.2%}  "
              f"跑赢比例 {(xv > 0).mean():.1%}")
        print(f"  t 值 {t:.2f}  (样本 {len(xv)};|t|<2 说明超额与 0 无法区分)")

    print(f"\n--- 分布 ---")
    q = r.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"  5% {q[0.05]:+.2%} | 25% {q[0.25]:+.2%} | 50% {q[0.5]:+.2%} "
          f"| 75% {q[0.75]:+.2%} | 95% {q[0.95]:+.2%}")
    print(f"  最好 {r.max():+.2%}  最差 {r.min():+.2%}")
    top = r.nlargest(max(1, len(r) // 20))
    print(f"  最好 5% 的 {len(top)} 笔贡献总收益的 "
          f"{top.sum() / r.sum():.0%}" if r.sum() != 0 else "")

    ev["month"] = ev["signal_date"].dt.to_period("M")
    by_month = ev.groupby("month").agg(
        次数=("ret", "size"), 毛收益=("ret", "mean"),
        超额=("excess", "mean"), 胜率=("ret", lambda s: (s > 0).mean())).round(4)
    if len(by_month) > 1:
        print(f"\n--- 按月(检查是否集中在某段行情)---\n{by_month.to_string()}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="bull", choices=["bull", "cross"])
    ap.add_argument("--hold", type=int, default=5, help="持有交易日数")
    ap.add_argument("--start", default="20240101")
    ap.add_argument("--end", default="")
    ap.add_argument("--codes", default="")
    ap.add_argument("--sector", default="沪深A股")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--csv-dir", default="", dest="csv_dir",
                    help="用本地CSV代替 xtdata(文件名即代码,需含 Date/OHLC/volume)")
    ap.add_argument("--cost", type=float, default=0.003, help="往返成本,默认 0.3%%")
    ap.add_argument("--profit-min", type=float, default=0.90, dest="profit_min")
    ap.add_argument("--volume-ratio", type=float, default=1.5, dest="volume_ratio")
    ap.add_argument("--volume-window", type=int, default=5, dest="volume_window")
    ap.add_argument("--no-flow", action="store_true",
                    help="[bull] 忽略资金流条件(无投研版/L2权限时用,结果不代表完整策略)")
    ap.add_argument("--out", default="backtest_events.csv")
    a = ap.parse_args(argv)

    p_seq = SeqParams()
    p_bull = BullParams(seq=p_seq, profit_min=a.profit_min,
                        volume_ratio=a.volume_ratio, volume_window=a.volume_window)

    all_events, fwd_by_code = [], {}
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
        ev, fwd = collect(code, bars, a.strategy, a.hold, float_shares, flow, p_bull, p_seq)
        if len(ev):
            all_events.append(ev)
        if len(fwd):
            fwd_by_code[code] = fwd

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
        print(f"标的 {len(codes)} 只 | 策略 {a.strategy} | 持有 {a.hold} 日")
        for i in range(0, len(codes), 200):
            chunk = codes[i:i + 200]
            print(f"  [{i + len(chunk)}/{len(codes)}] ...", flush=True)
            try:
                data = feed.fetch_daily(chunk, a.start, a.end)
            except Exception as exc:
                print(f"    行情批次失败:{type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            flows = (feed.fetch_money_flow(chunk, a.start, a.end, verbose=(i == 0))
                     if a.strategy == "bull" and not a.no_flow else {})
            floats = feed.fetch_float_shares(chunk) if a.strategy == "bull" else {}
            for code, bars in data.items():
                handle(code, bars, floats.get(code, 0.0), flows.get(code))

    if a.strategy == "bull" and no_flow_count:
        print(f"\n跳过 {no_flow_count} 只:无资金流数据。"
              f"若无投研版/Level2权限,可加 --no-flow 先看其余条件的效果"
              f"(但那不代表完整策略)。", file=sys.stderr)

    if not all_events:
        print("没有产生任何信号")
        return 1

    events = pd.concat(all_events, ignore_index=True)
    # 全市场等权基准:每个买入日,全部标的同口径 N 日收益的均值
    bench = pd.concat(fwd_by_code.values(), axis=1).mean(axis=1)

    events.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"已写出 {a.out}({len(events)} 行)")
    if a.no_flow:
        print("\n注意:--no-flow 已忽略资金流条件,以下结果不代表完整策略。")
    summarize(events, bench, a.hold, a.cost, universe_size=len(fwd_by_code))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
