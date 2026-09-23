# -*- coding: utf-8 -*-
"""组合回测:把选股信号跑成一条资金曲线。

    python scripts/backtest_portfolio.py --strategy bull --start 20240101 --end 20260904 \
        --hold 5 --max-positions 20 --sector 沪深300
    python scripts/backtest_portfolio.py --strategy bull --csv-dir data --hold 5 --no-flow

与 scripts/backtest.py 的区别:
    backtest.py           事件统计,每笔信号的前瞻收益,不受资金约束
    backtest_portfolio.py 组合模拟,有资金、仓位上限、并发持仓,给出年化/回撤/Sharpe

基准为全市场等权日收益累乘 —— 对只做多的选股器,这是最直接的对照。
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
from ema_strategy.portfolio import PortfolioParams, metrics, simulate   # noqa: E402
from ema_strategy.sequence import Params as SeqParams           # noqa: E402
from ema_strategy.sequence import run as cross_run              # noqa: E402

MIN_UNIVERSE = 30


def build_signals(code: str, bars: pd.DataFrame, strategy: str, float_shares: float,
                  flow, p_bull: BullParams, p_seq: SeqParams):
    """返回 (触发序列, 排序分, 次日可成交标志)。"""
    if strategy == "bull":
        if not float_shares or float_shares <= 0:
            return None
        daily = bull_run(bars, float_shares, flow, p_bull)["daily"]
        sig, score = daily["triggered"], daily["profit_ratio"].fillna(0.0)
    else:
        daily = cross_run(bars[feed.PRICE_COLS], p_seq)["daily"]
        sig = daily["triggered"]
        score = pd.Series(0.0, index=bars.index)

    nxt_open = bars["open"].shift(-1)
    pre = bars["preClose"].shift(-1) if "preClose" in bars.columns else bars["close"]
    ok = pd.Series(
        [feed.tradable_at_open(o, q, code) if pd.notna(o) and pd.notna(q) else False
         for o, q in zip(nxt_open, pre)], index=bars.index)
    return sig, score, ok


def report(res: dict, p: PortfolioParams) -> None:
    curve, bench, trades = res["equity"], res["benchmark"], res["trades"]
    m = metrics(curve, trades, res["holdings"])
    bm = metrics(bench, pd.DataFrame(columns=["ret"]), res["holdings"])

    print(f"\n{'=' * 66}")
    print(f"区间 {curve.index[0].date()} ~ {curve.index[-1].date()}  "
          f"({len(curve)} 个交易日)")
    print(f"持有 {p.hold_days} 日 | 最多 {p.max_positions} 个仓位 | "
          f"单日最多新开 {p.new_per_day} 个 | 往返成本 {p.cost:.2%}")
    if res["blocked"]:
        print(f"因次日开盘一字涨停放弃 {res['blocked']} 个信号")

    util = m["资金利用率"]
    print(f"\n--- 仓位 ---")
    print(f"  平均持仓 {m['平均持仓数']:.1f} 只 / 上限 {p.max_positions} 只,"
          f"资金利用率 {util:.0%}")
    if util < 0.3:
        print(f"  [警告] 资金利用率仅 {util:.0%},策略大部分时间空仓。")
        print(f"         此时拿总收益与满仓基准直接比较会严重低估策略 —— "
              f"差距主要来自没进场,而不是选股不行。")
        print(f"         应看下方「单笔平均」与 scripts/backtest.py 的超额收益,"
              f"或放宽条件/扩大股票池提高出手频率。")

    print(f"\n--- 组合 vs 全市场等权 ---")
    print(f"{'':12}{'策略':>12}{'全市场等权':>14}")
    for key in ("总收益", "年化收益", "年化波动", "最大回撤"):
        print(f"{key:12}{m[key]:>11.2%}{bm.get(key, float('nan')):>14.2%}")
    for key in ("Sharpe", "Calmar"):
        b = bm.get(key, float("nan"))
        print(f"{key:12}{m[key]:>11.2f}{b:>14.2f}")

    print(f"\n--- 交易 ---")
    print(f"  笔数 {m['交易笔数']}  单笔胜率 {m['单笔胜率']:.1%}  "
          f"单笔平均 {m['单笔平均']:+.2%}(已扣成本)")
    if len(trades):
        hd = trades["hold_days"].value_counts().sort_index()
        print(f"  持有天数分布 {hd.to_dict()}")

    if len(curve) > 250:
        yearly = curve.resample("YE").last().pct_change()
        yearly.iloc[0] = curve.resample("YE").last().iloc[0] / curve.iloc[0] - 1
        by = bench.resample("YE").last().pct_change()
        by.iloc[0] = bench.resample("YE").last().iloc[0] / bench.iloc[0] - 1
        print(f"\n--- 分年 ---")
        for ts in yearly.index:
            print(f"  {ts.year}  策略 {yearly[ts]:+7.2%}   基准 {by[ts]:+7.2%}   "
                  f"超额 {yearly[ts] - by[ts]:+7.2%}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="bull", choices=["bull", "cross"])
    ap.add_argument("--hold", type=int, default=5, help="持有交易日数(>=2,T+1)")
    ap.add_argument("--max-positions", type=int, default=20, dest="max_positions")
    ap.add_argument("--max-new-per-day", type=int, default=0, dest="max_new_per_day")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--cost", type=float, default=0.003)
    ap.add_argument("--start", default="20240101")
    ap.add_argument("--end", default="")
    ap.add_argument("--codes", default="")
    ap.add_argument("--sector", default="沪深A股")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--csv-dir", default="", dest="csv_dir")
    ap.add_argument("--profit-min", type=float, default=0.90, dest="profit_min")
    ap.add_argument("--volume-ratio", type=float, default=1.5, dest="volume_ratio")
    ap.add_argument("--volume-window", type=int, default=5, dest="volume_window")
    ap.add_argument("--no-flow", action="store_true",
                    help="[bull] 忽略资金流条件(无投研版/L2权限时用,不代表完整策略)")
    ap.add_argument("--out", default="portfolio_equity.csv")
    a = ap.parse_args(argv)

    p_seq = SeqParams()
    p_bull = BullParams(seq=p_seq, profit_min=a.profit_min,
                        volume_ratio=a.volume_ratio, volume_window=a.volume_window)
    try:
        p_port = PortfolioParams(hold_days=a.hold, max_positions=a.max_positions,
                                 max_new_per_day=a.max_new_per_day,
                                 cost=a.cost, init_capital=a.capital)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    bars_by_code, sigs, scores, oks = {}, {}, {}, {}
    no_flow = 0

    def handle(code: str, bars: pd.DataFrame, float_shares: float, flow) -> None:
        nonlocal no_flow
        if a.strategy == "bull" and flow is None:
            if not a.no_flow:
                no_flow += 1
                return
            flow = pd.DataFrame({"bidMostAmount": 1.0, "offMostAmount": 0.0},
                                index=bars.index)
        if len(bars) < (p_bull.warmup if a.strategy == "bull" else p_seq.warmup) + a.hold + 2:
            return
        built = build_signals(code, bars, a.strategy, float_shares, flow, p_bull, p_seq)
        if built is None:
            return
        sig, score, ok = built
        bars_by_code[code], sigs[code], scores[code], oks[code] = bars, sig, score, ok

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

    if no_flow:
        print(f"\n跳过 {no_flow} 只:无资金流数据。无投研版/Level2权限时可加 --no-flow "
              f"先看其余条件(但那不代表完整策略)。", file=sys.stderr)
    if not bars_by_code:
        print("没有可用标的")
        return 1
    if len(bars_by_code) < MIN_UNIVERSE:
        print(f"\n[警告] 股票池只有 {len(bars_by_code)} 只,「全市场等权」基准无意义 —— "
              f"等于拿这几只自己跟自己比。\n"
              f"        结论不可据此判断策略效果,请用 {MIN_UNIVERSE} 只以上重跑。")

    if a.no_flow:
        print("\n注意:--no-flow 已忽略资金流条件,以下结果不代表完整策略。")

    res = simulate(bars_by_code, sigs, p_port, scores, oks)
    pd.DataFrame({"equity": res["equity"], "benchmark": res["benchmark"],
                  "holdings": res["holdings"]}).to_csv(a.out, encoding="utf-8-sig")
    if len(res["trades"]):
        res["trades"].to_csv(a.out.replace(".csv", "_trades.csv"),
                             index=False, encoding="utf-8-sig")
    print(f"已写出 {a.out}")
    report(res, p_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
