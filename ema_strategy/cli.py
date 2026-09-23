# -*- coding: utf-8 -*-
"""命令行入口。

    python -m ema_strategy --date 20260904
    python -m ema_strategy --date 20260904 --explain 000001.SZ
    python -m ema_strategy --date 20260904 --codes 000001.SZ,600000.SH --no-start-bar
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import feed
from .bull import BullParams
from .bull_picker import explain as bull_explain
from .bull_picker import scan as bull_scan
from .picker import explain, scan
from .sequence import Params


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ema_strategy",
        description="黄金眼三金叉选股(1号 MA5上穿MA10 / 2号 MA5上穿MA30 / 3号 MA10上穿MA30)")
    ap.add_argument("--strategy", default="cross", choices=["cross", "bull"],
                    help="cross=三金叉两端形态(启动贯穿+确认跳空);"
                         "bull=多头黄金眼(形态维持+获利筹码+资金流入+放量)")
    ap.add_argument("--date", required=True, help="选股日 YYYYMMDD")
    ap.add_argument("--start", default="", help="行情起始日,缺省为选股日前 400 自然日")
    ap.add_argument("--codes", default="", help="逗号分隔;留空则取整个板块")
    ap.add_argument("--sector", default="沪深A股")
    ap.add_argument("--dividend", default="back", choices=["back", "front", "none"],
                    help="复权方式,信号生成必须用 back(后复权),否则历史会 repaint")
    ap.add_argument("--max-span", type=int, default=60, dest="max_span",
                    help="1号到3号金叉的最大间隔交易日")
    ap.add_argument("--no-start-bar", action="store_true",
                    help="放宽「启动点须贯穿」(实测该条件为负贡献,见 README)")
    ap.add_argument("--distinct-days", action="store_true",
                    help="要求三个金叉分属不同交易日")
    ap.add_argument("--limit", type=int, default=0,
                    help="只取股票池前 N 只,用于先小规模试跑(0=不限制)")
    ap.add_argument("--explain", default="", metavar="CODE", help="只诊断这一只股票")
    ap.add_argument("--profit-min", type=float, default=0.90, dest="profit_min",
                    help="[bull] 获利筹码下限,默认 0.90")
    ap.add_argument("--volume-ratio", type=float, default=1.5, dest="volume_ratio",
                    help="[bull] 放量倍数,默认 1.5")
    ap.add_argument("--volume-window", type=int, default=5, dest="volume_window",
                    help="[bull] 均量窗口(不含当日),默认 5")
    ap.add_argument("--out", default="picks.csv")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    params = Params(max_span=args.max_span, require_distinct_days=args.distinct_days)
    start = args.start or (pd.Timestamp(args.date) - pd.Timedelta(days=400)).strftime("%Y%m%d")

    if args.strategy == "bull":
        return _run_bull(args, params, start)

    if args.explain:
        bars = feed.fetch_daily([args.explain], start, args.date,
                                dividend_type=args.dividend).get(args.explain)
        if bars is None or not len(bars):
            print(f"{args.explain}: 无数据", file=sys.stderr)
            return 1
        print(explain(args.explain, bars, params, require_start_bar=not args.no_start_bar))
        return 0

    codes = ([c.strip() for c in args.codes.split(",") if c.strip()] if args.codes
             else feed.fetch_universe(args.sector, asof=args.date))
    if args.limit:
        codes = codes[:args.limit]
    print(f"选股日 {args.date} | 标的 {len(codes)} 只 | 行情自 {start} | 复权 {args.dividend}")

    picks = scan(args.date, codes, start, params, args.dividend,
                 require_start_bar=not args.no_start_bar)
    if not len(picks):
        print("当日无股票触发")
        return 1

    picks.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n触发 {len(picks)} 只,已写出 {args.out}\n")
    cols = ["code", "start_date", "confirm_date", "start_bar", "confirm_bar",
            "span_1_3", "regime", "close"]
    print(picks[cols].to_string(index=False))
    print("\n提示:次日开盘前用 attach_tradability() 过滤一字涨停后再下单。")
    return 0


def _run_bull(args, seq_params: Params, start: str) -> int:
    """多头黄金眼:形态维持 + 获利筹码 + 资金流入 + 放量。"""
    p = BullParams(seq=seq_params, profit_min=args.profit_min,
                   volume_ratio=args.volume_ratio, volume_window=args.volume_window)

    if args.explain:
        code = args.explain
        bars = feed.fetch_daily([code], start, args.date,
                                dividend_type=args.dividend).get(code)
        if bars is None or not len(bars):
            print(f"{code}: 无数据", file=sys.stderr)
            return 1
        floats = feed.fetch_float_shares([code])
        if code not in floats:
            print(f"{code}: 取不到流通股本,无法计算获利筹码", file=sys.stderr)
            return 1
        flow = feed.fetch_money_flow([code], start, args.date).get(code)
        print(bull_explain(code, bars, floats[code], flow, p))
        return 0

    codes = ([c.strip() for c in args.codes.split(",") if c.strip()] if args.codes
             else feed.fetch_universe(args.sector, asof=args.date))
    if args.limit:
        codes = codes[:args.limit]
    print(f"选股日 {args.date} | 标的 {len(codes)} 只 | 行情自 {start} | 复权 {args.dividend}")
    print(f"条件: 形态维持 且 获利筹码>{p.profit_min:.0%} 且 资金流入>0 "
          f"且 量>前{p.volume_window}日均量x{p.volume_ratio}")

    picks = bull_scan(args.date, codes, start, p, args.dividend)
    if not len(picks):
        print("当日无股票触发")
        return 1

    picks.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n触发 {len(picks)} 只,已写出 {args.out}\n")
    cols = ["code", "pattern_start", "days_in_pattern", "profit_ratio",
            "net_inflow", "vol_times", "close"]
    print(picks[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
