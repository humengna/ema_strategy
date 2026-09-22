# -*- coding: utf-8 -*-
"""环境自检:确认 QMT 能取数,并端到端跑通一次选股判定。

    python scripts/smoke_test.py
    python scripts/smoke_test.py --codes 000001.SZ,600519.SH --date 20260904

在跑全市场扫描之前先跑这个,能把问题定位到具体某一步,
而不是在一个跑了半小时的全市场任务里报错。
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CODES = ["000001.SZ", "600000.SH", "300750.SZ"]


def step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=",".join(DEFAULT_CODES))
    ap.add_argument("--date", default="", help="选股日 YYYYMMDD,缺省用最后一个有数据的交易日")
    ap.add_argument("--start", default="")
    a = ap.parse_args(argv)
    codes = [c.strip() for c in a.codes.split(",") if c.strip()]

    step(1, "导入 xtquant")
    try:
        from xtquant import xtdata
    except Exception as exc:
        print(f"  失败: {type(exc).__name__}: {exc}")
        print("  -> xtquant 只支持 Windows,且需要 QMT/miniQMT 客户端已启动并登录。")
        return 1
    print("  OK")

    step(2, "导入 ema_strategy")
    from ema_strategy import evaluate, explain, feed
    from ema_strategy.sequence import Params
    print("  OK")

    step(3, f"取数 {codes}")
    end = a.date or pd.Timestamp.today().strftime("%Y%m%d")
    start = a.start or (pd.Timestamp(end) - pd.Timedelta(days=400)).strftime("%Y%m%d")
    print(f"  区间 {start} ~ {end},后复权,不填充停牌")
    try:
        data = feed.fetch_daily(codes, start, end)
    except Exception as exc:
        print(f"  失败: {type(exc).__name__}: {exc}")
        print("  -> 确认 QMT 客户端在运行,且该账号有日线行情权限。")
        return 1

    ok = 0
    for code in codes:
        bars = data.get(code)
        if bars is None or not len(bars):
            print(f"  {code}: 无数据")
            continue
        ok += 1
        print(f"  {code}: {len(bars)} 根K线  {bars.index[0].date()} ~ {bars.index[-1].date()}"
              f"  最新收盘 {bars['close'].iloc[-1]:.3f}")
    if not ok:
        print("  -> 一只都没取到。先在 QMT 里手动下载一次日线数据,或检查代码格式(000001.SZ)。")
        return 1

    step(4, "跑规则判定")
    p = Params()
    for code in codes:
        bars = data.get(code)
        if bars is None or len(bars) < p.warmup:
            continue
        hit = evaluate(code, bars, p)
        print(f"  {code}: {'【触发】' if hit else '未触发'}")

    step(5, "逐条诊断(示例)")
    sample = next((c for c in codes if data.get(c) is not None and len(data[c]) >= p.warmup), None)
    if sample:
        print(explain(sample, data[sample], p))

    print("\n自检通过。接下来可以跑:")
    print("  python -m ema_strategy --date <YYYYMMDD> --sector 沪深300 --out picks.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
