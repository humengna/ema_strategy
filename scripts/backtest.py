# -*- coding: utf-8 -*-
"""对三金叉规则做样本内统计:触发组 vs 未触发组的前瞻收益。

    python scripts/backtest.py --start 20220101 --end 20260904
    python scripts/backtest.py --csv-dir ./data     # 用本地CSV,不依赖 QMT

本地CSV模式下,每个文件需含 Date/open/high/low/close 列,文件名即代码。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ema_strategy import feed                      # noqa: E402
from ema_strategy.sequence import Params, run      # noqa: E402

HORIZONS = (5, 10, 20, 30)


def collect(code: str, bars: pd.DataFrame, p: Params) -> list[dict]:
    """对一只股票,收集每条完整序列在确认点次日开盘买入后的前瞻收益。"""
    if len(bars) < p.warmup:
        return []
    res = run(bars[feed.PRICE_COLS], p)
    daily, seq = res["daily"], res["sequences"]
    idx, op, cl = daily.index, daily["open"], daily["close"]

    rows = []
    for _, q in seq.iterrows():
        i = idx.get_loc(q["confirm_date"])
        if i + 1 >= len(idx):
            continue
        entry = op.iat[i + 1]                       # T+1:确认点次日开盘买入
        if not np.isfinite(entry) or entry <= 0:
            continue
        for n in HORIZONS:
            j = min(i + 1 + n, len(idx) - 1)
            rows.append({"code": code, "triggered": bool(q["triggered"]),
                         "start_bar": q["start_bar"], "confirm_bar": q["confirm_bar"],
                         "span_1_3": int(q["span_1_3"]), "N": n,
                         "ret": cl.iat[j] / entry - 1})
    return rows


def report(df: pd.DataFrame) -> None:
    def agg(sub: pd.DataFrame, key) -> pd.DataFrame:
        return sub.groupby(key, observed=True)["ret"].agg(
            次数="size", 平均="mean", 中位="median",
            胜率=lambda s: (s > 0).mean()).round(4)

    total = len(df) // len(HORIZONS)
    print(f"\n完整三金叉序列 {total} 条,其中触发 {int(df[df.N == HORIZONS[0]].triggered.sum())} 条")
    for n in HORIZONS:
        print(f"\n-- 持有{n}日:触发 vs 未触发 --\n{agg(df[df.N == n], 'triggered').to_string()}")

    s20 = df[df.N == 20].copy()
    s20["启动贯穿"] = s20.start_bar == "贯穿"
    s20["确认跳空"] = s20.confirm_bar == "跳空"
    print(f"\n-- 持有20日:两个条件各自的贡献 --\n{agg(s20, '启动贯穿').to_string()}")
    print(f"\n{agg(s20, '确认跳空').to_string()}")
    print(f"\n-- 持有20日:两条件交叉 --\n"
          f"{s20.groupby(['启动贯穿', '确认跳空'])['ret'].agg(次数='size', 平均='mean', 胜率=lambda s: (s > 0).mean()).round(4).to_string()}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20220101")
    ap.add_argument("--end", default="")
    ap.add_argument("--codes", default="")
    ap.add_argument("--sector", default="沪深A股")
    ap.add_argument("--csv-dir", default="", dest="csv_dir",
                    help="用本地CSV目录代替 xtdata(便于无 QMT 环境验证)")
    ap.add_argument("--max-span", type=int, default=60, dest="max_span")
    ap.add_argument("--out", default="backtest_events.csv")
    a = ap.parse_args(argv)
    p = Params(max_span=a.max_span)

    rows = []
    if a.csv_dir:
        for path in sorted(glob.glob(os.path.join(a.csv_dir, "*.csv"))):
            code = os.path.splitext(os.path.basename(path))[0]
            d = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
            d = d.rename(columns=str.lower).sort_index()[feed.PRICE_COLS].dropna()
            rows += collect(code, d, p)
    else:
        codes = ([c.strip() for c in a.codes.split(",") if c.strip()] if a.codes
                 else feed.fetch_universe(a.sector, asof=a.end or None))
        print(f"标的 {len(codes)} 只")
        for i in range(0, len(codes), 200):
            chunk = codes[i:i + 200]
            print(f"  [{i + len(chunk)}/{len(codes)}] ...", flush=True)
            try:
                data = feed.fetch_daily(chunk, a.start, a.end)
            except Exception as exc:
                print(f"    批次失败:{type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            for code, bars in data.items():
                rows += collect(code, bars, p)

    if not rows:
        print("没有产生任何序列")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"已写出 {a.out}({len(df)} 行)")
    report(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
