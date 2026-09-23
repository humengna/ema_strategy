# -*- coding: utf-8 -*-
"""用已有的回测结果 CSV 直接算组合指标,不需要 QMT、不重新下载任何数据。

    python scripts/equity_from_events.py backtest_events.csv --hold 30
    python scripts/equity_from_events.py backtest_events_by_date.csv --hold 30 --strict

适用于 scripts/backtest.py 产出的两种文件(每条信号一行,或按选股日汇总)。

【口径与它的边界 —— 先读这段】
    events 里的 ret_Nd 是「该信号日的次日开盘买入、第N日收盘卖出」的收益,
    时间窗口锚定在**各自的信号日**上。

    --strict(默认):每 N 个交易日为一期,只取**恰好落在该期首日**的信号。
        此时 ret_Nd 的窗口与持仓期完全重合,期间收益精确,期与期之间不重叠。
        代价是样本变少(很多期可能空仓)。

    --loose:取该期内**全部**信号。样本大得多,但一个在期中第15天触发的信号,
        它的 ret_Nd 覆盖的是 t+1..t+30,而不是本期的持仓区间 ——
        存在时点错配,结果只能当粗略参考,不能当作真实可实现的收益。

    两种模式都无法做到日频盯市,所以夏普按「期收益」口径年化,
    而不是按日收益 —— 后者在只有区间收益时会严重低估波动、虚高夏普。

真正精确的日频资金曲线需要价格数据,用:
    python scripts/backtest_portfolio.py --mode rebalance --no-download
（--no-download 直接读 QMT 本地缓存,不重新下载)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ema_strategy.io_utils import safe_to_csv                   # noqa: E402

TRADING_DAYS_PER_YEAR = 242


def load_events(path: str, hold: int) -> pd.DataFrame:
    """读入 events,归一成 signal_date / weight / ret 三列。"""
    df = pd.read_csv(path, parse_dates=["signal_date"])
    col = f"ret_{hold}d"
    if col not in df.columns:
        have = [c for c in df.columns if c.startswith("ret_")]
        raise SystemExit(f"文件里没有 {col};可用的有 {have}")

    if "选出只数" in df.columns:                 # by_date 汇总文件
        out = df[["signal_date", "选出只数", col]].rename(
            columns={"选出只数": "weight", col: "ret"})
    else:                                        # 每条信号一行
        if "tradable" in df.columns:
            df = df[df["tradable"].astype(bool)]
        out = df[["signal_date", col]].rename(columns={col: "ret"})
        out["weight"] = 1.0
    return out.dropna(subset=["ret"]).sort_values("signal_date").reset_index(drop=True)


def period_series(ev: pd.DataFrame, hold: int, cost: float, strict: bool) -> pd.DataFrame:
    """切成互不重叠的持仓期,算每期等权收益。"""
    days = pd.DatetimeIndex(sorted(ev["signal_date"].unique()))
    axis = pd.date_range(days.min(), days.max(), freq="B")   # 以工作日近似交易日轴
    marks = list(range(0, len(axis), hold))

    rows = []
    for k in range(len(marks) - 1):
        t0, t1 = axis[marks[k]], axis[marks[k + 1]]
        if strict:
            sel = ev[ev["signal_date"] == t0]
            if not len(sel):                     # 首日无信号 -> 本期空仓
                sel = ev.iloc[0:0]
        else:
            sel = ev[(ev["signal_date"] >= t0) & (ev["signal_date"] < t1)]
        if len(sel):
            r = float(np.average(sel["ret"], weights=sel["weight"])) - cost
            n = float(sel["weight"].sum())
        else:
            r, n = 0.0, 0.0                      # 空仓期收益为 0
        rows.append({"period_start": t0, "n_signals": n, "ret": r})
    return pd.DataFrame(rows)


def report(p: pd.DataFrame, hold: int, cost: float, strict: bool, path: str) -> None:
    if not len(p):
        print("没有可用的持仓期")
        return
    eq = (1 + p["ret"]).cumprod()
    per_year = TRADING_DAYS_PER_YEAR / hold
    years = len(p) / per_year
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = p["ret"].std() * np.sqrt(per_year)
    dd = (eq / eq.cummax() - 1).min()
    invested = p[p["n_signals"] > 0]

    print(f"\n{'=' * 62}")
    print(f"{path} | 持有 {hold} 个交易日 | "
          f"{'strict(仅取每期首日信号)' if strict else 'loose(取每期全部信号)'} | "
          f"成本 {cost:.2%}")
    print(f"区间 {p['period_start'].iloc[0].date()} ~ {p['period_start'].iloc[-1].date()}")
    print(f"共 {len(p)} 期,其中有持仓 {len(invested)} 期、空仓 {len(p) - len(invested)} 期;"
          f"有持仓期平均 {invested['n_signals'].mean():.1f} 个信号"
          if len(invested) else f"共 {len(p)} 期,全部空仓")

    print(f"\n--- 组合指标 ---")
    print(f"  总收益     {eq.iloc[-1] - 1:+.2%}")
    print(f"  年化收益   {cagr:+.2%}")
    print(f"  年化波动   {vol:.2%}")
    print(f"  夏普       {cagr / vol if vol > 0 else float('nan'):.2f}   (按期收益年化,非日频)")
    print(f"  最大回撤   {dd:.2%}   (期末口径,期内浮亏看不到,真实回撤更深)")
    print(f"  Calmar     {cagr / abs(dd) if dd < 0 else float('nan'):.2f}")
    print(f"  期胜率     {(p['ret'] > 0).mean():.1%}  "
          f"最好 {p['ret'].max():+.2%}  最差 {p['ret'].min():+.2%}")

    if len(p) < 20:
        print(f"\n  [警告] 只有 {len(p)} 个持仓期,年化与夏普的抽样误差极大,"
              f"不足以判定策略优劣。")
    if not strict:
        print(f"\n  [警告] loose 模式存在时点错配:期中触发的信号,其 ret_{hold}d "
              f"覆盖的是自身信号日之后的 {hold} 天,\n"
              f"         并非本期持仓区间。结果只能当粗略参考,不是可实现收益。")

    p = p.copy()
    p["year"] = p["period_start"].dt.year
    print(f"\n--- 分年 ---")
    for y, g in p.groupby("year"):
        print(f"  {y}: {len(g)}期  收益 {(1 + g['ret']).prod() - 1:+7.2%}  "
              f"期均 {g['ret'].mean():+6.2%}  平均 {g['n_signals'].mean():.0f} 个信号/期")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("events_csv", help="backtest_events.csv 或 *_by_date.csv")
    ap.add_argument("--hold", type=int, default=30, help="持有交易日数,须是文件里已有的列")
    ap.add_argument("--cost", type=float, default=0.003, help="往返成本")
    ap.add_argument("--strict", action="store_true",
                    help="只取每期首日的信号(口径精确,样本少)")
    ap.add_argument("--loose", action="store_true",
                    help="取每期全部信号(样本多,存在时点错配)")
    ap.add_argument("--out", default="", help="把每期收益写出到 CSV")
    a = ap.parse_args(argv)

    if a.strict and a.loose:
        print("--strict 与 --loose 只能选一个", file=sys.stderr)
        return 2
    strict = not a.loose                          # 默认 strict

    ev = load_events(a.events_csv, a.hold)
    p = period_series(ev, a.hold, a.cost, strict)
    report(p, a.hold, a.cost, strict, a.events_csv)   # 先出报告,再落盘
    if a.out:
        written = safe_to_csv(p, a.out)
        if written:
            print(f"\n已写出 {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
