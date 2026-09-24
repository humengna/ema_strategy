# -*- coding: utf-8 -*-
"""多头黄金眼选股器:形态维持 + 获利筹码 + 资金流入 + 放量。"""
from __future__ import annotations

import sys
from typing import Optional

import pandas as pd

from . import feed
from .bull import BullParams, run

PICK_COLUMNS = ["code", "pick_date", "pattern_start", "days_in_pattern",
                "profit_ratio", "net_inflow", "volume", "vol_base", "vol_times",
                "close", "ma5", "ma10", "ma30"]


def evaluate(code: str, bars: pd.DataFrame, float_shares: float,
             flow: Optional[pd.DataFrame] = None, p: Optional[BullParams] = None,
             max_gap_days: int = 30,
             profit_series: Optional[pd.Series] = None) -> Optional[dict]:
    """bars 最后一日是否触发。不触发返回 None。

    profit_series 可注入外部计算的获利筹码比例;此时不需要 float_shares。
    """
    p = p or BullParams()
    if len(bars) < p.warmup:
        return None
    if profit_series is None and (not float_shares or float_shares <= 0):
        return None                               # 既无外部值也无流通股本,筹码无从算起

    res = run(bars, float_shares, flow, p, profit_series)
    daily = res["daily"]
    last = daily.iloc[-1]
    if not bool(last["triggered"]):
        return None
    if bool(feed.gap_flags(daily.index, max_gap_days).iloc[-p.seq.slow:].any()):
        return None

    vol = bars["volume"]
    base = vol.rolling(p.volume_window).mean().shift(1).iloc[-1]
    return {
        "code": code, "pick_date": daily.index[-1],
        "pattern_start": pd.Timestamp(last["pattern_start"]),
        "days_in_pattern": int(last["days_in_pattern"]),
        "profit_ratio": round(float(last["profit_ratio"]), 4),
        "net_inflow": float(last["net_inflow"]),
        "volume": float(vol.iloc[-1]), "vol_base": float(base),
        "vol_times": round(float(vol.iloc[-1] / base), 2) if base else float("nan"),
        "close": float(last["close"]), "ma5": float(last["ma_f"]),
        "ma10": float(last["ma_m"]), "ma30": float(last["ma_s"]),
    }


def explain(code: str, bars: pd.DataFrame, float_shares: float,
            flow: Optional[pd.DataFrame] = None, p: Optional[BullParams] = None,
            max_gap_days: int = 30,
            profit_series: Optional[pd.Series] = None) -> str:
    """逐条打印形态与三个触发条件的实际取值。与 evaluate() 共用口径。"""
    p = p or BullParams()
    lines = []

    def row(tag: str, ok_: bool, detail: str) -> None:
        lines.append(f"  [{'通过' if ok_ else '不通过'}] {tag:20s} {detail}")

    if len(bars) < p.warmup:
        return f"{code}: 数据不足({len(bars)}根,需 >= {p.warmup}根)"

    res = run(bars, float_shares, flow, p, profit_series)
    daily, seq = res["daily"], res["sequences"]
    last, asof = daily.iloc[-1], daily.index[-1]

    lines.append(f"{code}  {asof.date()}  收{last['close']:.3f}  "
                 f"MA5={last['ma_f']:.3f} MA10={last['ma_m']:.3f} MA30={last['ma_s']:.3f}")

    lines.append("  --- 形态 ---")
    if len(seq):
        q = seq.iloc[-1]
        lines.append(f"  最近三金叉: 1号 {pd.Timestamp(q['start_date']).date()}"
                     f" -> 2号 {pd.Timestamp(q['c2_date']).date()}"
                     f" -> 3号 {pd.Timestamp(q['confirm_date']).date()}(形态启动日)")
    else:
        lines.append("  样本内未出现完整的 5上穿10 -> 5上穿30 -> 10上穿30 序列")

    active = bool(last["active"])
    broken = (last["ma_f"] < last["ma_s"]) or (last["ma_m"] < last["ma_s"])
    row("形态维持中", active,
        (f"启动于 {pd.Timestamp(last['pattern_start']).date()},已第 {int(last['days_in_pattern'])} 天"
         if active else ("MA5或MA10已跌破MA30,形态破坏" if broken else "尚无有效形态")))

    lines.append("  --- 触发条件 ---")
    if p.require_pullback:
        pb = daily["pullback_day"]
        hist = pb.iloc[-p.pullback_window:]
        hit = [d.date() for d in hist.index[hist]]
        row(f"近{p.pullback_window}日内有回踩", bool(last["recent_pullback"]),
            f"回踩日 {hit}" if hit else
            f"近{p.pullback_window}日无回踩(需最低价跌破MA30后收盘站回)")

    if p.require_ma_up:
        d1 = daily.iloc[-2] if len(daily) > 1 else last
        row("三条均线均向上", bool(last["ma_up"]),
            f"MA5 {last['ma_f']:.3f}/{d1['ma_f']:.3f}  "
            f"MA10 {last['ma_m']:.3f}/{d1['ma_m']:.3f}  "
            f"MA30 {last['ma_s']:.3f}/{d1['ma_s']:.3f}(当日/前一日)")

    pr = last["profit_ratio"]
    src = last.get("profit_source", "内置换手衰减法")
    row(f"获利筹码 > {p.profit_min:.0%}", bool(last["cond_profit"]),
        f"获利筹码 = {pr:.1%}(来源:{src})" if pd.notna(pr)
        else f"获利筹码不可得(来源:{src});外部数据缺该日时不触发")

    inflow = last["net_inflow"]
    if pd.isna(inflow):
        row("当日资金流入", False,
            "缺少 bidMostAmount/offMostAmount —— 需投研版(transactioncount1d)"
            "或Level2权限(l2transactioncount)")
    else:
        row("当日资金流入", bool(last["cond_inflow"]),
            f"bidMostAmount - offMostAmount = {inflow:,.0f}")

    vol = bars["volume"]
    base = vol.rolling(p.volume_window).mean().shift(1).iloc[-1]
    times = vol.iloc[-1] / base if base else float("nan")
    row(f"放量 > 前{p.volume_window}日均量 x {p.volume_ratio}", bool(last["cond_volume"]),
        f"成交量 {vol.iloc[-1]:,.0f} / 均量 {base:,.0f} = {times:.2f} 倍"
        if pd.notna(base) else "均量不可得")

    lines.append("  ==> " + ("【入选】 触发多头黄金眼" if bool(last["triggered"]) else "【不入选】"))
    return "\n".join(lines)


def scan(asof: str, codes: list[str], start: str, p: Optional[BullParams] = None,
         dividend: str = "back", batch: int = 200, verbose: bool = True) -> pd.DataFrame:
    """全市场扫描(需 QMT 环境)。"""
    p = p or BullParams()
    asof_ts = pd.Timestamp(asof)
    picks, no_flow, no_float = [], 0, 0

    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        if verbose:
            print(f"  [{i + len(chunk)}/{len(codes)}] 下载并扫描 ...", flush=True)
        try:
            data = feed.fetch_daily(chunk, start, asof, dividend_type=dividend)
        except Exception as exc:
            print(f"    行情批次失败跳过:{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        flows = feed.fetch_money_flow(chunk, start, asof, verbose=verbose and i == 0)
        floats = feed.fetch_float_shares(chunk)

        for code, bars in data.items():
            if len(bars) == 0 or bars.index[-1] != asof_ts:
                continue
            if code not in floats:
                no_float += 1
                continue
            if code not in flows:
                no_flow += 1
            try:
                hit = evaluate(code, bars, floats[code], flows.get(code), p)
                if hit:
                    picks.append(hit)
            except Exception as exc:
                print(f"    {code} 扫描失败:{type(exc).__name__}: {exc}", file=sys.stderr)

    if verbose and (no_flow or no_float):
        print(f"  跳过:无资金流数据 {no_flow} 只、无流通股本 {no_float} 只")
    if not picks:
        return pd.DataFrame(columns=PICK_COLUMNS)
    return pd.DataFrame(picks).sort_values(
        ["profit_ratio", "net_inflow"], ascending=[False, False]).reset_index(drop=True)
