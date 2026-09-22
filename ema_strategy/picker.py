# -*- coding: utf-8 -*-
"""选股器:判断某只股票在指定交易日是否触发黄金眼,并给出可读的逐条诊断。"""
from __future__ import annotations

import sys
from typing import Optional

import pandas as pd

from . import feed
from .sequence import Params, run

PICK_COLUMNS = ["code", "pick_date", "start_date", "c2_date", "confirm_date",
                "start_bar", "confirm_bar", "span_1_3", "span_1_2", "span_2_3",
                "regime", "ma30_up", "close", "ma5", "ma10", "ma30"]


def evaluate(code: str, bars: pd.DataFrame, p: Optional[Params] = None,
             max_gap_days: int = 30, require_start_bar: bool = True) -> Optional[dict]:
    """bars 最后一日是否为触发日。不触发返回 None。

    require_start_bar=False 放宽「启动点须贯穿」—— 实测该条件为负贡献,见 README。
    """
    p = p or Params()
    if len(bars) < p.warmup:
        return None

    res = run(bars[feed.PRICE_COLS], p)
    daily, seq = res["daily"], res["sequences"]
    asof = daily.index[-1]
    if not len(seq):
        return None

    q = seq.iloc[-1]
    if q["confirm_date"] != asof:                       # 今日不是确认点
        return None
    if q["confirm_bar"] != p.confirm_bar:               # 确认点须跳空
        return None
    if require_start_bar and q["start_bar"] != p.start_bar:
        return None
    if bool(feed.gap_flags(daily.index, max_gap_days).iloc[-p.slow:].any()):
        return None

    last = daily.iloc[-1]
    return {
        "code": code, "pick_date": asof,
        "start_date": q["start_date"], "c2_date": q["c2_date"], "confirm_date": q["confirm_date"],
        "start_bar": q["start_bar"], "confirm_bar": q["confirm_bar"],
        "span_1_3": int(q["span_1_3"]), "span_1_2": int(q["span_1_2"]),
        "span_2_3": int(q["span_2_3"]),
        "regime": last["regime"],
        "ma30_up": bool(daily["ma_s"].iat[-1] > daily["ma_s"].iat[-2]),
        "close": float(last["close"]), "ma5": float(last["ma_f"]),
        "ma10": float(last["ma_m"]), "ma30": float(last["ma_s"]),
    }


def explain(code: str, bars: pd.DataFrame, p: Optional[Params] = None,
            max_gap_days: int = 30, require_start_bar: bool = True) -> str:
    """逐条打印三金叉序列与两端形态的实际取值。

    与 evaluate() 共用同一套判定,两者结论必须一致(tests/test_picker.py 逐日比对)。
    """
    p = p or Params()
    lines = []

    def row(tag: str, passed: bool, detail: str) -> None:
        lines.append(f"  [{'通过' if passed else '不通过'}] {tag:18s} {detail}")

    if len(bars) < p.warmup:
        return f"{code}: 数据不足({len(bars)}根,需 >= {p.warmup}根)"

    res = run(bars[feed.PRICE_COLS], p)
    daily, seq = res["daily"], res["sequences"]
    last, asof = daily.iloc[-1], daily.index[-1]

    lines.append(f"{code}  {asof.date()}  收{last['close']:.3f} 开{last['open']:.3f} "
                 f"高{last['high']:.3f} 低{last['low']:.3f}")
    lines.append(f"  MA{p.fast}={last['ma_f']:.3f}  MA{p.mid}={last['ma_m']:.3f}  "
                 f"MA{p.slow}={last['ma_s']:.3f}   当日K线={last['bar_type']}  排列={last['regime']}")

    if not len(seq):
        lines.append("  [不通过] 三金叉序列         样本内未出现完整的 1->2->3 序列")
        lines.append("  ==> 【不入选】")
        return "\n".join(lines)

    q = seq.iloc[-1]
    lines.append("  --- 最近一条完整序列 ---")
    lines.append(f"  1号金叉(启动) {pd.Timestamp(q['start_date']).date()}   MA{p.fast}上穿MA{p.mid}   K线={q['start_bar']}")
    lines.append(f"  2号金叉        {pd.Timestamp(q['c2_date']).date()}   MA{p.fast}上穿MA{p.slow}")
    lines.append(f"  3号金叉(确认) {pd.Timestamp(q['confirm_date']).date()}   MA{p.mid}上穿MA{p.slow}   K线={q['confirm_bar']}")
    lines.append(f"  间隔: 1->2 {int(q['span_1_2'])}日,2->3 {int(q['span_2_3'])}日,合计 {int(q['span_1_3'])}日")

    lines.append("  --- 触发条件 ---")
    today = q["confirm_date"] == asof
    row("确认点为当日", today, f"确认点={pd.Timestamp(q['confirm_date']).date()}"
        + ("" if today else "  (非当日,已过触发窗口)"))
    gap_ok = q["confirm_bar"] == p.confirm_bar
    row(f"确认点须{p.confirm_bar}", gap_ok, f"确认点K线={q['confirm_bar']}")
    start_ok = (not require_start_bar) or q["start_bar"] == p.start_bar
    row(f"启动点须{p.start_bar}", start_ok,
        f"启动点K线={q['start_bar']}" + ("" if require_start_bar else "  (已放宽)"))
    no_gap = not bool(feed.gap_flags(daily.index, max_gap_days).iloc[-p.slow:].any())
    row("无长期停牌", no_gap, f"近{p.slow}根K线内无超过{max_gap_days}自然日的缺口")

    selected = today and gap_ok and start_ok and no_gap
    lines.append("  ==> " + ("【入选】 触发黄金眼,次日开盘买入" if selected else "【不入选】"))
    return "\n".join(lines)


def scan(asof: str, codes: list[str], start: str, p: Optional[Params] = None,
         dividend: str = "back", batch: int = 200,
         require_start_bar: bool = True, verbose: bool = True) -> pd.DataFrame:
    """全市场扫描(需 QMT 环境)。asof/start 为 'YYYYMMDD'。"""
    p = p or Params()
    asof_ts = pd.Timestamp(asof)
    picks = []
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        if verbose:
            print(f"  [{i + len(chunk)}/{len(codes)}] 下载并扫描 ...", flush=True)
        try:
            data = feed.fetch_daily(chunk, start, asof, dividend_type=dividend)
        except Exception as exc:                      # 单批失败不影响整体
            print(f"    批次失败跳过:{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        for code, bars in data.items():
            if len(bars) == 0 or bars.index[-1] != asof_ts:
                continue                              # 当日停牌或无数据
            try:
                hit = evaluate(code, bars, p, require_start_bar=require_start_bar)
                if hit:
                    picks.append(hit)
            except Exception as exc:
                print(f"    {code} 扫描失败:{type(exc).__name__}: {exc}", file=sys.stderr)

    if not picks:
        return pd.DataFrame(columns=PICK_COLUMNS)
    return pd.DataFrame(picks).sort_values(["span_1_3", "code"]).reset_index(drop=True)


def attach_tradability(picks: pd.DataFrame, next_open: dict, pre_close: dict,
                       st_codes: Optional[set] = None) -> pd.DataFrame:
    """用次日开盘价标注能否买入(一字涨停买不进)。

    次日开盘价只有到次日才有,所以这是下单前的最后一道过滤,不在 evaluate() 内。
    """
    if not len(picks):
        return picks
    st_codes = st_codes or set()
    out = picks.copy()
    out["next_open"] = out["code"].map(next_open)
    out["tradable"] = [
        feed.tradable_at_open(o, pre_close.get(c, float("nan")), c, c in st_codes)
        if pd.notna(o) else False
        for c, o in zip(out["code"], out["next_open"])
    ]
    return out
