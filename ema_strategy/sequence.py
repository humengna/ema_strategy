# -*- coding: utf-8 -*-
"""三金叉序列 —— 黄金眼触发规则。

    1号金叉  MA5  上穿 MA10   -> 启动点,当日K线须为「贯穿」
    2号金叉  MA5  上穿 MA30
    3号金叉  MA10 上穿 MA30   -> 确认点,当日K线须为「跳空」

三者依次出现(1 -> 2 -> 3)且两端形态满足 -> 确认点当日触发。

原始规则未定义、本实现补充的约定(均可在 Params 调整):
  · abort_on_dead_cross:等待 2/3 号期间 MA5 下穿 MA10 则序列作废。
    不加这条的话,一个很久以前的1号金叉能和数月后的3号金叉凑成一对。
  · max_span:1号到3号的最大间隔交易日。实测真实间隔中位数 5-6 日、最大 13 日。
  · 期间再次出现1号金叉时,以最新的为准并重置2号。
  · 允许同日发生多个金叉(跳空高开可一次上穿两条均线);
    require_distinct_days=True 可要求三个金叉分属不同交易日。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import GAP, PIERCE, add_ma, classify_bar, cross_down, cross_up, ma_regime

SEQ_COLUMNS = ["start_date", "c2_date", "confirm_date", "start_bar", "confirm_bar",
               "span_1_3", "span_1_2", "span_2_3", "start_regime", "confirm_regime",
               "triggered"]


@dataclass(frozen=True)
class Params:
    fast: int = 5
    mid: int = 10
    slow: int = 30
    ma_kind: str = "sma"
    max_span: int = 60                    # 1号 -> 3号 最大间隔(交易日)
    abort_on_dead_cross: bool = True      # 等待期间 MA5 下穿 MA10 则作废
    require_distinct_days: bool = False   # 是否要求三个金叉不同日
    start_bar: str = PIERCE               # 启动点要求的K线形态
    confirm_bar: str = GAP                # 确认点要求的K线形态

    @property
    def warmup(self) -> int:
        """产生有效信号所需的最少K线数。"""
        return self.slow + 5


def prepare(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """附加均线、K线分类、均线排列。"""
    d = add_ma(df, p.fast, p.mid, p.slow, p.ma_kind)
    d["bar_type"] = classify_bar(d)
    d["regime"] = ma_regime(d)
    return d


def find_sequences(d: pd.DataFrame, p: Params) -> pd.DataFrame:
    """扫出所有「1号 -> 2号 -> 3号」依次完成的金叉序列。"""
    c1 = cross_up(d["ma_f"], d["ma_m"]).to_numpy()   # 1号:MA5 上穿 MA10
    c2 = cross_up(d["ma_f"], d["ma_s"]).to_numpy()   # 2号:MA5 上穿 MA30
    c3 = cross_up(d["ma_m"], d["ma_s"]).to_numpy()   # 3号:MA10 上穿 MA30
    dead = cross_down(d["ma_f"], d["ma_m"]).to_numpy()
    bar = d["bar_type"].to_numpy()
    regime = d["regime"].to_numpy()
    idx = d.index

    rows, i1, i2 = [], None, None
    for i in range(len(d)):
        if i1 is not None:
            if p.abort_on_dead_cross and dead[i] and not c1[i]:
                i1 = i2 = None
            elif i - i1 > p.max_span:
                i1 = i2 = None

        if c1[i]:
            i1, i2 = i, None                       # 以最新的1号金叉为准
        if c2[i] and i1 is not None and i >= i1:
            i2 = i
        if c3[i] and i1 is not None and i2 is not None and i >= i2:
            if not (p.require_distinct_days and not (i1 < i2 < i)):
                rows.append({
                    "start_date": idx[i1], "c2_date": idx[i2], "confirm_date": idx[i],
                    "start_bar": bar[i1], "confirm_bar": bar[i],
                    "span_1_3": i - i1, "span_1_2": i2 - i1, "span_2_3": i - i2,
                    "start_regime": regime[i1], "confirm_regime": regime[i],
                })
            i1 = i2 = None                         # 一条序列只用一次

    seq = pd.DataFrame(rows, columns=[c for c in SEQ_COLUMNS if c != "triggered"])
    seq["triggered"] = ((seq["start_bar"] == p.start_bar)
                        & (seq["confirm_bar"] == p.confirm_bar)) if len(seq) else pd.Series(dtype=bool)
    return seq


def run(df: pd.DataFrame, p: Params | None = None) -> dict:
    """完整扫描。df 需含 open/high/low/close,索引为升序 DatetimeIndex。"""
    p = p or Params()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index 必须是 DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df.index 必须按时间升序")

    d = prepare(df, p)
    seq = find_sequences(d, p)
    hits = set(seq.loc[seq["triggered"], "confirm_date"]) if len(seq) else set()
    d["triggered"] = d.index.isin(hits)
    return {"daily": d, "sequences": seq, "params": p}
