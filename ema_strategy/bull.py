# -*- coding: utf-8 -*-
"""多头黄金眼(形态维持 + 资金/筹码/量能触发)。

形态启动日
    5日线上穿10日线 -> 5日线上穿30日线 -> 10日线上穿30日线,三者依次出现,
    第三个金叉当天即为形态启动日。

形态破坏
    启动后,MA5 或 MA10 任一跌破 MA30,形态即告破坏,当日起不再维持。

触发条件(形态维持期间,当日须同时满足)
    1. 获利筹码 > 90%          —— 自行计算,见 chips.py
    2. 当日资金流入            —— bidMostAmount - offMostAmount > 0
    3. 放量                    —— 成交量 > 前 N 日均量 * ratio

规则未定义、本实现的约定(均可在 BullParams 调整):
  · 「放量」的口径:默认 成交量 > 前5日均量 * 1.5(不含当日,避免自我参照)
  · 「获利筹码」的算法:换手衰减法,见 chips.py
  · 三金叉的顺序约束与 max_span / abort_on_dead_cross 沿用 sequence.Params
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .chips import profit_ratio
from .sequence import Params as SeqParams
from .sequence import find_sequences, prepare


@dataclass(frozen=True)
class BullParams:
    seq: SeqParams = field(default_factory=SeqParams)
    profit_min: float = 0.90       # 获利筹码下限(严格大于)
    volume_ratio: float = 1.5      # 放量倍数
    volume_window: int = 5         # 均量窗口(不含当日)
    chip_decay: float = 1.0        # 筹码换手衰减系数
    chip_bin_pct: float = 0.002    # 筹码价格网格步长,见 chips.profit_ratio

    @property
    def warmup(self) -> int:
        return max(self.seq.warmup, self.volume_window + 1)


def pattern_state(daily: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
    """逐日标记形态是否维持。

    形态在启动日(三金叉完成当天)开启;自启动日起,任何一天出现
    MA5 < MA30 或 MA10 < MA30 即破坏,该日及之后不再维持,直到下一个启动日。
    """
    broken = ((daily["ma_f"] < daily["ma_s"]) | (daily["ma_m"] < daily["ma_s"])).to_numpy()
    starts = set(sequences["confirm_date"]) if len(sequences) else set()
    index = daily.index

    active = np.zeros(len(daily), dtype=bool)
    pid = np.full(len(daily), -1, dtype=int)
    start_of = np.full(len(daily), np.datetime64("NaT"), dtype="datetime64[ns]")
    days_in = np.zeros(len(daily), dtype=int)

    cur_id, cur_start, on = -1, None, False
    for i, ts in enumerate(index):
        if ts in starts and not on:               # 启动日
            on, cur_id, cur_start = True, cur_id + 1, ts
        if on and broken[i]:                      # 破坏日起不再维持
            on, cur_start = False, None
        if on:
            active[i] = True
            pid[i] = cur_id
            start_of[i] = np.datetime64(cur_start)
            days_in[i] = i - index.get_loc(cur_start) + 1

    return pd.DataFrame({"active": active, "pattern_id": pid,
                         "pattern_start": start_of, "days_in_pattern": days_in},
                        index=index)


def volume_surge(volume: pd.Series, window: int, ratio: float) -> pd.Series:
    """放量:当日成交量 > 前 window 日均量 * ratio。

    均量窗口经 shift(1) 排除当日,否则当日放量会把自己的均值抬高,条件被削弱。
    """
    base = volume.rolling(window).mean().shift(1)
    return (volume > base * ratio) & base.notna()


def net_inflow(flow: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.Series:
    """资金流入 = bidMostAmount - offMostAmount(主买特大单 - 主卖特大单)。

    flow 为 None 或缺列时,整列返回 NaN —— 由调用方决定是拒绝还是跳过该条件,
    绝不能当成 0 或 False 静默放过。
    """
    if flow is None or not len(flow):
        return pd.Series(np.nan, index=index, name="net_inflow")
    cols = {"bidMostAmount", "offMostAmount"}
    if not cols.issubset(flow.columns):
        return pd.Series(np.nan, index=index, name="net_inflow")
    diff = (pd.to_numeric(flow["bidMostAmount"], errors="coerce")
            - pd.to_numeric(flow["offMostAmount"], errors="coerce"))
    return diff.reindex(index).rename("net_inflow")


def run(bars: pd.DataFrame, float_shares: float | pd.Series,
        flow: pd.DataFrame | None = None, p: BullParams | None = None) -> dict:
    """完整扫描,返回逐日明细与触发标记。

    bars 需含 open/high/low/close/volume(amount 可选,用于成交均价)。
    """
    p = p or BullParams()
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars.index 必须是 DatetimeIndex")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars.index 必须按时间升序")

    daily = prepare(bars, p.seq)
    sequences = find_sequences(daily, p.seq)
    state = pattern_state(daily, sequences)

    daily["profit_ratio"] = profit_ratio(bars, float_shares,
                                         decay=p.chip_decay, bin_pct=p.chip_bin_pct)
    daily["net_inflow"] = net_inflow(flow, daily.index)
    daily["vol_surge"] = volume_surge(bars["volume"], p.volume_window, p.volume_ratio)

    daily["cond_profit"] = daily["profit_ratio"] > p.profit_min
    daily["cond_inflow"] = daily["net_inflow"] > 0
    daily["cond_volume"] = daily["vol_surge"]
    daily = pd.concat([daily, state], axis=1)

    daily["triggered"] = (daily["active"] & daily["cond_profit"]
                          & daily["cond_inflow"] & daily["cond_volume"])
    return {"daily": daily, "sequences": sequences, "params": p}
