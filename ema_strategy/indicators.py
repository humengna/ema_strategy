# -*- coding: utf-8 -*-
"""均线、金叉、K线与均线的位置关系。

比较口径统一取等号:价格或均线相等时一律归入 ">=" 一侧。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]

# K线与均线的位置关系
GAP = "跳空"          # low >= max(MA5, MA10, MA30)
PIERCE = "贯穿"       # 任一均线落在 [low, high] 内
BELOW = "均线下方"    # high < min(MA5, MA10, MA30)
OTHER = "其他"        # 整根K线夹在两条均线之间,不触碰任何一条

# 均线排列
BULL = "全多头"       # MA5 >= MA10 >= MA30
HALF = "半多头"
BEAR = "空头"         # MA5 < MA10 < MA30


def moving_average(s: pd.Series, n: int, kind: str = "sma") -> pd.Series:
    if kind == "ema":
        return s.ewm(span=n, adjust=False).mean()
    if kind == "sma":
        return s.rolling(n).mean()
    raise ValueError(f"未知均线类型: {kind}")


def add_ma(df: pd.DataFrame, fast: int = 5, mid: int = 10, slow: int = 30,
           kind: str = "sma") -> pd.DataFrame:
    """附加三条均线 ma_f / ma_m / ma_s。"""
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")
    out = df.copy()
    out["ma_f"] = moving_average(out["close"], fast, kind)
    out["ma_m"] = moving_average(out["close"], mid, kind)
    out["ma_s"] = moving_average(out["close"], slow, kind)
    return out


def _crossed(state: pd.Series, a: pd.Series, b: pd.Series) -> pd.Series:
    """由 state 的 False->True 跃迁判定穿越。

    必须同时要求「前一根K线两条均线都已有值」:否则均线预热期结束的第一根K线上,
    state 从 False(NaN 比较结果)跳到 True,会被误判成一次金叉 ——
    那不是穿越,只是均线刚开始有值。若不排除,每只股票都会在 MA30 预热期结束
    那天凭空多出 1/2/3 号金叉,足以凑出一条完整的假序列。
    """
    valid = a.notna() & b.notna()
    return (state & ~state.shift(1, fill_value=False)
            & valid & valid.shift(1, fill_value=False))


def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    """金叉:a 由下方穿到 b 之上。相等算作在上方。"""
    return _crossed(a >= b, a, b)


def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    """死叉:a 由上方穿到 b 之下。"""
    return _crossed(a < b, a, b)


def classify_bar(df: pd.DataFrame) -> pd.Series:
    """判定每根K线与三条均线的位置关系。

    跳空与贯穿在 low == max(MA) 的边界上同时成立,按「相等归入 >=」判为跳空。
    """
    mas = df[["ma_f", "ma_m", "ma_s"]]
    gap = df["low"] >= mas.max(axis=1)
    pierce = (mas.ge(df["low"], axis=0) & mas.le(df["high"], axis=0)).any(axis=1)
    below = df["high"] < mas.min(axis=1)

    out = pd.Series(OTHER, index=df.index, dtype=object)
    out[pierce] = PIERCE
    out[gap] = GAP                       # 跳空优先
    out[below & ~gap & ~pierce] = BELOW
    out[mas.isna().any(axis=1)] = np.nan
    return out


def ma_regime(df: pd.DataFrame) -> pd.Series:
    """均线多空排列。三线相等时归入全多头。"""
    f, m, s = df["ma_f"], df["ma_m"], df["ma_s"]
    out = pd.Series(HALF, index=df.index, dtype=object)
    out[(f < m) & (m < s)] = BEAR
    out[(f >= m) & (m >= s)] = BULL      # 全多头优先
    out[f.isna() | m.isna() | s.isna()] = np.nan
    return out
