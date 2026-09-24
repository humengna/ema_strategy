# -*- coding: utf-8 -*-
"""筹码分布与获利比例。

xtdata 不提供「获利筹码」字段,必须自行计算。这里用通行的换手衰减法:

    每个交易日,已有筹码按当日换手率衰减,腾出的比例由当日成交价格分布补上。
        w = clip(换手率 * decay, 0, 1)
        dist = dist * (1 - w) + w * 当日价格分布
    获利比例 = 成本低于当日收盘价的筹码 / 全部筹码

当日价格分布默认取三角分布:在 [low, high] 区间内以成交均价(amount/volume)
为峰值。均价缺失时退回 (high + low + close) / 3。

两个必须知道的口径问题:
  · xtdata 的 volume 单位是「手」(1手=100股),算换手率要乘 100。
    漏乘会让换手率小两个数量级,筹码几乎不衰减,早期低成本筹码一直留在分布里,
    上涨行情的获利比例被系统性高估(实测 0.895 -> 0.998)。
    对「获利筹码 > 90%」这类阈值筛选是危险方向:一批本不该入选的票会假装达标。
  · 流通股本只能取到当前值(get_instrument_detail 的 FloatVolume)。
    用它回溯历史换手率,在增发/解禁前后会偏差。历史流通股本若可得,
    应逐日传入 float_shares(Series)。

价格网格必须与未来数据无关:早期实现用整段数据的 min/max 划网格,
同一天的获利比例会因为后面多了几根K线而改变(截断到当天 63.4%,
用全历史 100%),那是彻头彻尾的未来函数。现改为以首个有效收盘价
为锚点的等比网格 —— 网格位置只由起点决定,后续数据只会落进已有的桶。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VOLUME_UNIT = 100          # xtdata 的 volume 以「手」计,1手 = 100股


def _triangle(seg: np.ndarray, low: float, high: float, peak: float) -> np.ndarray:
    """在给定的桶中心上铺一个三角形分布(峰值在 peak),并归一化。

    只作用于当日价格涉及的那十几个桶,不碰整条网格。
    """
    if len(seg) == 1:
        return np.ones(1)
    if peak < low:
        peak = low
    elif peak > high:
        peak = high
    left_w = peak - low if peak - low > 1e-12 else 1e-12
    right_w = high - peak if high - peak > 1e-12 else 1e-12

    vals = np.where(seg <= peak, (seg - low) / left_w, (high - seg) / right_w)
    np.clip(vals, 0.0, None, out=vals)
    total = vals.sum()
    return vals / total if total > 0 else np.ones(len(seg)) / len(seg)


def _price_grid(ref: float, bin_pct: float, span: float) -> tuple[np.ndarray, np.ndarray]:
    """以 ref 为锚点的等比价格网格,覆盖 [ref/span, ref*span]。

    只依赖 ref(首个有效收盘价),与后续价格无关,因此不存在未来函数。
    落在网格外的价格由调用方夹到首/末桶。
    """
    lo, hi = ref / span, ref * span
    n = int(np.ceil(np.log(hi / lo) / np.log1p(bin_pct)))
    edges = lo * np.power(1.0 + bin_pct, np.arange(n + 1, dtype=float))
    centers = np.sqrt(edges[:-1] * edges[1:])           # 等比区间的几何中点
    return edges, centers


def profit_ratio(bars: pd.DataFrame, float_shares: float | pd.Series,
                 decay: float = 1.0, bin_pct: float = 0.002,
                 grid_span: float = 50.0,
                 volume_unit: int = VOLUME_UNIT,
                 min_periods: int = 30) -> pd.Series:
    """逐日计算获利筹码比例(0~1)。

    bars        需含 high/low/close/volume,有 amount 时用于求成交均价
    float_shares 流通股本(股)。标量表示全程不变;Series 则逐日取值
    decay       换手衰减系数,1.0 表示换手多少就换掉多少筹码
    bin_pct     价格网格的等比步长(0.002 = 每桶 0.2%)。
                默认值经收敛性检验:横盘行情的理论获利比例应为 0.50,
                bin_pct=0.01/0.005 分别得 0.348/0.323(桶太粗,日内区间只覆盖几个桶),
                0.002 起稳定在 0.50。不要调大。
    grid_span   网格覆盖 [首日收盘/grid_span, 首日收盘*grid_span]
    min_periods 前 N 日筹码分布尚未稳定,返回 NaN
    """
    need = {"high", "low", "close", "volume"}
    missing = need - set(bars.columns)
    if missing:
        raise ValueError(f"缺少列: {sorted(missing)}")
    if len(bars) == 0:
        return pd.Series(dtype=float, index=bars.index)

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float)
    amount = (bars["amount"].to_numpy(dtype=float) if "amount" in bars.columns
              else np.full(len(bars), np.nan))

    if isinstance(float_shares, pd.Series):
        floats = float_shares.reindex(bars.index).to_numpy(dtype=float)
    else:
        floats = np.full(len(bars), float(float_shares) if float_shares else np.nan)

    valid_close = close[np.isfinite(close) & (close > 0)]
    if valid_close.size == 0:
        return pd.Series(np.nan, index=bars.index)
    edges, centers = _price_grid(float(valid_close[0]), bin_pct, grid_span)
    n_bins = len(centers)

    # --- 逐日索引与权重一次性向量化 ---
    # 循环内的标量 np.clip / np.searchsorted 是此前的主要开销:
    # numpy 标量运算要走通用机制(含 getlimits),比 Python 内建慢约两个数量级。
    lo_c = np.clip(low, edges[0], edges[-1])
    hi_c = np.clip(high, edges[0], edges[-1])
    i0_all = np.clip(np.searchsorted(edges, lo_c, side="right") - 1, 0, n_bins - 1)
    i1_all = np.clip(np.searchsorted(edges, hi_c, side="right") - 1, 0, n_bins - 1)
    k_all = np.searchsorted(centers, close, side="right")

    shares = volume * volume_unit
    with np.errstate(divide="ignore", invalid="ignore"):
        turnover = np.where(np.isfinite(floats) & (floats > 0), shares / floats, np.nan)
        w_all = np.clip(turnover * decay, 0.0, 1.0)
        peak_all = np.where(np.isfinite(amount) & (shares > 0),
                            amount / np.where(shares > 0, shares, 1.0),
                            (high + low + close) / 3.0)
    w_all = np.where(np.isfinite(w_all), w_all, 0.0)
    peak_all = np.where(np.isfinite(peak_all), peak_all, (high + low + close) / 3.0)
    bad = ~np.isfinite(low) | ~np.isfinite(high) | (high < low)

    dist = np.zeros(n_bins)
    out = np.full(len(bars), np.nan)
    lo_i, hi_i = n_bins, -1          # 已被触及的桶区间

    for i in range(len(bars)):
        if bad[i]:
            continue
        j0, j1 = int(i0_all[i]), int(i1_all[i])
        if j1 < j0:
            j0, j1 = j1, j0
        seg = _triangle(centers[j0:j1 + 1], lo_c[i], hi_c[i], peak_all[i])
        w = float(w_all[i])

        if hi_i < lo_i:                        # 首日:直接以当日分布起步
            dist[j0:j1 + 1] = seg
            lo_i, hi_i = j0, j1
        else:
            if w:
                dist[lo_i:hi_i + 1] *= (1.0 - w)
            dist[j0:j1 + 1] += w * seg
            if j0 < lo_i:
                lo_i = j0
            if j1 > hi_i:
                hi_i = j1

        if i + 1 >= min_periods:
            k = int(k_all[i])
            if k < lo_i:
                k = lo_i
            elif k > hi_i + 1:
                k = hi_i + 1
            total = dist[lo_i:hi_i + 1].sum()
            if total > 0:
                ratio = dist[lo_i:k].sum() / total
                out[i] = 0.0 if ratio < 0.0 else (1.0 if ratio > 1.0 else float(ratio))

    return pd.Series(out, index=bars.index, name="profit_ratio")
