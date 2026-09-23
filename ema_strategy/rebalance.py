# -*- coding: utf-8 -*-
"""定期调仓组合:每隔固定交易日整体换仓一次,期间不动。

与 portfolio.py 的区别:
    portfolio.py  每只票各自持有 N 天到期卖出,建仓时点分散(滚动)
    rebalance.py  全组合在调仓日一次性换掉,期间不调整(月度调仓常见做法)

调仓日做三件事(都在开盘):
    1. 卖出全部现有持仓
    2. 在回看窗口内触发过的股票中,按 score 降序取前 max_positions 只
    3. 等权买入

口径
    买卖都在调仓日开盘价撮合。持仓期内不做任何调整,不止盈不止损。
    成本按往返 cost 计,买卖各承担一半。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import metrics


@dataclass(frozen=True)
class RebalanceParams:
    freq_days: int = 21          # 调仓间隔(交易日)。月度约 21,半月约 10
    lookback: int = 21           # 信号回看窗口(交易日):调仓日往前多少天内触发过即候选
    max_positions: int = 20      # 每期最多持有只数
    cost: float = 0.003          # 往返成本
    init_capital: float = 1_000_000.0

    def __post_init__(self):
        if self.freq_days < 2:
            raise ValueError(f"freq_days={self.freq_days} 过短:买卖同在开盘,"
                             f"间隔不足2个交易日无意义")
        if self.lookback < 1:
            raise ValueError("lookback 至少为 1")


def _panel(bars_by_code: dict, field: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({c: b[field] for c, b in bars_by_code.items()}).reindex(index)


def simulate(bars_by_code: dict, signals_by_code: dict, p: RebalanceParams,
             scores_by_code: dict | None = None) -> dict:
    """逐日模拟定期调仓组合。"""
    if not bars_by_code:
        raise ValueError("bars_by_code 为空")

    index = pd.DatetimeIndex(sorted(set().union(*[b.index for b in bars_by_code.values()])))
    open_px = _panel(bars_by_code, "open", index)
    close_px = _panel(bars_by_code, "close", index)

    sig = pd.DataFrame(signals_by_code).reindex(index).fillna(False).astype(bool)
    score = (pd.DataFrame(scores_by_code).reindex(index)
             if scores_by_code else pd.DataFrame(0.0, index=index, columns=sig.columns))
    score = score.reindex(columns=sig.columns).fillna(-np.inf)

    half = p.cost / 2.0
    cash = p.init_capital
    positions: dict[str, float] = {}          # code -> shares
    equity, holdings, turnover_log, picks_log = [], [], [], []
    # 第一个调仓日至少要让回看窗口填满
    rebal = set(range(p.lookback, len(index), p.freq_days))

    for i, day in enumerate(index):
        if i in rebal:
            # 1) 开盘卖出全部
            for code, shares in list(positions.items()):
                px = open_px.iat[i, open_px.columns.get_loc(code)]
                if np.isfinite(px):
                    cash += shares * px * (1 - half)
                    del positions[code]
                # 停牌卖不掉的继续持有

            # 2) 回看窗口内触发过的股票
            lo = max(0, i - p.lookback)
            hit = sig.iloc[lo:i]
            cands = [c for c in sig.columns if hit[c].any()]
            cands = [c for c in cands
                     if np.isfinite(open_px.iat[i, open_px.columns.get_loc(c)])
                     and c not in positions]
            cands.sort(key=lambda c: float(score.iloc[lo:i][c].max()), reverse=True)
            chosen = cands[:p.max_positions]

            # 3) 等权买入
            if chosen:
                equity_now = cash + sum(
                    s * close_px.iat[i - 1, close_px.columns.get_loc(c)]
                    for c, s in positions.items()
                    if np.isfinite(close_px.iat[i - 1, close_px.columns.get_loc(c)]))
                target = equity_now / len(chosen)
                for code in chosen:
                    px = open_px.iat[i, open_px.columns.get_loc(code)]
                    notional = min(target, cash / (1 + half))
                    if notional <= 0 or px <= 0:
                        continue
                    shares = notional / px
                    cash -= shares * px * (1 + half)
                    positions[code] = positions.get(code, 0.0) + shares
            turnover_log.append(day)
            picks_log.append({"date": day, "n": len(chosen),
                              "codes": ",".join(sorted(chosen))})

        mv = 0.0
        for code, shares in positions.items():
            px = close_px.iat[i, close_px.columns.get_loc(code)]
            if np.isfinite(px):
                mv += shares * px
        equity.append(cash + mv)
        holdings.append(len(positions))

    curve = pd.Series(equity, index=index, name="equity")
    bench_ret = _panel(bars_by_code, "close", index).pct_change().mean(axis=1).fillna(0.0)
    bench = ((1 + bench_ret).cumprod() * p.init_capital).rename("benchmark")

    return {"equity": curve, "benchmark": bench,
            "holdings": pd.Series(holdings, index=index),
            "rebalance_dates": turnover_log,
            "picks": pd.DataFrame(picks_log), "params": p}


def period_returns(curve: pd.Series, rebalance_dates: list) -> pd.Series:
    """各调仓期的区间收益,用于算期胜率与更稳的夏普。"""
    if len(rebalance_dates) < 2:
        return pd.Series(dtype=float)
    marks = pd.DatetimeIndex(rebalance_dates)
    vals = curve.reindex(marks).dropna()
    return vals.pct_change().dropna()


def report_metrics(res: dict, periods_per_year: int = 242) -> dict:
    """组合指标 + 调仓期维度的补充指标。"""
    curve = res["equity"]
    m = metrics(curve, pd.DataFrame(columns=["ret"]), res["holdings"], periods_per_year)
    pr = period_returns(curve, res["rebalance_dates"])
    p = res["params"]
    if len(pr):
        per_year = periods_per_year / p.freq_days
        m["调仓期数"] = len(pr)
        m["期胜率"] = (pr > 0).mean()
        m["期平均收益"] = pr.mean()
        vol_p = pr.std() * np.sqrt(per_year)
        m["夏普(按期)"] = (m["年化收益"] / vol_p) if vol_p > 0 else np.nan
    return m
