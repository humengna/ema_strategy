# -*- coding: utf-8 -*-
"""组合回测:把选股信号变成一条可考核的资金曲线。

与事件统计(scripts/backtest.py)的区别:事件统计只看每笔信号的前瞻收益,
不受资金约束;组合回测要管资金、仓位上限、并发持仓与现金,
得到的年化/回撤/Sharpe 才是这套策略实际能拿到的东西。

交易口径(与事件统计保持一致,便于交叉验证)
    信号日 t 收盘后决策 -> t+1 开盘买入(A股 T+1)
    持有 hold 个交易日 -> 第 hold 日收盘卖出
    次日开盘一字涨停视为买不进,跳过
    成本按往返 cost 计,买卖各承担一半

仓位
    等权,单票目标市值 = 当前总权益 / max_positions。
    因为持有 hold 天,正常状态下每天约有 max_positions/hold 个仓位到期腾出,
    形成滚动建仓。单日新开仓位数受 max_new_per_day 限制。
    信号数超过空位时按 score 降序取前几个(score 由调用方给出)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioParams:
    hold_days: int = 5
    max_positions: int = 20        # 同时最多持仓只数
    max_new_per_day: int = 0       # 单日最多新开;0 表示取 ceil(max_positions/hold_days)
    cost: float = 0.003            # 往返成本,买卖各半
    init_capital: float = 1_000_000.0

    def __post_init__(self):
        # 买入在开盘、卖出在收盘,hold_days=1 等于当日买当日卖 —— A股 T+1 下不可执行。
        if self.hold_days < 2:
            raise ValueError(
                f"hold_days={self.hold_days} 在A股不可执行:买入在次日开盘、卖出在收盘,"
                f"持有1个交易日等于T+0。最小可执行持有期为 2。")

    @property
    def new_per_day(self) -> int:
        if self.max_new_per_day > 0:
            return self.max_new_per_day
        return max(1, int(np.ceil(self.max_positions / max(self.hold_days, 1))))


def _panel(bars_by_code: dict, field: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({c: b[field] for c, b in bars_by_code.items()}).reindex(index)


def simulate(bars_by_code: dict, signals_by_code: dict, p: PortfolioParams,
             scores_by_code: dict | None = None,
             tradable_by_code: dict | None = None) -> dict:
    """逐日模拟。

    bars_by_code     {code: DataFrame(open/close,...)},索引为交易日
    signals_by_code  {code: 逐日布尔 Series},True 表示当日收盘后触发
    scores_by_code   {code: 逐日数值 Series},空位不足时按当日 score 降序挑选
    tradable_by_code {code: 逐日布尔 Series},False 表示次日开盘买不进(一字板)
    """
    if not bars_by_code:
        raise ValueError("bars_by_code 为空")

    index = pd.DatetimeIndex(sorted(set().union(*[b.index for b in bars_by_code.values()])))
    open_px = _panel(bars_by_code, "open", index)
    close_px = _panel(bars_by_code, "close", index)

    sig = pd.DataFrame({c: s for c, s in signals_by_code.items()}).reindex(index).fillna(False)
    sig = sig.astype(bool)
    score = (pd.DataFrame(scores_by_code).reindex(index)
             if scores_by_code else pd.DataFrame(0.0, index=index, columns=sig.columns))
    score = score.reindex(columns=sig.columns).fillna(-np.inf)
    ok = (pd.DataFrame(tradable_by_code).reindex(index).fillna(False).astype(bool)
          if tradable_by_code else pd.DataFrame(True, index=index, columns=sig.columns))
    ok = ok.reindex(columns=sig.columns).fillna(False)

    half = p.cost / 2.0
    cash = p.init_capital
    positions: dict[str, dict] = {}          # code -> {shares, entry_px, exit_i, entry_i}
    trades, equity, holdings, blocked = [], [], [], 0

    # 日内顺序必须是「先开盘买、后收盘卖」。若反过来先卖后买,
    # 等于拿当日 15:00 的卖出款去买当日 9:30 的票,时序倒置;
    # 且同一只票会在卖出当日立刻被重新买入(收盘卖、开盘买)。
    for i, day in enumerate(index):
        # 1) 开盘:按昨日信号买入
        if i > 0:
            todays = sig.iloc[i - 1]
            cands = [c for c in sig.columns if todays.get(c, False)]
            cands = [c for c in cands if c not in positions]
            cands = [c for c in cands if np.isfinite(open_px.iat[i, open_px.columns.get_loc(c)])]
            blocked += sum(1 for c in cands if not ok.iat[i - 1, ok.columns.get_loc(c)])
            cands = [c for c in cands if ok.iat[i - 1, ok.columns.get_loc(c)]]
            cands.sort(key=lambda c: score.iat[i - 1, score.columns.get_loc(c)], reverse=True)

            slots = min(p.max_positions - len(positions), p.new_per_day)
            equity_now = cash + sum(
                pos["shares"] * close_px.iat[i - 1, close_px.columns.get_loc(c)]
                for c, pos in positions.items()
                if np.isfinite(close_px.iat[i - 1, close_px.columns.get_loc(c)]))
            target = equity_now / p.max_positions

            for code in cands[:max(slots, 0)]:
                px = open_px.iat[i, open_px.columns.get_loc(code)]
                notional = min(target, cash / (1 + half))
                if notional <= 0 or px <= 0:
                    continue
                shares = notional / px
                cash -= shares * px * (1 + half)
                positions[code] = {"shares": shares, "entry_px": px,
                                   "entry_i": i, "exit_i": min(i + p.hold_days - 1, len(index) - 1)}

        # 2) 收盘:到期卖出
        for code in [c for c, pos in positions.items() if pos["exit_i"] <= i]:
            pos = positions.pop(code)
            px = close_px.iat[i, close_px.columns.get_loc(code)]
            if not np.isfinite(px):          # 停牌:顺延到下一个有价格的交易日
                pos["exit_i"] = i + 1
                positions[code] = pos
                continue
            cash += pos["shares"] * px * (1 - half)
            trades.append({"code": code, "entry_date": index[pos["entry_i"]],
                           "exit_date": day, "entry_px": pos["entry_px"], "exit_px": px,
                           "ret": px / pos["entry_px"] - 1 - p.cost,
                           # 实际持有的交易日数(含买入日与卖出日)
                           "hold_days": i - pos["entry_i"] + 1})

        # 3) 收盘盯市
        mv = 0.0
        for code, pos in positions.items():
            px = close_px.iat[i, close_px.columns.get_loc(code)]
            if not np.isfinite(px):
                px = pos["entry_px"]
            mv += pos["shares"] * px
        equity.append(cash + mv)
        holdings.append(len(positions))

    curve = pd.Series(equity, index=index, name="equity")
    bench = _panel(bars_by_code, "close", index).pct_change().mean(axis=1).fillna(0.0)
    bench_curve = (1 + bench).cumprod() * p.init_capital

    return {"equity": curve, "benchmark": bench_curve.rename("benchmark"),
            "trades": pd.DataFrame(trades), "holdings": pd.Series(holdings, index=index),
            "blocked": blocked, "params": p}


def metrics(curve: pd.Series, trades: pd.DataFrame, holdings: pd.Series,
            periods_per_year: int = 242) -> dict:
    """资金曲线指标。"""
    if len(curve) < 2:
        return {}
    ret = curve.pct_change().dropna()
    years = len(curve) / periods_per_year
    total = curve.iloc[-1] / curve.iloc[0] - 1
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(periods_per_year)
    dd = (curve / curve.cummax() - 1).min()
    return {
        "总收益": total, "年化收益": cagr, "年化波动": vol,
        "Sharpe": cagr / vol if vol > 0 else np.nan,
        "最大回撤": dd, "Calmar": cagr / abs(dd) if dd < 0 else np.nan,
        "交易笔数": len(trades),
        "单笔胜率": (trades["ret"] > 0).mean() if len(trades) else np.nan,
        "单笔平均": trades["ret"].mean() if len(trades) else np.nan,
        "平均持仓数": holdings.mean(), "最大持仓数": holdings.max(),
        "资金利用率": holdings.mean() / max(1, holdings.max()),
    }
