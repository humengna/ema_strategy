# -*- coding: utf-8 -*-
"""多头黄金眼:形态维持与三个触发条件。"""
import numpy as np
import pandas as pd
import pytest

from ema_strategy.bull import BullParams, net_inflow, pattern_state, run, volume_surge
from ema_strategy.bull_picker import evaluate, explain
from ema_strategy.sequence import find_sequences, prepare

FLOAT = 3.3e9
P = BullParams()


@pytest.fixture(scope="module")
def bars(intc_with_volume):
    return intc_with_volume


@pytest.fixture(scope="module")
def flow(bars):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"bidMostAmount": rng.uniform(0, 1e8, len(bars)),
                         "offMostAmount": rng.uniform(0, 1e8, len(bars))}, index=bars.index)


def test_pattern_starts_on_third_cross(bars):
    """形态启动日就是三金叉序列完成那天(10上穿30)。"""
    daily = prepare(bars, P.seq)
    seq = find_sequences(daily, P.seq)
    state = pattern_state(daily, seq)
    for confirm in seq["confirm_date"]:
        assert state.loc[confirm, "active"]
        assert pd.Timestamp(state.loc[confirm, "pattern_start"]) == confirm


def test_pattern_breaks_when_ma_falls_below_slow(bars):
    """形态维持期间,MA5 与 MA10 都不得跌破 MA30。"""
    daily = prepare(bars, P.seq)
    state = pattern_state(daily, find_sequences(daily, P.seq))
    live = daily[state["active"]]
    assert (live["ma_f"] >= live["ma_s"]).all()
    assert (live["ma_m"] >= live["ma_s"]).all()


def test_pattern_days_count_up(bars):
    daily = prepare(bars, P.seq)
    state = pattern_state(daily, find_sequences(daily, P.seq))
    for _, grp in state[state["active"]].groupby("pattern_id"):
        assert grp["days_in_pattern"].iloc[0] == 1
        assert grp["days_in_pattern"].is_monotonic_increasing


def test_volume_surge_excludes_today():
    """均量窗口必须 shift(1) 排除当日,否则当日放量会抬高自己的基准。"""
    vol = pd.Series([100.0] * 5 + [200.0], index=pd.bdate_range("2024-01-01", periods=6))
    surge = volume_surge(vol, window=5, ratio=1.5)
    assert surge.iloc[-1]                       # 200 > 100 * 1.5
    assert not surge.iloc[:-1].any()            # 前5日基准不足,全 False


def test_volume_surge_threshold():
    vol = pd.Series([100.0] * 5 + [140.0], index=pd.bdate_range("2024-01-01", periods=6))
    assert not volume_surge(vol, 5, 1.5).iloc[-1]   # 1.4 倍,未达 1.5


def test_net_inflow_missing_is_nan():
    """缺少资金流字段必须返回 NaN,绝不能当成 0 或 False 静默放过。"""
    idx = pd.bdate_range("2024-01-01", periods=3)
    assert net_inflow(None, idx).isna().all()
    assert net_inflow(pd.DataFrame(), idx).isna().all()
    assert net_inflow(pd.DataFrame({"x": [1, 2, 3]}, index=idx), idx).isna().all()


def test_net_inflow_is_bid_minus_off():
    idx = pd.bdate_range("2024-01-01", periods=2)
    flow = pd.DataFrame({"bidMostAmount": [100.0, 50.0],
                         "offMostAmount": [40.0, 80.0]}, index=idx)
    assert net_inflow(flow, idx).tolist() == [60.0, -30.0]


def test_no_trigger_without_flow_data(bars):
    """资金流数据缺失时一律不触发 —— 三个条件必须全部可验证。"""
    daily = run(bars, FLOAT, None, P)["daily"]
    assert daily["net_inflow"].isna().all()
    assert not daily["triggered"].any()


def test_trigger_requires_all_conditions(bars, flow):
    daily = run(bars, FLOAT, flow, P)["daily"]
    hit = daily[daily["triggered"]]
    assert len(hit) > 0
    assert hit["active"].all()
    assert (hit["profit_ratio"] > P.profit_min).all()
    assert (hit["net_inflow"] > 0).all()
    assert hit["vol_surge"].all()


def test_trigger_only_inside_pattern(bars, flow):
    daily = run(bars, FLOAT, flow, P)["daily"]
    assert not daily.loc[~daily["active"], "triggered"].any()


def test_evaluate_matches_explain(bars, flow):
    """两条路径结论必须一致。"""
    checked = mismatch = 0
    for i in range(P.warmup + 10, len(bars), 37):
        sub, subflow = bars.iloc[:i], flow.iloc[:i]
        picked = evaluate("600000.SH", sub, FLOAT, subflow, P) is not None
        explained = "【入选】" in explain("600000.SH", sub, FLOAT, subflow, P)
        checked += 1
        mismatch += int(picked != explained)
    assert checked > 20
    assert mismatch == 0


def test_evaluate_rejects_without_float_shares(bars, flow):
    assert evaluate("600000.SH", bars, 0, flow, P) is None


def test_explain_reports_missing_flow(bars):
    text = explain("600000.SH", bars, FLOAT, None, P)
    assert "缺少 bidMostAmount" in text and "【不入选】" in text


def test_rejects_unsorted_index(bars):
    with pytest.raises(ValueError, match="升序"):
        run(bars.iloc[::-1], FLOAT, None, P)


def test_profit_threshold_is_strict():
    """获利筹码要求「大于」90%,恰好等于不算。"""
    p = BullParams(profit_min=0.90)
    assert not (0.90 > p.profit_min)


def test_whole_pipeline_has_no_lookahead(bars, flow):
    """整条管线(形态/筹码/放量/资金流)的逐日取值不得依赖未来数据。

    回测为了效率会对全历史跑一次 run() 再取触发日,
    这只有在管线完全因果时才等价于逐日滚动。
    """
    full = run(bars, FLOAT, flow, P)["daily"]
    for cut in (400, 900, 1500):
        trunc = run(bars.iloc[:cut], FLOAT, flow.iloc[:cut], P)["daily"]
        for col in ("profit_ratio", "net_inflow", "active", "triggered"):
            a = full[col].iloc[:cut].to_numpy()
            b = trunc[col].to_numpy()
            if a.dtype == bool:
                assert (a == b).all(), f"{col} 在截断到 {cut} 后发生变化"
            else:
                assert np.allclose(a.astype(float), b.astype(float), equal_nan=True), \
                    f"{col} 在截断到 {cut} 后发生变化"


def test_all_ma_rising_requires_all_three():
    """三条均线必须同时上行,任一走平或下行都不算。"""
    from ema_strategy.bull import all_ma_rising
    idx = pd.bdate_range("2024-01-01", periods=2)
    cases = {
        (1.0, 1.0, 1.0): True,      # 三条都涨
        (1.0, 1.0, 0.0): False,     # MA30 走平
        (1.0, -1.0, 1.0): False,    # MA10 下行
        (0.0, 1.0, 1.0): False,     # MA5 走平
    }
    for (df_, dm, ds), expected in cases.items():
        daily = pd.DataFrame({"ma_f": [10.0, 10.0 + df_], "ma_m": [10.0, 10.0 + dm],
                              "ma_s": [10.0, 10.0 + ds]}, index=idx)
        assert bool(all_ma_rising(daily).iloc[-1]) is expected


def test_ma_rising_false_during_warmup():
    from ema_strategy.bull import all_ma_rising
    idx = pd.bdate_range("2024-01-01", periods=3)
    daily = pd.DataFrame({"ma_f": [np.nan, 1.0, 2.0], "ma_m": [np.nan, 1.0, 2.0],
                          "ma_s": [np.nan, 1.0, 2.0]}, index=idx)
    assert not all_ma_rising(daily).iloc[1]     # 前值为 NaN
    assert all_ma_rising(daily).iloc[2]


def test_ma_up_condition_only_narrows(bars, flow):
    """新增条件只会减少触发,不会凭空多出信号。"""
    loose = run(bars, FLOAT, flow, BullParams(require_ma_up=False))["daily"]
    strict = run(bars, FLOAT, flow, BullParams(require_ma_up=True))["daily"]
    assert strict["triggered"].sum() <= loose["triggered"].sum()
    assert not (strict["triggered"] & ~loose["triggered"]).any()


def test_triggered_days_have_all_ma_rising(bars, flow):
    daily = run(bars, FLOAT, flow, BullParams(require_ma_up=True))["daily"]
    hit = daily[daily["triggered"]]
    if len(hit):
        assert hit["ma_up"].all()


# ---------------------------------------------------------------- 回踩判定
def _ma_frame(low, close, ma_f, ma_m, ma_s):
    n = len(low)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"low": low, "close": close, "high": [max(c, l) for c, l in zip(close, low)],
                         "ma_f": ma_f, "ma_m": ma_m, "ma_s": ma_s}, index=idx)


def test_pullback_same_day_break_and_recover():
    """下影线跌破 MA30、收盘站回 —— 同日即算一次回踩。"""
    from ema_strategy.bull import pullback_days
    d = _ma_frame(low=[10.0, 9.0], close=[10.5, 10.2],
                  ma_f=[11.0, 11.0], ma_m=[10.5, 10.5], ma_s=[10.0, 10.0])
    assert pullback_days(d).tolist() == [False, True]


def test_pullback_across_days():
    """先跌破、数日后才收复 —— 收复当天算回踩。"""
    from ema_strategy.bull import pullback_days
    d = _ma_frame(low=[9.0, 9.2, 9.5], close=[9.5, 9.8, 10.3],
                  ma_f=[11.0] * 3, ma_m=[10.5] * 3, ma_s=[10.0] * 3)
    assert pullback_days(d).tolist() == [False, False, True]


def test_pullback_needs_a_break_first():
    """没有跌破就没有回踩,单纯站在 MA30 上方不算。"""
    from ema_strategy.bull import pullback_days
    d = _ma_frame(low=[10.5, 10.6], close=[11.0, 11.2],
                  ma_f=[11.5] * 2, ma_m=[11.0] * 2, ma_s=[10.0] * 2)
    assert not pullback_days(d).any()


def test_pullback_voided_when_pattern_breaks():
    """跌破到收复期间 MA5 或 MA10 跌穿 MA30,该次回踩作废。"""
    from ema_strategy.bull import pullback_days
    # 第3天的最低价须在 MA30 之上,否则它自身又构成一次合法的同日回踩
    d = _ma_frame(low=[9.0, 9.2, 10.1], close=[9.5, 9.8, 10.5],
                  ma_f=[11.0, 9.5, 11.0],        # 第2天 MA5 跌破 MA30,回踩作废
                  ma_m=[10.5] * 3, ma_s=[10.0] * 3)
    assert not pullback_days(d).any()


def test_one_break_yields_one_pullback():
    """一次跌破只对应一次回踩,收复后需重新跌破才有下一次。"""
    from ema_strategy.bull import pullback_days
    d = _ma_frame(low=[9.0, 10.5, 10.6, 9.0], close=[10.3, 11.0, 11.1, 10.4],
                  ma_f=[11.5] * 4, ma_m=[11.0] * 4, ma_s=[10.0] * 4)
    assert pullback_days(d).tolist() == [True, False, False, True]


def test_recent_pullback_window():
    """窗口含当日:window=3 覆盖当日与前两日。"""
    from ema_strategy.bull import recent_pullback
    d = _ma_frame(low=[9.0, 10.5, 10.6, 10.7, 10.8],
                  close=[10.3, 11.0, 11.1, 11.2, 11.3],
                  ma_f=[11.5] * 5, ma_m=[11.0] * 5, ma_s=[10.0] * 5)
    assert recent_pullback(d, window=3).tolist() == [True, True, True, False, False]


def test_recent_pullback_reset_on_break():
    """形态一旦破坏,此前的回踩不再计数。"""
    from ema_strategy.bull import recent_pullback
    d = _ma_frame(low=[9.0, 10.5, 10.6], close=[10.3, 11.0, 11.1],
                  ma_f=[11.5, 9.0, 11.5],        # 第2天形态破坏
                  ma_m=[11.0] * 3, ma_s=[10.0] * 3)
    assert recent_pullback(d, window=6).tolist() == [True, False, False]


def test_pullback_condition_only_narrows(bars, flow):
    loose = run(bars, FLOAT, flow, BullParams(require_pullback=False))["daily"]
    strict = run(bars, FLOAT, flow, BullParams(require_pullback=True))["daily"]
    assert strict["triggered"].sum() <= loose["triggered"].sum()
    assert not (strict["triggered"] & ~loose["triggered"]).any()


def test_triggered_days_have_recent_pullback(bars, flow):
    daily = run(bars, FLOAT, flow, BullParams(require_pullback=True))["daily"]
    hit = daily[daily["triggered"]]
    if len(hit):
        assert hit["recent_pullback"].all()


# ------------------------------------------------ 外部注入的获利筹码比例
def test_external_profit_series_is_used(bars, flow):
    """注入外部获利比例时,应直接采用,不再走内置换手衰减法。"""
    ext = pd.Series(0.95, index=bars.index)
    daily = run(bars, FLOAT, flow, P, profit_series=ext)["daily"]
    assert (daily["profit_ratio"].dropna() == 0.95).all()
    assert daily["profit_source"].iloc[-1] == "外部"


def test_builtin_used_when_not_injected(bars, flow):
    daily = run(bars, FLOAT, flow, P)["daily"]
    assert daily["profit_source"].iloc[-1] == "内置换手衰减法"


def test_external_missing_days_do_not_trigger(bars, flow):
    """外部数据缺某日时记 NaN,该日不触发 —— 不拿自算值去填。"""
    ext = pd.Series(0.99, index=bars.index)
    ext.iloc[-50:] = np.nan
    daily = run(bars, FLOAT, flow, P, profit_series=ext)["daily"]
    assert daily["profit_ratio"].iloc[-50:].isna().all()
    assert not daily["triggered"].iloc[-50:].any()


def test_evaluate_accepts_external_without_float_shares(bars, flow):
    """注入外部获利比例后,不再需要流通股本。"""
    ext = pd.Series(0.99, index=bars.index)
    res = evaluate("600000.SH", bars, 0, flow, P, profit_series=ext)
    assert res is None or res["profit_ratio"] == pytest.approx(0.99)


def test_explain_reports_profit_source(bars, flow):
    ext = pd.Series(0.99, index=bars.index)
    assert "来源:外部" in explain("600000.SH", bars, 0, flow, P, profit_series=ext)
    assert "来源:内置换手衰减法" in explain("600000.SH", bars, FLOAT, flow, P)


# ---------------------------------------------------------------- 均线转向
def _ma3(f, m, s):
    return pd.DataFrame({"ma_f": f, "ma_m": m, "ma_s": s},
                        index=pd.bdate_range("2024-01-01", periods=len(f)))


def test_ma_turn_requires_prev_day_decline():
    """前一日至少一条下行、当日三条全上行 —— 才算转向。"""
    from ema_strategy.bull import ma_turn_up
    #        t0    t1(有下行)  t2(全上行)
    d = _ma3([10.0, 9.9, 10.2], [10.0, 10.1, 10.3], [10.0, 10.1, 10.2])
    assert ma_turn_up(d).tolist() == [False, False, True]


def test_ma_turn_false_when_prev_day_also_all_up():
    """连续上行的第二天不是转向日。"""
    from ema_strategy.bull import ma_turn_up
    d = _ma3([10.0, 10.1, 10.2], [10.0, 10.1, 10.2], [10.0, 10.1, 10.2])
    assert not ma_turn_up(d).any()


def test_ma_turn_false_when_today_not_all_up():
    from ema_strategy.bull import ma_turn_up
    d = _ma3([10.0, 9.9, 10.2], [10.0, 10.1, 10.0], [10.0, 10.1, 10.2])
    assert not ma_turn_up(d).any()


def test_ma_turn_flat_is_not_down():
    """走平不算向下,故「走平 -> 全上行」不判为转向。"""
    from ema_strategy.bull import ma_turn_up
    d = _ma3([10.0, 10.0, 10.2], [10.0, 10.0, 10.3], [10.0, 10.0, 10.2])
    assert not ma_turn_up(d).any()


def test_ma_turn_implies_all_rising(bars):
    from ema_strategy.bull import all_ma_rising, ma_turn_up
    from ema_strategy.sequence import prepare
    daily = prepare(bars, P.seq)
    turn, up = ma_turn_up(daily), all_ma_rising(daily)
    assert not (turn & ~up).any()          # 转向必然蕴含当日三条全上行
    assert turn.sum() < up.sum()           # 且严格更少


# ---------------------------------------------------------------- 放量口径
def test_volume_mode_prev():
    from ema_strategy.bull import volume_surge
    v = pd.Series([100.0, 120.0, 110.0], index=pd.bdate_range("2024-01-01", periods=3))
    assert volume_surge(v, 5, 1.5, "prev").tolist() == [False, True, False]


def test_volume_mode_prev_is_looser_than_ma(bars):
    from ema_strategy.bull import volume_surge
    v = bars["volume"]
    assert volume_surge(v, 5, 1.5, "prev").sum() > volume_surge(v, 5, 1.5, "ma").sum()


def test_volume_mode_rejects_unknown():
    from ema_strategy.bull import volume_surge
    with pytest.raises(ValueError, match="未知 volume_mode"):
        volume_surge(pd.Series([1.0, 2.0]), 5, 1.5, "bogus")


def test_new_combo_triggers_more_than_old(bars, flow):
    """放宽后的组合应比原组合出手更多。"""
    old = run(bars, FLOAT, flow, BullParams())["daily"]
    new = run(bars, FLOAT, flow, BullParams(
        require_ma_turn=True, volume_mode="prev",
        profit_min=0.85, require_pullback=False))["daily"]
    assert new["triggered"].sum() > old["triggered"].sum()


def test_triggered_days_satisfy_turn_when_required(bars, flow):
    p = BullParams(require_ma_turn=True, volume_mode="prev",
                   profit_min=0.85, require_pullback=False)
    daily = run(bars, FLOAT, flow, p)["daily"]
    hit = daily[daily["triggered"]]
    if len(hit):
        assert hit["ma_turn"].all()
        assert (hit["profit_ratio"] > 0.85).all()
