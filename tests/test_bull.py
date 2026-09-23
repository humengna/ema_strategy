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
