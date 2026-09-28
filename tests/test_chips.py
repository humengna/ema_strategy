# -*- coding: utf-8 -*-
"""筹码分布与获利比例。"""
import numpy as np
import pandas as pd
import pytest

from ema_strategy.chips import profit_ratio

N = 120
IDX = pd.bdate_range("2024-01-01", periods=N)
FLOAT = 1e9


def make(close: np.ndarray, amp: float = 0.01) -> pd.DataFrame:
    return pd.DataFrame({"high": close * (1 + amp), "low": close * (1 - amp),
                         "close": close, "volume": np.full(N, 2e6),
                         "amount": close * 2e6 * 100}, index=IDX)


def test_uptrend_mostly_profitable():
    assert profit_ratio(make(np.linspace(10, 30, N)), FLOAT).iloc[-1] > 0.85


def test_downtrend_mostly_losing():
    assert profit_ratio(make(np.linspace(30, 10, N)), FLOAT).iloc[-1] < 0.15


def test_flat_is_about_half():
    """横盘时成本集中在当前价附近,获利比例应接近 50%。"""
    assert profit_ratio(make(np.full(N, 20.0)), FLOAT).iloc[-1] == pytest.approx(0.5, abs=0.05)


def test_no_lookahead(intc):
    """同一天的获利比例不得因为后面多了K线而改变。

    回归测试:早期实现用整段数据的 min/max 划价格网格,
    截断到当天算得 63.4%、用全历史算得 100% —— 典型的未来函数。
    """
    bars = intc.copy()
    bars["volume"] = 2e6
    bars["amount"] = bars["close"] * 2e6 * 100
    full = profit_ratio(bars, 3.3e9)
    for cut in (200, 600, 1200):
        trunc = profit_ratio(bars.iloc[:cut], 3.3e9)
        assert np.allclose(full.iloc[:cut].to_numpy(), trunc.to_numpy(), equal_nan=True)


def test_bounded_0_1(intc):
    bars = intc.copy()
    bars["volume"] = 2e6
    ratio = profit_ratio(bars, 3.3e9).dropna()
    assert len(ratio) > 0
    assert ratio.between(0.0, 1.0).all()


def test_warmup_is_nan():
    ratio = profit_ratio(make(np.full(N, 20.0)), FLOAT, min_periods=30)
    assert ratio.iloc[:29].isna().all()
    assert ratio.iloc[29:].notna().all()


def test_missing_float_shares_gives_no_decay():
    """流通股本缺失时换手率无从计算,筹码不衰减,结果仍应有界。"""
    ratio = profit_ratio(make(np.linspace(10, 30, N)), float_shares=0)
    assert ratio.dropna().between(0.0, 1.0).all()


def test_requires_columns():
    with pytest.raises(ValueError, match="缺少列"):
        profit_ratio(pd.DataFrame({"close": [1.0]}, index=IDX[:1]), FLOAT)


def test_empty_input():
    empty = pd.DataFrame(columns=["high", "low", "close", "volume"],
                         index=pd.DatetimeIndex([]))
    assert len(profit_ratio(empty, FLOAT)) == 0


def test_limit_up_bar_does_not_crash():
    """一字板:high == low,分布退化为单个价位。"""
    close = np.full(N, 20.0)
    bars = pd.DataFrame({"high": close, "low": close, "close": close,
                         "volume": np.full(N, 2e6), "amount": close * 2e6 * 100}, index=IDX)
    assert profit_ratio(bars, FLOAT).dropna().between(0.0, 1.0).all()


def test_turnover_uses_hand_unit():
    """volume 以「手」计,换手率要乘 100;漏乘会高估获利比例。

    漏乘 -> 换手率小两个数量级 -> 筹码几乎不衰减 -> 早期低成本筹码一直留着
    -> 上涨行情的获利比例被系统性高估。对「获利筹码 > 90%」的阈值筛选,
    这是危险方向:一批本不该入选的票会假装达标。
    """
    bars = make(np.linspace(10, 30, N))
    correct = profit_ratio(bars, FLOAT, volume_unit=100).iloc[-1]
    wrong = profit_ratio(bars, FLOAT, volume_unit=1).iloc[-1]
    assert wrong > correct
    assert wrong > 0.99 and correct < 0.95


def test_optimization_preserves_semantics(intc):
    """优化后的实现必须与朴素实现逐位一致(容许浮点噪声)。

    朴素实现在此就地写出:每天在整条网格上做衰减、用布尔掩码求前缀和。
    这是优化前的语义,任何加速都不得改变结果。
    """
    bars = intc.copy()
    bars["volume"] = 2e6
    bars["amount"] = bars["close"] * 2e6 * 100
    from ema_strategy.chips import _price_grid, _triangle

    def naive(b, float_shares, decay=1.0, bin_pct=0.002,
              grid_span=50.0, volume_unit=100, min_periods=30):
        high = b["high"].to_numpy(float); low = b["low"].to_numpy(float)
        close = b["close"].to_numpy(float); vol = b["volume"].to_numpy(float)
        amt = b["amount"].to_numpy(float)
        edges, centers = _price_grid(float(close[0]), bin_pct, grid_span)
        dist = np.zeros(len(centers)); out = np.full(len(b), np.nan)
        for i in range(len(b)):
            shares = vol[i] * volume_unit
            w = float(np.clip(shares / float_shares * decay, 0.0, 1.0))
            peak = amt[i] / shares if shares > 0 else (high[i] + low[i] + close[i]) / 3
            lo = min(max(low[i], edges[0]), edges[-1])
            hi = min(max(high[i], edges[0]), edges[-1])
            j0 = int(np.clip(np.searchsorted(edges, lo, side="right") - 1, 0, len(centers) - 1))
            j1 = int(np.clip(np.searchsorted(edges, hi, side="right") - 1, 0, len(centers) - 1))
            seg = _triangle(centers[j0:j1 + 1], lo, hi, min(max(peak, lo), hi))
            today = np.zeros(len(centers)); today[j0:j1 + 1] = seg
            dist = today.copy() if dist.sum() <= 0 else dist * (1 - w) + w * today
            if i + 1 >= min_periods and dist.sum() > 0:
                out[i] = float(np.clip(dist[centers <= close[i]].sum() / dist.sum(), 0, 1))
        return pd.Series(out, index=b.index)

    for n in (200, 600):
        sub = bars.iloc[-n:]
        assert np.allclose(naive(sub, 3.3e9).to_numpy(),
                           profit_ratio(sub, 3.3e9).to_numpy(),
                           equal_nan=True, atol=1e-12)


def test_triangle_normalized():
    from ema_strategy.chips import _triangle
    seg = np.array([9.8, 9.9, 10.0, 10.1, 10.2])
    w = _triangle(seg, 9.75, 10.25, 10.0)
    assert w.sum() == pytest.approx(1.0)
    assert w.argmax() == 2                     # 峰值落在 10.0 所在的桶


def test_triangle_single_bin():
    from ema_strategy.chips import _triangle
    assert _triangle(np.array([10.0]), 10.0, 10.0, 10.0).tolist() == [1.0]


# ============================================================ 续算等价性
# 筹码分布是逐日向前滚的状态。把状态存下来续算,一旦有半点对不上
# 就是「看不出错的错值」—— 所以这里用随机序列 x 随机切点做穷举式比对。
def _mk_chip_bars(n, seed, bad_frac=0.0, nan_head=0, zero_vol=False):
    rng = np.random.default_rng(seed)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.03, n))
    vol = rng.integers(0, 2e5, n).astype(float)
    if zero_vol:
        vol[:] = 0.0
    idx = pd.bdate_range("2024-01-01", periods=n)
    b = pd.DataFrame({"high": close * 1.02, "low": close * 0.98, "close": close,
                      "volume": vol, "amount": close * vol * 100}, index=idx)
    if bad_frac:                        # 掺进坏K线(high 缺失)
        k = rng.choice(n, int(n * bad_frac), replace=False)
        b.iloc[k, b.columns.get_loc("high")] = np.nan
    if nan_head:                        # 开头没有有效收盘价,网格锚点靠后
        b.iloc[:nan_head, b.columns.get_loc("close")] = np.nan
    return b


@pytest.mark.parametrize("bad_frac", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("nan_head", [0, 1, 3, 10])
def test_resume_equals_full_recomputation(bad_frac, nan_head):
    """任意切点处分两段续算,结果必须与整段重算逐格相同。"""
    from ema_strategy.chips import profit_ratio_resumable
    rng = np.random.default_rng(hash((bad_frac, nan_head)) % 2**32)
    for seed in range(6):
        n = int(rng.integers(40, 200))
        b = _mk_chip_bars(n, seed, bad_frac, nan_head)
        full = profit_ratio(b, 1e9).to_numpy()
        # 把边界切点都覆盖到:0、1、锚点前后、末尾
        for cut in sorted({0, 1, nan_head, nan_head + 1, n - 1, n,
                           int(rng.integers(0, n + 1))}):
            if not 0 <= cut <= n:
                continue
            s1, state = profit_ratio_resumable(b.iloc[:cut], 1e9)
            s2, _ = profit_ratio_resumable(b.iloc[cut:], 1e9, state=state)
            got = np.concatenate([s1.to_numpy(), s2.to_numpy()])
            assert np.allclose(full, got, equal_nan=True), \
                f"seed={seed} n={n} cut={cut} 续算与整段不一致"


def test_resume_one_day_at_a_time():
    """逐日追加(每天只加一根K线)不能累积误差 —— 这正是日常的用法。"""
    from ema_strategy.chips import profit_ratio_resumable
    b = _mk_chip_bars(120, 3)
    full = profit_ratio(b, 1e9).to_numpy()

    parts, state = [], None
    for i in range(len(b)):
        s, state = profit_ratio_resumable(b.iloc[i:i + 1], 1e9, state=state)
        parts.append(s.to_numpy())
    assert np.allclose(full, np.concatenate(parts), equal_nan=True)


def test_resume_tracks_min_periods_across_segments():
    """min_periods 按全序列位置数。状态里漏记 offset 会让 NaN 段短掉一截。"""
    from ema_strategy.chips import profit_ratio_resumable
    b = _mk_chip_bars(60, 11)
    full = profit_ratio(b, 1e9, min_periods=30).to_numpy()
    s1, state = profit_ratio_resumable(b.iloc[:20], 1e9, min_periods=30)
    s2, _ = profit_ratio_resumable(b.iloc[20:], 1e9, min_periods=30, state=state)
    got = np.concatenate([s1.to_numpy(), s2.to_numpy()])
    assert np.isnan(got[:29]).all()
    assert np.allclose(full, got, equal_nan=True)


def test_state_keeps_only_the_active_range():
    """状态只存活跃桶,不是整条网格 —— 否则 5000 只的检查点会大到没法用。"""
    from ema_strategy.chips import _price_grid, profit_ratio_resumable
    b = _mk_chip_bars(200, 4)
    _, state = profit_ratio_resumable(b, 1e9)
    _, centers = _price_grid(float(b["close"].iloc[0]), 0.002, 50.0)
    assert len(state.dist) == state.hi_i - state.lo_i + 1
    assert len(state.dist) < len(centers) / 2
    assert state.offset == len(b)


def test_bars_without_valid_close_are_excluded():
    """收盘价无效的K线不计入分布,等价于压根没有这几行。

    价格网格锚在首个有效收盘价上,锚点之前的K线没有自洽的位置可放;
    原先它们仍按 high/low 计入分布,导致整段算与分两段续算对不上
    (续算时前一段还没有锚点,那几根就丢了)。
    """
    from ema_strategy.chips import profit_ratio_resumable
    b = _mk_chip_bars(80, 6, nan_head=5)

    # min_periods=0 排除掉位置计数的影响,只比分布本身
    with_head = profit_ratio(b, 1e9, min_periods=0).to_numpy()
    without_head = profit_ratio(b.iloc[5:], 1e9, min_periods=0).to_numpy()
    assert np.isnan(with_head[:5]).all()
    assert np.allclose(with_head[5:], without_head, equal_nan=True)

    # 那几根算「消费过」(offset 要走),但还没锚定网格
    _, state = profit_ratio_resumable(b.iloc[:5], 1e9)
    assert state.anchor is None and state.offset == 5
