# -*- coding: utf-8 -*-
"""QMT 内置 Python 版必须与 ema_strategy 包算出完全相同的结果。

QMT 环境里没有本包,那份文件是自包含的重写。两边若出现偏差,
QMT 选出的票就和回测结果对不上 —— 这是最容易出、也最难察觉的错。
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from ema_strategy.bull import (BullParams, all_ma_rising, ma_turn_up,
                               pattern_state, volume_surge)
from ema_strategy.bull import run as bull_run
from ema_strategy.chips import profit_ratio
from ema_strategy.sequence import Params as SeqParams
from ema_strategy.sequence import find_sequences, prepare

QMT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "qmt", "golden_eye_qmt.py")


@pytest.fixture(scope="module")
def qmt():
    spec = importlib.util.spec_from_file_location("golden_eye_qmt", QMT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bars(intc_with_volume):
    return intc_with_volume


@pytest.fixture(scope="module")
def daily(qmt, bars):
    d = bars.copy()
    d["ma_f"] = qmt._ma(d["close"], qmt.FAST)
    d["ma_m"] = qmt._ma(d["close"], qmt.MID)
    d["ma_s"] = qmt._ma(d["close"], qmt.SLOW)
    return d


def test_params_match_python_version(qmt):
    """参数必须与放宽后的那版一致,否则两边选出的票不同。"""
    assert (qmt.FAST, qmt.MID, qmt.SLOW) == (5, 10, 30)
    assert qmt.PROFIT_MIN == 0.85
    assert qmt.CHIP_BIN_PCT == 0.002
    assert qmt.VOLUME_UNIT == 100
    assert qmt.MAX_SPAN == SeqParams().max_span


def test_cross_up_matches(qmt, daily):
    from ema_strategy.indicators import cross_up
    for a, b in (("ma_f", "ma_m"), ("ma_f", "ma_s"), ("ma_m", "ma_s")):
        assert (qmt._cross_up(daily[a], daily[b]).to_numpy()
                == cross_up(daily[a], daily[b]).to_numpy()).all()


def test_cross_down_matches(qmt, daily):
    from ema_strategy.indicators import cross_down
    assert (qmt._cross_down(daily["ma_f"], daily["ma_m"]).to_numpy()
            == cross_down(daily["ma_f"], daily["ma_m"]).to_numpy()).all()


def test_sequences_match(qmt, bars, daily):
    """三金叉序列的启动日必须完全一致。"""
    seq = find_sequences(prepare(bars, SeqParams()), SeqParams())
    expected = [daily.index.get_loc(t) for t in seq["confirm_date"]]
    assert qmt._find_sequences(daily) == expected


def test_pattern_active_matches(qmt, bars, daily):
    seq = find_sequences(prepare(bars, SeqParams()), SeqParams())
    ref = pattern_state(prepare(bars, SeqParams()), seq)["active"].to_numpy()
    got = qmt._pattern_active(daily, qmt._find_sequences(daily)).to_numpy()
    assert (got == ref).all()


def test_all_ma_rising_matches(qmt, daily):
    assert (qmt._all_ma_rising(daily).to_numpy()
            == all_ma_rising(daily).to_numpy()).all()


def test_ma_turn_matches(qmt, daily):
    assert (qmt._ma_turn_up(daily).to_numpy() == ma_turn_up(daily).to_numpy()).all()


def test_volume_surge_matches(qmt, bars):
    ref = volume_surge(bars["volume"], 5, 1.5, "prev").to_numpy()
    assert (qmt._volume_surge_prev(bars["volume"]).to_numpy() == ref).all()


def test_price_grid_matches(qmt, bars):
    from ema_strategy.chips import _price_grid
    e1, c1 = qmt._price_grid(float(bars["close"].iloc[0]), 0.002, 50.0)
    e2, c2 = _price_grid(float(bars["close"].iloc[0]), 0.002, 50.0)
    assert np.allclose(e1, e2) and np.allclose(c1, c2)


def test_profit_ratio_matches(qmt, bars):
    """获利筹码是最复杂的一段,必须逐位一致。"""
    ref = profit_ratio(bars, 3.3e9, decay=1.0, bin_pct=0.002).to_numpy()
    got = qmt._profit_ratio(bars, 3.3e9).to_numpy()
    assert np.allclose(got, ref, equal_nan=True, atol=1e-12)


def test_evaluate_one_matches_package(qmt, bars):
    """端到端:逐日比对 QMT 版与包版的入选判定。"""
    rng = np.random.default_rng(0)
    flow = pd.DataFrame({"bidMostAmount": rng.uniform(0, 1e8, len(bars)),
                         "offMostAmount": rng.uniform(0, 1e8, len(bars))},
                        index=bars.index)
    p = BullParams(require_ma_turn=True, volume_mode="prev",
                   profit_min=0.85, require_pullback=False)
    ref = bull_run(bars, 3.3e9, flow, p)["daily"]["triggered"]

    checked = mismatch = 0
    for i in range(60, len(bars), 7):
        sub, subflow = bars.iloc[:i], flow.iloc[:i]
        got = qmt.evaluate_one(sub, 3.3e9, subflow) is not None
        checked += 1
        mismatch += int(got != bool(ref.iloc[i - 1]))
    assert checked > 200
    assert mismatch == 0


def test_no_flow_never_triggers(qmt, bars):
    """资金流缺失时一律不入选,不能当成「有流入」放过。"""
    for i in range(60, len(bars), 53):
        assert qmt.evaluate_one(bars.iloc[:i], 3.3e9, None) is None
        assert qmt.evaluate_one(bars.iloc[:i], 3.3e9, pd.DataFrame()) is None


def test_no_float_shares_never_triggers(qmt, bars):
    rng = np.random.default_rng(0)
    flow = pd.DataFrame({"bidMostAmount": rng.uniform(0, 1e8, len(bars)),
                         "offMostAmount": rng.uniform(0, 1e8, len(bars))},
                        index=bars.index)
    for i in range(60, len(bars), 53):
        assert qmt.evaluate_one(bars.iloc[:i], 0, flow.iloc[:i]) is None


def test_short_history_returns_none(qmt, bars):
    assert qmt.evaluate_one(bars.iloc[:20], 3.3e9, None) is None


# ----------------------------------------------- 最小上下文:防属性缺失
class MinimalContext:
    """只提供确信存在的方法,模拟 QMT 真实的 __PyContext。

    QMT 里的上下文是 C++ 对象,并非 xtquant 的 qmttools.ContextInfo,
    属性集更窄 —— 曾因用了 C.trade_mode 直接抛 AttributeError。
    本类刻意不提供 trade_mode、universe 等,凡多访问一个就会立刻炸出来。
    """

    def __init__(self, bars_by_code, flows=None, details=None, with_last_bar=True):
        self._bars = bars_by_code
        self._flows = flows or {}
        self._details = details or {}
        if with_last_bar:
            self.is_last_bar = lambda: True

    def get_stock_list_in_sector(self, sector):
        return list(self._bars)

    def get_instrument_detail(self, code, iscomplete=False):
        return self._details.get(code)

    def get_market_data_ex(self, fields, stock_code, period='1d', start_time='',
                           end_time='', count=-1, dividend_type='', fill_data=True,
                           subscribe=True):
        src = self._flows if period == 'transactioncount1d' else self._bars
        return {c: src[c] for c in stock_code if c in src}

    def get_bar_timetag(self, barpos=None):
        return 1735689600000            # 2025-01-01

    def __getattr__(self, name):        # 任何未提供的属性都当作不存在
        raise AttributeError(
            f"'MinimalContext' object has no attribute '{name}' —— "
            f"QMT 的 __PyContext 同样可能没有,请改用防御性访问")


@pytest.fixture
def ctx_data(bars):
    rng = np.random.default_rng(0)
    flow = pd.DataFrame({"bidMostAmount": rng.uniform(0, 1e8, len(bars)),
                         "offMostAmount": rng.uniform(0, 1e8, len(bars))},
                        index=bars.index)
    code = "000001.SZ"
    detail = {"InstrumentName": "测试股", "FloatVolume": 3.3e9, "OpenDate": "20100101"}
    return {code: bars}, {code: flow}, {code: detail}


def test_init_and_handlebar_survive_minimal_context(qmt, ctx_data, capsys):
    """init 与 handlebar 只能用最小上下文提供的方法,多碰一个就报错。"""
    b, f, d = ctx_data
    C = MinimalContext(b, f, d)
    qmt.init(C)
    qmt.handlebar(C)
    out = capsys.readouterr().out
    assert "股票池" in out and "入选" in out


def test_handlebar_without_is_last_bar(qmt, ctx_data, capsys):
    """连 is_last_bar 都没有时也不能崩,应按「扫描」处理。"""
    b, f, d = ctx_data
    C = MinimalContext(b, f, d, with_last_bar=False)
    qmt.init(C)
    qmt.handlebar(C)
    assert "入选" in capsys.readouterr().out


def test_no_trade_mode_access(qmt):
    """源码里不得再出现 C.trade_mode / C.universe 的实际访问。"""
    import ast
    src = open(QMT_PATH, encoding="utf-8").read()
    used = {n.attr for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "C"}
    assert "trade_mode" not in used
    assert "universe" not in used


def test_only_known_context_members(qmt):
    """限制 C 上可用的成员,新增前须先确认 QMT 真的提供。"""
    import ast
    allowed = {"get_stock_list_in_sector", "get_instrument_detail",
               "get_market_data_ex", "is_last_bar", "get_bar_timetag", "barpos"}
    src = open(QMT_PATH, encoding="utf-8").read()
    used = {n.attr for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "C"}
    assert used <= allowed, f"用到了未确认的上下文成员: {used - allowed}"
