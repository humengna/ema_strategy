# -*- coding: utf-8 -*-
"""QMT 内置 Python 版必须与 ema_strategy 包算出完全相同的结果。

QMT 环境里没有本包,那份文件是自包含的重写。两边若出现偏差,
QMT 选出的票就和回测结果对不上 —— 这是最容易出、也最难察觉的错。
"""
import ast
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
    """按 GBK 读入后执行。

    文件首行声明 #coding:gbk 且实际以 GBK 存储(官方文档硬性要求),
    Python 默认按 UTF-8 读源码会直接报 SyntaxError,故显式解码后 exec。
    """
    src = open(QMT_PATH, encoding="gbk").read()
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("golden_eye_qmt", loader=None))
    exec(compile(src, QMT_PATH, "exec"), mod.__dict__)
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

    def __init__(self, bars_by_code, flows=None, details=None, with_last_bar=True,
                 flow_period="transactioncount1d"):
        self._bars = bars_by_code
        self._flows = flows or {}
        self._details = details or {}
        # 只有这个周期能取到资金流,用来测 FLOW_PERIODS 的顺序与兜底
        self._flow_period = flow_period
        self.flow_periods_tried = []
        if with_last_bar:
            self.is_last_bar = lambda: True

    def get_stock_list_in_sector(self, sector):
        return list(self._bars)

    def get_instrument_detail(self, code, iscomplete=False):
        return self._details.get(code)

    def get_market_data_ex(self, fields, stock_code, period='1d', start_time='',
                           end_time='', count=-1, dividend_type='', fill_data=True,
                           subscribe=True):
        if period in ('transactioncount1d', 'transactioncount1m',
                      'l2transactioncount'):
            self.flow_periods_tried.append(period)
            if period != self._flow_period:
                return {}
            src = self._flows
        else:
            src = self._bars
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
    src = open(QMT_PATH, encoding="gbk").read()
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
    src = open(QMT_PATH, encoding="gbk").read()
    used = {n.attr for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "C"}
    assert used <= allowed, f"用到了未确认的上下文成员: {used - allowed}"


# ------------------------------------------- 回测提速:缓存不得改变结果
def test_evaluate_series_last_row_matches_evaluate_one(qmt, bars):
    """整段计算的最后一行,必须等于只喂到那天的单日判定。"""
    rng = np.random.default_rng(0)
    flow = pd.DataFrame({"bidMostAmount": rng.uniform(0, 1e8, len(bars)),
                         "offMostAmount": rng.uniform(0, 1e8, len(bars))},
                        index=bars.index)
    checked = mismatch = 0
    for i in range(60, len(bars), 29):
        sub, subflow = bars.iloc[:i], flow.iloc[:i]
        series_hit = bool(qmt.evaluate_series(sub, 3.3e9, subflow)["triggered"].iloc[-1])
        one_hit = qmt.evaluate_one(sub, 3.3e9, subflow) is not None
        checked += 1
        mismatch += int(series_hit != one_hit)
    assert checked > 50 and mismatch == 0


def test_series_has_no_lookahead(qmt, bars):
    """整段一次算出的第 i 日取值,必须等于只喂到第 i 日算出的取值。

    这是「整段只算一次」得以成立的前提 —— 若不成立,缓存就是在用未来数据。
    """
    rng = np.random.default_rng(0)
    flow = pd.DataFrame({"bidMostAmount": rng.uniform(0, 1e8, len(bars)),
                         "offMostAmount": rng.uniform(0, 1e8, len(bars))},
                        index=bars.index)
    full = qmt.evaluate_series(bars, 3.3e9, flow)["triggered"].to_numpy()
    for cut in (400, 900, 1500):
        trunc = qmt.evaluate_series(bars.iloc[:cut], 3.3e9,
                                    flow.iloc[:cut])["triggered"].to_numpy()
        assert (full[:cut] == trunc).all(), f"截断到 {cut} 后结果改变"


def test_cached_backtest_equals_per_bar_rescan(qmt, ctx_data, monkeypatch, capsys):
    """缓存版(整段扫一次)与逐根重算,选出的票必须完全一致。

    提速的底线:不能改变结果。
    """
    b, f, d = ctx_data
    code = next(iter(b))
    bars = b[code]

    # 逐根重算:对每个交易日只喂到当天,调 evaluate_one
    expected = {}
    for i in range(60, len(bars)):
        sub, subflow = bars.iloc[:i], f[code].iloc[:i]
        hit = qmt.evaluate_one(sub, 3.3e9, subflow)
        if hit:
            day = pd.Timestamp(bars.index[i - 1]).strftime("%Y-%m-%d")
            expected[day] = code

    # 缓存版:整段扫一次
    C = MinimalContext(b, f, d)
    qmt.init(C)
    got_map = qmt._scan_all(C, [code])
    got = {day: rows[0]["code"] for day, rows in got_map.items()}

    # 缓存版覆盖完整历史,逐根版从第60根起,故只比对交集之外的差异
    common_days = {pd.Timestamp(bars.index[i - 1]).strftime("%Y-%m-%d")
                   for i in range(60, len(bars))}
    assert {k: v for k, v in got.items() if k in common_days} == expected


def test_scan_cache_resets_on_init(qmt, ctx_data):
    """重跑时必须清掉缓存,否则会沿用上一轮结果。"""
    b, f, d = ctx_data
    C = MinimalContext(b, f, d)
    qmt.init(C)
    qmt._PICKS_BY_DAY = {"2020-01-01": [{"code": "X"}]}
    qmt._SCANNED = True
    qmt.init(C)
    assert qmt._SCANNED is False
    assert qmt._PICKS_BY_DAY == {}


def test_float_shares_cached_in_init(qmt, ctx_data):
    """init 应顺手存下流通股本,扫描时不再重复调 get_instrument_detail。

    全市场近 5000 只,重复调用就是近万次。
    """
    b, f, d = ctx_data
    calls = {"n": 0}

    class CountingContext(MinimalContext):
        def get_instrument_detail(self, code, iscomplete=False):
            calls["n"] += 1
            return self._details.get(code)

    C = CountingContext(b, f, d)
    qmt.init(C)
    after_init = calls["n"]
    assert after_init == len(b)              # init 每只票一次

    qmt._scan_all(C, list(b), verbose=False)
    assert calls["n"] == after_init          # 扫描阶段不再调用
    assert qmt._FLOAT_SHARES                 # 且确实缓存下来了


def test_universe_excludes_missing_float_shares(qmt, bars):
    """取不到流通股本的票直接剔出股票池,而不是扫描时才发现。"""
    code = "999999.SZ"
    b = {code: bars}
    d = {code: {"InstrumentName": "无股本", "OpenDate": "20100101"}}   # 无 FloatVolume
    C = MinimalContext(b, {}, d)
    qmt.init(C)
    assert code not in qmt._UNIVERSE


# -------------------------------------------- 与官方文档对齐的检查
def test_coding_declaration_is_gbk():
    """官方《快速开始》要求首行 #coding:gbk,且脚本本身必须是 GBK 编码。

    两者必须一致 —— 声明 gbk 却存成 UTF-8,QMT 会直接报解码错。
    """
    raw = open(QMT_PATH, "rb").read()
    assert raw.split(b"\n", 1)[0].strip() == b"#coding:gbk"
    raw.decode("gbk")                       # 能按 GBK 解出来才算数
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")                 # 且确实不是 UTF-8


def test_market_data_uses_subscribe_false():
    """官方文档:回测取本地数据应指定 subscribe=False。

    订阅模式还有股票数量上限,全市场扫描必须关掉。
    """
    src = open(QMT_PATH, encoding="gbk").read()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "get_market_data_ex"]
    assert calls, "未找到 get_market_data_ex 调用"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "subscribe" in kw, "get_market_data_ex 必须显式传 subscribe"
        assert kw["subscribe"].value is False


def test_flow_periods_prefer_transactioncount1d(qmt):
    """官方数据字典里内置 get_market_data_ex 可以传 transactioncount1d。

    它本来就是日级、历史长,且不需要 Level2 权限,所以排在
    l2transactioncount(盘中累计值、需 Level2)之前。
    """
    assert qmt.FLOW_PERIODS[0] == "transactioncount1d"
    assert "l2transactioncount" in qmt.FLOW_PERIODS


def test_fetch_flows_falls_back_to_l2(qmt, ctx_data):
    """首选周期取不到时要退到 l2transactioncount,而不是当成没有资金流。"""
    b, f, d = ctx_data
    C = MinimalContext(b, f, d, flow_period="l2transactioncount")
    qmt._FLOW_PERIOD = None
    flows = qmt._fetch_flows(C, list(b))
    assert set(flows) == set(b)
    assert C.flow_periods_tried == ["transactioncount1d", "l2transactioncount"]
    assert qmt._FLOW_PERIOD == "l2transactioncount"


def test_fetch_flows_pins_period_after_probe(qmt, ctx_data):
    """探到可用周期后就固定下来,后续批次不再把另一个周期重试一遍。"""
    b, f, d = ctx_data
    C = MinimalContext(b, f, d, flow_period="l2transactioncount")
    qmt._FLOW_PERIOD = None
    qmt._fetch_flows(C, list(b))
    C.flow_periods_tried = []
    qmt._fetch_flows(C, list(b))
    assert C.flow_periods_tried == ["l2transactioncount"]


def test_fetch_flows_empty_when_no_period_works(qmt, ctx_data):
    """两个周期都取不到时返回空,由上层保证「缺数据不入选」。"""
    b, f, d = ctx_data
    C = MinimalContext(b, f, d, flow_period="__none__")
    qmt._FLOW_PERIOD = None
    assert qmt._fetch_flows(C, list(b)) == {}
    assert C.flow_periods_tried == list(qmt.FLOW_PERIODS)


def test_init_resets_probed_flow_period(qmt, ctx_data):
    """重跑时必须重新探测周期,不能沿用上一轮(可能换了行情权限/数据源)。"""
    b, f, d = ctx_data
    qmt._FLOW_PERIOD = "l2transactioncount"
    qmt.init(MinimalContext(b, f, d))
    assert qmt._FLOW_PERIOD in (None, "transactioncount1d")


def test_normalize_flow_takes_last_per_day(qmt):
    """L2 大单统计是盘中累计值,同日多条应取末值。"""
    raw = pd.DataFrame({"bidMostAmount": [1.0, 5.0, 2.0, 9.0],
                        "offMostAmount": [1.0, 2.0, 1.0, 4.0]},
                       index=["20240102093000", "20240102150000",
                              "20240103093000", "20240103150000"])
    out = qmt._normalize_flow(raw)
    assert out["bidMostAmount"].tolist() == [5.0, 9.0]
    assert out["offMostAmount"].tolist() == [2.0, 4.0]


def test_normalize_flow_rejects_bad_input(qmt):
    assert qmt._normalize_flow(None) is None
    assert qmt._normalize_flow(pd.DataFrame()) is None
    assert qmt._normalize_flow(pd.DataFrame({"x": [1]}, index=["20240102"])) is None


def test_python36_compatible_syntax():
    """QMT 内置 Python 为 3.6,不得使用 3.7+ 语法(如 dataclasses、海象运算符)。"""
    src = open(QMT_PATH, encoding="gbk").read()
    assert "dataclass" not in src
    assert ":=" not in src
    assert "from __future__" not in src
