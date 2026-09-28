# -*- coding: utf-8 -*-
"""scripts/check_one.py:单只股票的逐日入选诊断。"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ema_strategy.bull import BullParams, run          # noqa: E402
from ema_strategy.chips import profit_ratio            # noqa: E402


def _load_script():
    path = os.path.join(ROOT, "scripts", "check_one.py")
    spec = importlib.util.spec_from_file_location("check_one", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check_one = _load_script()
CSV = os.path.join(ROOT, "tests", "intc.csv")


@pytest.fixture
def flow_csv(tmp_path):
    """给本地CSV配一份资金流,否则资金流条件恒不通过,永远选不出东西。"""
    d = pd.read_csv(CSV, parse_dates=["Date"])[["Date"]]
    rng = np.random.default_rng(7)
    d["bidMostAmount"] = rng.uniform(0, 1e8, len(d))
    d["offMostAmount"] = rng.uniform(0, 1e8, len(d))
    path = tmp_path / "flow.csv"
    d.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------- 显示宽度
def test_pad_counts_cjk_as_two_columns():
    """汉字占两格。按字符数对齐会让整张表歪掉,这是表格可读的前提。"""
    assert check_one._pad("日期", 6) == "  日期"      # 显示宽度 4,补 2 格
    assert check_one._pad("ab", 6) == "    ab"        # 显示宽度 2,补 4 格
    assert check_one._pad("第1天", 8, right=False) == "第1天   "


def test_pad_never_truncates_when_too_narrow():
    assert check_one._pad("获利筹码", 2) == "获利筹码"


# ---------------------------------------------------------------- CSV 读入
def test_load_csv_amount_is_price_times_shares():
    """volume 已是股数时 amount 不能再乘 100。

    多乘 100 会让均价 amount/股数 变成收盘价的 100 倍,被夹到当日最高价,
    筹码三角分布的峰值就永远贴着上沿。
    """
    d = check_one.load_csv(CSV)
    assert np.allclose(d["amount"], d["close"] * d["volume"])


def test_load_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Date": ["2020-01-01"], "Close": [1.0]}).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        check_one.load_csv(str(path))


def test_load_flow_csv_rejects_missing_columns(tmp_path):
    """缺列必须报错。静默返回空会让资金流条件恒不通过,查起来要命。"""
    path = tmp_path / "bad_flow.csv"
    pd.DataFrame({"Date": ["2020-01-01"], "bidMostAmount": [1.0]}).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        check_one.load_flow_csv(str(path))


# ---------------------------------------------------------------- 端到端
def test_range_mode_runs_and_reports(capsys, flow_csv):
    rc = check_one.main(["INTC", "--csv", CSV, "--flow-csv", flow_csv,
                         "--from", "20030401", "--to", "20030630"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2003-05-02" in out and "【入选】" in out
    assert "入选 1 天" in out


def test_single_day_agrees_with_range(capsys, flow_csv):
    """同一天,单日模式(截断到当日)与区间模式(整段切片)结论必须一致。"""
    rc = check_one.main(["INTC", "20030502", "--csv", CSV, "--flow-csv", flow_csv])
    out = capsys.readouterr().out
    assert rc == 0 and "【入选】 触发多头黄金眼" in out

    rc = check_one.main(["INTC", "20030505", "--csv", CSV, "--flow-csv", flow_csv])
    out = capsys.readouterr().out
    assert rc == 0 and "【不入选】" in out


def test_verify_finds_no_lookahead(capsys, flow_csv):
    rc = check_one.main(["INTC", "--csv", CSV, "--flow-csv", flow_csv,
                         "--from", "20030401", "--to", "20030630", "--verify"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "结论全部一致" in out


def test_missing_flow_blocks_every_day(capsys):
    """不给资金流时任何一天都不该入选 —— 缺数据不能当成「有流入」。"""
    rc = check_one.main(["INTC", "--csv", CSV,
                         "--from", "20030401", "--to", "20030630"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "入选 0 天" in out
    assert "该条件恒不通过" in out


def test_requires_a_date():
    assert check_one.main(["INTC", "--csv", CSV]) == 2


def test_csv_defaults_volume_unit_to_one(capsys, flow_csv):
    """--csv 默认 volume_unit=1。

    沿用 A 股的 100 会把换手率放大两个数量级,筹码每天被整段冲掉,
    获利比例逐日乱跳(实测同一段里 96.9% 与 6.4% 相邻),阈值筛选彻底失真。
    """
    bars = check_one.load_csv(CSV).loc[:"2003-06-30"]
    unit1 = profit_ratio(bars, 1e9, volume_unit=1)
    unit100 = profit_ratio(bars, 1e9, volume_unit=100)
    span = slice("2003-06-02", "2003-06-30")
    assert unit1[span].diff().abs().max() < unit100[span].diff().abs().max()

    check_one.main(["INTC", "--csv", CSV, "--flow-csv", flow_csv,
                    "--from", "20030602", "--to", "20030630"])
    out = capsys.readouterr().out
    # 表里的获利筹码应与 volume_unit=1 的计算对得上
    assert "79.6%" in out


# ---------------------------------------------------------------- 参数透传
def test_bull_params_threads_volume_unit_to_chips():
    """BullParams.chip_volume_unit 必须真的传到 profit_ratio,不能只是个摆设。"""
    bars = check_one.load_csv(CSV).loc[:"2003-06-30"]
    got = run(bars, 1e9, None, BullParams(chip_volume_unit=1))["daily"]["profit_ratio"]
    assert np.allclose(got.to_numpy(), profit_ratio(bars, 1e9, volume_unit=1).to_numpy(),
                       equal_nan=True)


def test_bull_params_default_volume_unit_is_a_share():
    """默认仍是 A 股口径 100 —— 改默认值会静默改变全市场扫描的结果。"""
    assert BullParams().chip_volume_unit == 100
