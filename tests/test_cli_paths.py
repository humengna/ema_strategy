# -*- coding: utf-8 -*-
"""跑通 CLI 的取数路径(用假的 feed 代替 QMT)。

这条路径是实际使用时走的那条。之前只用 --csv-dir 测过,
导致 `--holds` 重构后残留的 `a.hold` 没被发现,直到真机上才报 AttributeError。
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import backtest  # noqa: E402
import backtest_portfolio  # noqa: E402

CODES = [f"00000{i}.SZ" for i in range(1, 7)]


def _bars(seed: int) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=400)
    rng = np.random.default_rng(seed)
    close = 20 * np.cumprod(1 + rng.normal(0.0008, 0.02, len(idx)))
    vol = rng.uniform(5e4, 5e5, len(idx))
    return pd.DataFrame({
        "open": close * 0.997, "high": close * 1.02, "low": close * 0.98,
        "close": close, "volume": vol, "amount": close * vol * 100,
        "preClose": np.r_[close[0], close[:-1]],
    }, index=idx)


@pytest.fixture
def fake_feed(monkeypatch):
    """把 feed 的取数函数换成本地构造的数据,不接触 QMT。"""
    data = {c: _bars(i) for i, c in enumerate(CODES)}

    def fetch_daily(codes, start, end, dividend_type="back", download=True):
        return {c: data[c] for c in codes if c in data}

    def fetch_universe(sector="沪深A股", **kw):
        return list(CODES)

    def fetch_money_flow(codes, start, end, **kw):
        out = {}
        for i, c in enumerate(codes):
            idx = data[c].index
            rng = np.random.default_rng(100 + i)
            out[c] = pd.DataFrame({"bidMostAmount": rng.uniform(0, 1e8, len(idx)),
                                   "offMostAmount": rng.uniform(0, 1e8, len(idx))}, index=idx)
        return out

    def fetch_float_shares(codes, **kw):
        return {c: 5e8 for c in codes}

    for mod in (backtest, backtest_portfolio):
        monkeypatch.setattr(mod.feed, "fetch_daily", fetch_daily)
        monkeypatch.setattr(mod.feed, "fetch_universe", fetch_universe)
        monkeypatch.setattr(mod.feed, "fetch_money_flow", fetch_money_flow)
        monkeypatch.setattr(mod.feed, "fetch_float_shares", fetch_float_shares)
    return data


@pytest.mark.parametrize("strategy", ["bull", "cross"])
def test_backtest_universe_path_runs(fake_feed, tmp_path, strategy):
    out = tmp_path / "ev.csv"
    rc = backtest.main(["--strategy", strategy, "--sector", "沪深A股",
                        "--holds", "1,3,5,7,10,15,30", "--out", str(out)])
    assert rc in (0, 1)          # 1 = 没有信号,也算正常跑完


def test_backtest_codes_path_runs(fake_feed, tmp_path):
    out = tmp_path / "ev.csv"
    rc = backtest.main(["--strategy", "bull", "--codes", ",".join(CODES),
                        "--holds", "3,5", "--out", str(out)])
    assert rc in (0, 1)


def test_backtest_limit_applies(fake_feed, tmp_path):
    out = tmp_path / "ev.csv"
    rc = backtest.main(["--strategy", "cross", "--sector", "沪深A股",
                        "--limit", "2", "--holds", "5", "--out", str(out)])
    assert rc in (0, 1)


def test_backtest_rejects_empty_holds(fake_feed, tmp_path):
    rc = backtest.main(["--strategy", "cross", "--holds", ",", "--out", str(tmp_path / "x.csv")])
    assert rc == 2


def test_portfolio_universe_path_runs(fake_feed, tmp_path):
    out = tmp_path / "pf.csv"
    rc = backtest_portfolio.main(["--strategy", "bull", "--sector", "沪深A股",
                                  "--hold", "5", "--max-positions", "5",
                                  "--out", str(out)])
    assert rc in (0, 1)


def test_portfolio_rejects_t_plus_zero(fake_feed, tmp_path):
    rc = backtest_portfolio.main(["--strategy", "cross", "--hold", "1",
                                  "--out", str(tmp_path / "pf.csv")])
    assert rc == 2


def test_portfolio_rebalance_mode_runs(fake_feed, tmp_path):
    """月度调仓模式的取数路径。"""
    out = tmp_path / "rb.csv"
    rc = backtest_portfolio.main(["--strategy", "cross", "--mode", "rebalance",
                                  "--freq-days", "21", "--lookback", "21",
                                  "--max-positions", "5", "--sector", "沪深A股",
                                  "--out", str(out)])
    assert rc in (0, 1)


def test_portfolio_rebalance_rejects_short_freq(fake_feed, tmp_path):
    rc = backtest_portfolio.main(["--strategy", "cross", "--mode", "rebalance",
                                  "--freq-days", "1", "--out", str(tmp_path / "rb.csv")])
    assert rc == 2
