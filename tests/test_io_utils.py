# -*- coding: utf-8 -*-
"""落盘容错:目标被占用时不得丢掉计算结果。"""
import os

import pandas as pd

from ema_strategy.io_utils import safe_to_csv

DF = pd.DataFrame({"a": [1, 2], "b": [3, 4]})


def test_writes_normally(tmp_path):
    path = str(tmp_path / "out.csv")
    assert safe_to_csv(DF, path) == path
    assert pd.read_csv(path).equals(DF)


def test_falls_back_when_path_unwritable(tmp_path, capsys):
    """目标不可写时换名重试,且不抛异常。"""
    bad = str(tmp_path / "nodir" / "sub" / "out.csv")   # 父目录不存在
    got = safe_to_csv(DF, bad)
    assert got is None                                   # 备用路径同样在缺失目录下
    assert "失败" in capsys.readouterr().err


def test_fallback_path_used(tmp_path, monkeypatch):
    """第一次写失败、备用路径可写时,返回备用路径。"""
    path = str(tmp_path / "out.csv")
    calls = {"n": 0}
    real = pd.DataFrame.to_csv

    def flaky(self, target=None, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("locked by Excel")
        return real(self, target, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", flaky)
    got = safe_to_csv(DF, path)
    assert got is not None and got != path and os.path.exists(got)


def test_never_raises(tmp_path, monkeypatch):
    def always_fail(self, *args, **kwargs):
        raise PermissionError("locked")
    monkeypatch.setattr(pd.DataFrame, "to_csv", always_fail)
    assert safe_to_csv(DF, str(tmp_path / "out.csv")) is None
