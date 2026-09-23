# -*- coding: utf-8 -*-
"""黄金眼三金叉选股策略(数据源:xtdata / 迅投 QMT)。"""
from .indicators import BELOW, BULL, GAP, HALF, OTHER, PIERCE, add_ma, classify_bar, ma_regime
from .bull import BullParams
from .bull import run as bull_run
from .bull_picker import evaluate as bull_evaluate
from .bull_picker import explain as bull_explain
from .bull_picker import scan as bull_scan
from .chips import profit_ratio
from .picker import attach_tradability, evaluate, explain, scan
from .sequence import Params, find_sequences, prepare, run

__version__ = "0.1.0"
__all__ = [
    "Params", "run", "prepare", "find_sequences",
    "BullParams", "bull_run", "bull_evaluate", "bull_explain", "bull_scan", "profit_ratio",
    "evaluate", "explain", "scan", "attach_tradability",
    "add_ma", "classify_bar", "ma_regime",
    "GAP", "PIERCE", "BELOW", "OTHER", "BULL", "HALF",
]
