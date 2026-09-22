# -*- coding: utf-8 -*-
"""xtdata(迅投 QMT / miniQMT)接入层。

分两部分:
  · fetch_*  调用 xtdata,只能在装有 QMT 客户端的 Windows 上运行
  · 其余     纯函数,不依赖 xtdata,可离线测试

已核对 xtquant 250807.1.2 的真实签名:
  get_market_data_ex(field_list, stock_list, period='1d', start_time='', end_time='',
                     count=-1, dividend_type='none', fill_data=True)
      -> {stock_code: DataFrame},1d 周期 index 为 '%Y%m%d' 字符串
  download_history_data2(stock_list, period, start_time='', end_time='', callback=None, incrementally=None)
  get_stock_list_in_sector(sector_name) / get_instrument_detail(code)

两个会静默污染结果的默认值必须改:
  1) dividend_type 默认 'none'(不复权):除权日会出现假金叉/假死叉。
     回测与选股一律用 'back'(后复权),历史不 repaint。
     'front'(前复权)每遇新除权就把全部历史重算,过去的均线值会变,
     用于信号生成等于引入未来信息,只适合看图。
  2) fill_data 默认 True:停牌日被前值填充成量为0的假K线。
     这种K线 low==high==close,会被判成「贯穿」,直接污染信号。
     必须置 False,并按 volume<=0 兜底过滤。
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Optional

import numpy as np
import pandas as pd

FIELDS = ["time", "open", "high", "low", "close", "volume", "amount", "preClose"]
PRICE_COLS = ["open", "high", "low", "close"]


# ------------------------------------------------------------------ 板块 / 涨跌停
def board_of(code: str) -> str:
    """按代码判板块。code 形如 '000001.SZ' / '600000.SH' / '430047.BJ'。"""
    num, _, mkt = code.partition(".")
    if mkt == "BJ" or num[:2] in ("43", "83", "87", "88", "92"):
        return "北交所"
    if num[:3] in ("300", "301"):
        return "创业板"
    if num[:3] in ("688", "689"):
        return "科创板"
    return "主板"


def limit_ratio(code: str, is_st: bool = False) -> float:
    """涨跌停比例。ST 主板 5%,ST 创业板/科创板仍为 20%,北交所 30%。"""
    b = board_of(code)
    if b == "北交所":
        return 0.30
    if b in ("创业板", "科创板"):
        return 0.20
    return 0.05 if is_st else 0.10


def limit_prices(pre_close: float, ratio: float) -> tuple[float, float]:
    """涨停价 / 跌停价,四舍五入到分。

    必须用 Decimal:Python 内建 round() 是银行家舍入,且浮点表示会让
    round(3.465, 2) 得到 3.46,与交易所的 3.47 不符,差一分会导致一字板漏判。
    """
    def r2(x: float) -> float:
        return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return r2(pre_close * (1 + ratio)), r2(pre_close * (1 - ratio))


def tradable_at_open(open_px: float, pre_close: float, code: str,
                     is_st: bool = False, eps: float = 1e-6) -> bool:
    """次日开盘能否买入:开盘即涨停(一字板)则买不进。"""
    if not np.isfinite(open_px) or not np.isfinite(pre_close) or pre_close <= 0:
        return False
    up, _ = limit_prices(pre_close, limit_ratio(code, is_st))
    return open_px < up - eps


# ------------------------------------------------------------------------ 清洗
def normalize_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """把 get_market_data_ex 返回的单只 DataFrame 规整成策略输入。

    索引转 DatetimeIndex、剔除停牌与无效K线、去重、升序。
    """
    cols = PRICE_COLS + ["volume", "preClose"]
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=cols)

    df = raw.copy()
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
    df = df[df.index.notna()]

    for c in PRICE_COLS + ["volume", "amount", "preClose"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[[c for c in cols if c in df.columns]]

    prices = df[PRICE_COLS]
    bad = prices.isna().any(axis=1) | (prices <= 0).any(axis=1)
    if "volume" in df.columns:
        bad |= df["volume"].fillna(0) <= 0          # fill_data 残留的假K线
    df = df[~bad]

    df = df[(df["high"] >= df["low"])
            & (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9)
            & (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9)]
    return df[~df.index.duplicated(keep="last")].sort_index()


def gap_flags(index: pd.DatetimeIndex, max_gap_days: int = 30) -> pd.Series:
    """标记距前一根K线超过 max_gap_days 自然日的位置(长期停牌)。

    跨越长期停牌的均线没有意义,这些位置上产生的信号应剔除。
    """
    if len(index) == 0:
        return pd.Series(dtype=bool)
    gaps = pd.Series(index, index=index).diff().dt.days
    return (gaps > max_gap_days).fillna(False)


def is_st_name(name: Optional[str]) -> bool:
    return bool(name) and ("ST" in name.upper() or "退" in name)


# ------------------------------------------------------------- 取数(需 Windows + QMT)
def fetch_daily(codes: Iterable[str], start: str, end: str,
                dividend_type: str = "back", download: bool = True) -> dict:
    """下载并读取日线,返回 {code: 已清洗的 DataFrame}。"""
    from xtquant import xtdata          # 延迟导入:非 Windows 环境 import 即失败

    codes = list(codes)
    if download:
        xtdata.download_history_data2(codes, period="1d", start_time=start, end_time=end)
    raw = xtdata.get_market_data_ex(
        FIELDS, codes, period="1d", start_time=start, end_time=end,
        count=-1, dividend_type=dividend_type, fill_data=False,
    )
    return {c: normalize_bars(v) for c, v in (raw or {}).items()}


def fetch_universe(sector: str = "沪深A股", exclude_st: bool = True,
                   min_listed_days: int = 120, asof: Optional[str] = None) -> list[str]:
    """取板块成分并剔除 ST / 次新 / 已退市。"""
    from xtquant import xtdata

    asof_ts = pd.Timestamp(asof) if asof else pd.Timestamp.today()
    out = []
    for code in xtdata.get_stock_list_in_sector(sector) or []:
        info = xtdata.get_instrument_detail(code)
        if not info:
            continue
        if exclude_st and is_st_name(info.get("InstrumentName")):
            continue
        opened = str(info.get("OpenDate") or "")
        if len(opened) == 8 and (asof_ts - pd.to_datetime(opened, format="%Y%m%d")).days < min_listed_days:
            continue
        expire = str(info.get("ExpireDate") or "")
        if len(expire) == 8 and pd.to_datetime(expire, format="%Y%m%d") <= asof_ts:
            continue
        out.append(code)
    return out


def fetch_next_open(codes: Iterable[str], date: str) -> dict:
    """取指定交易日的开盘价,用于下单前过滤一字涨停。"""
    from xtquant import xtdata

    raw = xtdata.get_market_data_ex(
        ["time", "open"], list(codes), period="1d",
        start_time=date, end_time=date, count=-1,
        dividend_type="back", fill_data=False,
    ) or {}
    out = {}
    for code, v in raw.items():
        if v is not None and len(v):
            out[code] = float(v["open"].iloc[-1])
    return out
