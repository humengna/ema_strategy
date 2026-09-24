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

import sys
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


def parse_ymd(value) -> Optional[pd.Timestamp]:
    """把 xtdata 的 YYYYMMDD 日期字段解析为 Timestamp,无法解析时返回 None。

    OpenDate / ExpireDate 并不总是合法日期:未退市的合约 ExpireDate 常填
    哨兵值 99999999,也可能是 0、空串或其他长度。这些值用
    pd.to_datetime(..., format="%Y%m%d") 直接解析会抛 ValueError
    ("unconverted data remains: 99"),足以中断整轮全市场扫描。
    一律返回 None,由调用方按「未知」处理。
    """
    text = str(value or "").strip()
    if len(text) != 8 or not text.isdigit():
        return None
    ts = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return None if pd.isna(ts) else ts


# ------------------------------------------------------------- 取数(需 Windows + QMT)
_DETAIL_CACHE: dict = {}


def instrument_detail(code: str) -> Optional[dict]:
    """带缓存的合约信息。

    get_instrument_detail_list 内部只是对 get_instrument_detail 的循环,没有批量优势;
    而股票池筛选与流通股本各要取一遍,全市场就是近万次重复调用。
    合约信息在一次回测内不会变,缓存即可。
    """
    if code in _DETAIL_CACHE:
        return _DETAIL_CACHE[code]
    from xtquant import xtdata
    try:
        info = xtdata.get_instrument_detail(code)
    except Exception:
        info = None
    _DETAIL_CACHE[code] = info
    return info


def clear_detail_cache() -> None:
    _DETAIL_CACHE.clear()


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
                   min_listed_days: int = 120, asof: Optional[str] = None,
                   verbose: bool = True) -> list[str]:
    """取板块成分并剔除 ST / 次新 / 已退市。

    单只合约的信息异常不会中断整轮扫描,只记数并跳过。
    """
    from xtquant import xtdata

    asof_ts = pd.Timestamp(asof) if asof else pd.Timestamp.today()
    out, skipped = [], {"no_detail": 0, "st": 0, "new": 0, "delisted": 0, "error": 0}

    for code in xtdata.get_stock_list_in_sector(sector) or []:
        try:
            info = instrument_detail(code)
            if not info:
                skipped["no_detail"] += 1
                continue
            if exclude_st and is_st_name(info.get("InstrumentName")):
                skipped["st"] += 1
                continue

            opened = parse_ymd(info.get("OpenDate"))
            if opened is not None and (asof_ts - opened).days < min_listed_days:
                skipped["new"] += 1
                continue

            expire = parse_ymd(info.get("ExpireDate"))
            if expire is not None and expire <= asof_ts:
                skipped["delisted"] += 1
                continue

            out.append(code)
        except Exception as exc:                 # 单只异常不应中断整轮扫描
            skipped["error"] += 1
            print(f"  跳过 {code}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if verbose and any(skipped.values()):
        print(f"  股票池筛选: 保留 {len(out)} 只 | 剔除 "
              f"ST {skipped['st']}、次新 {skipped['new']}、退市 {skipped['delisted']}"
              f"、无合约信息 {skipped['no_detail']}、异常 {skipped['error']}")
    return out


# bidMostAmount / offMostAmount 所在的周期,按优先级尝试:
#   transactioncount1d  Level1 逐笔成交统计(日线)—— 投研版特色数据
#   l2transactioncount  Level2 大单统计 —— 需 Level2 行情权限,盘中累计值
# 两者都有门槛,取不到时 fetch_money_flow 返回空,由调用方显式拒绝该条件,
# 绝不能当成「无流入」或「有流入」静默放过。
FLOW_PERIODS = ("transactioncount1d", "l2transactioncount")
FLOW_FIELDS = ["time", "bidMostAmount", "offMostAmount"]


def fetch_money_flow(codes: Iterable[str], start: str, end: str,
                     periods: Iterable[str] = FLOW_PERIODS,
                     verbose: bool = True, download: bool = True) -> dict:
    """取主买/主卖特大单成交额,返回 {code: DataFrame(bidMostAmount, offMostAmount)}。

    逐个周期尝试,第一个取到数据的即采用。全部失败则返回 {}。
    l2transactioncount 是盘中累计值,按日取末值汇总为当日口径。
    """
    from xtquant import xtdata

    codes = list(codes)
    for period in periods:
        if download:
            try:
                xtdata.download_history_data2(codes, period=period,
                                              start_time=start, end_time=end)
            except Exception:
                pass                              # 下载失败仍尝试直接读本地缓存
        try:
            data = xtdata.get_market_data_ex(
                FLOW_FIELDS, codes, period=period, start_time=start,
                end_time=end, count=-1, fill_data=False) or {}
        except Exception as exc:
            if verbose:
                print(f"  资金流周期 {period} 不可用: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
            continue

        out = {}
        for code, df in data.items():
            norm = _normalize_flow(df)
            if norm is not None and len(norm):
                out[code] = norm
        if out:
            if verbose:
                print(f"  资金流数据源: {period}(取到 {len(out)}/{len(codes)} 只)")
            return out

    if verbose:
        print("  未取到资金流数据(transactioncount1d 需投研版,"
              "l2transactioncount 需Level2权限)", file=sys.stderr)
    return {}


def _normalize_flow(df) -> Optional[pd.DataFrame]:
    """把资金流原始数据归整到日频:索引转日期,盘中多条则取当日末值。"""
    if df is None or len(df) == 0:
        return None
    if not {"bidMostAmount", "offMostAmount"}.issubset(df.columns):
        return None

    out = df.copy()
    idx = pd.to_datetime(out.index.astype(str).str.slice(0, 8),
                         format="%Y%m%d", errors="coerce")
    out.index = idx
    out = out[out.index.notna()]
    for col in ("bidMostAmount", "offMostAmount"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[["bidMostAmount", "offMostAmount"]]
    return out.groupby(level=0).last().sort_index()


def fetch_winner_chips(codes: Iterable[str], start: str, end: str,
                       func_name: str = "get_winner_chips", **kwargs) -> dict:
    """取外部计算好的获利筹码比例,返回 {code: Series(0~1, 索引为日期)}。

    xtquant 250807.1.2 里并没有 get_winner_chips —— .py、文档与 .pyd 中均无此名。
    因此这里按名字动态查找:装了带该接口的版本就能直接用,没有则明确报错,
    而不是悄悄退回自算、让两套口径混在一起。

    若该函数的参数或返回结构与此处假设不同,改这一个函数即可,
    上层(bull.run 的 profit_series 参数)不用动。
    """
    from xtquant import xtdata

    func = getattr(xtdata, func_name, None)
    if func is None:
        raise AttributeError(
            f"当前 xtquant 没有 {func_name}。可选做法:\n"
            f"  1) 升级到提供该接口的 xtquant 版本\n"
            f"  2) 自行取得获利比例后,用 bull.run(..., profit_series=...) 注入\n"
            f"  3) 不传,使用内置的换手衰减法(chips.profit_ratio)")

    raw = func(list(codes), start_time=start, end_time=end, **kwargs)
    return {c: _normalize_winner(v) for c, v in (raw or {}).items()
            if _normalize_winner(v) is not None}


def _normalize_winner(value) -> Optional[pd.Series]:
    """把获利比例归一成:DatetimeIndex 的 Series,取值 0~1。

    容忍百分数(>1 视为百分比自动除 100)与 DataFrame/Series 两种返回形态。
    """
    if value is None or len(value) == 0:
        return None
    if isinstance(value, pd.DataFrame):
        col = next((c for c in value.columns
                    if str(c).lower() in ("winner", "winner_chips", "ratio", "value")),
                   value.columns[-1])
        s = value[col]
    else:
        s = pd.Series(value)

    s = pd.to_numeric(s, errors="coerce")
    idx = pd.to_datetime(pd.Index(s.index).astype(str).str.slice(0, 8),
                         format="%Y%m%d", errors="coerce")
    s.index = idx
    s = s[s.index.notna()].sort_index()
    if s.dropna().gt(1.0).any():          # 返回的是百分数
        s = s / 100.0
    return s.clip(0.0, 1.0)


def fetch_float_shares(codes: Iterable[str], verbose: bool = True) -> dict:
    """取流通股本(股),用于换手率与筹码分布。

    只能取到当前值;增发/解禁前后回溯历史换手率会有偏差。
    走 instrument_detail 的缓存,与股票池筛选共用同一次调用。
    """
    out = {}
    for code in codes:
        try:
            info = instrument_detail(code)
            value = float(info.get("FloatVolume") or 0) if info else 0.0
            if value > 0:
                out[code] = value
        except Exception:
            continue
    if verbose and len(out) < len(list(codes)):
        pass
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
