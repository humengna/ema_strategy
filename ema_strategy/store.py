# -*- coding: utf-8 -*-
"""本地 Parquet 缓存 —— 把「每次回测都重新问 QMT 要数据」变成「读一次文件」。

全市场 4992 只跑一遍,实测取数占 59%(行情 42s + 资金流/股本 265s),
计算占 41%(218s)。取数那一半每次都在重复做同样的事,应该落盘。

## 存储布局

    <root>/
      bars/2024.parquet      长表 [code, date, open, high, low, close, volume, amount]
      bars/2025.parquet
      bars/_meta.json        复权方式、字段、写入时间
      flow/2024.parquet      长表 [code, date, bidMostAmount, offMostAmount]
      floats.parquet         [code, float_shares, updated]
      profit/...             获利筹码缓存,见 ProfitCache

**长表 + 按年分文件**,不是一只股票一个文件。5000 个小文件在 Windows 上
光是开关文件句柄就要好几秒,而单个长表又会让「补一天数据」变成重写整个文件。
按年切开兼顾两头:平时只重写当年那一个文件。

## 为什么后复权可以安全增量追加

后复权把复权因子累乘到**未来**,历史价格不随新的分红除权变化
(前复权正相反 —— 每次除权都会重刷整段历史,那既是未来函数也没法增量)。
所以 dividend_type='back' 的缓存追加新的一天是安全的,
而 'front' / 'none' 会在 _meta.json 里被记下来,换了口径就整段作废重取。

即便如此,缓存与行情源之间仍可能因为数据修订而漂移,
所以配了 verify():抽样重取并逐格比对,不一致就报出来。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "preClose"]
FLOW_COLUMNS = ["bidMostAmount", "offMostAmount"]

# 各数据集的值列。key 即目录名。
DATASETS = {"bars": BAR_COLUMNS, "flow": FLOW_COLUMNS}


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            raise SystemExit(
                "本地缓存需要 Parquet 引擎,请先装一个:\n"
                "    pip install pyarrow\n"
                "(或 pip install fastparquet)")


def _years(start: Optional[str], end: Optional[str]) -> Optional[range]:
    """要读哪几年。任一端缺失就返回 None,表示全读。"""
    if not start or not end:
        return None
    return range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)


def _split_by_code(df: pd.DataFrame, value_cols: list) -> dict:
    """按 code 把长表切成 {code: DataFrame}。

    用 numpy 找边界再切,比 groupby 快一倍(实测 2.0M 行 4.8s -> 2.4s)。
    groupby 每组都要走一遍 pandas 的索引与类型机制,而这里每组只剩一次
    DataFrame 构造 —— 5000 只的规模下,省掉的就是几秒。

    前提:df 已按 code 排好序,同一 code 的行必须连续。
    """
    if not len(df):
        return {}
    codes = df["code"].to_numpy()
    # 不强制时间精度:pandas 2 默认 ns、pandas 3 默认 us,
    # 硬转会让读回来的索引与现取的行情精度不一致,拼接时平白多一次转换。
    dates = df["date"].to_numpy()
    vals = df[value_cols].to_numpy(dtype="float64")

    bounds = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(codes)]])
    return {str(codes[s]): pd.DataFrame(vals[s:e], columns=value_cols,
                                        index=pd.DatetimeIndex(dates[s:e]))
            for s, e in zip(starts, ends)}


def digest_bars(bars: pd.DataFrame) -> str:
    """一段K线的指纹,用于判断缓存的派生结果(如获利筹码)是否还有效。

    只取收盘价与日期:派生结果只依赖它们,取多了会因为无关字段的微小修订
    白白失效。float64 的字节表示是确定的,不涉及浮点比较。
    """
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    h.update(bars["close"].to_numpy(dtype="float64").tobytes())
    h.update(bars.index.to_numpy(dtype="datetime64[ns]").tobytes())
    return h.hexdigest()


class ParquetStore:
    """行情 / 资金流 / 流通股本的本地缓存。"""

    def __init__(self, root: str = "cache", dividend_type: str = "back"):
        _require_parquet()
        self.root = os.path.abspath(root)
        self.dividend_type = dividend_type

    # ------------------------------------------------------------ 路径
    def _dir(self, name: str) -> str:
        return os.path.join(self.root, name)

    def _part(self, name: str, year: int) -> str:
        return os.path.join(self._dir(name), f"{year}.parquet")

    def _meta_path(self, name: str) -> str:
        return os.path.join(self._dir(name), "_meta.json")

    # ------------------------------------------------------------ 元数据
    def read_meta(self, name: str) -> dict:
        try:
            with open(self._meta_path(name), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def write_meta(self, name: str, **kw) -> None:
        os.makedirs(self._dir(name), exist_ok=True)
        meta = self.read_meta(name)
        meta.update(kw)
        meta["updated"] = pd.Timestamp.now().isoformat(timespec="seconds")
        with open(self._meta_path(name), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def check_dividend(self, name: str = "bars") -> None:
        """复权口径变了就必须整段作废 —— 混着用会凭空造出金叉。"""
        got = self.read_meta(name).get("dividend_type")
        if got is not None and got != self.dividend_type:
            raise SystemExit(
                f"缓存 {self._dir(name)} 是 dividend_type={got!r} 写的,"
                f"当前要 {self.dividend_type!r}。\n"
                f"两种复权的价格不能混用(会凭空造出金叉),"
                f"请换个 --cache 目录,或先删掉重建。")

    # ------------------------------------------------------------ 读
    def _read_parts(self, name: str, start: Optional[str],
                    end: Optional[str]) -> pd.DataFrame:
        d = self._dir(name)
        if not os.path.isdir(d):
            return pd.DataFrame(columns=["code", "date"] + DATASETS[name])

        want = _years(start, end)
        frames = []
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".parquet"):
                continue
            try:
                year = int(os.path.splitext(fn)[0])
            except ValueError:
                continue
            if want is not None and year not in want:
                continue
            frames.append(pd.read_parquet(os.path.join(d, fn)))
        if not frames:
            return pd.DataFrame(columns=["code", "date"] + DATASETS[name])
        return pd.concat(frames, ignore_index=True)

    def load(self, name: str, codes: Optional[Iterable[str]] = None,
             start: Optional[str] = None, end: Optional[str] = None) -> dict:
        """读出 {code: DataFrame(索引为日期)}。缺的 code 直接不出现在结果里。"""
        if name == "bars":
            self.check_dividend(name)
        df = self._read_parts(name, start, end)
        if not len(df):
            return {}

        if codes is not None:
            df = df[df["code"].isin(set(codes))]
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        if not len(df):
            return {}

        value_cols = [c for c in DATASETS[name] if c in df.columns]
        # 年份文件按年序读入、各自内部已按 (code, date) 排好,
        # 所以对 code 做一次**稳定**排序就够了,同一 code 的日期仍是升序。
        df = df.sort_values("code", kind="stable")
        return _split_by_code(df, value_cols)

    def coverage(self, name: str) -> pd.DataFrame:
        """每只股票缓存到哪天,用来决定增量只补哪一段。"""
        df = self._read_parts(name, None, None)
        if not len(df):
            return pd.DataFrame(columns=["code", "first", "last", "rows"])
        g = df.groupby("code", observed=True)["date"]
        return pd.DataFrame({"first": g.min(), "last": g.max(),
                             "rows": g.size()}).reset_index()

    # ------------------------------------------------------------ 写
    def save(self, name: str, data: dict, verbose: bool = True) -> int:
        """把 {code: DataFrame} 合并进缓存,返回写入的行数。

        同一个 (code, date) 以新数据为准 —— 行情会被数据源事后修订,
        缓存必须跟着改,不能保留第一次取到的值。
        """
        if not data:
            return 0
        value_cols = DATASETS[name]
        rows = []
        for code, df in data.items():
            if df is None or not len(df):
                continue
            sub = df.reindex(columns=[c for c in value_cols if c in df.columns]).copy()
            sub.insert(0, "date", pd.DatetimeIndex(df.index))
            sub.insert(0, "code", str(code))
            rows.append(sub)
        if not rows:
            return 0

        fresh = pd.concat(rows, ignore_index=True)
        fresh = fresh[fresh["date"].notna()]
        os.makedirs(self._dir(name), exist_ok=True)

        written = 0
        for year, chunk in fresh.groupby(fresh["date"].dt.year, sort=True):
            path = self._part(name, int(year))
            if os.path.exists(path):
                chunk = pd.concat([pd.read_parquet(path), chunk], ignore_index=True)
            chunk = (chunk.drop_duplicates(["code", "date"], keep="last")
                          .sort_values(["code", "date"]))
            self._atomic_write(chunk, path)
            written += len(chunk)

        if name == "bars":
            self.write_meta(name, dividend_type=self.dividend_type)
        else:
            self.write_meta(name)
        if verbose:
            print(f"  缓存 {name}: {len(data)} 只,落盘 {written} 行 -> {self._dir(name)}")
        return written

    @staticmethod
    def _atomic_write(df: pd.DataFrame, path: str) -> None:
        """先写临时文件再改名。中途断电/Ctrl-C 不会留下半个损坏的 parquet。"""
        tmp = path + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    def drop(self, name: str) -> None:
        shutil.rmtree(self._dir(name), ignore_errors=True)

    # ------------------------------------------------------- 取数水位线
    # 「缓存到哪天」和「问到哪天」是两回事:停牌的票、周末、当天还没收盘,
    # 都会让「问到了但没有数据」。只看数据的最后一天,就会每跑一次都
    # 重新去问那段本来就空的尾巴 —— 全市场 5000 只,每次白等一轮。
    def _asked_path(self, name: str) -> str:
        return os.path.join(self._dir(name), "_asked.parquet")

    def load_asked(self, name: str) -> dict:
        path = self._asked_path(name)
        if not os.path.exists(path):
            return {}
        df = pd.read_parquet(path)
        return {str(r.code): (pd.Timestamp(r.asked_from), pd.Timestamp(r.asked_through))
                for r in df.itertuples()}

    def record_asked(self, name: str, codes: Iterable[str],
                     start: str, end: str) -> None:
        """记下「这批 code 的 [start, end] 已经问过了」,不管有没有取到数据。"""
        codes = [str(c) for c in codes]
        if not codes:
            return
        os.makedirs(self._dir(name), exist_ok=True)
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        have = self.load_asked(name)
        for c in codes:
            old = have.get(c)
            have[c] = ((min(old[0], lo), max(old[1], hi)) if old else (lo, hi))
        df = pd.DataFrame({"code": list(have),
                           "asked_from": [v[0] for v in have.values()],
                           "asked_through": [v[1] for v in have.values()]})
        self._atomic_write(df.sort_values("code"), self._asked_path(name))

    # ------------------------------------------------------------ 流通股本
    def load_floats(self) -> dict:
        path = os.path.join(self.root, "floats.parquet")
        if not os.path.exists(path):
            return {}
        df = pd.read_parquet(path)
        return {str(r.code): float(r.float_shares)
                for r in df.itertuples() if r.float_shares > 0}

    def save_floats(self, floats: dict, verbose: bool = True) -> int:
        if not floats:
            return 0
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, "floats.parquet")
        fresh = pd.DataFrame({"code": list(floats), "float_shares": list(floats.values())})
        fresh["updated"] = pd.Timestamp.now()
        if os.path.exists(path):
            fresh = pd.concat([pd.read_parquet(path), fresh], ignore_index=True)
        fresh = fresh.drop_duplicates("code", keep="last").sort_values("code")
        self._atomic_write(fresh, path)
        if verbose:
            print(f"  缓存 floats: {len(fresh)} 只 -> {path}")
        return len(fresh)


class ProfitCache:
    """获利筹码的结果缓存。

    筹码分布是逐日向前滚的状态,没法只算新增的那几天,但它完全由
    (收盘价序列, 流通股本, 参数) 决定 —— 只要这三样没变就可以直接复用。
    K线指纹一变就整只重算,不做「续算」,那需要把分布数组也存下来,
    存量大得多,且一旦对不齐就是静默错值。

    实测计算那一半里筹码占大头;命中缓存后全市场一轮从 3.6min 降到秒级。
    """

    def __init__(self, store: ParquetStore, params_key: str):
        self.store = store
        self.params_key = params_key
        self.dir = os.path.join(store.root, "profit")
        self._path = os.path.join(self.dir, "profit.parquet")
        self._meta = os.path.join(self.dir, "_meta.json")

    def _valid_store(self) -> bool:
        """参数换了(阈值、衰减、网格步长…)缓存就整体作废。"""
        try:
            with open(self._meta, encoding="utf-8") as f:
                return json.load(f).get("params_key") == self.params_key
        except (OSError, ValueError):
            return False

    def load(self) -> dict:
        """返回 {code: (digest, Series)}。参数不匹配时返回空。"""
        if not self._valid_store() or not os.path.exists(self._path):
            return {}
        df = pd.read_parquet(self._path)
        out = {}
        for code, g in df.groupby("code", sort=False, observed=True):
            s = pd.Series(g["profit_ratio"].to_numpy(),
                          index=pd.DatetimeIndex(g["date"].to_numpy()))
            out[str(code)] = (str(g["digest"].iloc[0]), s)
        return out

    def save(self, entries: dict, verbose: bool = True) -> int:
        """entries: {code: (digest, Series)}。整体重写,不做合并。

        调用方本来就是拿着全量结果来写的,合并只会把过期的脏数据留下来。
        """
        if not entries:
            return 0
        os.makedirs(self.dir, exist_ok=True)
        rows = []
        for code, (digest, s) in entries.items():
            if s is None or not len(s):
                continue
            rows.append(pd.DataFrame({
                "code": str(code), "digest": str(digest),
                "date": pd.DatetimeIndex(s.index),
                "profit_ratio": pd.to_numeric(s.to_numpy(), errors="coerce"),
            }))
        if not rows:
            return 0
        df = pd.concat(rows, ignore_index=True).sort_values(["code", "date"])
        ParquetStore._atomic_write(df, self._path)
        with open(self._meta, "w", encoding="utf-8") as f:
            json.dump({"params_key": self.params_key,
                       "updated": pd.Timestamp.now().isoformat(timespec="seconds")},
                      f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"  缓存 profit: {len(entries)} 只,{len(df)} 行 -> {self._path}")
        return len(df)


def missing_ranges(store: ParquetStore, name: str, codes: Iterable[str],
                   start: str, end: str) -> dict:
    """算出每只股票还缺哪一段,返回 {code: (start, end)}。

    判断用的是**问到哪天**(_asked 水位线),不是**有数据到哪天**。
    两者经常不等:停牌的票、周末、当天尚未收盘,都是「问过了但没数据」。
    按数据末日去判,这些空尾巴会被每一次回测重新问一遍 ——
    全市场 5000 只,每跑一次就白等一整轮。

    只补尾巴:问到 T,要到 T+n,就从 T 的次日开始取。
    要的区间比问过的起点还早,则整段重取 —— 中间挖空的情形没法用
    「一个起止区间」表达,与其猜不如老老实实重来。

    没有水位线时(老缓存)退回按数据覆盖判断,行为与之前一致。
    """
    asked = store.load_asked(name)
    if not asked:
        cov = store.coverage(name)
        asked = {str(r.code): (pd.Timestamp(r.first), pd.Timestamp(r.last))
                 for r in cov.itertuples()} if len(cov) else {}
    want_lo, want_hi = pd.Timestamp(start), pd.Timestamp(end)

    todo = {}
    for code in codes:
        rng = asked.get(str(code))
        if rng is None:
            todo[code] = (start, end)
            continue
        first, last = rng
        if first > want_lo:
            todo[code] = (start, end)
        elif last < want_hi:
            nxt = (last + pd.Timedelta(days=1)).strftime("%Y%m%d")
            todo[code] = (nxt, end)
    return todo


def verify(store: ParquetStore, name: str, fetched: dict,
           tol: float = 1e-6, verbose: bool = True) -> list:
    """把刚从行情源取到的数据与缓存逐格比对,返回有出入的 code 列表。

    缓存最怕的不是慢,是**悄悄读到过期或错位的数据**。
    数据源事后修订、复权因子变化、缓存写了一半,都会让回测结果不知不觉变样。
    所以留一条随时能跑的校验路径,别等到结论出来了才怀疑数据。
    """
    cached = store.load(name, codes=list(fetched), start=None, end=None)
    bad = []
    for code, fresh in fetched.items():
        old = cached.get(code)
        if old is None or fresh is None or not len(fresh):
            continue
        common = old.index.intersection(fresh.index)
        if not len(common):
            continue
        cols = [c for c in DATASETS[name] if c in old.columns and c in fresh.columns]
        a = old.loc[common, cols].to_numpy(dtype="float64")
        b = fresh.loc[common, cols].to_numpy(dtype="float64")
        same = np.isclose(a, b, rtol=tol, atol=tol, equal_nan=True)
        if not same.all():
            n = int((~same).any(axis=1).sum())
            bad.append(code)
            if verbose:
                print(f"  [!] {code}: {n}/{len(common)} 天与缓存不一致", file=sys.stderr)
    return bad
