# -*- coding: utf-8 -*-
"""查一只股票在某一天(或某个区间的每一天)是否入选,并给出每个条件的实际取值。

    # 单日:逐条列出五个条件过没过
    python scripts/check_one.py 000001.SZ 20260904

    # 区间:一次列出每天的结论,顺带统计是哪个条件卡住的
    python scripts/check_one.py 000001.SZ --from 20260801 --to 20260930

    # 不碰 QMT,用本地 CSV 验证脚本本身(列名 Date/Open/High/Low/Close/Volume)
    python scripts/check_one.py INTC --csv tests/intc.csv --from 20170101 --to 20171231

默认参数就是最终敲定的那一版口径:均线转向、放量=高于前一日、获利筹码>85%、
不要求回踩。要改用别的口径,见 --help。

区间模式整段只算一次,再按日切片。所有条件都只用当日及之前的数据,
所以切出来的第 i 日结论与「只喂到第 i 日再算」完全相同;
不放心可以加 --verify,它会逐日截断重算一遍并比对。
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ema_strategy import feed                      # noqa: E402
from ema_strategy.bull import BullParams, run      # noqa: E402
from ema_strategy.bull_picker import explain       # noqa: E402
from ema_strategy.sequence import Params           # noqa: E402

# 区间表格里展示的条件列:(表头, daily 里的列名)
COND_COLS = [("形态维持", "active"), ("均线", "cond_ma_up"), ("筹码", "cond_profit"),
             ("流入", "cond_inflow"), ("放量", "cond_volume")]


def _pad(text: str, width: int, right: bool = True) -> str:
    """按显示宽度对齐:CJK 与全角符号占 2 列,f-string 的 :>10 只会数字符个数。

    一个汉字算 1 个字符但占 2 格,直接用格式化宽度会让整张表歪掉。
    """
    w = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    pad = " " * max(0, width - w)
    return pad + text if right else text + pad


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="check_one.py",
        description="查单只股票某日/某区间是否入选多头黄金眼,并列出每个条件的取值")
    ap.add_argument("code", help="股票代码,如 000001.SZ;用 --csv 时是任意标签")
    ap.add_argument("date", nargs="?", default="", help="要查的那一天 YYYYMMDD")
    ap.add_argument("--from", dest="date_from", default="", help="区间起(YYYYMMDD)")
    ap.add_argument("--to", dest="date_to", default="", help="区间止(YYYYMMDD)")
    ap.add_argument("--csv", default="", help="从本地CSV读行情,不连 QMT")
    ap.add_argument("--flow-csv", default="", dest="flow_csv",
                    help="配合 --csv:资金流CSV,列 Date,bidMostAmount,offMostAmount。"
                         "不给则资金流缺失,该条件恒不通过")
    ap.add_argument("--warmup-days", type=int, default=400, dest="warmup_days",
                    help="往前多取多少自然日给均线和筹码预热,默认 400")
    ap.add_argument("--dividend", default="back", choices=["back", "front", "none"],
                    help="复权方式;信号判定必须 back,否则历史会 repaint")
    ap.add_argument("--no-download", action="store_true", dest="no_download",
                    help="跳过下载,直接读 QMT 本地缓存")
    ap.add_argument("--float-shares", type=float, default=0.0, dest="float_shares",
                    help="手工指定流通股本(股);--csv 模式下必填或用默认 1e9")
    ap.add_argument("--verify", action="store_true",
                    help="区间模式下逐日截断重算并比对,用来自查未来函数")
    # ---- 策略口径,默认即最终版 ----
    ap.add_argument("--profit-min", type=float, default=0.85, dest="profit_min",
                    help="获利筹码下限,默认 0.85")
    ap.add_argument("--volume-mode", default="prev", choices=["prev", "ma"],
                    dest="volume_mode", help="放量口径,默认 prev(高于前一日)")
    ap.add_argument("--volume-ratio", type=float, default=1.5, dest="volume_ratio")
    ap.add_argument("--volume-window", type=int, default=5, dest="volume_window")
    ap.add_argument("--ma-up", action="store_true", dest="ma_up",
                    help="改用较宽的「三线均向上」,替代默认的「均线转向」")
    ap.add_argument("--pullback", action="store_true",
                    help="打开回踩要求(默认关闭)")
    ap.add_argument("--pullback-window", type=int, default=6, dest="pullback_window")
    ap.add_argument("--max-span", type=int, default=60, dest="max_span")
    ap.add_argument("--allow-same-day", action="store_true", dest="allow_same_day",
                    help="允许两个金叉同日(默认要求三个金叉分属不同交易日,"
                         "严格 1号日 < 2号日 < 3号日)")
    ap.add_argument("--volume-unit", type=int, default=0, dest="volume_unit",
                    help="volume 的单位:A股(QMT)按手计=100,已是股数则=1。"
                         "缺省按数据源自动取(QMT=100,--csv=1)")
    return ap


def load_csv(path: str) -> pd.DataFrame:
    """读本地 CSV。列名不分大小写,volume 缺失时无法判放量,直接报错。"""
    d = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    d = d.rename(columns=str.lower).sort_index()
    missing = [c for c in feed.PRICE_COLS + ["volume"] if c not in d.columns]
    if missing:
        raise SystemExit(f"{path} 缺列:{missing}")
    d = d[feed.PRICE_COLS + ["volume"]].dropna()
    # CSV 的 volume 已是股数(见 --volume-unit),amount 就是 收盘价 x 股数。
    # 多乘 100 会让均价 amount/股数 被夹到当日最高价,筹码三角分布的峰值永远贴上沿。
    d["amount"] = d["close"] * d["volume"]
    return d


def load_flow_csv(path: str) -> pd.DataFrame:
    """读资金流 CSV。缺列直接报错 —— 静默返回空会让资金流条件恒不通过,更难查。"""
    d = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    need = ["bidMostAmount", "offMostAmount"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise SystemExit(f"{path} 缺列:{missing}")
    return d[need].apply(pd.to_numeric, errors="coerce")


def load_qmt(code: str, start: str, end: str, a) -> tuple:
    """从 QMT 取行情 / 流通股本 / 资金流。返回 (bars, float_shares, flow)。"""
    bars = feed.fetch_daily([code], start, end, dividend_type=a.dividend,
                            download=not a.no_download).get(code)
    if bars is None or not len(bars):
        raise SystemExit(f"{code}: {start}~{end} 无行情数据")

    fs = a.float_shares
    if fs <= 0:
        fs = feed.fetch_float_shares([code]).get(code, 0.0)
    if fs <= 0:
        raise SystemExit(f"{code}: 取不到流通股本,获利筹码无从算起"
                         f"(可用 --float-shares 手工指定)")

    flow = feed.fetch_money_flow([code], start, end,
                                 download=not a.no_download).get(code)
    if flow is None:
        print(f"[!] {code} 取不到资金流(bidMostAmount/offMostAmount)。"
              f"该条件恒不通过,任何一天都不会入选 —— 缺数据不当成有流入。",
              file=sys.stderr)
    return bars, fs, flow


def describe_params(p: BullParams) -> str:
    conds = ["形态维持" + ("(三金叉分属不同日)" if p.seq.require_distinct_days
                          else "(允许金叉同日)")]
    if p.require_pullback:
        conds.append(f"近{p.pullback_window}日内有回踩")
    conds.append("均线转向(昨有下行今全上)" if p.require_ma_turn else "三线均向上")
    conds += [f"获利筹码>{p.profit_min:.0%}", "资金流入>0",
              "量>前一日" if p.volume_mode == "prev"
              else f"量>前{p.volume_window}日均量x{p.volume_ratio}"]
    return " 且 ".join(conds)


def one_day(code: str, bars: pd.DataFrame, fs: float, flow, p: BullParams,
            date: str) -> int:
    """单日诊断。把行情截到该日再算,与 `python -m ema_strategy --explain` 同口径。"""
    asof = pd.Timestamp(date)
    sub = bars.loc[:asof]
    if not len(sub):
        print(f"{code}: {asof.date()} 及之前没有K线", file=sys.stderr)
        return 1
    actual = sub.index[-1]
    if actual != asof:
        print(f"[!] {code} 在 {asof.date()} 无K线(停牌或非交易日),"
              f"下面诊断的是最近的 {actual.date()}。", file=sys.stderr)

    sub_flow = flow.loc[:asof] if flow is not None and len(flow) else flow
    print(explain(code, sub, fs, sub_flow, p))

    # explain() 不含停牌缺口检查,而 scan() 会因此剔除信号,必须单独提示
    if bool(feed.gap_flags(sub.index).iloc[-p.seq.slow:].any()):
        print(f"  [!] 近 {p.seq.slow} 根K线内存在超过 30 天的停牌缺口,"
              f"均线失真,全市场扫描会剔除该信号。")
    return 0


def a_range(code: str, bars: pd.DataFrame, fs: float, flow, p: BullParams,
            date_from: str, date_to: str, verify: bool) -> int:
    """区间模式:整段算一次,再切出 [from, to] 的逐日结论。"""
    res = run(bars, fs, flow, p)
    daily = res["daily"]
    lo = pd.Timestamp(date_from) if date_from else daily.index[0]
    hi = pd.Timestamp(date_to) if date_to else daily.index[-1]
    view = daily.loc[lo:hi]
    if not len(view):
        print(f"{code}: {lo.date()}~{hi.date()} 区间内没有K线", file=sys.stderr)
        return 1

    seq = res["sequences"]
    if len(seq):
        q = seq.iloc[-1]
        print(f"最近一次三金叉: 1号 {pd.Timestamp(q['start_date']).date()}"
              f" -> 2号 {pd.Timestamp(q['c2_date']).date()}"
              f" -> 3号 {pd.Timestamp(q['confirm_date']).date()}(形态启动日)")
    else:
        print("样本内未出现完整的 5上穿10 -> 5上穿30 -> 10上穿30 序列")

    widths = [12, 9, 10] + [8] * (len(COND_COLS) - 1) + [10, 14, 10]
    heads = (["日期", "收盘", "形态"] + [n for n, _ in COND_COLS[1:]]
             + ["获利筹码", "净流入", "结论"])
    print()
    print("".join(_pad(h, w, right=(i > 0)) for i, (h, w) in enumerate(zip(heads, widths))))
    print("-" * sum(widths))
    for ts, r in view.iterrows():
        pr, ni = r["profit_ratio"], r["net_inflow"]
        cells = ([str(ts.date()), f"{r['close']:.3f}",
                  "第%d天" % int(r["days_in_pattern"]) if r["active"] else "未维持"]
                 + ["Y" if bool(r[c]) else "-" for _, c in COND_COLS[1:]]
                 + [f"{pr:.1%}" if pd.notna(pr) else "缺",
                    f"{ni:,.0f}" if pd.notna(ni) else "缺数据",
                    "【入选】" if bool(r["triggered"]) else ""])
        print("".join(_pad(c, w, right=(i > 0))
                      for i, (c, w) in enumerate(zip(cells, widths))))

    hits = view.index[view["triggered"]]
    print(f"\n区间 {lo.date()}~{hi.date()} 共 {len(view)} 个交易日,"
          f"入选 {len(hits)} 天"
          + (":" + " ".join(str(d.date()) for d in hits) if len(hits) else ""))

    # 没入选时最想知道的是「差在哪一条」——按未通过次数排个序
    if len(hits) < len(view):
        miss = view[~view["triggered"]]
        tally = sorted(((n, int((~miss[c].astype(bool)).sum())) for n, c in COND_COLS),
                       key=lambda kv: -kv[1])
        print("未入选的 %d 天里各条件未通过次数:" % len(miss)
              + " ".join(f"{n}={k}" for n, k in tally if k))

    if verify:
        return _verify(code, bars, fs, flow, p, view)
    return 0


def _verify(code: str, bars: pd.DataFrame, fs: float, flow, p: BullParams,
            view: pd.DataFrame) -> int:
    """逐日截断重算,与整段一次算出的结论比对 —— 自查有没有用到未来数据。"""
    bad = []
    for ts in view.index:
        sub = bars.loc[:ts]
        sub_flow = flow.loc[:ts] if flow is not None and len(flow) else flow
        got = run(sub, fs, sub_flow, p)["daily"].iloc[-1]["triggered"]
        if bool(got) != bool(view.loc[ts, "triggered"]):
            bad.append(ts.date())
    if bad:
        print(f"\n[!!] 截断重算结论不一致的日期:{bad} —— 存在未来函数,别用这个结果。")
        return 1
    print(f"\n截断一致性自查:{len(view)} 天逐日重算结论全部一致,未用到未来数据。")
    return 0


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if not a.date and not (a.date_from or a.date_to):
        print("要么给一个日期,要么给 --from/--to 区间", file=sys.stderr)
        return 2

    # CSV 一般是美股/导出数据,volume 已是股数;QMT 的 volume 以「手」计。
    unit = a.volume_unit or (1 if a.csv else 100)
    p = BullParams(seq=Params(max_span=a.max_span,
                              require_distinct_days=not a.allow_same_day),
                   require_pullback=a.pullback, pullback_window=a.pullback_window,
                   require_ma_turn=not a.ma_up, profit_min=a.profit_min,
                   volume_mode=a.volume_mode, volume_ratio=a.volume_ratio,
                   volume_window=a.volume_window, chip_volume_unit=unit)
    print(f"条件: {describe_params(p)}")

    end = a.date or a.date_to or pd.Timestamp.today().strftime("%Y%m%d")
    first = a.date_from or a.date or end
    start = (pd.Timestamp(first) - pd.Timedelta(days=a.warmup_days)).strftime("%Y%m%d")

    if a.csv:
        bars = load_csv(a.csv)
        bars = bars.loc[:pd.Timestamp(end)]
        fs = a.float_shares if a.float_shares > 0 else 1e9
        flow = load_flow_csv(a.flow_csv) if a.flow_csv else None
        print(f"行情来自 {a.csv}({len(bars)} 根,至 {bars.index[-1].date()});"
              f"流通股本按 {fs:,.0f} 股计;"
              + (f"资金流来自 {a.flow_csv}" if flow is not None
                 else "无资金流数据 —— 该条件恒不通过"))
    else:
        bars, fs, flow = load_qmt(a.code, start, end, a)
        print(f"{a.code} 行情 {bars.index[0].date()}~{bars.index[-1].date()}"
              f"({len(bars)} 根) | 流通股本 {fs:,.0f} 股 | 复权 {a.dividend}")

    if len(bars) < p.warmup:
        print(f"数据不足:{len(bars)} 根,至少需要 {p.warmup} 根"
              f"(可调大 --warmup-days)", file=sys.stderr)
        return 1

    if a.date and not (a.date_from or a.date_to):
        return one_day(a.code, bars, fs, flow, p, a.date)
    return a_range(a.code, bars, fs, flow, p, a.date_from, a.date_to, a.verify)


if __name__ == "__main__":
    raise SystemExit(main())
