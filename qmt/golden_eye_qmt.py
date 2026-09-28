#encoding:utf-8
"""
黄金眼选股 —— QMT 内置 Python 版

放进 QMT「策略编辑器」即可运行。与仓库里的 ema_strategy 包是同一套逻辑,
但完全自包含(QMT 环境里没有该包),所有计算函数都在本文件内。

【选股条件】(当日须全部满足)
  1. 处于黄金眼形态维持期间
     形态启动:MA5上穿MA10 -> MA5上穿MA30 -> MA10上穿MA30 依次出现,第三个金叉当天
     形态破坏:MA5 或 MA10 跌破 MA30
  2. 均线转向:前一日至少一条均线下行,当日三条全部上行
  3. 获利筹码 > 85%
  4. 当日资金流入:bidMostAmount - offMostAmount > 0
  5. 放量:成交量 > 前一日成交量

【编码】
  本文件以 UTF-8 保存,首行声明 utf-8。若 QMT 报编码错,
  把文件另存为 GBK 并把首行改成 #encoding:gbk。

【运行方式】
  周期选「日线」。init 里取好股票池,handlebar 在最后一根K线上做全市场扫描,
  把当日入选股票打印出来。回测模式下逐根K线都会输出当日选股。

【与 Python 版的一致性】
  本文件的纯计算函数与 ema_strategy 包逐位对齐,由 tests/test_qmt_port.py 校验。
"""
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ 参数
SECTOR = '沪深A股'          # 股票池;调试可改 '沪深300'
FAST, MID, SLOW = 5, 10, 30  # 黄金眼三条均线
MAX_SPAN = 60                # 1号到3号金叉的最大间隔(交易日)
PROFIT_MIN = 0.85            # 获利筹码下限(严格大于)
CHIP_BIN_PCT = 0.002         # 筹码价格网格步长,别调大(见仓库 README 的收敛性说明)
CHIP_GRID_SPAN = 50.0
CHIP_MIN_PERIODS = 30
VOLUME_UNIT = 100            # QMT 的 volume 以「手」计,换手率要乘 100
HISTORY_BARS = 400           # 每只票取多少根日线
EXCLUDE_ST = True
MIN_LISTED_DAYS = 120
PRINT_LIMIT = 50             # 每日最多打印多少只
SCAN_EVERY_BAR = False       # False=只在最新K线选股(实盘);True=每根K线都扫(回测逐日输出)


# ------------------------------------------------------- 纯计算(与Python版一致)
def _ma(s, n):
    return s.rolling(n).mean()


def _crossed(state, a, b):
    """由 state 的 False->True 跃迁判定穿越。

    必须要求前一根K线两条均线都已有值:否则均线预热期结束的第一根K线上,
    state 从 False(NaN 比较结果)跳到 True,会被误判成一次金叉。
    """
    valid = a.notna() & b.notna()
    return (state & ~state.shift(1, fill_value=False)
            & valid & valid.shift(1, fill_value=False))


def _cross_up(a, b):
    return _crossed(a >= b, a, b)          # 相等算作在上方


def _cross_down(a, b):
    return _crossed(a < b, a, b)


def _find_sequences(d):
    """扫出 1号(5上穿10) -> 2号(5上穿30) -> 3号(10上穿30) 依次完成的序列。

    返回第三个金叉(形态启动日)的下标列表。
    """
    c1 = _cross_up(d['ma_f'], d['ma_m']).to_numpy()
    c2 = _cross_up(d['ma_f'], d['ma_s']).to_numpy()
    c3 = _cross_up(d['ma_m'], d['ma_s']).to_numpy()
    dead = _cross_down(d['ma_f'], d['ma_m']).to_numpy()

    starts, i1, i2 = [], None, None
    for i in range(len(d)):
        if i1 is not None:
            if (dead[i] and not c1[i]) or (i - i1 > MAX_SPAN):
                i1 = i2 = None             # 启动条件反转或超期,本轮作废
        if c1[i]:
            i1, i2 = i, None               # 以最新的1号金叉为准
        if c2[i] and i1 is not None and i >= i1:
            i2 = i
        if c3[i] and i1 is not None and i2 is not None and i >= i2:
            starts.append(i)
            i1 = i2 = None
    return starts


def _pattern_active(d, starts):
    """形态自启动日开启,MA5 或 MA10 跌破 MA30 即破坏。"""
    broken = ((d['ma_f'] < d['ma_s']) | (d['ma_m'] < d['ma_s'])).to_numpy()
    active = np.zeros(len(d), dtype=bool)
    start_set, on = set(starts), False
    for i in range(len(d)):
        if i in start_set and not on:
            on = True
        if on and broken[i]:
            on = False
        active[i] = on
    return pd.Series(active, index=d.index)


def _all_ma_rising(d):
    cols = ('ma_f', 'ma_m', 'ma_s')
    ups = [(d[c] > d[c].shift(1)) & d[c].notna() & d[c].shift(1).notna() for c in cols]
    return ups[0] & ups[1] & ups[2]


def _ma_turn_up(d):
    """前一日至少一条均线下行,当日三条全部上行。

    「至少一条向下」按字面取严格小于;走平不算向下。
    """
    cols = ('ma_f', 'ma_m', 'ma_s')
    any_down = pd.concat([d[c] < d[c].shift(1) for c in cols], axis=1).any(axis=1)
    prev_valid = pd.concat([d[c].shift(2).notna() for c in cols], axis=1).all(axis=1)
    return _all_ma_rising(d) & any_down.shift(1, fill_value=False) & prev_valid


def _volume_surge_prev(volume):
    """放量:当日成交量高于前一日。"""
    prev = volume.shift(1)
    return (volume > prev) & prev.notna()


def _price_grid(ref, bin_pct, span):
    """以首个有效收盘价为锚点的等比价格网格。

    网格位置只由起点决定,与后续价格无关 —— 若改用全段 min/max 划网格,
    同一天的获利比例会随后续K线变化,那是未来函数。
    """
    lo, hi = ref / span, ref * span
    n = int(np.ceil(np.log(hi / lo) / np.log1p(bin_pct)))
    edges = lo * np.power(1.0 + bin_pct, np.arange(n + 1, dtype=float))
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def _triangle(seg, low, high, peak):
    """当日成交在价格桶上的三角分布(峰值在成交均价),已归一化。"""
    if len(seg) == 1:
        return np.ones(1)
    peak = low if peak < low else (high if peak > high else peak)
    lw = peak - low if peak - low > 1e-12 else 1e-12
    rw = high - peak if high - peak > 1e-12 else 1e-12
    vals = np.where(seg <= peak, (seg - low) / lw, (high - seg) / rw)
    np.clip(vals, 0.0, None, out=vals)
    total = vals.sum()
    return vals / total if total > 0 else np.ones(len(seg)) / len(seg)


def _profit_ratio(bars, float_shares):
    """获利筹码比例(换手衰减法)。

    每日已有筹码按换手率衰减,腾出的比例由当日价格分布补上;
    获利比例 = 成本低于当日收盘价的筹码占比。

    注意 volume 以「手」计,换手率须乘 VOLUME_UNIT。漏乘会让筹码几乎不衰减,
    上涨行情的获利比例被系统性高估,一批不该入选的票会假装达标。
    """
    n = len(bars)
    if n == 0 or not float_shares or float_shares <= 0:
        return pd.Series([np.nan] * n, index=bars.index)

    high = bars['high'].to_numpy(float)
    low = bars['low'].to_numpy(float)
    close = bars['close'].to_numpy(float)
    volume = bars['volume'].to_numpy(float)
    amount = (bars['amount'].to_numpy(float) if 'amount' in bars.columns
              else np.full(n, np.nan))

    ok_close = close[np.isfinite(close) & (close > 0)]
    if ok_close.size == 0:
        return pd.Series([np.nan] * n, index=bars.index)
    edges, centers = _price_grid(float(ok_close[0]), CHIP_BIN_PCT, CHIP_GRID_SPAN)
    n_bins = len(centers)

    lo_c = np.clip(low, edges[0], edges[-1])
    hi_c = np.clip(high, edges[0], edges[-1])
    i0_all = np.clip(np.searchsorted(edges, lo_c, side='right') - 1, 0, n_bins - 1)
    i1_all = np.clip(np.searchsorted(edges, hi_c, side='right') - 1, 0, n_bins - 1)
    k_all = np.searchsorted(centers, close, side='right')

    shares = volume * VOLUME_UNIT
    with np.errstate(divide='ignore', invalid='ignore'):
        w_all = np.clip(shares / float_shares, 0.0, 1.0)
        peak_all = np.where(np.isfinite(amount) & (shares > 0),
                            amount / np.where(shares > 0, shares, 1.0),
                            (high + low + close) / 3.0)
    w_all = np.where(np.isfinite(w_all), w_all, 0.0)
    peak_all = np.where(np.isfinite(peak_all), peak_all, (high + low + close) / 3.0)
    bad = ~np.isfinite(low) | ~np.isfinite(high) | (high < low)

    dist = np.zeros(n_bins)
    out = np.full(n, np.nan)
    lo_i, hi_i = n_bins, -1

    for i in range(n):
        if bad[i]:
            continue
        j0, j1 = int(i0_all[i]), int(i1_all[i])
        if j1 < j0:
            j0, j1 = j1, j0
        seg = _triangle(centers[j0:j1 + 1], lo_c[i], hi_c[i], peak_all[i])
        w = float(w_all[i])

        if hi_i < lo_i:
            dist[j0:j1 + 1] = seg
            lo_i, hi_i = j0, j1
        else:
            if w:
                dist[lo_i:hi_i + 1] *= (1.0 - w)
            dist[j0:j1 + 1] += w * seg
            lo_i = j0 if j0 < lo_i else lo_i
            hi_i = j1 if j1 > hi_i else hi_i

        if i + 1 >= CHIP_MIN_PERIODS:
            k = int(k_all[i])
            k = lo_i if k < lo_i else (hi_i + 1 if k > hi_i + 1 else k)
            total = dist[lo_i:hi_i + 1].sum()
            if total > 0:
                r = dist[lo_i:k].sum() / total
                out[i] = 0.0 if r < 0 else (1.0 if r > 1 else float(r))
    return pd.Series(out, index=bars.index)


def evaluate_one(bars, float_shares, flow):
    """判断 bars 最后一根K线当日是否入选,返回 dict 或 None。

    bars 需含 open/high/low/close/volume(amount 可选);
    flow 为含 bidMostAmount / offMostAmount 的 DataFrame,可为 None。
    """
    if len(bars) < SLOW + 5:
        return None

    d = bars.copy()
    d['ma_f'] = _ma(d['close'], FAST)
    d['ma_m'] = _ma(d['close'], MID)
    d['ma_s'] = _ma(d['close'], SLOW)

    active = _pattern_active(d, _find_sequences(d))
    if not bool(active.iloc[-1]):
        return None

    turn = _ma_turn_up(d)
    if not bool(turn.iloc[-1]):
        return None

    if not bool(_volume_surge_prev(d['volume']).iloc[-1]):
        return None

    # 资金流缺失时不触发 —— 不把缺数据当成「有流入」放过
    if flow is None or len(flow) == 0:
        return None
    if 'bidMostAmount' not in flow.columns or 'offMostAmount' not in flow.columns:
        return None
    inflow = flow['bidMostAmount'].reindex(d.index) - flow['offMostAmount'].reindex(d.index)
    if not (pd.notna(inflow.iloc[-1]) and inflow.iloc[-1] > 0):
        return None

    pr = _profit_ratio(d, float_shares)
    if not (pd.notna(pr.iloc[-1]) and pr.iloc[-1] > PROFIT_MIN):
        return None

    last = d.iloc[-1]
    return {
        'close': float(last['close']),
        'ma5': float(last['ma_f']), 'ma10': float(last['ma_m']),
        'ma30': float(last['ma_s']),
        'profit_ratio': float(pr.iloc[-1]),
        'net_inflow': float(inflow.iloc[-1]),
        'volume': float(last['volume']),
    }


# ------------------------------------------------------------------ QMT 入口
# 股票池存模块级变量,不挂到 C 上。
# QMT 里真正的上下文是 __PyContext(C++ 对象),不是 xtquant 的 qmttools.ContextInfo,
# 属性集不同,也未必允许挂自定义属性。凡是 C 上的东西一律防御性访问。
_UNIVERSE = []


def init(C):
    global _UNIVERSE
    _UNIVERSE = []
    try:
        codes = C.get_stock_list_in_sector(SECTOR) or []
    except Exception as e:
        print('取股票池失败:%s' % e)
        codes = []

    kept, skip_st, skip_new = [], 0, 0
    for code in codes:
        try:
            info = C.get_instrument_detail(code)
        except Exception:
            info = None
        if not info:
            continue
        name = info.get('InstrumentName') or ''
        if EXCLUDE_ST and ('ST' in name.upper() or '退' in name):
            skip_st += 1
            continue
        opened = str(info.get('OpenDate') or '')
        # 未退市合约的 ExpireDate 常填哨兵值 99999999,直接解析会抛异常
        if len(opened) == 8 and opened.isdigit():
            try:
                days = (pd.Timestamp.today() - pd.Timestamp(opened)).days
                if days < MIN_LISTED_DAYS:
                    skip_new += 1
                    continue
            except Exception:
                pass
        kept.append(code)

    _UNIVERSE = kept
    print('股票池 %d 只(剔除 ST %d、次新 %d)' % (len(kept), skip_st, skip_new))
    print('条件: 形态维持 且 均线转向 且 获利筹码>%.0f%% 且 资金流入>0 且 量>前一日'
          % (PROFIT_MIN * 100))


def _should_scan(C):
    """本根K线是否要做扫描。

    不使用 C.trade_mode —— QMT 的 __PyContext 没有该属性(曾因此报
    AttributeError)。改由 SCAN_EVERY_BAR 显式控制:
      False(默认)只在最新K线扫描,实盘/盘后选股用;
      True 每根K线都扫,回测时用来逐日输出选股。
    is_last_bar 也可能缺失,取不到时按「扫描」处理,宁可多跑不可不跑。
    """
    if SCAN_EVERY_BAR:
        return True
    try:
        return bool(C.is_last_bar())
    except Exception:
        return True


def _bar_date(C):
    """当前K线日期;取不到就返回空串,不影响选股。"""
    try:
        tt = C.get_bar_timetag(C.barpos)
    except Exception:
        try:
            tt = C.get_bar_timetag()
        except Exception:
            return ''
    try:
        return pd.Timestamp(tt, unit='ms').strftime('%Y-%m-%d')
    except Exception:
        return ''


def handlebar(C):
    if not _should_scan(C):
        return

    day = _bar_date(C)
    codes = _UNIVERSE
    if not codes:
        print('股票池为空,请确认 init 是否正常执行、SECTOR 是否正确')
        return

    picks = []
    for i in range(0, len(codes), 200):
        chunk = codes[i:i + 200]
        try:
            data = C.get_market_data_ex(
                ['open', 'high', 'low', 'close', 'volume', 'amount'],
                chunk, period='1d', count=HISTORY_BARS,
                dividend_type='back', fill_data=False) or {}
        except Exception as e:
            print('取行情失败:%s' % e)
            continue

        try:
            flows = C.get_market_data_ex(
                ['bidMostAmount', 'offMostAmount'], chunk,
                period='transactioncount1d', count=HISTORY_BARS,
                fill_data=False) or {}
        except Exception:
            flows = {}          # 无投研版/L2权限时取不到,下面一律不触发

        for code in chunk:
            bars = data.get(code)
            if bars is None or len(bars) < SLOW + 5:
                continue
            try:
                info = C.get_instrument_detail(code)
                fs = float(info.get('FloatVolume') or 0) if info else 0.0
                if fs <= 0:
                    continue
                hit = evaluate_one(bars, fs, flows.get(code))
                if hit:
                    hit['code'] = code
                    picks.append(hit)
            except Exception as e:
                print('%s 计算失败:%s' % (code, e))

    picks.sort(key=lambda x: x['profit_ratio'], reverse=True)
    print('=' * 60)
    print('%s 入选 %d 只' % (day, len(picks)))
    for p in picks[:PRINT_LIMIT]:
        print('  %s  收%.2f  获利筹码%.1f%%  净流入%.0f  量%.0f'
              % (p['code'], p['close'], p['profit_ratio'] * 100,
                 p['net_inflow'], p['volume']))
    if len(picks) > PRINT_LIMIT:
        print('  ...(其余 %d 只已省略)' % (len(picks) - PRINT_LIMIT))
