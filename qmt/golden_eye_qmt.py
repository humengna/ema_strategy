#coding:gbk
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
  官方文档《快速开始》明确要求:首行写 #coding:gbk,脚本统一 GBK 编码。
  本文件请以 GBK 另存后再贴进 QMT 策略编辑器。

【运行方式】
  周期选「日线」,且回测必须以「副图模式」执行(官方文档原文:
  回测必须以 副图模式 执行,不要选择主图/主图叠加)。
  init 里取好股票池,handlebar 扫描并打印当日入选股票。

【资金流数据】
  bidMostAmount / offMostAmount 按 FLOW_PERIODS 的顺序试:
    transactioncount1d  逐笔成交统计(日级)—— 已是日频、历史长,优先用;
    l2transactioncount  Level2 大单统计 —— 需 Level2 权限,盘中累计值,兜底用。
  官方数据字典里这两个周期都可以传给内置 get_market_data_ex
  (get_market_data_ex 条目下的周期枚举只列了K线周期,特色数据另见数据字典)。
  第一批探到哪个周期有数据就固定用它,之后不再重试另一个。
  两个都取不到时该股一律不入选,不把缺数据当成「有流入」放过。

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
BATCH_SIZE = 50              # 每批处理多少只。调小=进度更密、更早看到是否卡住
PROBE_FIRST = True           # 先跑一批测速并给出全量预估,再决定要不要等下去
# 资金流周期,按顺序试,第一个取到 bidMostAmount/offMostAmount 的就固定下来。
# transactioncount1d 是日级、历史长,优先;l2transactioncount 需 Level2 权限、
# 是盘中累计值,作为兜底(按日取末值归到当日口径)。
FLOW_PERIODS = ('transactioncount1d', 'l2transactioncount')
# 各周期每日约几条:日级 1 条;L2 是盘中累计,多取几条按日取末值也不改口径。
FLOW_BARS_PER_DAY = {'transactioncount1d': 1, 'l2transactioncount': 1}


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


def _normalize_flow(df):
    """把资金流数据归到日频:索引转日期,同日多条取末值。

    transactioncount1d 本来就一天一条,groupby 是恒等变换;
    l2transactioncount 是盘中累计值,一天可能有多条,
    当日口径取收盘时的累计值,即同日最后一条。
    """
    if df is None or len(df) == 0:
        return None
    if 'bidMostAmount' not in df.columns or 'offMostAmount' not in df.columns:
        return None
    out = df[['bidMostAmount', 'offMostAmount']].copy()
    # 索引可能是 '20240102150000' 这样的时间戳字符串,也可能已经是 DatetimeIndex。
    # 对后者做字符串切片会得到 '2024-01-',解析失败后整列变 NaT,
    # 结果是资金流全部丢失、永不触发 —— 两种都要能处理。
    try:
        src_idx = pd.Index(out.index)
        if isinstance(src_idx, pd.DatetimeIndex):
            idx = src_idx.normalize()
        else:
            idx = pd.to_datetime(src_idx.astype(str).str.slice(0, 8),
                                 format='%Y%m%d', errors='coerce')
    except Exception:
        return None
    out.index = idx
    out = out[out.index.notna()]
    if not len(out):
        return None
    for c in ('bidMostAmount', 'offMostAmount'):
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out.groupby(level=0).last().sort_index()


def evaluate_series(bars, float_shares, flow):
    """一次算出整段的逐日入选判定,返回 DataFrame(索引同 bars)。

    回测的关键:整段只算一次。若每根K线都重算一遍全市场,
    一年 242 根K线就是 242 次全市场扫描,耗时相差两个数量级。

    各条件都是因果的(只用当日及之前的数据),所以整段一次算出的第 i 日取值,
    与只喂到第 i 日再算的结果完全相同 —— 不存在未来函数,
    由 tests/test_qmt_port.py 的截断一致性测试保证。
    """
    n = len(bars)
    idx = bars.index
    empty = pd.DataFrame({'triggered': np.zeros(n, dtype=bool)}, index=idx)
    if n < SLOW + 5:
        return empty

    d = bars.copy()
    d['ma_f'] = _ma(d['close'], FAST)
    d['ma_m'] = _ma(d['close'], MID)
    d['ma_s'] = _ma(d['close'], SLOW)

    active = _pattern_active(d, _find_sequences(d))
    turn = _ma_turn_up(d)
    vol_ok = _volume_surge_prev(d['volume'])

    # 资金流缺失时整列为 NaN,该日不触发 —— 不把缺数据当成「有流入」放过
    if (flow is None or len(flow) == 0
            or 'bidMostAmount' not in flow.columns
            or 'offMostAmount' not in flow.columns):
        inflow = pd.Series(np.nan, index=idx)
    else:
        inflow = (flow['bidMostAmount'].reindex(idx)
                  - flow['offMostAmount'].reindex(idx))

    pr = _profit_ratio(d, float_shares)

    out = pd.DataFrame({
        'close': d['close'], 'ma5': d['ma_f'], 'ma10': d['ma_m'], 'ma30': d['ma_s'],
        'volume': d['volume'], 'profit_ratio': pr, 'net_inflow': inflow,
    }, index=idx)
    out['triggered'] = (active & turn & vol_ok
                        & inflow.notna() & (inflow > 0)
                        & pr.notna() & (pr > PROFIT_MIN))
    return out


def evaluate_one(bars, float_shares, flow):
    """判断 bars 最后一根K线当日是否入选,返回 dict 或 None。

    供单只诊断使用;批量场景请用 evaluate_series,避免重复计算。
    """
    res = evaluate_series(bars, float_shares, flow)
    if not len(res) or not bool(res['triggered'].iloc[-1]):
        return None
    last = res.iloc[-1]
    return {
        'close': float(last['close']), 'ma5': float(last['ma5']),
        'ma10': float(last['ma10']), 'ma30': float(last['ma30']),
        'profit_ratio': float(last['profit_ratio']),
        'net_inflow': float(last['net_inflow']),
        'volume': float(last['volume']),
    }


# ------------------------------------------------------------------ QMT 入口
# 股票池存模块级变量,不挂到 C 上。
# QMT 里真正的上下文是 __PyContext(C++ 对象),不是 xtquant 的 qmttools.ContextInfo,
# 属性集不同,也未必允许挂自定义属性。凡是 C 上的东西一律防御性访问。
_UNIVERSE = []
_PICKS_BY_DAY = {}        # 回测缓存:{日期: [入选记录]},整段只算一次
_SCANNED = False
_FLOAT_SHARES = {}        # init 里顺手存下,避免扫描时再次调 get_instrument_detail
_FLOW_PERIOD = None       # 探测到的可用资金流周期;None=还没探过


def init(C):
    global _UNIVERSE, _PICKS_BY_DAY, _SCANNED, _FLOAT_SHARES, _FLOW_PERIOD
    _UNIVERSE = []
    _FLOW_PERIOD = None      # 重跑时重新探测,别沿用上一轮的周期
    _PICKS_BY_DAY = {}
    _FLOAT_SHARES = {}
    _SCANNED = False          # 重跑时必须清掉,否则沿用上一轮的结果
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
        try:
            fs = float(info.get('FloatVolume') or 0)
        except Exception:
            fs = 0.0
        if fs <= 0:
            continue                      # 无流通股本则筹码无从算起
        _FLOAT_SHARES[code] = fs
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
    """当前K线日期(YYYY-MM-DD);取不到返回空串。

    官方示例用的是全局函数 timetag_to_datetime(C.get_bar_timetag(C.barpos), fmt),
    它在 QMT 内置环境是内建的;本地跑测试时不存在,故降级到 pandas 解析。
    """
    try:
        tt = C.get_bar_timetag(C.barpos)
    except Exception:
        return ''
    try:
        return timetag_to_datetime(tt, '%Y-%m-%d')      # noqa: F821  QMT 内建
    except Exception:
        pass
    try:
        return pd.Timestamp(tt, unit='ms').strftime('%Y-%m-%d')
    except Exception:
        return ''


def _fetch_flows(C, chunk, verbose=False):
    """取一批股票的资金流,返回 {code: 日频DataFrame}。

    FLOW_PERIODS 里按顺序试,第一个真正取到 bidMostAmount/offMostAmount 的
    周期记进 _FLOW_PERIOD,之后只用它 —— 每批都把两个周期都试一遍太慢。
    两个都取不到就返回空字典,对应的股票一律不入选。
    """
    global _FLOW_PERIOD
    periods = (_FLOW_PERIOD,) if _FLOW_PERIOD else FLOW_PERIODS
    for period in periods:
        try:
            raw = C.get_market_data_ex(
                ['bidMostAmount', 'offMostAmount'], chunk,
                period=period,
                count=HISTORY_BARS * FLOW_BARS_PER_DAY.get(period, 1),
                fill_data=False, subscribe=False) or {}
        except Exception as e:
            if verbose:
                print('资金流周期 %s 取数失败:%s' % (period, e))
            continue
        flows = {}
        for c, v in raw.items():
            nv = _normalize_flow(v)
            if nv is not None:
                flows[c] = nv
        if flows:
            if _FLOW_PERIOD != period:
                _FLOW_PERIOD = period
                if verbose:
                    print('资金流数据源:%s(取到 %d/%d 只)'
                          % (period, len(flows), len(chunk)))
            return flows
        if verbose:
            print('资金流周期 %s 无数据,换下一个' % period)
    if verbose:
        print('资金流全部周期都取不到(transactioncount1d 需投研版,'
              'l2transactioncount 需Level2权限)—— 本批一律不入选')
    return {}


def _scan_all(C, codes, verbose=True):
    """对股票池整段扫描一次,返回 {日期字符串: [入选记录, ...]}。

    这是回测提速的关键。原先 handlebar 每根K线都重算一遍全市场,
    一年 242 根K线 = 242 次全市场扫描;现在整段只算一次,
    之后每根K线只做一次字典查表。实测差约两个数量级。

    分批打印各阶段耗时与预计剩余 —— 全市场要跑好几分钟,
    没有进度就分不清是在算还是卡死了。
    """
    import time
    by_day = {}
    t_start = time.time()
    n_hit = 0

    for i in range(0, len(codes), BATCH_SIZE):
        chunk = codes[i:i + BATCH_SIZE]
        t0 = time.time()
        try:
            # subscribe=False:官方文档「回测模型取本地数据遍历,不需要向服务器
            # 订阅实时行情,应使用 get_market_data_ex 并指定 subscribe 为 False」。
            # 订阅模式还有股票数量上限,全市场扫描必须关掉。
            data = C.get_market_data_ex(
                ['open', 'high', 'low', 'close', 'volume', 'amount'],
                chunk, period='1d', count=HISTORY_BARS,
                dividend_type='back', fill_data=False, subscribe=False) or {}
        except Exception as e:
            print('取行情失败:%s' % e)
            continue
        t_bars = time.time() - t0

        t0 = time.time()
        flows = _fetch_flows(C, chunk, verbose=verbose and i == 0)
        t_flow = time.time() - t0
        t0 = time.time()

        for code in chunk:
            bars = data.get(code)
            if bars is None or len(bars) < SLOW + 5:
                continue
            fs = _FLOAT_SHARES.get(code, 0.0)
            if fs <= 0:
                continue
            try:
                res = evaluate_series(bars, fs, flows.get(code))
                hits = res[res['triggered']]
                for ts, row in hits.iterrows():
                    day = pd.Timestamp(ts).strftime('%Y-%m-%d')
                    by_day.setdefault(day, []).append({
                        'code': code, 'close': float(row['close']),
                        'profit_ratio': float(row['profit_ratio']),
                        'net_inflow': float(row['net_inflow']),
                        'volume': float(row['volume']),
                    })
            except Exception as e:
                print('%s 计算失败:%s' % (code, e))
        t_calc = time.time() - t0

        if verbose:
            done = min(i + BATCH_SIZE, len(codes))
            elapsed = time.time() - t_start
            eta = elapsed / done * (len(codes) - done) if done else 0
            n_hit = sum(len(v) for v in by_day.values())
            print('  [%d/%d] 行情%.1fs 资金流%.1fs 计算%.1fs | 累计命中%d | '
                  '已用%.1fmin 预计还需%.1fmin'
                  % (done, len(codes), t_bars, t_flow, t_calc, n_hit,
                     elapsed / 60.0, eta / 60.0))
    return by_day


def handlebar(C):
    global _PICKS_BY_DAY, _SCANNED

    if not _should_scan(C):
        return

    day = _bar_date(C)
    codes = _UNIVERSE
    if not codes:
        print('股票池为空,请确认 init 是否正常执行、SECTOR 是否正确')
        return

    if SCAN_EVERY_BAR:
        # 回测:整段只扫一次,之后每根K线查表
        if not _SCANNED:
            if PROBE_FIRST and len(codes) > BATCH_SIZE:
                # 先跑一批,立刻给出全量耗时预估 —— 避免傻等几分钟才知道要跑多久
                import time
                t0 = time.time()
                _scan_all(C, codes[:BATCH_SIZE], verbose=False)
                per = (time.time() - t0) / BATCH_SIZE
                print('测速:%d 只用时 %.1fs,全量 %d 只预计约 %.1f 分钟'
                      % (BATCH_SIZE, per * BATCH_SIZE, len(codes),
                         per * len(codes) / 60.0))
            print('整段扫描中(只做一次)...')
            _PICKS_BY_DAY = _scan_all(C, codes)
            _SCANNED = True
            print('扫描完成,共 %d 个交易日有入选' % len(_PICKS_BY_DAY))
        picks = list(_PICKS_BY_DAY.get(day, []))
    else:
        # 实盘/盘后:只看最新一根K线,直接算
        picks = []
        scanned = _scan_all(C, codes)
        if day:
            picks = list(scanned.get(day, []))
        elif scanned:
            picks = list(scanned[max(scanned)])

    picks.sort(key=lambda x: x['profit_ratio'], reverse=True)
    print('=' * 60)
    print('%s 入选 %d 只' % (day, len(picks)))
    for p in picks[:PRINT_LIMIT]:
        print('  %s  收%.2f  获利筹码%.1f%%  净流入%.0f  量%.0f'
              % (p['code'], p['close'], p['profit_ratio'] * 100,
                 p['net_inflow'], p['volume']))
    if len(picks) > PRINT_LIMIT:
        print('  ...(其余 %d 只已省略)' % (len(picks) - PRINT_LIMIT))
