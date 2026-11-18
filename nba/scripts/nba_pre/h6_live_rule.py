"""H6 (side deliverable) - replay of the CURRENT live entry rule on every 2025-26 and 2026-27 core game, using only
information available at the scheduled start, with the current exit.

Live bot (tools/nba-live-bot.mjs, unit nba-live-bot-20261104: --max-rank 5 --max-spread 0.03 --max-line 10
--min-bid 0.10 --min-price 0.10 --max-ask 0.45 --limit-window-s 420 --limit-offset 0.01 --entry-grace-s 600
--q4-exit 0.50 --stop-bid 0.20):
  at the scheduled start rank the event's spread markets by CURRENT volume; walk ranks 1..5 keeping markets with a
  two-sided book of spread <= 3c until 4 are kept; drop |line| > 10; candidate tokens: bid > 0.10, ask - bid <= 0.03,
  ask <= 0.45, mid >= 0.10; buy the lowest-mid candidate (post-only at ask - 1c for 7 min, then taker; skip after
  10 min). Exit: from Q3 (ESPN period >= 3) sell everything when bid <= 0.20; from Q4 sell everything when
  bid >= 0.50; otherwise hold to settlement.
Replay
  volume at the scheduled start = sum of print shares up to then (Gamma volume = print shares); book proxy: a CLOB
  minute mid exists (trailing 5-point median) and is inside (0.01, 0.99); bid / ask = next tick below / above the mid
  (pre-game spreads are 1 tick ~90% of the time); eligibility re-checked at sched + 0, 2.5, 5, 7.5, 10 min.
  Entry E1 (main, conservative): taker at the ask at the first eligible check. Entry E2 (optimistic live-like):
  post-only at ask - 1c, filled if within 7 min a print trades through that price (queue ignored), else taker at the
  ask 7 min later.
  Exit on the token's print-revealed bid (taker SELL of our token at b, or taker BUY of the other token at 1 - b);
  Q3 / Q4 start = first ESPN play of the period (wallclock fix applied), fallback sched + 95 / 140 min.
  Fee 0.05 p (1-p) on every taker leg (entry taker, both exits); 10 USDC per trade.
Outputs: h6_trades.csv (one row per game), h6_summary.csv
"""
import numpy as np
import pandas as pd

from common_pre import DATA, RES, SRC, STAKE, ask_of_mid, bid_of_mid, boot, core_games, fee

g = core_games()
g = g[g.season.isin(['2025-26', '2026-27'])].copy()
g['s0'] = g.sched.astype('int64') // 10**6
g['tip_ts'] = g.tip.astype('int64') // 10**6
m = pd.read_parquet(SRC / 'markets.parquet')
m = m[(m.kind == 'spreads') & m.event_role.isin(['main', 'v2']) & m.game_id.isin(g.game_id)].copy()
m = m.merge(g[['game_id', 's0', 'tip_ts']], on='game_id')

# ---------------- volume and mids at the decision times ----------------
st = pd.read_parquet(DATA / 'spread_trades.parquet')
st = st[st.market_id.isin(m.market_id)]
st = st.merge(m[['market_id', 's0']], on='market_id')
vol0 = st[st.ts <= st.s0].groupby('market_id')['size'].sum()
m['vol_s0'] = m.market_id.map(vol0).fillna(0.0)
lp = pd.read_parquet(DATA / 'line_prices.parquet')
lp = lp[lp.market_id.isin(m.market_id)].sort_values(['market_id', 't'])
lp['p0'] = lp.groupby('market_id', sort=False).p0.transform(lambda s: s.rolling(5, min_periods=1).median())
lp = lp.sort_values('t')
CHECKS = [0, 150, 300, 450, 600, 420]          # last one: the end of the 7-min limit window (E2 taker fallback)
q = pd.concat([pd.DataFrame({'market_id': m.market_id.values, 'k': k, 'tq': (m.s0 + k).values}) for k in CHECKS])
q = q.drop_duplicates(['market_id', 'k']).sort_values('tq')
mm = pd.merge_asof(q, lp, left_on='tq', right_on='t', by='market_id', direction='backward', tolerance=900)
mids = mm.pivot(index='market_id', columns='k', values='p0')

# ---------------- ESPN period starts ----------------
pbp = pd.read_parquet(SRC / 'espn_pbp.parquet', columns=['espn_event_id', 'period', 'wallclock_utc',
                                                          'wallclock_shift_suspect_s'])
pbp = pbp[pbp.espn_event_id.isin(g.espn_event_id) & pbp.period.isin([3, 4]) & pbp.wallclock_utc.notna()]
pbp['wc'] = pbp.wallclock_utc.astype('int64') // 10**6 - pbp.wallclock_shift_suspect_s.fillna(0).astype('int64')
ps = pbp.groupby(['espn_event_id', 'period']).wc.min().unstack()
g['t3'] = g.espn_event_id.map(ps.get(3)).fillna(g.s0 + 95 * 60)
g['t4'] = g.espn_event_id.map(ps.get(4)).fillna(g.s0 + 140 * 60)
g['t3_src'] = np.where(g.espn_event_id.map(ps.get(3)).notna(), 'espn', 'fallback')

stg = {k: v for k, v in st.groupby('market_id')}
rows = []
for gid, lines in m.groupby('game_id'):
    gg = g.loc[g.game_id == gid].iloc[0]
    lines = lines.sort_values(['vol_s0', 'market_slug'], ascending=[False, True]).reset_index(drop=True)
    kept = []
    for i in range(min(len(lines), 5)):
        if len(kept) == 4:
            break
        r = lines.loc[i]
        mid = mids.at[r.market_id, 0] if r.market_id in mids.index else np.nan
        if pd.notna(mid) and 0.01 < mid < 0.99 and r.vol_s0 > 0:
            kept.append((i + 1, r))
    kept = [(rk, r) for rk, r in kept if abs(r.handicap0) <= 10]
    rec = dict(game_id=gid, slug=gg.slug, date=gg.game_date_et, season=gg.season, split=gg.split, month=gg.month,
               n_lines=len(lines), n_kept=len(kept), gap_flag=bool(gg.flag_print_gap_in_play))
    pick = None
    for k in CHECKS[:5]:
        if gg.s0 + k >= gg.tip_ts:                      # never after the real tip
            break
        cands = []
        for rk, r in kept:
            mid0 = mids.at[r.market_id, k] if r.market_id in mids.index else np.nan
            if pd.isna(mid0):
                continue
            for ti in (0, 1):
                mid = mid0 if ti == 0 else 1 - mid0
                a, b = float(ask_of_mid(mid)), float(bid_of_mid(mid))
                if b > 0.10 and a - b <= 0.03 + 1e-9 and a <= 0.45 + 1e-9 and mid >= 0.10:
                    cands.append((mid, rk, ti, a, b, r))
        if cands:
            cands.sort(key=lambda c: (c[0], c[1]))
            pick = (k,) + cands[0]
            break
    if pick is None:
        rows.append(rec | dict(traded=False))
        continue
    k, mid, rk, ti, a, b, r = pick
    y = r.winner_index
    y = np.nan if pd.isna(y) or int(y) not in (0, 1) else float(int(y) == ti)
    t_in = gg.s0 + k
    rec.update(traded=True, check_s=k, rank=rk, market_id=r.market_id, question=r.question,
               outcome=r[f'outcome_{ti}'], handicap0=r.handicap0, token_index=ti, mid=mid, ask=a, bid=b, y=y)
    tr = stg.get(r.market_id, pd.DataFrame(columns=st.columns))
    # E2: post-only at ask - 1c for 7 min (filled when a print trades at or through it), else taker 7 min later
    lim = round(a - 0.01, 2)
    w = tr[(tr.ts > t_in) & (tr.ts <= t_in + 420)]
    hit = w[((w.token_index == ti) & (w.side == 'SELL') & (w.price <= lim + 1e-9)) |
            ((w.token_index != ti) & (w.side == 'BUY') & (w.price >= 1 - lim - 1e-9))]
    if len(hit):
        e2_price, e2_fee, e2_t = lim, 0.0, int(hit.ts.iloc[0])
    else:
        mid7 = mids.at[r.market_id, 420] if r.market_id in mids.index else np.nan
        mid7 = mid if pd.isna(mid7) else (mid7 if ti == 0 else 1 - mid7)
        a7 = float(ask_of_mid(mid7))
        e2_price, e2_fee, e2_t = (a7, fee(a7), t_in + 420) if a7 <= 0.47 + 1e-9 else (np.nan, np.nan, np.nan)
    rec.update(e2_price=e2_price, e2_maker=bool(len(hit)), e2_t=e2_t)
    # exits on the print-revealed bid
    ing = tr[tr.ts > t_in]
    isb = ((ing.token_index == ti) & (ing.side == 'SELL')) | ((ing.token_index != ti) & (ing.side == 'BUY'))
    bp = ing[isb].assign(bid=lambda d: np.where(d.token_index == ti, d.price, 1 - d.price))
    st_ = bp[(bp.ts >= gg.t3) & (bp.bid <= 0.20 + 1e-9)]
    tp_ = bp[(bp.ts >= gg.t4) & (bp.bid >= 0.50 - 1e-9)]
    ts_stop = st_.ts.iloc[0] if len(st_) else np.inf
    ts_tp = tp_.ts.iloc[0] if len(tp_) else np.inf
    if ts_stop == np.inf and ts_tp == np.inf:
        ex = ('settle', np.nan, y)
    elif ts_stop < ts_tp:
        ex = ('stop', ts_stop, float(st_.bid.iloc[0]))
    else:
        ex = ('q4tp', ts_tp, float(tp_.bid.iloc[0]))
    # optimistic stop fill: if the bid was seen above 0.20 after Q3 started and before the trigger, the 2-s poller
    # would have sold near the 0.20 level -> fill at max(trigger print, 0.19); else the trigger print itself
    if len(st_):
        before = bp[(bp.ts >= gg.t3) & (bp.ts < st_.ts.iloc[0])]
        opt = max(float(st_.bid.iloc[0]), 0.19) if (before.bid > 0.20).any() else float(st_.bid.iloc[0])
    else:
        opt = np.nan
    rec.update(stop_price_opt=opt)
    rec.update(stop_price=float(st_.bid.iloc[0]) if len(st_) else np.nan,
               tp_price=float(tp_.bid.iloc[0]) if len(tp_) else np.nan)
    rec.update(exit=ex[0], exit_ts=ex[1], exit_price=ex[2], t3_src=gg.t3_src,
               max_bid_after_t4=float(bp[bp.ts >= gg.t4].bid.max()) if len(bp[bp.ts >= gg.t4]) else np.nan,
               min_bid_after_t3=float(bp[bp.ts >= gg.t3].bid.min()) if len(bp[bp.ts >= gg.t3]) else np.nan)
    rows.append(rec)

T = pd.DataFrame(rows)
tt = T[T.traded & T.y.notna()].copy()


def pnl(entry, entry_fee_ps, exit_kind, exit_price, y):
    sh = STAKE / entry
    cost = STAKE + sh * entry_fee_ps
    proceeds = np.where(exit_kind == 'settle', sh * y, sh * (exit_price - fee(exit_price)))
    return proceeds - cost, cost, np.where(exit_kind == 'settle', sh * y, sh * exit_price) - STAKE


tt['pnl_hold'], tt['cost_hold'], tt['gross_hold'] = pnl(tt.ask, fee(tt.ask), np.array(['settle'] * len(tt)), 0, tt.y)
tt['pnl_cur'], tt['cost_cur'], tt['gross_cur'] = pnl(tt.ask, fee(tt.ask), tt.exit.values, tt.exit_price.fillna(0), tt.y)
for v, col in (('stoponly', 'stop_price'), ('tponly', 'tp_price')):
    kind = np.where(tt[col].notna(), 'x', 'settle')
    tt[f'pnl_{v}'], tt[f'cost_{v}'], tt[f'gross_{v}'] = pnl(tt.ask, fee(tt.ask), kind, tt[col].fillna(0), tt.y)
xp = np.where(tt.exit == 'stop', tt.stop_price_opt, tt.exit_price)
tt['pnl_curopt'], tt['cost_curopt'], tt['gross_curopt'] = pnl(tt.ask, fee(tt.ask), tt.exit.values,
                                                               np.nan_to_num(xp), tt.y)
e2 = tt.e2_price.notna()
fe2 = np.where(tt.e2_maker, 0.0, fee(tt.e2_price.fillna(0.5)))
p2, c2, g2 = pnl(tt.e2_price.fillna(0.5), fe2, tt.exit.values, tt.exit_price.fillna(0), tt.y)
tt['pnl_e2cur'] = np.where(e2, p2, np.nan)
tt['cost_e2cur'] = np.where(e2, c2, np.nan)
T = T.merge(tt[['game_id', 'pnl_hold', 'cost_hold', 'gross_hold', 'pnl_cur', 'cost_cur', 'gross_cur', 'pnl_e2cur',
                'cost_e2cur', 'pnl_stoponly', 'pnl_tponly', 'pnl_curopt']], on='game_id', how='left')
T.to_csv(RES / 'h6_trades.csv', index=False)

out = []
for by in (None, 'split', 'season', 'month', 'exit'):
    groups = [('all', tt)] if by is None else list(tt.groupby(by))
    for k, d in groups:
        for v in ('hold', 'cur', 'curopt', 'e2cur', 'stoponly', 'tponly'):
            dd = d.dropna(subset=[f'pnl_{v}'])
            if not len(dd):
                continue
            b = boot(dd, f'pnl_{v}', f'cost_{v}', B=1000)
            out.append(dict(by=by or 'all', level=k, variant=v, n=len(dd), win=dd.y.mean(), avg_ask=dd.ask.mean(),
                            pnl=b['pnl'], roi=b['roi'], roi_lo=b['roi_lo'], roi_hi=b['roi_hi'],
                            pnl_lo=b['pnl_lo'], pnl_hi=b['pnl_hi'],
                            gross=dd[f'gross_{v}'].sum() if f'gross_{v}' in dd else np.nan))
S = pd.DataFrame(out)
S.round(4).to_csv(RES / 'h6_summary.csv', index=False)
print(f"games {len(T)}, traded {int(T.traded.sum())}, with outcome {len(tt)}; exits {tt.exit.value_counts().to_dict()}")
print(S[S.by.isin(['all', 'split', 'season'])].round(3).to_string(index=False))
