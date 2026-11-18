"""Build the moneyline pre-game panel: one row per core game (2,750) -> data/nba_pre/ml_panel.parquet.

Columns (token 0 = Polymarket outcome 0, normally the away team; home_idx gives the home token):
- times: sched (ESPN scheduled start, else Polymarket gameStartTime), tip (ESPN first-play wallclock with the nba_data
  fix, else sched + 11 min), tip_source; first_print_ts (listing activity)
- mid0_m<k>: CLOB minute mid of token 0 at tip - k minutes (last point at or before, max age 10 min),
  k in OFFS; mid0_s0 = at the scheduled start (the live bot's entry moment), mid0_s60 = sched - 60 min
- pask0_m<k>, pbid0_m<k>: last print-revealed ask / bid of token 0 in the 10 min before tip - k
  (ask0 from taker BUY token0 or taker SELL token1 (1 - price); bid0 from SELL token0 / BUY token1)
- pmid0_h<k>: median print mid-proxy (ask - 0.5c / bid + 0.5c) of token 0 in the hour before tip - k hours (k=3,6,12,24)
- vol_sh_pre, vol_usdc_pre: all pre-tip prints; vol_usdc_60m, vol_usdc_15m, n_prints_15m; max_print_usdc_15m
- book (ESPN odds, no live rows; priority core_odds DraftKings > summary pickcenter DraftKings > core_odds ESPN BET):
  book_src, ml/spread/total open & close, pbook0_open / pbook0_close (multiplicative de-vig, token 0),
  pbook0_open_pow / pbook0_close_pow (power de-vig), vig_open / vig_close
- rest: home_rest / away_rest (days since the team's previous ESPN game incl. preseason; NaN = first game),
  home_b2b / away_b2b
"""
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from common_pre import DATA, SRC, american_to_prob, core_games, devig_mult, devig_power

OFFS = [1, 2, 5, 10, 15, 30, 60, 90, 120, 180, 240, 300]
t0 = time.time()
g = core_games()
g['tip_ts'] = g.tip.astype('int64') // 10**6
g['sched_ts'] = g.sched.astype('int64') // 10**6
keep = ['game_id', 'tip_ts', 'sched_ts']

# ---------------- minute mids ----------------
ph = pd.read_parquet(SRC / 'price_history', columns=['game_id', 'market_role', 'token_index', 't', 'p'],
                     filters=[('market_role', '=', 'moneyline')])
ph = ph[(ph.token_index == 0) & ph.game_id.isin(g.game_id)][['game_id', 't', 'p']].copy()
ph['game_id'] = ph.game_id.astype(str)
ph['p'] = ph.p.clip(0, 1)
# Single-minute spikes exist (thin 2024-25 books: e.g. 0.14 -> 0.56 -> 0.17); use a TRAILING 5-point median
# (causal, ~5 minutes) as the mid at time t.
ph = ph.sort_values(['game_id', 't'])
ph['p'] = ph.groupby('game_id', sort=False).p.transform(lambda s: s.rolling(5, min_periods=1).median())
ph = ph.sort_values('t')
q = []
for k in OFFS:
    q.append(g[keep].assign(label=f'm{k}', tq=g.tip_ts - 60 * k))
q.append(g[keep].assign(label='s0', tq=g.sched_ts))
q.append(g[keep].assign(label='s60', tq=g.sched_ts - 3600))
q = pd.concat(q, ignore_index=True)[['game_id', 'label', 'tq']].sort_values('tq')
q['game_id'] = q.game_id.astype(str)
mm = pd.merge_asof(q, ph, left_on='tq', right_on='t', by='game_id', direction='backward', tolerance=600)
w = mm.pivot(index='game_id', columns='label', values='p').add_prefix('mid0_')
g = g.merge(w, left_on='game_id', right_index=True, how='left')
print(f'minute mids done {time.time() - t0:.0f}s', flush=True)

# ---------------- prints ----------------
d = ds.dataset(SRC / 'trades', format='parquet', partitioning='hive')
tr = d.to_table(columns=['game_id', 'ts', 'token_index', 'side', 'price', 'size'],
                filter=(ds.field('market_role') == 'moneyline')).to_pandas()
tr['game_id'] = tr.game_id.astype(str)
tr = tr.merge(g[['game_id', 'tip_ts']], on='game_id')
tr = tr[tr.ts < tr.tip_ts].copy()
buy = tr.side.eq('BUY').to_numpy()
tok0 = tr.token_index.eq(0).to_numpy()
tr['is_ask0'] = (tok0 & buy) | (~tok0 & ~buy)
tr['px0'] = np.where(tok0, tr.price, 1 - tr.price)
tr['mid0p'] = np.where(tr.is_ask0, tr.px0 - 0.005, tr.px0 + 0.005)
tr['usdc'] = tr['size'] * tr.price
tr = tr.sort_values('ts')
print(f'pre-tip prints {len(tr):,} {time.time() - t0:.0f}s', flush=True)

fp = tr.groupby('game_id').ts.min().rename('first_print_ts')
g = g.merge(fp, left_on='game_id', right_index=True, how='left')
agg = tr.groupby('game_id').agg(vol_sh_pre=('size', 'sum'), vol_usdc_pre=('usdc', 'sum'), n_prints_pre=('ts', 'size'))
g = g.merge(agg, left_on='game_id', right_index=True, how='left')
for mins in (60, 15):
    x = tr[tr.ts >= tr.tip_ts - 60 * mins]
    a = x.groupby('game_id').agg(**{f'vol_usdc_{mins}m': ('usdc', 'sum'), f'n_prints_{mins}m': ('ts', 'size'),
                                    f'max_print_usdc_{mins}m': ('usdc', 'max')})
    g = g.merge(a, left_on='game_id', right_index=True, how='left')

qq = q[q.label.isin([f'm{k}' for k in OFFS])].copy()
for nm, sub in (('pask0', tr[tr.is_ask0]), ('pbid0', tr[~tr.is_ask0])):
    s = sub[['game_id', 'ts', 'px0']].rename(columns={'px0': nm})
    m2 = pd.merge_asof(qq, s, left_on='tq', right_on='ts', by='game_id', direction='backward', tolerance=600)
    g = g.merge(m2.pivot(index='game_id', columns='label', values=nm).add_prefix(nm + '_'), left_on='game_id',
                right_index=True, how='left')

for h in (3, 6, 12, 24):
    x = tr[['game_id', 'ts', 'tip_ts', 'mid0p']]
    x = x[(x.ts <= x.tip_ts - 3600 * h) & (x.ts > x.tip_ts - 3600 * (h + 1))]
    g = g.merge(x.groupby('game_id').mid0p.median().rename(f'pmid0_h{h}'), left_on='game_id', right_index=True,
                how='left')
print(f'print features done {time.time() - t0:.0f}s', flush=True)

# ---------------- sportsbook ----------------
pc = pd.read_parquet(SRC / 'pickcenter.parquet')
pc = pc[~pc.provider_name.str.contains('Live', na=False)].copy()
prio = {('core_odds', 'DraftKings'): 0, ('summary_pickcenter', 'DraftKings'): 1, ('core_odds', 'ESPN BET'): 2}
pc['prio'] = [prio.get((s, p), 9) for s, p in zip(pc.source, pc.provider_name)]
pc = pc[(pc.prio < 9) & pc.ml_home_close.notna()]
pc = pc.sort_values(['espn_event_id', 'prio']).drop_duplicates('espn_event_id')
bk = pc[['espn_event_id', 'source', 'provider_name', 'ml_home_open', 'ml_away_open', 'ml_home_close', 'ml_away_close',
         'spread_home_open', 'spread_home_close', 'total_open', 'total_close', 'spread_home_close_odds',
         'spread_away_close_odds', 'over_close_odds', 'under_close_odds', 'spread_home_open_odds',
         'spread_away_open_odds', 'over_open_odds', 'under_open_odds']].copy()
bk['book_src'] = bk.source + '/' + bk.provider_name
for w_ in ('open', 'close'):
    ih, ia = american_to_prob(bk[f'ml_home_{w_}']), american_to_prob(bk[f'ml_away_{w_}'])
    bk[f'pbook_home_{w_}'] = devig_mult(ih, ia)
    bk[f'pbook_home_{w_}_pow'] = devig_power(ih, ia)
    bk[f'vig_{w_}'] = ih + ia - 1
bk = bk.drop(columns=['source', 'provider_name'])
g = g.merge(bk, on='espn_event_id', how='left')
for w_ in ('open', 'close'):
    for suf in ('', '_pow'):
        ph_ = g[f'pbook_home_{w_}{suf}']
        g[f'pbook0_{w_}{suf}'] = np.where(g.home_idx == 0, ph_, 1 - ph_)

# ---------------- rest days ----------------
e = pd.read_parquet(SRC / 'work' / 'espn_games.parquet')
e = e[e.completed.fillna(False) & (e.comp_type.fillna('') != 'ALLSTAR')]
long = pd.concat([e[['espn_event_id', 'espn_date_et', 'home_abbr']].rename(columns={'home_abbr': 'team'}),
                  e[['espn_event_id', 'espn_date_et', 'away_abbr']].rename(columns={'away_abbr': 'team'})])
long['d'] = pd.to_datetime(long.espn_date_et)
long = long.sort_values(['team', 'd'])
long['rest'] = long.groupby('team').d.diff().dt.days
long.loc[long.rest > 30, 'rest'] = np.nan
r = long.set_index(['espn_event_id', 'team']).rest
g['home_rest'] = [r.get((i, t), np.nan) for i, t in zip(g.espn_event_id, g.espn_home_abbr)]
g['away_rest'] = [r.get((i, t), np.nan) for i, t in zip(g.espn_event_id, g.espn_away_abbr)]
g['home_b2b'] = g.home_rest.eq(1)
g['away_b2b'] = g.away_rest.eq(1)

cols = [c for c in g.columns if c not in ('sibling_roles',)]
g[cols].to_parquet(DATA / 'ml_panel.parquet', index=False)
print(f'saved {len(g)} rows, {time.time() - t0:.0f}s', flush=True)
print(g[['mid0_m1', 'mid0_m5', 'mid0_s0', 'pask0_m5', 'pbid0_m5', 'pmid0_h24', 'pbook0_close', 'pbook0_open']]
      .notna().groupby(g.season).sum().to_string())
