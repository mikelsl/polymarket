"""Build base tables for nba_live from data/nba_data (no network).

Outputs (data/nba_live/):
  games.parquet            core games + derived columns (home token, outcome, split, tip/end epoch, pre-game prices)
  prints_moneyline.parquet  ML taker prints in HOME terms: game_id, rt (real time = ts-LAG), d (+1 ask side), ph, size
  prints_spread.parquet / prints_total.parquet  main spread / total prints in token0 terms (p0, d)
  states.parquet           ESPN plays with fixed wallclock, in wallclock order, with the 'known' game state at each play
"""
import time
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from common import NBA, DL, LAG, SPLIT_DATE, to_epoch, game_slices, last_at_or_before

t0 = time.time()
g = pd.read_parquet(NBA / 'games.parquet')
c = g[g.is_core_game].copy()
c['home_tok'] = np.where(c.ml_outcome_0 == c.espn_home_name, 0, 1)
# sanity: outcome_1 should be the home name except the Twolves spelling
c['home_win'] = (c.ml_winner_index == c.home_tok).astype(int)
espn_home_win = (c.espn_home_score > c.espn_away_score).astype(int)
assert (c.home_win == espn_home_win).mean() > 0.999, (c.home_win != espn_home_win).sum()
c['split'] = np.where(c.game_date_et < SPLIT_DATE, 'train', 'test')
c['month'] = c.game_date_et.str[:7]
c['tip'] = to_epoch(c.espn_tipoff_utc_fixed)
c['sched'] = to_epoch(c.espn_sched_utc)
c['ml_closed_ts'] = to_epoch(c.ml_closed_time)
keep = ['game_id', 'slug', 'season', 'split', 'month', 'game_date_et', 'game_type', 'espn_event_id', 'home_tok', 'home_win',
        'espn_home_abbr', 'espn_away_abbr', 'espn_home_score', 'espn_away_score', 'espn_max_period', 'tip', 'sched',
        'ml_closed_ts', 'ml_market_id', 'ml_fee_type', 'main_spread_market_id', 'main_spread_team0', 'main_spread_handicap0',
        'main_spread_home_handicap', 'main_total_market_id', 'main_total_line', 'book_spread_home_close', 'book_total_close',
        'main_spread_minus_book', 'main_total_minus_book', 'flag_print_gap_in_play', 'wallclock_shift_suspect_s',
        'flag_wallclock_plays_shifted', 'flag_wallclock_plays_out_of_window', 'flag_final_score_pbp_mismatch',
        'flag_first_play_not_jumpball', 'espn_n_score_backsteps_scoring_wallclock', 'espn_neutral_site']
c = c[keep].reset_index(drop=True)
print('core games', len(c), round(time.time() - t0, 1), flush=True)

# ---------------- prints ----------------
dset = ds.dataset(NBA / 'trades', format='parquet', partitioning='hive')
tb = dset.to_table(columns=['game_id', 'market_id', 'market_role', 'ts', 'token_index', 'side', 'price', 'size'])
tr = tb.to_pandas()
tr = tr[tr.game_id.isin(set(c.game_id))]
print('prints loaded', len(tr), round(time.time() - t0, 1), flush=True)

ml = tr[tr.market_role == 'moneyline'].merge(c[['game_id', 'home_tok', 'ml_market_id']], on='game_id')
ml = ml[ml.market_id == ml.ml_market_id]
is_home_tok = ml.token_index.values == ml.home_tok.values
buy = ml.side.values == 'BUY'
ph = np.where(is_home_tok, ml.price.values, 1 - ml.price.values)
d = np.where(is_home_tok == buy, 1, -1).astype(np.int8)
P = pd.DataFrame({'game_id': ml.game_id.values, 'rt': ml.ts.values.astype(float) - LAG, 'd': d, 'ph': ph,
                  'size': ml['size'].values.astype(float)})
P = P.sort_values(['game_id', 'rt'], kind='mergesort').reset_index(drop=True)
P.to_parquet(DL / 'prints_moneyline.parquet', index=False)
print('ML prints', len(P), round(time.time() - t0, 1), flush=True)

for role, col in [('main_spread', 'main_spread_market_id'), ('main_total', 'main_total_market_id')]:
    x = tr[tr.market_role == role].merge(c[['game_id', col]], on='game_id')
    x = x[x.market_id == x[col]]
    buy = x.side.values == 'BUY'
    tok0 = x.token_index.values == 0
    p0 = np.where(tok0, x.price.values, 1 - x.price.values)
    dd = np.where(tok0 == buy, 1, -1).astype(np.int8)
    X = pd.DataFrame({'game_id': x.game_id.values, 'market_id': x.market_id.values, 'rt': x.ts.values.astype(float) - LAG,
                      'd': dd, 'p0': p0, 'size': x['size'].values.astype(float)})
    X = X.sort_values(['game_id', 'rt'], kind='mergesort').reset_index(drop=True)
    X.to_parquet(DL / f'prints_{role.split("_")[1]}.parquet', index=False)
    print(role, len(X), flush=True)
del tr, tb

# ---------------- pre-game market price ----------------
sl = game_slices(P)
pre_mid, pre_n, pre_stale = [], [], []
rt_all, d_all, ph_all = P.rt.values, P.d.values, P.ph.values
for gid, tip in zip(c.game_id, c.tip):
    if gid not in sl or np.isnan(tip):
        pre_mid.append(np.nan); pre_n.append(0); pre_stale.append(np.nan); continue
    a, b = sl[gid]
    rt, dd, pp = rt_all[a:b], d_all[a:b], ph_all[a:b]
    # mid of last ask-side and last bid-side print before tip (each within 15 min)
    ma = (dd == 1)
    ta, va = last_at_or_before(rt[ma], pp[ma], np.array([tip]), within=900)
    tb_, vb = last_at_or_before(rt[~ma], pp[~ma], np.array([tip]), within=900)
    w = (rt >= tip - 900) & (rt < tip)
    pre_n.append(int(w.sum()))
    if np.isnan(va[0]) or np.isnan(vb[0]):
        # fallback: size-weighted mean of prints in the last 30 min
        w2 = (rt >= tip - 1800) & (rt < tip)
        pre_mid.append(np.average(pp[w2], weights=P['size'].values[a:b][w2]) if w2.any() else np.nan)
        pre_stale.append(np.nan)
    else:
        pre_mid.append((va[0] + vb[0]) / 2)
        pre_stale.append(tip - min(ta[0], tb_[0]))
c['pre_ph'] = pre_mid
c['pre_n_prints_15m'] = pre_n
c['pre_stale_s'] = pre_stale
print('pre-game price missing', int(np.isnan(c.pre_ph).sum()), flush=True)

# ---------------- ESPN states ----------------
pbp = pd.read_parquet(NBA / 'espn_pbp.parquet', columns=['espn_event_id', 'seq', 'wallclock_utc', 'period', 'clock_s', 'elapsed_s',
                                                          'home_score', 'away_score', 'scoring_play', 'score_value', 'play_type_id',
                                                          'play_type', 'team_side', 'wallclock_shift_suspect_s'])
pbp = pbp[pbp.espn_event_id.isin(set(c.espn_event_id))].copy()
pbp['wc'] = to_epoch(pbp.wallclock_utc) - pbp.wallclock_shift_suspect_s.fillna(0).values
pbp = pbp[~np.isnan(pbp.wc)].copy()
pbp = pbp.sort_values(['espn_event_id', 'wc', 'elapsed_s', 'seq'], kind='mergesort').reset_index(drop=True)
# known state = play with max (elapsed_s, seq) seen so far in wallclock order
ord_key = pbp.elapsed_s.values.astype(float) * 1e5 + pbp.seq.values.astype(float)
out_idx = np.empty(len(pbp), dtype=np.int64)
gsl = game_slices(pbp, 'espn_event_id')
for e, (a, b) in gsl.items():
    k = ord_key[a:b]
    cm = np.maximum.accumulate(k)
    # index of the row achieving the running max
    is_new = k >= cm
    idx = np.where(is_new, np.arange(a, b), -1)
    idx = np.maximum.accumulate(idx)
    out_idx[a:b] = idx
for col in ['period', 'clock_s', 'elapsed_s', 'home_score', 'away_score']:
    pbp['k_' + col] = pbp[col].values[out_idx]
pbp = pbp.merge(c[['espn_event_id', 'game_id']], on='espn_event_id')
pbp = pbp.sort_values(['game_id', 'wc', 'elapsed_s', 'seq'], kind='mergesort').reset_index(drop=True)
cols = ['game_id', 'espn_event_id', 'seq', 'wc', 'period', 'clock_s', 'elapsed_s', 'home_score', 'away_score', 'scoring_play',
        'score_value', 'play_type_id', 'play_type', 'team_side', 'k_period', 'k_clock_s', 'k_elapsed_s', 'k_home_score', 'k_away_score']
pbp[cols].to_parquet(DL / 'states.parquet', index=False)
endt = pbp.groupby('game_id').wc.max()
c['end_wc'] = c.game_id.map(endt)
c.to_parquet(DL / 'games.parquet', index=False)
print('states', len(pbp), 'games with states', pbp.game_id.nunique(), round(time.time() - t0, 1), flush=True)
