"""Honest summary: for each hypothesis family pick the variant with the best TRAIN mean P&L (n_train >= 100), then report
its untouched TEST result (and test by sub-period: 2026-02..06 = 2025-26 part, 2026-10..11 = 2026-27).
USDC per 10 USDC trade, 0.05 taker fee, bootstrap 95% CI clustered by game."""
import numpy as np
import pandas as pd
from common import DL, RL, summarize

fam = []
g = pd.read_parquet(DL / 'games.parquet')[['game_id', 'season']]


def add(name, T, keys, pnl_cols):
    T = T.copy()
    if 'season' not in T.columns:
        T = T.merge(g, on='game_id', how='left')
    if 'split' not in T.columns:
        T['split'] = np.where(T.grp == 'test', 'test', 'train')
    for pc in pnl_cols:
        for key, d in T.groupby(keys):
            tr, te = d[d.split == 'train'], d[d.split == 'test']
            if len(tr) < 100 or tr[pc].notna().sum() < 100:
                continue
            x = tr[tr[pc].notna()]
            gs = x.groupby('game_id')[pc].agg(['sum', 'size'])
            mu = x[pc].mean()
            se = np.sqrt(((gs['sum'] - gs['size'] * mu) ** 2).sum()) / len(x)
            fam.append(dict(family=name, variant=str(dict(zip(keys, key if isinstance(key, tuple) else (key,)))) + f' exit={pc}',
                            pc=pc, key=key, n_tr=int(len(x)), mean_tr=mu, t_tr=mu / se if se > 0 else 0, d=d))


add('H1 model-vs-market (L=15)', pd.read_parquet(DL / 'h1_trades.parquet'), ['model', 'theta', 'window'], ['pnl_hold', 'pnl_5m', 'pnl_10m'])
add('H2 price-bucket calibration', pd.read_parquet(DL / 'h2_trades.parquet'), ['rule'], ['pnl'])
add('H3 swings (follow/fade)', pd.read_parquet(DL / 'h3_events.parquet'), ['W', 'X', 'D', 'mode'], ['pnl_60', 'pnl_180', 'pnl_300', 'pnl_600', 'pnl_hold'])
add('H4 breaks', pd.read_parquet(DL / 'h4_breaks.parquet'), ['kind', 'side'], ['pnl_break', 'pnl_hold'])
add('H6c fade pre-game gap', pd.read_parquet(DL / 'h6_fade_gap_trades.parquet'), ['window', 'G'], ['pnl_hold', 'pnl_10m'])
for k in ('total', 'spread'):
    f = DL / f'h5_trades_{k}.parquet'
    if f.exists():
        add(f'H5 in-play {k}s', pd.read_parquet(f), ['theta', 'window'], ['pnl'])
for k in ('total', 'spread'):
    f = DL / f'h5b_trades_{k}.parquet'
    if f.exists():
        add(f'H5b in-play {k}s vs executable price', pd.read_parquet(f), ['theta', 'window'], ['pnl'])
f = DL / 'h5c_trades.parquet'
if f.exists():
    add('H5c spread vs ML-implied', pd.read_parquet(f), ['rule', 'theta', 'window'], ['pnl'])
f = DL / 'h1b_trades_ml.parquet'
if f.exists():
    add('H1b ML vs executable price', pd.read_parquet(f), ['theta', 'window'], ['pnl'])
F = pd.DataFrame(fam)
out = []
import sys
CRIT = sys.argv[1] if len(sys.argv) > 1 else 't_tr'
for name, d in F.groupby('family', sort=False):
    best = d.sort_values(CRIT, ascending=False).iloc[0]
    D = best.d; pc = best.pc
    tr, te = D[D.split == 'train'], D[D.split == 'test']
    s_tr = summarize(tr[tr[pc].notna()], pc, n_boot=2000); s_te = summarize(te[te[pc].notna()], pc, n_boot=2000)
    r = dict(family=name, n_variants=len(d), chosen=best.variant, n_train=s_tr.n[0], train_mean=s_tr.pnl_mean[0], train_lo=s_tr.ci_lo[0], train_hi=s_tr.ci_hi[0],
             n_test=s_te.n[0] if len(s_te) else 0, test_mean=s_te.pnl_mean[0] if len(s_te) else np.nan,
             test_lo=s_te.ci_lo[0] if len(s_te) else np.nan, test_hi=s_te.ci_hi[0] if len(s_te) else np.nan)
    for sea in ('2025-26', '2026-27'):
        x = te[(te.season == sea) & te[pc].notna()]
        r[f'test_{sea}_n'] = len(x); r[f'test_{sea}_mean'] = x[pc].mean() if len(x) else np.nan
    out.append(r)
O = pd.DataFrame(out)
O['criterion'] = CRIT
O.to_csv(RL / f'best_of_train_{CRIT}.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
print(O.round(3).to_string())
