"""H5b: spreads/totals in play, signal against the EXECUTABLE price instead of the stale print mid.

At each 30-s grid time t (aligned games, ESPN info delay 15 s + game lag), for each side s of the decision-time line, the
fill is the first print on that side at rt >= t + 2 s; only fills within 10 s are used (proxy for the ask the bot would see
at t). Edge = model probability of side s - fill price. Rule: first time per game with edge >= theta inside the window ->
buy side s at that fill, hold to resolution (0.05 fee). theta grid {0.04, 0.06, 0.08, 0.10, 0.12} x window {all reg >2m,
Q1-Q2, Q3-Q4 >2m}. Train = 2025-26 games before 2026-02-01; test = 2026-02-01 .. 2026-11-02.
Also the descriptive residual (win - fill) by edge bin, all eligible rows.
Usage: h5b_exec.py spread|total
"""
import sys
import numpy as np
import pandas as pd
from common import DL, RL, summarize
from features import pnl_hold, pnl_gross_hold

KIND = sys.argv[1] if len(sys.argv) > 1 else 'spread'
SFX = sys.argv[2] if len(sys.argv) > 2 else ''          # e.g. _L30 (panel built with a longer ESPN info delay)
WMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0  # max seconds from t to the fill print
MODEL = sys.argv[4] if len(sys.argv) > 4 else 'pm'        # pm (fitted) or pm_naive
P = pd.read_parquet(DL / f'h5_panel_{KIND}{SFX}.parquet')
TAG = f'{KIND}{SFX}' + ('' if WMAX == 10 else f'_w{int(WMAX)}') + ('' if MODEL == 'pm' else '_naive')
P = P[~P.ot & (P.rem >= 120)].sort_values(['game_id', 't'])
rows = []
for s in (0, 1):
    px = P[f'ask{s}']; w = P[f'wait{s}']
    pm = P[MODEL] if s == 0 else 1 - P[MODEL]
    win = P.win0 if s == 0 else 1 - P.win0
    ok = px.between(0.03, 0.97) & (w <= WMAX) & pm.notna()
    rows.append(pd.DataFrame({'game_id': P.game_id[ok], 'split': P.split[ok], 'season': P.season[ok], 'month': P.month[ok], 't': P.t[ok], 'rem': P.rem[ok],
                              'side': s, 'px': px[ok], 'pm': pm[ok], 'win': win[ok], 'edge': (pm - px)[ok],
                              'm_home': P.m_home[ok], 't0_home': P.t0_home[ok], 'line0': P.line0[ok], 'mu_home': P.mu_home[ok]}))
E = pd.concat(rows, ignore_index=True).sort_values(['game_id', 't'])
E['res'] = E.win - E.px
E['ebin'] = pd.cut(E.edge, [-1, -0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.2, 1])
D = summarize(E, 'res', by=['split', 'ebin'], n_boot=300)
D.to_csv(RL / f'h5b_resid_{TAG}.csv', index=False)
pd.set_option('display.width', 250)
print(D[['split', 'ebin', 'n', 'games', 'pnl_mean', 'ci_lo', 'ci_hi']].round(4).to_string())
T = []
for th in (0.04, 0.06, 0.08, 0.10, 0.12):
    for wn, (lo, hi) in {'all(>2m)': (120, 2881), 'Q1-Q2': (1440, 2881), 'Q3-Q4(>2m)': (120, 1440)}.items():
        x = E[(E.edge >= th) & (E.rem >= lo) & (E.rem < hi)].groupby('game_id').head(1)
        T.append(x.assign(theta=th, window=wn, pnl=pnl_hold(x.px, x.win), gross=pnl_gross_hold(x.px, x.win)))
T = pd.concat(T, ignore_index=True)
T.drop(columns=['ebin']).to_parquet(DL / f'h5b_trades_{TAG}.parquet', index=False)
R = summarize(T, 'pnl', by=['theta', 'window', 'split'], n_boot=1000)
R = R.merge(T.groupby(['theta', 'window', 'split']).agg(px=('px', 'mean'), win=('win', 'mean'), gross=('gross', 'mean'), edge=('edge', 'mean')).reset_index(),
            on=['theta', 'window', 'split'])
R.to_csv(RL / f'h5b_rules_{TAG}.csv', index=False)
print(R[['theta', 'window', 'split', 'n', 'px', 'win', 'edge', 'gross', 'pnl_mean', 'ci_lo', 'ci_hi']].round(3).to_string())
