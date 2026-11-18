"""H5c: spread line vs the moneyline (cross-market relative value), no team-strength model needed.

At each 30-s grid time t the moneyline home mid p_ml (prints, real time) gives the market's expected final home margin
M* = sd(t) * Phi^-1(p_ml), with sd(t) = residual SD of the final margin given the state (train fit, by remaining time,
h5_fit_spread.csv). ML-implied probability that token 0 of the decision-time spread line covers:
  home team0: Phi(Phi^-1(p_ml) + h0 / sd);  away team0: Phi(Phi^-1(1 - p_ml) + h0 / sd)   (h0 = token-0 handicap)
Edge = implied(side) - fill price (first spread print on our side within 10 s after t+2). Same rules as H5b (first trigger
per game, buy, hold, 0.05 fee). Optional arg 'state': also require the state-model edge (H5b) to agree in sign.
Outputs: results/nba_live/h5c_resid.csv, h5c_rules.csv, data/nba_live/h5c_trades.parquet
"""
import sys
import ast
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from common import DL, RL, summarize
from features import pnl_hold, pnl_gross_hold

SFX = sys.argv[1] if len(sys.argv) > 1 else ''
P = pd.read_parquet(DL / f'h5_panel_spread{SFX}.parquet')
ML = pd.read_parquet(DL / 'panel.parquet', columns=['game_id', 't', 'mid'])
ML = ML.rename(columns={'mid': 'p_ml'})
P = P.merge(ML, on=['game_id', 't'], how='left')
fit = pd.read_csv(RL / 'h5_fit_spread.csv')
edges = list(fit.bucket_rem_from) + [2881]
sds = fit.resid_sd.values
P = P[~P.ot & (P.rem >= 120) & P.p_ml.between(0.02, 0.98)].sort_values(['game_id', 't']).copy()
bk = np.clip(np.searchsorted(edges, P.rem.values, side='right') - 1, 0, len(sds) - 1)
sd = sds[bk]
z_home = ndtri(P.p_ml.values)
z_t0 = np.where(P.t0_home.values, z_home, -z_home)
P['p0_x'] = ndtr(z_t0 + P.line0.values / sd)
rows = []
for s in (0, 1):
    px = P[f'ask{s}']; w = P[f'wait{s}']
    imp = P.p0_x if s == 0 else 1 - P.p0_x
    st = P.pm if s == 0 else 1 - P.pm
    win = P.win0 if s == 0 else 1 - P.win0
    ok = px.between(0.03, 0.97) & (w <= 10)
    rows.append(pd.DataFrame({'game_id': P.game_id[ok], 'split': P.split[ok], 'season': P.season[ok], 'month': P.month[ok], 't': P.t[ok], 'rem': P.rem[ok],
                              'side': s, 'px': px[ok], 'win': win[ok], 'edge': (imp - px)[ok], 'edge_state': (st - px)[ok], 'p_ml': P.p_ml[ok]}))
E = pd.concat(rows, ignore_index=True).sort_values(['game_id', 't'])
E['res'] = E.win - E.px
E['ebin'] = pd.cut(E.edge, [-1, -0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.2, 1]).astype(str)
D = summarize(E, 'res', by=['split', 'ebin'], n_boot=300); D.to_csv(RL / f'h5c_resid{SFX}.csv', index=False)
pd.set_option('display.width', 250)
print(D[['split', 'ebin', 'n', 'games', 'pnl_mean', 'ci_lo', 'ci_hi']].round(4).to_string())
print('corr(edge_x, edge_state):', round(np.corrcoef(E.edge, E.edge_state)[0, 1], 3))
T = []
for both in (False, True):
    for th in (0.04, 0.06, 0.08, 0.10, 0.12):
        for wn, (lo, hi) in {'all(>2m)': (120, 2881), 'Q1-Q2': (1440, 2881), 'Q3-Q4(>2m)': (120, 1440)}.items():
            c = (E.edge >= th) & (E.rem >= lo) & (E.rem < hi)
            if both:
                c &= E.edge_state > 0
            x = E[c].groupby('game_id').head(1)
            T.append(x.assign(theta=th, window=wn, rule=('xmkt+state' if both else 'xmkt'), pnl=pnl_hold(x.px, x.win), gross=pnl_gross_hold(x.px, x.win)))
T = pd.concat(T, ignore_index=True)
T.drop(columns=['ebin']).to_parquet(DL / f'h5c_trades{SFX}.parquet', index=False)
R = summarize(T, 'pnl', by=['rule', 'theta', 'window', 'split'], n_boot=1000)
R = R.merge(T.groupby(['rule', 'theta', 'window', 'split']).agg(px=('px', 'mean'), win=('win', 'mean'), gross=('gross', 'mean')).reset_index(),
            on=['rule', 'theta', 'window', 'split'])
R.to_csv(RL / f'h5c_rules{SFX}.csv', index=False)
print(R[['rule', 'theta', 'window', 'split', 'n', 'px', 'win', 'gross', 'pnl_mean', 'ci_lo', 'ci_hi']].round(3).to_string())
