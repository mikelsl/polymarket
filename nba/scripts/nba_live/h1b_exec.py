"""H1b: moneyline analogue of H5b (signal against the executable price). Panel (30 s), aligned games, ESPN info delay
15 s + game lag; fill = first print on our side at rt >= t + 2 s within 10 s; edge = p_emp(side) - fill. First trigger per
game with edge >= theta in the window -> buy at the fill, hold to resolution. Same grid as H5b."""
import numpy as np, pandas as pd
from features import load_panel, p_emp_rows, pnl_hold, pnl_gross_hold
from common import summarize, RL, DL
p = load_panel(L=15, align_filter=True)
p = p[~p.ot & (p.rem >= 120)].sort_values(['game_id', 't']).copy()
p['p_emp'] = p_emp_rows(p.m.values, p.elapsed.values, p.mu.values)
rows = []
for s in ('H', 'A'):
    px = p[f'buy{s}_px']; w = p[f'buy{s}_wait']
    pm = p.p_emp if s == 'H' else 1 - p.p_emp
    win = p.home_win if s == 'H' else 1 - p.home_win
    ok = px.between(0.03, 0.97) & (w <= 10)
    rows.append(pd.DataFrame({'game_id': p.game_id[ok], 'split': p.split[ok], 'season': p.season[ok], 't': p.t[ok], 'rem': p.rem[ok], 'px': px[ok], 'win': win[ok], 'edge': (pm - px)[ok]}))
E = pd.concat(rows, ignore_index=True).sort_values(['game_id', 't'])
E['grp'] = np.where(E.season == '2024-25', 'tr24', np.where(E.split == 'train', 'tr25', 'test'))
E['res'] = E.win - E.px
E['ebin'] = pd.cut(E.edge, [-1, -0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.2, 1]).astype(str)
D = summarize(E, 'res', by=['grp', 'ebin'], n_boot=300); D.to_csv(RL / 'h1b_resid_ml.csv', index=False)
print(D[['grp', 'ebin', 'n', 'games', 'pnl_mean', 'ci_lo', 'ci_hi']].round(4).to_string())
T = []
for th in (0.04, 0.06, 0.08, 0.10, 0.12):
    for wn, (lo, hi) in {'all(>2m)': (120, 2881), 'Q1-Q2': (1440, 2881), 'Q3-Q4(>2m)': (120, 1440)}.items():
        x = E[(E.edge >= th) & (E.rem >= lo) & (E.rem < hi)].groupby('game_id').head(1)
        T.append(x.assign(theta=th, window=wn, pnl=pnl_hold(x.px, x.win)))
T = pd.concat(T, ignore_index=True)
T.to_parquet(DL / 'h1b_trades_ml.parquet', index=False)
R = summarize(T, 'pnl', by=['theta', 'window', 'grp'], n_boot=500); R.to_csv(RL / 'h1b_rules_ml.csv', index=False)
print(R.pivot_table(index=['theta', 'window'], columns='grp', values='pnl_mean').round(2).to_string())
