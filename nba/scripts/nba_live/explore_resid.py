"""Exploration (train only for decisions): does game state predict the outcome beyond the market mid?
Logit of home_win on logit(mid) + state features, cluster-robust by game, by remaining-time bucket."""
import numpy as np, pandas as pd, statsmodels.api as sm
from features import load_panel
p = load_panel(L=15, align_filter=True)
p = p[p.mid.between(0.03, 0.97) & ~p.ot].copy()
p['lm'] = np.log(p.mid / (1 - p.mid))
p['sq'] = np.sqrt(np.maximum(p.rem, 30) / 2880)
p['ms'] = p.m / p.sq / 10
p['mus'] = p.mu * p.sq / 10
p['lbm'] = np.log(p.p_bm / (1 - p.p_bm)).clip(-8, 8)
for sp in ['train', 'test']:
    for (lo, hi) in [(2160, 2880), (1440, 2160), (720, 1440), (360, 720), (120, 360), (0, 120)]:
        d = p[(p.split == sp) & (p.rem >= lo) & (p.rem < hi)]
        X = sm.add_constant(d[['lm', 'ms', 'mus']])
        m = sm.Logit(d.home_win, X).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.game_id)[0]})
        X2 = sm.add_constant(d[['lm', 'lbm']])
        m2 = sm.Logit(d.home_win, X2).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.game_id)[0]})
        f = lambda mm, k: f"{mm.params[k]:+.3f}({mm.bse[k]:.3f})"
        print(sp, f'{lo//60}-{hi//60}m', 'n', len(d), 'games', d.game_id.nunique(), '| lm', f(m, 'lm'), 'm/sqrt', f(m, 'ms'), 'mu*sqrt', f(m, 'mus'),
              '| lm', f(m2, 'lm'), 'logit p_bm', f(m2, 'lbm'))
