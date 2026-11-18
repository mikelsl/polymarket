"""H6b: in-play anchoring to the pre-game price. Residual (home_win - mid) vs (pre-game price - mid), by time bucket,
aligned games, clean rows (market stable, no known score in 30 s). OLS slope with game-clustered SE.
Slope > 0: the market moves too far from the pre-game price (in-play overreaction); slope < 0: too little."""
import numpy as np, pandas as pd, statsmodels.api as sm
from features import load_panel
from common import RL
p = load_panel(L=15, align_filter=True)
p = p[p.mid.between(0.03, 0.97) & ((p.mid - p.mid_m20).abs() <= 0.02) & (p.since_score >= 30) & ~p.ot].copy()
p['grp'] = np.where(p.season == '2024-25', 'tr24', np.where(p.split == 'train', 'tr25', 'test'))
p['gap'] = p.pre_ph - p.mid
p['res'] = p.home_win - p.mid
rows = []
for grp, d0 in p.groupby('grp'):
    for (lo, hi, nm) in [(2160, 2881, 'Q1'), (1440, 2160, 'Q2'), (720, 1440, 'Q3'), (0, 720, 'Q4')]:
        d = d0[(d0.rem >= lo) & (d0.rem < hi)]
        m = sm.OLS(d.res, sm.add_constant(d.gap)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.game_id)[0]})
        big = d[d.gap.abs() >= 0.25]
        rows.append(dict(grp=grp, q=nm, rows=len(d), games=d.game_id.nunique(), slope=m.params['gap'], se=m.bse['gap'],
                         big_rows=len(big), big_res_signed=(np.sign(big.gap) * big.res).mean()))
O = pd.DataFrame(rows); O.to_csv(RL / 'h6_anchor.csv', index=False)
print(O.round(3).to_string())
