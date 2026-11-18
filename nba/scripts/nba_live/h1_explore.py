"""H1 exploration: market vs models (log loss) and outcome residual by model-minus-market deviation."""
import numpy as np, pandas as pd
from features import load_panel, p_emp_rows
from common import summarize, RL
p = load_panel(L=15, align_filter=True)
p = p[p.mid.between(0.02, 0.98)].copy()
p['p_emp'] = p_emp_rows(p.m.values, p.elapsed.values, p.mu.values)
ll = lambda q, y: -(y * np.log(np.clip(q, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(q, 1e-4, 1 - 1e-4)))
p['grp'] = np.where(p.season == '2024-25', '2024-25(train)', np.where(p.split == 'train', '2025-26(train)', 'test'))
for gname, d in p.groupby('grp'):
    print(gname, 'rows', len(d), 'games', d.game_id.nunique(), 'LL mid %.4f emp %.4f bm %.4f' % (ll(d.mid, d.home_win).mean(), ll(d.p_emp, d.home_win).mean(), ll(d.p_bm, d.home_win).mean()))
p['dev'] = p.p_emp - p.mid
p['res'] = p.home_win - p.mid
p['dbin'] = pd.cut(p.dev, [-1, -0.15, -0.08, -0.04, -0.02, 0.02, 0.04, 0.08, 0.15, 1])
p['stable'] = (p.mid - p.mid_m20).abs() <= 0.02
out = []
for (gname, db), d in p[p.stable & (p.since_score >= 30)].groupby(['grp', 'dbin'], observed=True):
    s = summarize(d, 'res', n_boot=300)
    out.append(dict(grp=gname, dbin=str(db), rows=len(d), games=d.game_id.nunique(), dev=d.dev.mean(), res=s.pnl_mean[0], lo=s.ci_lo[0], hi=s.ci_hi[0]))
O = pd.DataFrame(out); O.to_csv(RL / 'h1_explore_devbins.csv', index=False)
print(O.round(4).to_string())
