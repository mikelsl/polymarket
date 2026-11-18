"""Depth proxy at simulated entries: for panel decision rows (every 30 s, in play), the first ask-side moneyline print at
rt >= t + 2 s (our fill), its notional (USDC), and the ask-side notional printed within 30 s after it at <= fill price + 1c.
By season x time bucket x price band. If >= 100 USDC printed at <= +1c within 30 s, a 100 USDC taker order would likely
have filled within 1c (lower bound on displayed depth; the book itself is not observed before 2026-10-29)."""
import numpy as np, pandas as pd
from features import load_panel
from common import RL, load_prints, game_slices
p = load_panel(L=15, align_filter=False)
p = p[p.buyH_px.notna() & p.mid.between(0.02, 0.98)].sample(frac=0.25, random_state=1).sort_values(['game_id', 't'])
P = load_prints(); sl = game_slices(P)
rows = []
for gid, d in p.groupby('game_id'):
    if gid not in sl: continue
    a, b = sl[gid]
    rt, ph, dd, sz = P.rt.values[a:b], P.ph.values[a:b], P.d.values[a:b], P['size'].values[a:b]
    A = dd == 1
    At, Ap, As = rt[A], ph[A], sz[A]
    cum = np.r_[0, np.cumsum(As * Ap)]
    for t, px, wait in zip(d.t.values, d.buyH_px.values, d.buyH_wait.values):
        i = np.searchsorted(At, t + 2, side='left')
        if i >= len(At): continue
        j = np.searchsorted(At, At[i] + 30, side='right')
        sel = Ap[i:j] <= Ap[i] + 0.01 + 1e-9
        rows.append((gid, t, Ap[i], As[i] * Ap[i], float((As[i:j][sel] * Ap[i:j][sel]).sum())))
D = pd.DataFrame(rows, columns=['game_id', 't', 'px', 'fill_notional', 'notional_30s_1c'])
D = D.merge(p[['game_id', 't', 'season', 'tb']], on=['game_id', 't'])
D['band'] = pd.cut(D.px, [0, 0.1, 0.3, 0.7, 0.9, 0.97, 1.0])
def agg(x):
    return pd.Series({'n': len(x), 'fill_med_usdc': x.fill_notional.median(), 'share_fill_ge10': (x.fill_notional >= 10).mean(),
                      'n30_med_usdc': x.notional_30s_1c.median(), 'share_30s_ge100': (x.notional_30s_1c >= 100).mean(),
                      'share_30s_ge1000': (x.notional_30s_1c >= 1000).mean()})
O1 = D.groupby(['season', 'tb']).apply(agg).reset_index()
O2 = D.groupby(['season', 'band'], observed=True).apply(agg).reset_index()
O1.to_csv(RL / 'depth_by_time.csv', index=False); O2.to_csv(RL / 'depth_by_price.csv', index=False)
pd.set_option('display.width', 200)
print(O1.round(3).to_string()); print(O2.round(3).to_string())
