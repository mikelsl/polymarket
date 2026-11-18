"""H5d: deep dive on the H5b spread result (state margin model vs executable spread price).
- train-selected variant (best train t-stat among the 15 H5b spread variants) and its test result
- characterisation of trades: side (fav token = negative handicap / dog token), leader/trailer, price band, quarter
- stability: test by season and month; train by month
- entry print size (USDC) for the depth check
"""
import numpy as np, pandas as pd
from common import DL, RL, ROOT, LAG, summarize
T = pd.read_parquet(DL / 'h5b_trades_spread.parquet')
def tstat(d):
    gs = d.groupby('game_id').pnl.agg(['sum', 'size']); mu = d.pnl.mean()
    se = np.sqrt(((gs['sum'] - gs['size'] * mu) ** 2).sum()) / len(d)
    return mu / se
V = []
for (th, wn), d in T.groupby(['theta', 'window']):
    tr = d[d.split == 'train']
    V.append(dict(theta=th, window=wn, n_tr=len(tr), mean_tr=tr.pnl.mean(), t_tr=tstat(tr)))
V = pd.DataFrame(V).sort_values('t_tr', ascending=False)
print(V.round(3).to_string())
best = V.iloc[0]
d = T[(T.theta == best.theta) & (T.window == best.window)].copy()
print('\nTRAIN-SELECTED:', best.theta, best.window)
print(pd.concat([summarize(d, 'pnl', by=['split'], n_boot=2000), summarize(d[d.split == 'test'], 'pnl', by=['season'], n_boot=2000)]).round(3).to_string())
print(summarize(d, 'pnl', by=['split', 'month'], n_boot=300)[['split', 'month', 'n', 'pnl_mean', 'ci_lo', 'ci_hi']].round(2).to_string())
# characterise all moderate-theta trades (theta 0.06, all(>2m))
x = T[(T.theta == 0.06) & (T.window == 'all(>2m)')].copy()
x['tok_is_fav'] = x.side == 0                       # token 0 carries the negative handicap (favourite side of the line)
team_home = np.where(x.side == 0, x.t0_home, ~x.t0_home.astype(bool))
x['team_margin'] = np.where(team_home, x.m_home, -x.m_home)
x['team_mu'] = np.where(team_home, x.mu_home, -x.mu_home)
x['state'] = np.where(x.team_margin > 0, 'leading', np.where(x.team_margin < 0, 'trailing', 'tied'))
x['pre'] = np.where(x.team_mu > 0, 'pre-fav', 'pre-dog')
x['pband'] = pd.cut(x.px, [0, 0.3, 0.45, 0.55, 0.7, 1]).astype(str)
x['q'] = pd.cut(x.rem, [0, 720, 1440, 2160, 2881], labels=['Q4', 'Q3', 'Q2', 'Q1']).astype(str)
for col in ['tok_is_fav', 'state', 'pre', 'pband', 'q']:
    print('\n', col); print(summarize(x, 'pnl', by=['split', col], n_boot=300)[['split', col, 'n', 'pnl_mean', 'ci_lo', 'ci_hi']].round(2).to_string())
# entry print size: reload the line prints and find the print at the fill
sel = pd.read_parquet(DL / 'lines_selected.parquet'); sel = sel[sel.sel_kind == 'spread'].set_index('game_id')
sizes = []
for r in x.itertuples(index=False):
    mid_ = sel.market_id.get(r.game_id)
    f1 = ROOT / 'data' / 'nba_data' / 'work' / 'trades_parts' / f'{mid_}.parquet'; f2 = DL / 'trades_lines' / f'{mid_}.parquet'
    f = f1 if f1.exists() else f2
    q = pd.read_parquet(f, columns=['ts', 'token_index', 'side', 'price', 'size'])
    rt = q.ts.values - LAG
    tok0 = q.token_index.values == 0; buy = q.side.values == 'BUY'
    ours = (tok0 == buy) if r.side == 0 else (tok0 != buy)
    w = ours & (rt >= r.t + 2) & (rt <= r.t + 40)
    usd = (q['size'].values * np.where(tok0, q.price.values, 1 - q.price.values))
    usd = usd if r.side == 0 else q['size'].values * np.where(tok0, 1 - q.price.values, q.price.values)
    first = np.flatnonzero(w)
    sizes.append((usd[first[0]] if len(first) else np.nan, usd[w].sum()))
x['fill_usd'] = [s[0] for s in sizes]; x['usd_40s'] = [s[1] for s in sizes]
print('\nentry print USDC: median %.1f, share >=10: %.2f, share >=100: %.2f; same-side USDC within 40 s: median %.1f, share >=100: %.2f' % (
    x.fill_usd.median(), (x.fill_usd >= 10).mean(), (x.fill_usd >= 100).mean(), x.usd_40s.median(), (x.usd_40s >= 100).mean()))
x.to_parquet(DL / 'h5d_trades_theta006_all.parquet', index=False)
