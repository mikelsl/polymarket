"""Per-game check of ESPN wallclock alignment against market price moves (moneyline prints, real time rt=ts-4 s).

For each core game: a simple win-probability series from ESPN states (Brownian margin model, sigma 12.34, pre-game drift
from the pre-tip price) and the market price series (last print, either side) on a 2-s grid from tip to end.
Cross-correlate 10-s changes for lags -180..+180 s (market later = positive lag) plus +-3600 s (DST) and +-86400 s;
the best lag and the peak correlation are written to data/nba_live/alignment.parquet.
"""
import numpy as np
import pandas as pd
from common import DL, RL, load_games, load_prints, load_states, game_slices, norm_cdf, norm_ppf

SIG = 12.34
STEP = 2.0
LAGS = np.arange(-180, 181, 2)

g = load_games()
P = load_prints()
S = load_states()
psl, ssl = game_slices(P), game_slices(S)


def model_p(margin, elapsed, mu):
    tau = np.clip((2880.0 - elapsed) / 2880.0, 0, None)
    # OT: remaining OT time, 300 s periods
    ot = elapsed >= 2880
    ot_rem = np.where(ot, (300 - ((elapsed - 2880) % 300)) / 2880.0, 0)
    tau = np.where(ot, ot_rem, tau)
    sd = SIG * np.sqrt(np.maximum(tau, 1e-6))
    return norm_cdf((margin + 0.5 * 0 + mu * tau) / sd)


rows = []
for _, r in g.iterrows():
    gid = r.game_id
    if gid not in psl or gid not in ssl or np.isnan(r.tip) or np.isnan(r.pre_ph):
        rows.append(dict(game_id=gid, lag=np.nan, peak=np.nan)); continue
    a, b = psl[gid]
    rt, ph, dd = P.rt.values[a:b], P.ph.values[a:b], P.d.values[a:b]
    rta, pha, rtb, phb = rt[dd == 1], ph[dd == 1], rt[dd == -1], ph[dd == -1]
    a2, b2 = ssl[gid]
    wc = S.wc.values[a2:b2]
    margin = (S.k_home_score.values[a2:b2] - S.k_away_score.values[a2:b2]).astype(float)
    el = S.k_elapsed_s.values[a2:b2].astype(float)
    mu = SIG * norm_ppf(np.clip(r.pre_ph, 0.02, 0.98))
    pm = model_p(margin, el, mu)
    t0, t1 = r.tip, r.end_wc
    grid = np.arange(t0, t1, STEP)
    if len(grid) < 600:
        rows.append(dict(game_id=gid, lag=np.nan, peak=np.nan)); continue
    ix = np.searchsorted(wc, grid, side='right') - 1
    x = np.where(ix >= 0, pm[np.clip(ix, 0, None)], np.nan)

    def mkt(shift):
        # last print (either side): reacts at the first print after an event (a two-sided mid lags)
        iy = np.searchsorted(rt, grid + shift, side='right') - 1
        return np.where(iy >= 0, ph[np.clip(iy, 0, None)], np.nan)

    k = int(10 / STEP)
    dx = x[k:] - x[:-k]
    dx = np.nan_to_num(dx)
    best, bestc, prof = np.nan, -9, {}
    for L in list(LAGS) + [-3600, 3600, -86400, 86400]:
        y = mkt(L)
        dy = np.nan_to_num(y[k:] - y[:-k])
        if dx.std() == 0 or dy.std() == 0:
            cc = np.nan
        else:
            cc = np.corrcoef(dx, dy)[0, 1]
        prof[L] = cc
        if cc > bestc:
            bestc, best = cc, L
    # per-period lag (periods 1-4, OT pooled as 5)
    per_at = S.k_period.values[a2:b2]
    pg = np.where(ix >= 0, per_at[np.clip(ix, 0, None)], 0)[k:]
    pg = np.where(pg <= 2, 1, 2)   # halves: periods 1-2 and 3+ (OT included)
    Ys = {L: np.nan_to_num(mkt(L)[k:] - mkt(L)[:-k]) for L in LAGS}
    per_res = {}
    for q in (1, 2):
        msk = pg == q
        if msk.sum() < 100 or dx[msk].std() == 0:
            per_res[q] = (np.nan, np.nan); continue
        bq, bc = np.nan, -9
        for L in LAGS:
            yy = Ys[L][msk]
            if yy.std() == 0: continue
            cq = np.corrcoef(dx[msk], yy)[0, 1]
            if cq > bc: bc, bq = cq, L
        per_res[q] = (bq, bc)
    in_rng = {L: v for L, v in prof.items() if abs(L) <= 180}
    bl = max(in_rng, key=lambda L: -9 if np.isnan(in_rng[L]) else in_rng[L])
    rows.append(dict(game_id=gid, lag=best, peak=bestc, lag_local=bl, peak_local=in_rng[bl], c0=prof[0], c10=prof[10],
                     c_m3600=prof[-3600], c_p3600=prof[3600], c_m86400=prof[-86400], c_p86400=prof[86400],
                     **{f'lag_q{q}': per_res[q][0] for q in per_res}, **{f'peak_q{q}': per_res[q][1] for q in per_res}))
A = pd.DataFrame(rows)
A.to_parquet(DL / 'alignment.parquet', index=False)
print(A.describe().T.to_string())
print('lag distribution (local):')
print(pd.cut(A.lag_local, [-181, -60, -20, -10, -4, 0, 4, 10, 20, 40, 60, 181]).value_counts().sort_index().to_string())
print('gross best lag:', A.lag.value_counts().loc[lambda s: s.index.isin([-3600, 3600, -86400, 86400])].to_dict())
