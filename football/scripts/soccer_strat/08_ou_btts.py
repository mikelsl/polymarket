"""H4: Over/Under 2.5 and BTTS (minute price history only; no prints were collected for these markets).

Pre-match: PM Over/BTTS-Yes price ~T-5m vs bookmaker fair Over (Pinnacle close train only, Betfair close, market
average close) and a Poisson BTTS fair (lambdas fitted to fair 1X2 + Over). In-play: PM price at fixed minutes in
quiet states (no goal in the previous 180 s, located goal times) vs a Poisson remaining-goals model and outcomes.
Because there are no prints, every entry here assumes a fill at minute price + h (h = assumed half-spread 0.5c or
1c) plus the 0.05 p(1-p) fee; results are indicative only.
Outputs: results/soccer_strat/h4_*.csv
"""
import json
import sys

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import numpy as np
import pandas as pd
from scipy.stats import poisson

from lib import DS, RS, SD, STAKE, cluster_boot, fee_per_share, load_matches, log

KMAX = 11


def poisson_grid():
    g = np.round(np.arange(0.10, 4.01, 0.02), 2)
    LH, LA = np.meshgrid(g, g, indexing='ij')
    lh, la = LH.ravel(), LA.ravel()
    k = np.arange(KMAX)
    ph = poisson.pmf(k[None, :], lh[:, None])
    pa = poisson.pmf(k[None, :], la[:, None])
    J = ph[:, :, None] * pa[:, None, :]
    i, j = np.meshgrid(k, k, indexing='ij')
    H = (J * (i > j)).sum((1, 2))
    A = (J * (i < j)).sum((1, 2))
    O = (J * ((i + j) >= 3)).sum((1, 2))
    B = (1 - ph[:, 0]) * (1 - pa[:, 0])
    return lh, la, H, A, O, B


def fit_btts(m):
    lh, la, H, A, O, B = poisson_grid()
    out = np.full((len(m), 3), np.nan)
    fh, fa, fo = m.bfx_c_home.values, m.bfx_c_away.values, m.bfx_c_over.values
    for n in range(len(m)):
        if np.isnan(fh[n]) or np.isnan(fo[n]):
            continue
        err = (H - fh[n]) ** 2 + (A - fa[n]) ** 2 + (O - fo[n]) ** 2
        k = np.argmin(err)
        out[n] = (B[k], lh[k], la[k])
    return out


def main():
    m = load_matches()
    fb = fit_btts(m)
    m['fair_btts'], m['lh_ou'], m['la_ou'] = fb[:, 0], fb[:, 1], fb[:, 2]
    ph = pd.read_parquet(f'{SD}/price_history.parquet', columns=['match_id', 'role', 'ts', 'price', 't_rel_kickoff_s'],
                         filters=[('role', 'in', ['ou25', 'btts'])])
    ph['match_id'] = ph.match_id.astype(str)
    ph['role'] = ph.role.astype(str)
    ph = ph[ph.match_id.isin(m.match_id)].sort_values(['match_id', 'role', 'ts'])
    log('price points', len(ph))
    # ---------- pre-match
    pre = ph[ph.t_rel_kickoff_s.between(-900, -60)]
    pre = pre.groupby(['match_id', 'role']).tail(1).pivot_table(index='match_id', columns='role', values='price')
    P = m.set_index('match_id').join(pre, how='inner')
    rows, trades = [], []
    for mk, fair_cols, yc in [('ou25', ['pin_c_over', 'bfe_c_over', 'avg_c_over', 'bfx_c_over'], 'res_over'),
                              ('btts', ['fair_btts'], 'res_btts')]:
        d = P[P[mk].notna()]
        for per in ['train', 'test']:
            dd = d[d.period == per]
            for fc in fair_cols:
                ok = dd[fc].notna()
                if ok.sum() < 30:
                    continue
                y = dd.loc[ok, yc]
                rows.append(dict(market=mk, period=per, fair=fc, n=int(ok.sum()), freq=y.mean(),
                                 pm_mean=dd.loc[ok, mk].mean(), fair_mean=dd.loc[ok, fc].mean(),
                                 brier_pm=((dd.loc[ok, mk] - y) ** 2).mean(), brier_fair=((dd.loc[ok, fc] - y) ** 2).mean(),
                                 mean_abs_gap=(dd.loc[ok, mk] - dd.loc[ok, fc]).abs().mean()))
        # value rule with assumed half-spread h
        for fc in fair_cols:
            for h in [0.005, 0.01]:
                for dthr in [0.0, 0.01, 0.02, 0.03]:
                    pa_ = (d[mk] + h).clip(0.01, 0.99)
                    pb_ = (1 - d[mk] + h).clip(0.01, 0.99)
                    e_o = d[fc] - pa_ - fee_per_share(pa_)
                    e_u = (1 - d[fc]) - pb_ - fee_per_share(pb_)
                    t1 = pd.DataFrame({'match_id': d.index, 'period': d.period, 'price': pa_, 'yy': d[yc], 'tk': e_o >= dthr})
                    t2 = pd.DataFrame({'match_id': d.index, 'period': d.period, 'price': pb_, 'yy': 1 - d[yc], 'tk': e_u >= dthr})
                    t = pd.concat([t1, t2]).reset_index(drop=True)
                    t = t[t.tk & d.reindex(t.match_id)[fc].notna().values]
                    sh = STAKE / t.price
                    t = t.assign(stake=STAKE, pnl=sh * (t.yy - t.price) - sh * fee_per_share(t.price))
                    for per in ['train', 'test']:
                        tt = t[t.period == per]
                        if len(tt) < 20:
                            continue
                        r = cluster_boot(tt, n_boot=1000)
                        trades.append(dict(market=mk, fair=fc, h=h, d=dthr, period=per, n=r['n'], roi=r['roi'],
                                           lo=r['ci_roi'][0], hi=r['ci_roi'][1]))
    C = pd.DataFrame(rows)
    C.to_csv(f'{RS}/h4_prematch_calibration.csv', index=False)
    V = pd.DataFrame(trades)
    V.to_csv(f'{RS}/h4_prematch_value_rules.csv', index=False)
    print(C.round(4).to_string(index=False))
    # ---------- in-play: quiet states at fixed minutes
    E = pd.read_parquet(f'{DS}/events_located.parquet')
    G = E[E.kind == 'goal']
    prm = json.load(open(f'{DS}/model_params.json'))
    prof = prm['profile']
    grid, W1, W2 = np.array(prof['grid']), np.array(prof['W1']), np.array(prof['W2'])
    mm = m.set_index('match_id')
    recs = []
    phg = {k: g for k, g in ph.groupby(['match_id', 'role'], sort=False)}
    gg = {k: g.ev_ts.values for k, g in G.groupby('match_id')}
    gside = {k: g.side.values for k, g in G.groupby('match_id')}
    for mid in mm.index:
        r = mm.loc[mid]
        if np.isnan(r.lh_ou):
            continue
        evt = gg.get(mid, np.array([]))
        sides = gside.get(mid, np.array([]))
        for M in [20, 30, 40, 55, 65, 75, 85]:
            half = 1 if M <= 45 else 2
            t = r.kickoff_wall_ts + M * 60 if half == 1 else r.second_half_wall_ts + (M - 45) * 60
            if t >= r.fulltime_wall_ts:
                continue
            k = np.searchsorted(evt, t, 'right')
            if k > 0 and t - evt[k - 1] < 180:
                continue
            hg = int((sides[:k] == 'home').sum())
            ag = int((sides[:k] == 'away').sum())
            el = M if half == 1 else M - 45
            wr = np.interp(el, grid, W1 if half == 1 else W2)
            lt = (r.lh_ou + r.la_ou) * wr
            p_over = 1 - poisson.cdf(2 - (hg + ag), lt) if hg + ag < 3 else 1.0
            ph_ = 1 - np.exp(-r.lh_ou * wr)
            pa_ = 1 - np.exp(-r.la_ou * wr)
            p_btts = (1.0 if hg > 0 else ph_) * (1.0 if ag > 0 else pa_)
            for mk, pm_, yc in [('ou25', p_over, 'res_over'), ('btts', p_btts, 'res_btts')]:
                g = phg.get((mid, mk))
                if g is None:
                    continue
                i = np.searchsorted(g.ts.values, t, 'right') - 1
                if i < 0 or t - g.ts.values[i] > 120:
                    continue
                recs.append(dict(match_id=mid, period=r.period, market=mk, M=M, goals=hg + ag,
                                 scored=int(hg > 0) + int(ag > 0), price=g.price.values[i], model=pm_, y=r[yc]))
    I = pd.DataFrame(recs)
    I = I[(I.price > 0.02) & (I.price < 0.98)]
    I['state'] = np.where(I.market == 'ou25', 'g' + I.goals.clip(0, 2).astype(str), 's' + I.scored.astype(str))
    cal = I.groupby(['market', 'period', 'M', 'state']).agg(n=('y', 'size'), freq=('y', 'mean'), price=('price', 'mean'),
                                                             model=('model', 'mean')).reset_index()
    cal['freq_minus_price'] = cal.freq - cal.price
    cal.to_csv(f'{RS}/h4_inplay_calibration.csv', index=False)
    ll = lambda p, y: -(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
    I['ll_pm'], I['ll_mod'] = ll(I.price, I.y), ll(I.model, I.y)
    print(I.groupby(['market', 'period'])[['ll_pm', 'll_mod']].mean().round(4).to_string())
    # in-play rules: buy the side the model favours by >= d (one entry per match/market at the first qualifying minute)
    out = []
    nvar = 0
    for mk in ['ou25', 'btts']:
        for h in [0.005, 0.01]:
            for dthr in [0.02, 0.05]:
                for side in ['yes', 'no']:
                    d = I[I.market == mk].copy()
                    px = (d.price + h) if side == 'yes' else (1 - d.price + h)
                    fair = d.model if side == 'yes' else 1 - d.model
                    d = d.assign(price_in=px.clip(0.01, 0.99), yy=d.y if side == 'yes' else 1 - d.y)
                    d = d[(fair - d.price_in - fee_per_share(d.price_in)) >= dthr]
                    d = d.sort_values('M').groupby('match_id').head(1)
                    sh = STAKE / d.price_in
                    d = d.assign(stake=STAKE, pnl=sh * (d.yy - d.price_in) - sh * fee_per_share(d.price_in))
                    nvar += 1
                    for per in ['train', 'test']:
                        tt = d[d.period == per]
                        if len(tt) < 20:
                            continue
                        rr = cluster_boot(tt, n_boot=1000)
                        out.append(dict(market=mk, h=h, d=dthr, side=side, period=per, n=rr['n'], roi=rr['roi'],
                                        lo=rr['ci_roi'][0], hi=rr['ci_roi'][1]))
        # model-free: buy Under / BTTS-No in the late 0-goal state (time decay) at minute >= 65
        for side in ['yes', 'no']:
            for h in [0.005, 0.01]:
                d = I[(I.market == mk) & (I.M >= 65) & (((I.goals == 0) & (mk == 'ou25')) | ((I.scored == 0) & (mk == 'btts')))]
                d = d.sort_values('M').groupby('match_id').head(1)
                px = ((d.price + h) if side == 'yes' else (1 - d.price + h)).clip(0.01, 0.99)
                yy = d.y if side == 'yes' else 1 - d.y
                sh = STAKE / px
                d = d.assign(stake=STAKE, pnl=sh * (yy - px) - sh * fee_per_share(px))
                nvar += 1
                for per in ['train', 'test']:
                    tt = d[d.period == per]
                    if len(tt) < 20:
                        continue
                    rr = cluster_boot(tt, n_boot=1000)
                    out.append(dict(market=mk, h=h, d='late_0goal', side=side, period=per, n=rr['n'], roi=rr['roi'],
                                    lo=rr['ci_roi'][0], hi=rr['ci_roi'][1]))
    O = pd.DataFrame(out)
    O.to_csv(f'{RS}/h4_inplay_rules.csv', index=False)
    log('in-play variants', nvar, 'pre-match variants', len(V) // 2)
    print(V[(V.h == 0.01) & (V.d.isin([0.0, 0.02]))].round(3).to_string(index=False))
    print(O[O.h == 0.01].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
