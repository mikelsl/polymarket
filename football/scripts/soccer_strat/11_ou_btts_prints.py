"""H4 follow-up: re-test the in-play O/U 2.5 and BTTS rules from 08 with REAL taker prints (09 fetch).

Decision points as in 08 (minutes 20..85, no located goal in the previous 180 s). Model: Poisson remaining goals
with lambdas fitted to fair 1X2 + Over (08.fit_btts) and the train goal-time profile. Entry = first print at/after
t + 2 s within 60 s: buy Over/Yes at the Over/Yes-scale price of a print where the taker gained Over/Yes; buy
Under/No at 1 - (Over/Yes-scale price of a print where the taker gained Under/No). Hold to resolution, fee
0.05 p(1-p) per share. One entry per match x market x rule (first qualifying minute).
Outputs: results/soccer_strat/h4b_prints_rules.csv, h4b_prints_coverage.csv
"""
import glob
import importlib.util
import json
import sys

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import numpy as np
import pandas as pd
from scipy.stats import poisson

from lib import DS, RS, STAKE, cluster_boot, fee_per_share, load_matches, log

spec = importlib.util.spec_from_file_location('h4', f'{sys.path[0]}/08_ou_btts.py')
h4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h4)


def main():
    m = load_matches()
    fb = h4.fit_btts(m)
    m['lh_ou'], m['la_ou'] = fb[:, 1], fb[:, 2]
    fs = glob.glob(f'{DS}/ou_btts_trades/*.parquet')
    tr = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    tr = tr.dropna(subset=['price', 'timestamp'])
    tr['ts'] = tr.timestamp.astype(np.int64)
    tr['p0'] = np.where(tr.outcome_index == 0, tr.price, 1 - tr.price)          # Over / Yes scale
    tr['gain0'] = ((tr.side == 'BUY') & (tr.outcome_index == 0)) | ((tr.side == 'SELL') & (tr.outcome_index == 1))
    tr['usd'] = tr.price * tr['size']
    tr = tr.sort_values(['match_id', 'role', 'ts'])
    log('prints', len(tr), 'markets', tr.groupby(['match_id', 'role']).ngroups)
    E = pd.read_parquet(f'{DS}/events_located.parquet')
    G = E[E.kind == 'goal']
    prof = json.load(open(f'{DS}/model_params.json'))['profile']
    grid, W1, W2 = np.array(prof['grid']), np.array(prof['W1']), np.array(prof['W2'])
    gg = {k: (g.ev_ts.values, g.side.values) for k, g in G.groupby('match_id')}
    tg = {k: g for k, g in tr.groupby(['match_id', 'role'], sort=False)}
    recs = []
    for r in m.itertuples(index=False):
        if np.isnan(r.lh_ou):
            continue
        evt, sides = gg.get(r.match_id, (np.array([]), np.array([])))
        for M in [20, 30, 40, 55, 65, 75, 85]:
            half = 1 if M <= 45 else 2
            t = r.kickoff_wall_ts + M * 60 if half == 1 else r.second_half_wall_ts + (M - 45) * 60
            if t >= r.fulltime_wall_ts:
                continue
            k = np.searchsorted(evt, t, 'right')
            if k > 0 and t - evt[k - 1] < 180:
                continue
            hg, ag = int((sides[:k] == 'home').sum()), int((sides[:k] == 'away').sum())
            wr = np.interp(M if half == 1 else M - 45, grid, W1 if half == 1 else W2)
            lt = (r.lh_ou + r.la_ou) * wr
            p_over = 1 - poisson.cdf(2 - (hg + ag), lt) if hg + ag < 3 else 1.0
            p_btts = (1.0 if hg > 0 else 1 - np.exp(-r.lh_ou * wr)) * (1.0 if ag > 0 else 1 - np.exp(-r.la_ou * wr))
            for mk, pm_, y in [('ou25', p_over, r.res_over), ('btts', p_btts, r.res_btts)]:
                g = tg.get((r.match_id, mk))
                if g is None:
                    continue
                rec = dict(match_id=r.match_id, period=r.period, league=r.league, month=r.month, market=mk, M=M,
                           goals=hg + ag, scored=int(hg > 0) + int(ag > 0), model=pm_, y=y)
                for nm, msk in [('ask0', g.gain0.values), ('bid0', ~g.gain0.values)]:
                    ts, px, us = g.ts.values[msk], g.p0.values[msk], g.usd.values[msk]
                    i = np.searchsorted(ts, t + 2, 'left')
                    ok = i < len(ts) and ts[i] - t <= 60
                    rec[nm] = px[i] if ok else np.nan
                    rec[nm + '_usd'] = us[i] if ok else np.nan
                    j = np.searchsorted(ts, t, 'right') - 1
                    rec[nm + '_last'] = px[j] if (j >= 0 and t - ts[j] <= 120) else np.nan
                recs.append(rec)
    I = pd.DataFrame(recs)
    I.to_parquet(f'{DS}/h4b_points.parquet', index=False)
    cov = I.groupby(['market', 'period']).agg(points=('y', 'size'), ask_fill=('ask0', lambda v: v.notna().mean()),
                                              bid_fill=('bid0', lambda v: v.notna().mean()),
                                              spread_med=('ask0_last', lambda v: np.nan)).reset_index()
    I['spr'] = I.ask0_last - I.bid0_last
    cov['spread_med'] = I.groupby(['market', 'period']).spr.median().values
    cov.to_csv(f'{RS}/h4b_prints_coverage.csv', index=False)
    print(cov.round(3).to_string(index=False))
    out = []
    for mk in ['ou25', 'btts']:
        for side in ['yes', 'no']:
            for rule in ['model_d0.02', 'model_d0.05', 'late_0goal']:
                d = I[I.market == mk].copy()
                px = d.ask0 if side == 'yes' else 1 - d.bid0
                fair = d.model if side == 'yes' else 1 - d.model
                d = d.assign(price=px, yy=d.y if side == 'yes' else 1 - d.y,
                             usd=d.ask0_usd if side == 'yes' else d.bid0_usd)
                d = d[d.price.between(0.03, 0.97)]
                if rule.startswith('model'):
                    thr = float(rule.split('_d')[1])
                    d = d[(fair[d.index] - d.price - fee_per_share(d.price)) >= thr]
                else:
                    z = (d.goals == 0) if mk == 'ou25' else (d.scored == 0)
                    d = d[z & (d.M >= 65)]
                d = d.sort_values('M').groupby('match_id').head(1)
                sh = STAKE / d.price
                d = d.assign(stake=STAKE, pnl=sh * (d.yy - d.price) - sh * fee_per_share(d.price))
                for per in ['train', 'test']:
                    tt = d[d.period == per]
                    if len(tt) < 15:
                        continue
                    rr = cluster_boot(tt, n_boot=2000)
                    out.append(dict(market=mk, side=side, rule=rule, period=per, n=rr['n'], roi=rr['roi'],
                                    lo=rr['ci_roi'][0], hi=rr['ci_roi'][1], pnl=rr['pnl'], mean_px=tt.price.mean(),
                                    share_print_ge10usd=(tt.usd >= 10).mean()))
    O = pd.DataFrame(out)
    O.to_csv(f'{RS}/h4b_prints_rules.csv', index=False)
    print(O.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
