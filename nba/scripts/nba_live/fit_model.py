"""Win-probability models from game state, fitted on TRAIN games only (game date < 2026-02-01).

Data: ESPN play-by-play in game-clock order, state at every 15 s of game clock (regulation and OT), outcome home_win.
Pre-game strength: mu = SIG0 * Phi^-1(pre-tip market price) (pre-tip mid from prints; SIG0 = 12.34 from book spread vs price).

Models
  BM   : Brownian margin. Final margin - m ~ N(mu * tau, sigma^2 * tau) with tau = remaining fraction of 48 min; continuity
         correction (+-0.5) and P(OT) * q_ot for ties; sigma fitted by MLE; OT periods use remaining OT time and
         sigma_ot (fitted) with mu scaled the same way.
  LOGIT: per remaining-time bucket logistic recalibration logit P = a_k + b_k * Phi^-1(P_BM) (fitted on train).
Writes data/nba_live/model_params.json and results/nba_live/model_fit.csv (log loss / Brier by split and time bucket).
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.special import ndtr, ndtri
from common import DL, RL, NBA, load_games

SIG0 = 12.34
g = load_games()
pbp = pd.read_parquet(NBA / 'espn_pbp.parquet', columns=['espn_event_id', 'seq', 'period', 'clock_s', 'elapsed_s', 'home_score', 'away_score'])
bad_games = set(g.loc[g.flag_final_score_pbp_mismatch, 'espn_event_id'])
pbp = pbp[~pbp.espn_event_id.isin(bad_games)]
pbp = pbp.merge(g[['espn_event_id', 'game_id', 'split', 'season', 'pre_ph', 'home_win']], on='espn_event_id')
pbp = pbp.sort_values(['game_id', 'elapsed_s', 'seq'], kind='mergesort')
rows = []
for gid, d in pbp.groupby('game_id', sort=False):
    el = d.elapsed_s.values.astype(float)
    mx = el.max()
    cps = np.arange(0, mx + 1, 15.0)
    ix = np.searchsorted(el, cps, side='right') - 1
    ix = np.clip(ix, 0, None)
    r0 = d.iloc[0]
    rows.append(pd.DataFrame({'game_id': gid, 'split': r0.split, 'season': r0.season, 'pre_ph': r0.pre_ph, 'home_win': r0.home_win,
                              'elapsed': cps, 'm': (d.home_score.values[ix] - d.away_score.values[ix]).astype(float)}))
X = pd.concat(rows, ignore_index=True)
X = X[~X.pre_ph.isna()].copy()
X['mu'] = SIG0 * ndtri(X.pre_ph.clip(0.02, 0.98))


def tau_of(elapsed):
    reg = np.clip(2880.0 - elapsed, 0, None) / 2880.0
    ot = elapsed >= 2880
    ot_rem = (300.0 - ((elapsed - 2880.0) % 300.0)) / 2880.0
    return np.where(ot, ot_rem, reg), ot


def p_bm(m, elapsed, mu, sig, sig_ot=None, q_tie=0.5):
    tau, ot = tau_of(np.asarray(elapsed, float))
    s = np.where(ot, sig if sig_ot is None else sig_ot, sig) * np.sqrt(np.maximum(tau, 1e-9))
    mean = m + mu * tau
    p_gt = 1 - ndtr((0.5 - mean) / s)          # final margin >= 1
    p_tie = ndtr((0.5 - mean) / s) - ndtr((-0.5 - mean) / s)
    # tie at end of regulation/OT -> another OT: approx win prob 0.5 + mu-based edge (5-min period)
    q = ndtr(mu * (300 / 2880) / (sig * np.sqrt(300 / 2880)))
    return np.clip(p_gt + p_tie * q, 1e-6, 1 - 1e-6)


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


tr = X[X.split == 'train']
reg = tr[tr.elapsed < 2880 - 30]
f = lambda s: ll(p_bm(reg.m.values, reg.elapsed.values, reg.mu.values, s), reg.home_win.values).mean()
res = minimize_scalar(f, bounds=(6, 20), method='bounded')
SIG = float(res.x)
ot = tr[tr.elapsed >= 2880]
f2 = lambda s: ll(p_bm(ot.m.values, ot.elapsed.values, ot.mu.values, SIG, s), ot.home_win.values).mean()
SIG_OT = float(minimize_scalar(f2, bounds=(6, 30), method='bounded').x)
print('sigma (regulation, per 48 min)', round(SIG, 3), 'sigma OT-equivalent', round(SIG_OT, 3))
# diagnostic: sigma fitted separately by remaining-time bucket
for lo, hi in [(30, 120), (120, 300), (300, 720), (720, 1440), (1440, 2160), (2160, 2880)]:
    d = tr[(2880 - tr.elapsed >= lo) & (2880 - tr.elapsed < hi)]
    fb = lambda s: ll(p_bm(d.m.values, d.elapsed.values, d.mu.values, s), d.home_win.values).mean()
    print('  remaining', lo, hi, 'sigma', round(float(minimize_scalar(fb, bounds=(4, 30), method='bounded').x), 2), 'n', len(d))

# logistic recalibration per remaining-time bucket
EDGES = [0, 30, 60, 120, 180, 300, 480, 720, 1080, 1440, 1800, 2160, 2520, 2881]


def bucket(elapsed):
    rem = np.clip(2880.0 - np.asarray(elapsed, float), 0, None)
    ot = np.asarray(elapsed) >= 2880
    rem_ot = 300.0 - ((np.asarray(elapsed, float) - 2880.0) % 300.0)
    rem = np.where(ot, rem_ot, rem)
    b = np.searchsorted(EDGES, rem, side='right') - 1
    return np.where(ot, 100 + np.searchsorted(EDGES, rem_ot, side='right') - 1, b)


X['p_bm'] = p_bm(X.m.values, X.elapsed.values, X.mu.values, SIG, SIG_OT)
X['zb'] = ndtri(X.p_bm.clip(1e-6, 1 - 1e-6))
X['bk'] = bucket(X.elapsed.values)
params = {}
for bk, d in X[X.split == 'train'].groupby('bk'):
    y = d.home_win.values; z = d.zb.values

    def nll(th):
        pp = 1 / (1 + np.exp(-(th[0] + th[1] * z)))
        pp = np.clip(pp, 1e-9, 1 - 1e-9)
        return ll(pp, y).mean()
    th = minimize(nll, x0=[0.0, 1.6], method='Nelder-Mead').x
    params[int(bk)] = [float(th[0]), float(th[1])]


def p_logit(zb, bk):
    a = np.array([params.get(int(k), [0.0, 1.6])[0] for k in bk])
    b = np.array([params.get(int(k), [0.0, 1.6])[1] for k in bk])
    return 1 / (1 + np.exp(-(a + b * zb)))


X['p_lg'] = p_logit(X.zb.values, X.bk.values)

# EMP: per regulation bucket, logistic on [1, m/sqrt(tau'), mu*sqrt(tau')], tau' = max(rem, 10)/2880 (OT: BM)
import statsmodels.api as sm


def emp_feats(m, elapsed, mu):
    rem = np.clip(2880.0 - np.asarray(elapsed, float), 10, None) / 2880.0
    sq = np.sqrt(rem)
    return np.column_stack([np.ones(len(sq)), np.asarray(m, float) / sq / 10, np.asarray(mu, float) * sq / 10])


emp = {}
for bk, d in X[(X.split == 'train') & (X.bk < 100)].groupby('bk'):
    F = emp_feats(d.m.values, d.elapsed.values, d.mu.values)
    mm = sm.Logit(d.home_win.values, F).fit(disp=0)
    emp[int(bk)] = [float(v) for v in mm.params]


def p_emp(m, elapsed, mu, bk, p_ot):
    F = emp_feats(m, elapsed, mu)
    th = np.array([emp.get(int(k), [np.nan] * 3) for k in bk])
    z = (F * th).sum(1)
    return np.where(np.asarray(bk) >= 100, p_ot, 1 / (1 + np.exp(-z)))


X['p_emp'] = p_emp(X.m.values, X.elapsed.values, X.mu.values, X.bk.values, X.p_bm.values)
json.dump({'SIG0': SIG0, 'SIG': SIG, 'SIG_OT': SIG_OT, 'EDGES': EDGES, 'logit': params, 'emp': emp}, open(DL / 'model_params.json', 'w'), indent=1)
print('EMP params (bucket: const, m/sqrt/10, mu*sqrt/10):')
for k, v in emp.items():
    print('  ', k, EDGES[k], [round(x, 3) for x in v])
out = []
for (sp, bk), d in X.groupby(['split', 'bk']):
    y = d.home_win.values
    out.append(dict(split=sp, bk=bk, n=len(d), games=d.game_id.nunique(), ll_bm=ll(d.p_bm.values, y).mean(),
                    ll_lg=ll(d.p_lg.values, y).mean(), ll_emp=ll(d.p_emp.values, y).mean(), br_bm=((d.p_bm - y) ** 2).mean(), br_lg=((d.p_lg - y) ** 2).mean(),
                    mean_y=y.mean(), mean_bm=d.p_bm.mean(), mean_lg=d.p_lg.mean()))
O = pd.DataFrame(out)
O.to_csv(RL / 'model_fit.csv', index=False)
print(O.groupby('split')[['ll_bm', 'll_lg', 'll_emp', 'br_bm', 'br_lg']].mean())
print(O[O.split == 'test'][['bk', 'n', 'll_bm', 'll_lg', 'll_emp', 'mean_y', 'mean_bm', 'mean_lg']].to_string())
