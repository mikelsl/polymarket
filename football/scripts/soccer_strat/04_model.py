"""In-play goal model (fit on TRAIN only) and model probabilities for snapshots and event entries.

Model: goal-difference Markov chain. Team scoring intensity = lambda_team x w(t) x m(goal diff from the team's view,
red-card balance). w(t) is the empirical goal-time profile (train goals, elapsed minutes within each half incl.
stoppage); m() are multipliers from a Poisson GLM on train team-intervals. Pre-match lambdas are solved per match so
that the model reproduces the de-vigged closing 1X2 (Betfair exchange close, else market average, else the PM mid at
kick-off). Only the pre-match prior uses bookmaker data; everything in-play is our own model.
Outputs: data/soccer_strat/model_params.json, lambdas.parquet, snaps_model.parquet, event_trades_model.parquet
"""
import json
import sys

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import numpy as np
import pandas as pd
import statsmodels.api as sm

from lib import DS, RS, ROLES, load_matches, log

DMAX = 9
ND = 2 * DMAX + 1
K = 40  # steps over the remaining goal mass


def goal_profile(E, m):
    """Empirical share of goals remaining after (half, elapsed minutes): returns grid + W arrays."""
    g = E[(E.kind == 'goal')].merge(m[['match_id', 'kickoff_wall_ts', 'second_half_wall_ts', 'train']], on='match_id')
    g = g[g.train]
    el = np.where(g.period == 1, (g.ev_ts - g.kickoff_wall_ts) / 60.0, (g.ev_ts - g.second_half_wall_ts) / 60.0)
    g = g.assign(el=np.clip(el, 0, 60))
    s1 = (g.period == 1).mean()
    grid = np.arange(0, 60.01, 0.25)
    F1 = np.array([(g.el[g.period == 1] <= x).mean() for x in grid])
    F2 = np.array([(g.el[g.period == 2] <= x).mean() for x in grid])
    W1 = s1 * (1 - F1) + (1 - s1)
    W2 = (1 - s1) * (1 - F2)
    return dict(grid=grid.tolist(), W1=W1.tolist(), W2=W2.tolist(), share_h1=float(s1), n_goals=int(len(g)))


def w_rem(prof, half, el):
    grid = np.array(prof['grid'])
    W1, W2 = np.array(prof['W1']), np.array(prof['W2'])
    el = np.clip(np.asarray(el, float), 0, 60)
    return np.where(np.asarray(half) == 1, np.interp(el, grid, W1), np.interp(el, grid, W2))


def mult_table(params):
    """Multiplier arrays indexed by goal diff (team view, -DMAX..DMAX) and red balance (-1, 0, +1)."""
    gd = np.arange(-DMAX, DMAX + 1)
    out = {}
    for rb in (-1, 0, 1):
        lg = np.zeros(ND)
        lg += np.where(gd <= -2, params['gd_m2'], 0) + np.where(gd == -1, params['gd_m1'], 0)
        lg += np.where(gd == 1, params['gd_p1'], 0) + np.where(gd >= 2, params['gd_p2'], 0)
        lg += params['red_down'] if rb > 0 else 0   # team has more red cards (a man down)
        lg += params['red_up'] if rb < 0 else 0     # opponent a man down
        out[rb] = np.exp(lg)
    return out


def chain(lh, la, d0, wr, rbal, params):
    """P(final goal diff >0, =0, <0) for arrays lh, la (pre-match lambdas), current diff d0 (home-away),
    remaining goal mass wr and red balance rbal (home reds - away reds)."""
    n = len(lh)
    mt = mult_table(params)
    P = np.zeros((n, ND))
    P[np.arange(n), np.clip(d0, -DMAX, DMAX).astype(int) + DMAX] = 1.0
    rb_h = np.sign(rbal).astype(int)       # home view: +1 if home is a man down
    mh = np.stack([mt[int(v)] for v in rb_h])           # (n, ND) indexed by home-view diff
    ma = np.stack([mt[int(-v)][::-1] for v in rb_h])    # away view diff = -d -> reversed index
    dt = wr / K
    ph = (lh * dt)[:, None] * mh
    pa = (la * dt)[:, None] * ma
    ph = np.minimum(ph, 0.45)
    pa = np.minimum(pa, 0.45)
    for _ in range(K):
        Q = P * (1 - ph - pa)
        Q[:, 1:] += (P * ph)[:, :-1]
        Q[:, -1] += (P * ph)[:, -1]
        Q[:, :-1] += (P * pa)[:, 1:]
        Q[:, 0] += (P * pa)[:, 0]
        P = Q
    return np.stack([P[:, DMAX + 1:].sum(1), P[:, DMAX], P[:, :DMAX].sum(1)], axis=1)


def solve_lambdas(fh, fa, params, iters=8):
    n = len(fh)
    lh = np.full(n, 1.45)
    la = np.full(n, 1.15)
    z0 = np.zeros(n)
    w1 = np.ones(n)
    for _ in range(iters):
        p = chain(lh, la, z0, w1, z0, params)
        e = 1e-3
        ph_ = chain(lh + e, la, z0, w1, z0, params)
        pa_ = chain(lh, la + e, z0, w1, z0, params)
        r = np.stack([p[:, 0] - fh, p[:, 2] - fa], 1)
        J = np.stack([np.stack([(ph_[:, 0] - p[:, 0]) / e, (pa_[:, 0] - p[:, 0]) / e], 1),
                      np.stack([(ph_[:, 2] - p[:, 2]) / e, (pa_[:, 2] - p[:, 2]) / e], 1)], 1)
        step = np.linalg.solve(J, r[:, :, None])[:, :, 0]
        lh = np.clip(lh - step[:, 0], 0.05, 6)
        la = np.clip(la - step[:, 1], 0.05, 6)
    p = chain(lh, la, z0, w1, z0, params)
    err = np.abs(p[:, 0] - fh) + np.abs(p[:, 2] - fa)
    return lh, la, err


def team_intervals(E, m, lam, prof):
    """Team-level intervals between events (train) for the multiplier GLM."""
    rows = []
    mm = m.set_index('match_id')
    for mid, g in E[E.match_id.isin(m.match_id[m.train])].groupby('match_id'):
        if mid not in lam.index:
            continue
        r = mm.loc[mid]
        kw, sh = r.kickoff_wall_ts, r.second_half_wall_ts
        g = g.sort_values('ev_ts')
        bounds = [(1, 0.0)]
        for e in g.itertuples():
            el = (e.ev_ts - kw) / 60 if e.period == 1 else (e.ev_ts - sh) / 60
            bounds.append((e.period, el))
        Wb = [float(w_rem(prof, h, x)) for h, x in bounds] + [0.0]
        hg = ag = hr = ar = 0
        evl = list(g.itertuples())
        for i in range(len(Wb) - 1):
            expo = max(Wb[i] - Wb[i + 1], 0)
            gh = ga = 0
            if i < len(evl):
                e = evl[i]
                if e.kind == 'goal':
                    gh, ga = int(e.side == 'home'), int(e.side == 'away')
            for team, lamt, goals, gd, rb in (('h', lam.at[mid, 'lh'], gh, hg - ag, hr - ar),
                                              ('a', lam.at[mid, 'la'], ga, ag - hg, ar - hr)):
                if expo > 0:
                    rows.append(dict(match_id=mid, goals=goals, expo=lamt * expo, gd=gd, rb=rb, home=int(team == 'h')))
            if i < len(evl):
                e = evl[i]
                if e.kind == 'goal':
                    hg += e.side == 'home'
                    ag += e.side == 'away'
                else:
                    hr += e.side == 'home'
                    ar += e.side == 'away'
    return pd.DataFrame(rows)


def fit_mults(TI):
    X = pd.DataFrame({'gd_m2': (TI.gd <= -2).astype(float), 'gd_m1': (TI.gd == -1).astype(float),
                      'gd_p1': (TI.gd == 1).astype(float), 'gd_p2': (TI.gd >= 2).astype(float),
                      'red_down': (TI.rb > 0).astype(float), 'red_up': (TI.rb < 0).astype(float)})
    X = sm.add_constant(X)
    res = sm.GLM(TI.goals, X, family=sm.families.Poisson(), offset=np.log(TI.expo)).fit()
    p = res.params.to_dict()
    return p, res


def main():
    m = load_matches()
    E = pd.read_parquet(f'{DS}/events_located.parquet')
    prof = goal_profile(E, m)
    log('goal profile share_h1', prof['share_h1'], 'n', prof['n_goals'])
    # prior fair 1X2: Betfair exchange close, else average close (power de-vig), else PM kick-off mid
    snap = pd.read_parquet(f'{DS}/prematch_snap.parquet')
    pmk = snap[snap.offset_s == -60].pivot_table(index='match_id', columns='role', values='mid_med')
    pmk = pmk.div(pmk.sum(1), axis=0)
    fh = m.bfx_c_home.copy()
    fa = m.bfx_c_away.copy()
    src = np.where(fh.notna(), 'bfx', 'pm')
    fh = fh.fillna(m.match_id.map(pmk['home']))
    fa = fa.fillna(m.match_id.map(pmk['away']))
    ok = fh.notna() & fa.notna()
    params = dict(const=0.0, gd_m2=0.0, gd_m1=0.0, gd_p1=0.0, gd_p2=0.0, red_down=0.0, red_up=0.0)
    for it in range(3):
        lh, la, err = solve_lambdas(fh[ok].values, fa[ok].values, params)
        lam = pd.DataFrame({'lh': lh, 'la': la, 'err': err}, index=m.match_id[ok].values)
        TI = team_intervals(E, m[ok], lam, prof)
        p, res = fit_mults(TI)
        log('iter', it, 'max fit err', err.max(), {k: round(v, 3) for k, v in p.items()})
        params = {k: float(v) for k, v in p.items()}
        params['const'] = 0.0  # the constant is absorbed by the per-match lambdas (fit to the prior)
    with open(f'{DS}/model_params.json', 'w') as f:
        json.dump(dict(params=params, profile=prof, glm=str(res.summary())), f)
    with open(f'{RS}/h2_model_glm.txt', 'w') as f:
        f.write(str(res.summary()))
    lam['prior_src'] = pd.Series(src, index=m.match_id.values)[lam.index]
    lam.index.name = 'match_id'
    lam.reset_index().to_parquet(f'{DS}/lambdas.parquet', index=False)
    # ---- snapshots
    S = pd.read_parquet(f'{DS}/snaps.parquet')
    S = S[S.match_id.isin(lam.index)]
    U = S[S.role == 'home'][['match_id', 't', 'half', 'el', 'gh', 'ga', 'rh', 'ra']].copy()
    wr = w_rem(prof, U.half.values, U.el.values)
    P = chain(U.match_id.map(lam.lh).values, U.match_id.map(lam.la).values, (U.gh - U.ga).values, wr,
              (U.rh - U.ra).values, params)
    U['wr'] = wr
    for i, r in enumerate(ROLES):
        U[f'pm_{r}'] = P[:, i]
    S = S.merge(U[['match_id', 't', 'wr', 'pm_home', 'pm_draw', 'pm_away']], on=['match_id', 't'])
    S['model'] = np.select([S.role == 'home', S.role == 'draw'], [S.pm_home, S.pm_draw], S.pm_away)
    S.to_parquet(f'{DS}/snaps_model.parquet', index=False)
    log('snaps with model', len(S))
    # ---- event entries: state right after the event (event included), at the entry decision time
    T = pd.read_parquet(f'{DS}/event_trades.parquet')
    T = T[T.match_id.isin(lam.index)].copy()
    Ev = E.reset_index(drop=True)
    Ev['ev_idx'] = Ev.groupby('match_id').cumcount()
    T = T.merge(Ev[['match_id', 'ev_idx', 'period', 'minute', 'added', 'h_after', 'a_after', 'h_before', 'a_before',
                    'hr_after', 'ar_after', 'how', 'pre', 'post']], on=['match_id', 'ev_idx'], how='left')
    mm = m.set_index('match_id')
    # running score at the event (fill red-card rows from the previous goal state)
    T = T.sort_values(['match_id', 'ev_idx'])
    T['gh'] = T.groupby('match_id').h_after.transform(lambda s: s.ffill().fillna(0))
    T['ga'] = T.groupby('match_id').a_after.transform(lambda s: s.ffill().fillna(0))
    T['half'] = np.where(T.ev_ts + T.lag >= T.match_id.map(mm.second_half_wall_ts), 2, 1)
    T['el'] = np.where(T.half == 1, (T.ev_ts + T.lag - T.match_id.map(mm.kickoff_wall_ts)) / 60,
                       (T.ev_ts + T.lag - T.match_id.map(mm.second_half_wall_ts)) / 60)
    wr = w_rem(prof, T.half.values, T.el.values)
    P = chain(T.match_id.map(lam.lh).values, T.match_id.map(lam.la).values, (T.gh - T.ga).values, wr,
              (T.hr_after - T.ar_after).values, params)
    T['wr'] = wr
    T['model'] = np.select([T.role == 'home', T.role == 'draw'], [P[:, 0], P[:, 1]], P[:, 2])
    T.to_parquet(f'{DS}/event_trades_model.parquet', index=False)
    log('event trades with model', len(T))


if __name__ == '__main__':
    main()
