"""H3 follow-up: post-goal 'model edge' rule with conservative anchors, fills and depth.

Decision time t = anchor + L, anchor = located print jump ('jump') or max(ESPN wallclock, jump) ('late').
Fill variants: 'first' = first print at/after t (as in 06); 'worst15' = worst print price in [t, t+15 s]
(max ask for buys of YES, min bid for buys of NO). Depth = USDC printed on that side in [t, t+60 s] at prices no
worse than the fill + 1c. Model state at t uses all events with located time <= t.
Rules (pre-registered from 06 train results): buy the scorer's YES if model - ask - fee >= d; buy the conceding
team's NO if bid - model - fee >= d. Hold to resolution or exit after 300/600 s (fee on both legs).
Outputs: data/soccer_strat/goal_entries.parquet, results/soccer_strat/h3b_*.csv
"""
import importlib.util
import json
import sys

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import numpy as np
import pandas as pd

from lib import DS, RS, ROLES, STAKE, cluster_boot, fee_per_share, load_matches, load_trades, log

spec = importlib.util.spec_from_file_location('mdl', f'{sys.path[0]}/04_model.py')
mdl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdl)


def side_arrays(g):
    b = g.taker_buys_yes.values
    return dict(ta=g.ts.values[b], pa=g.yes_price.values[b], ua=(g.yes_price * g['size']).values[b],
                tb=g.ts.values[~b], pb=g.yes_price.values[~b], ub=((1 - g.yes_price) * g['size']).values[~b])


def fills(ts, px, usd, t, worst):
    i = np.searchsorted(ts, t, 'left')
    if i >= len(ts) or ts[i] - t > 60:
        return np.nan, np.nan, np.nan
    first = px[i]
    j = np.searchsorted(ts, t + 15, 'right')
    seg = px[i:max(j, i + 1)]
    w = seg.max() if worst == 'max' else seg.min()
    k = np.searchsorted(ts, t + 60, 'right')
    seg_p, seg_u = px[i:k], usd[i:k]
    depth = seg_u[(seg_p <= w + 0.01)].sum() if worst == 'max' else seg_u[(seg_p >= w - 0.01)].sum()
    return first, w, depth


def nxt(ts, px, t, win=180):
    i = np.searchsorted(ts, t, 'left')
    return px[i] if (i < len(ts) and ts[i] - t <= win) else np.nan


def main():
    m = load_matches()
    P = json.load(open(f'{DS}/model_params.json'))
    params, prof = P['params'], P['profile']
    lam = pd.read_parquet(f'{DS}/lambdas.parquet').set_index('match_id')
    lam = lam[lam.err < 0.01]
    E = pd.read_parquet(f'{DS}/events_located.parquet')
    E = E[E.match_id.isin(lam.index)].sort_values(['match_id', 'ev_ts', 'seq'])
    tr = load_trades(filters=[('match_id', 'in', list(lam.index)), ('t_rel_kickoff_s', '>=', 0),
                              ('t_rel_kickoff_s', '<=', 12000)])
    tr = tr.sort_values(['match_id', 'role', 'ts'], kind='mergesort')
    trg = {k: g for k, g in tr.groupby('match_id', sort=False)}
    mm = m.set_index('match_id')
    rows = []
    for mid, eg in E.groupby('match_id', sort=False):
        if mid not in trg:
            continue
        books = {r: side_arrays(g) for r, g in trg[mid].groupby('role', sort=False)}
        if len(books) < 3:
            continue
        r_ = mm.loc[mid]
        kw, sh, ft = r_.kickoff_wall_ts, r_.second_half_wall_ts, r_.fulltime_wall_ts
        evt = eg.ev_ts.values
        isg = (eg.kind == 'goal').values
        hs = np.cumsum(isg & (eg.side == 'home').values)
        as_ = np.cumsum(isg & (eg.side == 'away').values)
        hr = np.cumsum(~isg & (eg.side == 'home').values)
        ar = np.cumsum(~isg & (eg.side == 'away').values)
        for k, e in enumerate(eg.itertuples(index=False)):
            if e.kind != 'goal':
                continue
            for anchor in ['jump', 'late']:
                a = e.ev_ts if anchor == 'jump' else max(e.ev_ts, e.wc)
                for L in [30, 60, 120]:
                    t = a + L
                    if t >= ft - 60:
                        continue
                    n = np.searchsorted(evt, t, 'right')  # events known at t
                    gh, ga, rh, ra = hs[n - 1], as_[n - 1], hr[n - 1], ar[n - 1]
                    half = 2 if t >= sh else 1
                    el = (t - kw) / 60 if half == 1 else (t - sh) / 60
                    wr = float(mdl.w_rem(prof, np.array([half]), np.array([el]))[0])
                    pr = mdl.chain(np.array([lam.at[mid, 'lh']]), np.array([lam.at[mid, 'la']]), np.array([gh - ga]),
                                   np.array([wr]), np.array([rh - ra]), params)[0]
                    for ri, role in enumerate(ROLES):
                        b = books[role]
                        a1, aw, ad = fills(b['ta'], b['pa'], b['ua'], t, 'max')
                        b1, bw, bd = fills(b['tb'], b['pb'], b['ub'], t, 'min')
                        rec = dict(match_id=mid, ev_k=k, side=e.side, anchor=anchor, L=L, t=t, role=role,
                                   rel='draw' if role == 'draw' else ('team' if role == e.side else 'opp'),
                                   model=pr[ri], ask1=a1, askw=aw, ask_depth=ad, bid1=b1, bidw=bw, bid_depth=bd,
                                   y=int(r_[f'res_{role}']), minute=e.minute,
                                   period_half=e.period, gd_after=(gh - ga), later_goal_before_t=int(n - 1 > k),
                                   wc_minus_jump=e.wc - e.ev_ts)
                        for H in [300, 600]:
                            tx = t + H
                            if tx >= ft - 30:
                                rec[f'xb_{H}'], rec[f'xa_{H}'], rec[f'xres_{H}'] = np.nan, np.nan, 1
                            else:
                                rec[f'xb_{H}'] = nxt(b['tb'], b['pb'], tx)
                                rec[f'xa_{H}'] = nxt(b['ta'], b['pa'], tx)
                                rec[f'xres_{H}'] = 0
                        rows.append(rec)
    G = pd.DataFrame(rows)
    G = G.merge(m[['match_id', 'period', 'league', 'month', 'bfx_c_home', 'bfx_c_away']], on='match_id')
    G.to_parquet(f'{DS}/goal_entries.parquet', index=False)
    log('goal entry rows', len(G))


def pnl(T, direction, fill, H):
    if direction == 'yes':
        p_in, y = (T.ask1 if fill == 'first' else T.askw), T.y
        p_out = T.get(f'xb_{H}')
    else:
        p_in, y = 1 - (T.bid1 if fill == 'first' else T.bidw), 1 - T.y
        p_out = 1 - T[f'xa_{H}'] if H != 'res' else None
    sh = STAKE / p_in
    res = sh * (y - p_in) - sh * fee_per_share(p_in)
    if H == 'res':
        return res
    trade = sh * (p_out - p_in) - sh * fee_per_share(p_in) - sh * fee_per_share(p_out)
    return pd.Series(np.where(T[f'xres_{H}'] == 1, res, trade), index=T.index)


def evaluate():
    G = pd.read_parquet(f'{DS}/goal_entries.parquet')
    out, keep = [], []
    for anchor in ['jump', 'late']:
        for L in [30, 60, 120]:
            for fill in ['first', 'worst15']:
                for d in [0.02, 0.03, 0.05]:
                    for H in [300, 600, 'res']:
                        g = G[(G.anchor == anchor) & (G.L == L)]
                        ay = g.ask1 if fill == 'first' else g.askw
                        bn = g.bid1 if fill == 'first' else g.bidw
                        e_yes = g.model - ay - fee_per_share(ay)
                        e_no = bn - g.model - fee_per_share(bn)
                        y_ = g[(g.rel == 'team') & (e_yes >= d) & ay.between(0.03, 0.97)].assign(dir='yes')
                        n_ = g[(g.rel == 'opp') & (e_no >= d) & bn.between(0.03, 0.97)].assign(dir='no')
                        parts = []
                        for part, dr in [(y_, 'yes'), (n_, 'no')]:
                            part = part.assign(pnl=pnl(part, dr, fill, H), stake=STAKE,
                                               edge=(e_yes if dr == 'yes' else e_no)[part.index],
                                               depth=part.ask_depth if dr == 'yes' else part.bid_depth)
                            parts.append(part)
                        T = pd.concat(parts).dropna(subset=['pnl'])
                        for comp, TT in [('both', T), ('team_yes', T[T.dir == 'yes']), ('opp_no', T[T.dir == 'no'])]:
                            for s in ['train', 'test']:
                                tt = TT[TT.period == s]
                                if len(tt) < 20:
                                    continue
                                r = cluster_boot(tt, n_boot=1000)
                                out.append(dict(anchor=anchor, L=L, fill=fill, d=d, H=str(H), comp=comp, set=s, n=r['n'],
                                                matches=r['matches'], roi=r['roi'], lo=r['ci_roi'][0], hi=r['ci_roi'][1],
                                                pnl=r['pnl'], depth_med=tt.depth.median(),
                                                share_depth_ge10=(tt.depth >= 10).mean(),
                                                share_depth_ge100=(tt.depth >= 100).mean()))
                        if anchor == 'late' and L == 60 and fill == 'worst15' and d == 0.03:
                            keep.append(T.assign(H=str(H)))
    R = pd.DataFrame(out)
    R.to_csv(f'{RS}/h3b_goal_rule_grid.csv', index=False)
    K = pd.concat(keep)
    K.to_parquet(f'{DS}/h3b_main_trades.parquet', index=False)
    return R, K


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] != 'eval':
        main()
    R, K = evaluate()
    v = R[(R.comp == 'both') & (R.d == 0.03)]
    print(v[['anchor', 'L', 'fill', 'H', 'set', 'n', 'roi', 'lo', 'hi', 'depth_med', 'share_depth_ge10']].round(3).to_string(index=False))
