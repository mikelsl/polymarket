"""H3: price reactions after goals and red cards, entries >= 30 s after the print-located jump.

Input: data/soccer_strat/event_trades_model.parquet (03 + 04). For each event x 1X2 market x entry lag:
entry at the first print at/after jump + lag (ask for YES, 1 - bid for NO); exit at the first print at/after
entry time + H (bid for YES, 1 - ask for NO; fee on both legs) or at resolution ('res', or when the exit time falls
after full time). Market roles relative to the event team: 'team' (scorer / carded team), 'opp', 'draw'.
Outputs: results/soccer_strat/h3_*.csv
"""
import sys

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import numpy as np
import pandas as pd

from lib import DS, RS, STAKE, cluster_boot, fee_per_share, load_matches, log

HS = [120, 300, 600, 1200, 1800, 'res']


def build():
    m = load_matches()
    T = pd.read_parquet(f'{DS}/event_trades_model.parquet')
    T = T.merge(m[['match_id', 'period', 'league', 'month', 'bfx_c_home', 'bfx_c_away']].rename(columns={'period': 'set'}),
                on='match_id')
    T['rel'] = np.where(T.role == 'draw', 'draw', np.where(T.role == T.side, 'team', 'opp'))
    T['minute_x'] = np.where(T.half == 1, T.el, 45 + T.el)
    fav_side = np.where(T.bfx_c_home >= T.bfx_c_away, 'home', 'away')
    T['ev_by_fav'] = T.side == fav_side
    gd_after = T.gh - T.ga
    T['gd_team_after'] = np.where(T.side == 'home', gd_after, -gd_after)  # from the event team's view
    return T


def pnl(T, direction, H):
    if direction == 'yes':
        p_in, y = T.ask, T.y
        p_out = T[f'xb_{H}'] if H != 'res' else None
    else:
        p_in, y = 1 - T.bid, 1 - T.y
        p_out = (1 - T[f'xa_{H}']) if H != 'res' else None
    sh = STAKE / p_in
    if H == 'res':
        out = sh * (y - p_in) - sh * fee_per_share(p_in)
    else:
        res_leg = T[f'xres_{H}'] == 1
        out_trade = sh * (p_out - p_in) - sh * fee_per_share(p_in) - sh * fee_per_share(p_out)
        out_res = sh * (y - p_in) - sh * fee_per_share(p_in)
        out = np.where(res_leg, out_res, out_trade)
    return pd.Series(out, index=T.index)


def main():
    T = build()
    log('event-market rows', len(T), T.groupby('kind').ev_idx.size().to_dict())
    rows = []
    nvar = 0
    for kind in ['goal', 'red']:
        for lag in [30, 60, 120, 300]:
            for rel in ['team', 'opp', 'draw']:
                for direction in ['yes', 'no']:
                    for H in HS:
                        for filt in ['all', 'model_edge>=0.03']:
                            sub = T[(T.kind == kind) & (T.lag == lag) & (T.rel == rel)]
                            if direction == 'yes':
                                sub = sub[sub.ask.between(0.03, 0.97)]
                                edge = sub.model - sub.ask - fee_per_share(sub.ask)
                            else:
                                sub = sub[sub.bid.between(0.03, 0.97)]
                                edge = sub.bid - sub.model - fee_per_share(sub.bid)
                            if filt != 'all':
                                sub = sub[edge >= 0.03]
                            sub = sub.assign(pnl=pnl(sub, direction, H), stake=STAKE).dropna(subset=['pnl'])
                            nvar += 1
                            for s in ['train', 'test']:
                                tt = sub[sub.set == s]
                                if len(tt) < 20:
                                    continue
                                r = cluster_boot(tt, n_boot=1000)
                                rows.append(dict(kind=kind, lag=lag, rel=rel, dir=direction, H=str(H), filt=filt, set=s,
                                                 n=r['n'], matches=r['matches'], roi=r['roi'], lo=r['ci_roi'][0],
                                                 hi=r['ci_roi'][1], pnl_tr=r['pnl_per_trade']))
    R = pd.DataFrame(rows)
    R.to_csv(f'{RS}/h3_event_rules.csv', index=False)
    log('variants', nvar)
    W = R.pivot_table(index=['kind', 'lag', 'rel', 'dir', 'H', 'filt'], columns='set', values=['n', 'roi', 'lo', 'hi'])
    W.columns = [f'{a}_{b}' for a, b in W.columns]
    W = W.reset_index()
    W.to_csv(f'{RS}/h3_event_rules_wide.csv', index=False)
    # pre-registered selection: train CI lower bound > 0
    sel = W[(W.lo_train > 0)]
    print('selected on train (CI>0):', len(sel))
    print(sel[['kind', 'lag', 'rel', 'dir', 'H', 'filt', 'n_train', 'roi_train', 'lo_train', 'n_test', 'roi_test', 'lo_test', 'hi_test']].round(3).to_string(index=False))
    # compact unconditional view, goals, lag 60
    v = W[(W.kind == 'goal') & (W.lag == 60) & (W.filt == 'all') & (W.dir == 'yes')]
    print(v[['rel', 'H', 'n_train', 'roi_train', 'lo_train', 'hi_train', 'n_test', 'roi_test', 'lo_test', 'hi_test']].round(3).to_string(index=False))
    # drift diagnostics: mid at entry vs later (YES scale), scorer market, goals, by lag
    d = T[(T.kind == 'goal') & (T.rel == 'team')].copy()
    for H in [300, 600, 1800]:
        d[f'drift_{H}'] = ((d[f'xb_{H}'] + d[f'xa_{H}']) / 2) - (d.ask + d.bid) / 2
    dg = d.groupby(['set', 'lag'])[[f'drift_{H}' for H in [300, 600, 1800]]].agg(['mean', 'count']).round(4)
    dg.to_csv(f'{RS}/h3_drift_scorer.csv')
    print(dg.to_string())


if __name__ == '__main__':
    main()
