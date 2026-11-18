"""H5 checks: (a) 1X2 cross-market sum (buy all three YES / all three NO) from prints; (b) stale prices after
in-play trading halts. Outputs: results/soccer_strat/h5_*.csv
"""
import sys

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import numpy as np
import pandas as pd

from lib import DS, RS, ROLES, fee_per_share, load_matches, load_trades, log


def main():
    m = load_matches()
    # (a) pre-match T-5m: first asks after t for the three YES tokens
    s = pd.read_parquet(f'{DS}/prematch_snap.parquet')
    rows = []
    for off in [-3600, -300, -60]:
        w = s[s.offset_s == off].pivot_table(index='match_id', columns='role', values=['ask_yes', 'bid_yes'])
        sa = w['ask_yes'][ROLES].dropna()
        sb = w['bid_yes'][ROLES].dropna()
        fa = sum(fee_per_share(sa[r]) for r in ROLES)
        fb = sum(fee_per_share(sb[r]) for r in ROLES)
        rows.append(dict(phase=f'pre {off}s', n_yes=len(sa), sum_ask_med=sa.sum(1).median(),
                         share_buy3yes_profit=float(((sa.sum(1) + fa) < 1).mean()),
                         n_no=len(sb), sum_bid_med=sb.sum(1).median(),
                         share_buy3no_profit=float(((3 - sb.sum(1) + fb) < 2).mean())))
    # in-play snapshots: last ask / bid prints within 120 s (not simultaneous -> optimistic)
    S = pd.read_parquet(f'{DS}/snaps.parquet', columns=['match_id', 't', 'role', 'last_ask', 'last_bid'])
    w = S.pivot_table(index=['match_id', 't'], columns='role', values=['last_ask', 'last_bid'])
    sa = w['last_ask'][ROLES].dropna()
    sb = w['last_bid'][ROLES].dropna()
    fa = sum(fee_per_share(sa[r]) for r in ROLES)
    fb = sum(fee_per_share(sb[r]) for r in ROLES)
    rows.append(dict(phase='in-play (last prints <=120 s)', n_yes=len(sa), sum_ask_med=sa.sum(1).median(),
                     share_buy3yes_profit=float(((sa.sum(1) + fa) < 1).mean()), n_no=len(sb),
                     sum_bid_med=sb.sum(1).median(), share_buy3no_profit=float(((3 - sb.sum(1) + fb) < 2).mean())))
    A = pd.DataFrame(rows)
    A.to_csv(f'{RS}/h5_cross_market_sum.csv', index=False)
    print(A.round(4).to_string(index=False))
    # (b) halts: reopen price vs 5 and 15 min later, home market YES scale
    h = m[(m.inplay_max_gap_s >= 900)][['match_id', 'inplay_max_gap_s', 'inplay_max_gap_start_ts', 'period']]
    tr = load_trades(filters=[('match_id', 'in', list(h.match_id)), ('role', '=', 'home')])
    tr = tr.sort_values('ts')
    out = []
    for r in h.itertuples(index=False):
        g = tr[tr.match_id == r.match_id]
        t_re = r.inplay_max_gap_start_ts + r.inplay_max_gap_s
        before = g[g.ts <= r.inplay_max_gap_start_ts].yes_price.tail(5).median()
        first = g[(g.ts >= t_re) & (g.ts < t_re + 60)].yes_price.median()
        p5 = g[(g.ts >= t_re + 300) & (g.ts < t_re + 420)].yes_price.median()
        p15 = g[(g.ts >= t_re + 900) & (g.ts < t_re + 1020)].yes_price.median()
        n1 = int(((g.ts >= t_re) & (g.ts < t_re + 60)).sum())
        out.append(dict(match_id=r.match_id, period=r.period, gap_min=r.inplay_max_gap_s / 60, before=before,
                        reopen_1min=first, after_5min=p5, after_15min=p15, prints_first_min=n1))
    Hh = pd.DataFrame(out)
    Hh['abs_move_5'] = (Hh.after_5min - Hh.reopen_1min).abs()
    Hh.to_csv(f'{RS}/h5_halts.csv', index=False)
    print(len(Hh), Hh[['gap_min', 'abs_move_5']].describe().round(3).to_string())
    print(Hh.sort_values('abs_move_5', ascending=False).head(5).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
