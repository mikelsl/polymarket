"""Pre-match executable price snapshots from taker prints (1X2 markets).

For each analysis-ready match, 1X2 role and offset x before the scheduled kickoff (t = kickoff_ts + x):
  ask_yes  = YES price of the first print at/after t where the taker gained YES exposure (BUY YES or SELL NO)
  bid_yes  = YES price of the first print at/after t where the taker gained NO exposure (SELL YES or BUY NO)
  mid_med  = median YES price of prints in [t - 10 min, t)  (both sides; a mid proxy, not tradable)
  vwap_pre = size-weighted YES price of prints in [t - 10 min, t)
  cum_usdc = taker notional of this market before t (no look-ahead liquidity measure)
Buying YES costs ask_yes, buying NO costs 1 - bid_yes, selling YES gets bid_yes, selling NO gets 1 - ask_yes.
Output: data/soccer_strat/prematch_snap.parquet
"""
import sys

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import numpy as np
import pandas as pd

from lib import DS, load_matches, load_trades, log

OFFSETS = [-7 * 86400, -3 * 86400, -2 * 86400, -86400, -12 * 3600, -6 * 3600, -3 * 3600, -2 * 3600, -3600, -1800,
           -900, -600, -300, -120, -60, 0]
MAXLAG = 900  # first print must come within 15 min after t


def main():
    m = load_matches()
    ko = dict(zip(m.match_id, m.kickoff_ts))
    kw = dict(zip(m.match_id, m.kickoff_wall_ts))
    tr = load_trades(filters=[('match_id', 'in', list(m.match_id))])
    tr = tr[tr.t_rel_kickoff_s < 600].copy()
    tr['usdc'] = tr.price * tr['size']
    tr = tr.sort_values(['match_id', 'role', 'ts'], kind='mergesort')
    log('trades', len(tr))
    out = []
    for (mid, role), g in tr.groupby(['match_id', 'role'], sort=False):
        k = ko[mid]
        ts = g.ts.values
        yp = g.yes_price.values
        sz = g['size'].values
        bys = g.taker_buys_yes.values
        cu = np.cumsum(g.usdc.values)
        ts_b, yp_b, sz_b = ts[bys], yp[bys], sz[bys]
        ts_s, yp_s, sz_s = ts[~bys], yp[~bys], sz[~bys]
        for x in OFFSETS:
            t = k + x
            r = dict(match_id=mid, role=role, offset_s=x, t=t)
            i = np.searchsorted(ts_b, t, 'left')
            if i < len(ts_b) and ts_b[i] - t <= MAXLAG:
                r.update(ask_yes=yp_b[i], ask_ts=ts_b[i], ask_size=sz_b[i])
            j = np.searchsorted(ts_s, t, 'left')
            if j < len(ts_s) and ts_s[j] - t <= MAXLAG:
                r.update(bid_yes=yp_s[j], bid_ts=ts_s[j], bid_size=sz_s[j])
            a, b = np.searchsorted(ts, t - 600, 'left'), np.searchsorted(ts, t, 'left')
            if b > a:
                r.update(mid_med=float(np.median(yp[a:b])), vwap_pre=float(np.sum(yp[a:b] * sz[a:b]) / np.sum(sz[a:b])),
                         n_pre=int(b - a))
            # last print before t, and spread proxy from the last buy-YES and sell-YES prints in the previous 10 min
            if b > 0:
                r.update(last_px=yp[b - 1], last_ts=ts[b - 1], cum_usdc=float(cu[b - 1]))
            ib, js = np.searchsorted(ts_b, t, 'left'), np.searchsorted(ts_s, t, 'left')
            if ib > 0 and js > 0 and t - ts_b[ib - 1] <= 600 and t - ts_s[js - 1] <= 600:
                r.update(last_ask=yp_b[ib - 1], last_bid=yp_s[js - 1])
            out.append(r)
    df = pd.DataFrame(out)
    df['kickoff_wall_ts'] = df.match_id.map(kw)
    df.to_parquet(f'{DS}/prematch_snap.parquet', index=False)
    log('rows', len(df))
    log(df.groupby('offset_s')[['ask_yes', 'bid_yes', 'mid_med', 'last_ask']].count())


if __name__ == '__main__':
    main()
