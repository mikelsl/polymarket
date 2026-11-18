"""H3 diagnosis: split the jump-anchored post-goal rule (L=60 s, worst-of-15 s fill, model edge >= 3c, scorer YES +
conceder NO) by the ESPN delay (ESPN goal wallclock minus print-located jump). Large delays = goals confirmed late
(VAR checks): the jump-anchored sample keeps only goals that stood, a look-ahead. Also counts ESPN VAR rows.
Output: results/soccer_strat/h3c_delay_split.csv
"""
import sys

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import numpy as np
import pandas as pd

from lib import DS, RS, SD, STAKE, cluster_boot, fee_per_share


def main():
    G = pd.read_parquet(f'{DS}/goal_entries.parquet')
    g = G[(G.anchor == 'jump') & (G.L == 60)]
    e_yes = g.model - g.askw - fee_per_share(g.askw)
    e_no = g.bidw - g.model - fee_per_share(g.bidw)
    parts = []
    for dr, sel in [('yes', (g.rel == 'team') & (e_yes >= 0.03) & g.askw.between(0.03, 0.97)),
                    ('no', (g.rel == 'opp') & (e_no >= 0.03) & g.bidw.between(0.03, 0.97))]:
        T = g[sel]
        pin = T.askw if dr == 'yes' else 1 - T.bidw
        y = T.y if dr == 'yes' else 1 - T.y
        pout = T.xb_300 if dr == 'yes' else 1 - T.xa_300
        sh = STAKE / pin
        res = sh * (y - pin) - sh * fee_per_share(pin)
        p300 = np.where(T.xres_300 == 1, res, sh * (pout - pin) - sh * fee_per_share(pin) - sh * fee_per_share(pout))
        parts.append(T.assign(pnl_res=res, pnl_300=p300, stake=STAKE))
    T = pd.concat(parts)
    T['delay'] = pd.cut(T.wc_minus_jump, [-1e9, -10, 10, 60, 1e9], labels=['<-10s', '-10..10s', '10..60s', '>60s'])
    rows = []
    for (k, s), d in T.groupby(['delay', 'period'], observed=True):
        for col in ['pnl_300', 'pnl_res']:
            dd = d.dropna(subset=[col]).rename(columns={col: 'pnl'})
            if len(dd) < 10:
                continue
            r = cluster_boot(dd, n_boot=1000)
            rows.append(dict(delay=k, set=s, horizon=col, n=r['n'], roi=r['roi'], lo=r['ci_roi'][0], hi=r['ci_roi'][1]))
    R = pd.DataFrame(rows)
    R.to_csv(f'{RS}/h3c_delay_split.csv', index=False)
    print(R.round(3).to_string(index=False))
    ev = pd.read_parquet(f'{SD}/events.parquet', columns=['type', 'source'])
    print(ev[ev.source == 'commentary'].type.value_counts().to_dict())


if __name__ == '__main__':
    main()
