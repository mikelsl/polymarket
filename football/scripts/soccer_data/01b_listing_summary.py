"""Step 1b: what Polymarket lists per league and month (from raw/gamma_index.json and the raw main events).

Writes results/soccer_data/pm_listing_by_month.csv: main events, '-more-markets' events, other sibling event kinds,
share of 1X2 markets tagged sportsMarketType='moneyline', fee settings. No network.
Usage: .venv/bin/python -I scripts/soccer_data/01b_listing_summary.py
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import collections
import json
import re

import pandas as pd

from common import DATA, RAW, RES, log

idx = json.load(open(f'{RAW}/gamma_index.json'))
rows = []
for r in idx:
    m = re.search(r'-(\d{4}-\d{2})-\d{2}(?:-|$)', r['slug'] or '')
    rows.append(dict(league=r['series_league'], month=m.group(1) if m else 'n/a', kind=r['kind']))
df = pd.DataFrame(rows)
df['k'] = df.kind.where(df.kind.isin(['main', 'more', 'other']), 'sibling_other')
t = df.pivot_table(index=['league', 'month'], columns='k', aggfunc='size', fill_value=0)
kinds = df[df.kind.str.startswith('sub:')].groupby(['league', 'month']).kind.apply(lambda s: ';'.join(sorted({x[4:] for x in s})))
mk = pd.read_parquet(f'{DATA}/markets.parquet')
mk = mk[mk.role.isin(['home', 'draw', 'away'])]
mk['month'] = mk.match_id.str[-10:-3]
g = mk.groupby(['league', 'month']).agg(markets_1x2=('condition_id', 'size'),
                                        share_typed_moneyline=('sports_market_type', lambda s: round(float((s == 'moneyline').mean()), 3)),
                                        share_fees_enabled=('fees_enabled', lambda s: round(float(s.mean()), 3)),
                                        fee_rates=('fee_rate', lambda s: ';'.join(f'{x:g}:{n}' for x, n in s.fillna(0).value_counts().items())))
out = t.join(kinds.rename('sibling_kinds')).join(g).reset_index()
out.to_csv(f'{RES}/pm_listing_by_month.csv', index=False)
log('written', len(out), 'rows; other (non-match) events:', [r['slug'] for r in idx if r['kind'] == 'other'][:20])
first_more = df[df.kind == 'more'].groupby('league').month.min().to_dict()
first_sib = df[df.kind.str.startswith('sub:')].groupby('league').month.min().to_dict()
log('first more-markets month', first_more, 'first other sibling month', first_sib)
