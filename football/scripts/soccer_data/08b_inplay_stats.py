"""Step 8b: per-match in-play trading statistics from trades.parquet, merged into matches.parquet.

inplay_prints          taker prints on the three 1X2 markets between ESPN kickoff and full-time wallclocks
inplay_max_gap_s       longest interval without any print in that window (window edges count as boundaries)
inplay_max_gap_start_ts start of that interval (epoch s UTC)
Long gaps mark in-play trading halts (single market or platform-wide), see REPORT.
Usage: .venv/bin/python -I scripts/soccer_data/08b_inplay_stats.py
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import numpy as np
import pandas as pd

from common import DATA, log

m = pd.read_parquet(f'{DATA}/matches.parquet')
m = m.drop(columns=[c for c in ('inplay_prints', 'inplay_max_gap_s', 'inplay_max_gap_start_ts') if c in m.columns])
tr = pd.read_parquet(f'{DATA}/trades.parquet', columns=['match_id', 'ts'])
win = m.set_index('match_id')[['kickoff_wall_ts', 'fulltime_wall_ts']].dropna().astype('int64')
out = []
for mid, g in tr.groupby('match_id', observed=True):
    if mid not in win.index:
        continue
    a, b = win.loc[mid]
    t = np.sort(g.ts.to_numpy())
    t = t[(t >= a) & (t <= b)]
    edges = np.concatenate([[a], t, [b]])
    gaps = np.diff(edges)
    j = int(np.argmax(gaps))
    out.append(dict(match_id=mid, inplay_prints=len(t), inplay_max_gap_s=int(gaps[j]), inplay_max_gap_start_ts=int(edges[j])))
s = pd.DataFrame(out)
m = m.merge(s, on='match_id', how='left')
for c in ('inplay_prints', 'inplay_max_gap_s', 'inplay_max_gap_start_ts'):
    m[c] = m[c].astype('Int64')
m.to_parquet(f'{DATA}/matches.parquet', index=False)
log('in-play stats for', len(s), 'matches; max gap quantiles', s.inplay_max_gap_s.describe(percentiles=[.5, .9, .95, .99]).round(0).to_dict(),
    'matches with gap >= 600 s:', int((s.inplay_max_gap_s >= 600).sum()))
