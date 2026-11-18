"""Spot checks across tables for a few games (no network). Usage: .venv/bin/python scripts/nba_data/spot_check.py [slug ...]"""
import sys

import pandas as pd
import pyarrow.dataset as ds

from common import DATA

slugs = sys.argv[1:] or ['nba-nyk-bos-2024-10-22', 'nba-cle-nyk-2025-12-25', 'nba-mia-lal-2026-11-02']
g = pd.read_parquet(DATA / 'games.parquet')
mk = pd.read_parquet(DATA / 'markets.parquet')
tr = ds.dataset(DATA / 'trades', format='parquet', partitioning='hive')
ph = ds.dataset(DATA / 'price_history', format='parquet', partitioning='hive')
pbp = pd.read_parquet(DATA / 'espn_pbp.parquet')
pc = pd.read_parquet(DATA / 'pickcenter.parquet')
for s in slugs:
    r = g[g.slug == s].iloc[0]
    print(f'== {s} game_id={r.game_id} type={r.game_type} start={r.game_start_utc} espn={r.espn_event_id} '
          f'final away-home {r.espn_away_score}-{r.espn_home_score} ml_winner={r.ml_winner_index}/{r.ml_winner_espn}')
    t = tr.to_table(filter=ds.field('game_id') == r.game_id).to_pandas()
    p = ph.to_table(filter=ds.field('game_id') == r.game_id).to_pandas()
    for role, grp in t.groupby('market_role', observed=True):
        m = mk[mk.market_id == grp.market_id.iloc[0]].iloc[0]
        pp = p[p.market_role == role]
        print(f'  {role:12s} {m.question!r:40s} prints={len(grp):6d} sum_size={grp["size"].sum():14,.2f} gamma_vol={m.volume} '
              f'price_pts={pp.groupby("token_index").size().to_dict()} last_p={pp.sort_values("t").groupby("token_index").p.last().round(3).to_dict()} '
              f'winner={m.winner_index}')
    b = pbp[pbp.espn_event_id == r.espn_event_id].sort_values(['elapsed_s', 'seq'])
    if len(b):
        print(f'  pbp plays={len(b)} first={b.wallclock_utc.min()} tipoff={r.espn_tipoff_utc} last={b.wallclock_utc.max()} '
              f'final pbp {b.away_score.iloc[-1]}-{b.home_score.iloc[-1]}')
    print('  odds', pc[pc.espn_event_id == r.espn_event_id][['source', 'provider_name', 'details', 'spread', 'over_under',
                                                               'home_ml', 'away_ml', 'ml_home_open', 'ml_home_close']].to_dict('records'))
