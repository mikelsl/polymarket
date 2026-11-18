"""Print the headline numbers used in the final hand-off (no network)."""
import json

import pandas as pd
import pyarrow.dataset as ds

from common import DATA, RES

g = pd.read_parquet(DATA / 'games.parquet')
mk = pd.read_parquet(DATA / 'markets.parquet')
c = g[g.is_core_game]
print('events', len(g), 'core', len(c))
print(g.groupby(['season', 'game_type']).size().unstack(fill_value=0).to_string())
print('core by season:', c.groupby('season').size().to_dict())
print('core with main spread:', c.groupby('season').main_spread_market_id.count().to_dict(),
      'with main total:', c.groupby('season').main_total_market_id.count().to_dict())
print('all events with main spread / total:', int(g.main_spread_market_id.notna().sum()), int(g.main_total_market_id.notna().sum()))
sp = mk[mk.kind.isin(['spreads', 'totals']) & mk.event_role.isin(['main', 'v2'])].merge(g[['game_id', 'season', 'is_core_game']], on='game_id')
print('spread/total lines per core game (median):',
      sp[sp.is_core_game].groupby(['season', 'kind']).game_id.value_counts().groupby(['season', 'kind']).median().to_dict())
tr = ds.dataset(DATA / 'trades', format='parquet', partitioning='hive')
t = tr.to_table(columns=['season', 'market_role', 'size']).to_pandas()
print('trades rows', len(t)); print(t.groupby(['season', 'market_role'], observed=True)['size'].agg(['size', 'sum']).round(0).to_string())
ph = ds.dataset(DATA / 'price_history', format='parquet', partitioning='hive')
print('price rows', ph.count_rows())
du = lambda p: sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
for p in ('trades', 'price_history', 'raw', 'work'):
    print(p, round(du(DATA / p) / 1e6, 1), 'MB')
print('total data/nba_data', round(du(DATA) / 1e6, 1), 'MB')
print(json.dumps(json.loads((RES / 'qa_summary.json').read_text()), indent=0))
