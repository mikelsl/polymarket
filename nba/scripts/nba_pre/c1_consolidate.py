"""Consolidate the per-market pulls into two tables (run after f1_fetch.py finished):

data/nba_pre/line_prices.parquet : market_id, t, p0 (token-0 CLOB minute mid) for every traded full-game spread / total
                                   line of the 2025-26 and 2026-27 core games (nba_data main lines + nba_pre pulls)
data/nba_pre/spread_trades.parquet / total_trades.parquet: market_id, ts, token_index, side, price, size for every
                                   traded full-game spread / total line of those games (nba_data main-line prints +
                                   nba_pre pulls; totals only where pulled)
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common_pre import DATA, SRC

parts = []
for f in sorted((DATA / 'prices_parts').glob('*.parquet')):
    t = pq.read_table(f)
    parts.append(t.append_column('market_id', pa.array([f.stem] * t.num_rows, pa.string())))
mine = pa.concat_tables(parts).to_pandas().rename(columns={'p': 'p0'})
ph = pd.read_parquet(SRC / 'price_history', columns=['market_id', 'market_role', 'token_index', 't', 'p'],
                     filters=[('market_role', 'in', ['main_spread', 'main_total'])])
ph = ph[ph.token_index == 0][['market_id', 't', 'p']].rename(columns={'p': 'p0'})
ph['market_id'] = ph.market_id.astype(str)
lp = pd.concat([mine[['market_id', 't', 'p0']], ph], ignore_index=True)
lp['p0'] = lp.p0.clip(0, 1)
lp = lp.drop_duplicates(['market_id', 't']).sort_values(['market_id', 't'])
lp.to_parquet(DATA / 'line_prices.parquet', index=False)
print('line_prices', len(lp), lp.market_id.nunique())

# prints: nba_pre pulls first, else the nba_data per-market part (main lines and the 2026-03 null-volume probe lines)
g = pd.read_parquet(SRC / 'games.parquet', columns=['game_id', 'is_core_game', 'season'])
g = g[g.is_core_game & g.season.isin(['2025-26', '2026-27'])]
mk = pd.read_parquet(SRC / 'markets.parquet', columns=['market_id', 'game_id', 'kind', 'event_role'])
mk = mk[mk.game_id.isin(g.game_id) & mk.kind.isin(['spreads', 'totals']) & mk.event_role.isin(['main', 'v2'])]
cols = ['ts', 'token_index', 'side', 'price', 'size']
out = {'spreads': [], 'totals': []}
for mid, kind in zip(mk.market_id, mk.kind):
    f = DATA / 'trades_parts' / f'{mid}.parquet'
    if not f.exists():
        f = SRC / 'work' / 'trades_parts' / f'{mid}.parquet'
    if not f.exists():
        continue
    t = pq.read_table(f, columns=cols).to_pandas()
    t['side'] = t.side.astype(str)
    t['market_id'] = mid
    out[kind].append(t)
for kind, name in (('spreads', 'spread_trades'), ('totals', 'total_trades')):
    st = pd.concat(out[kind], ignore_index=True)
    st = st.sort_values(['market_id', 'ts'], kind='stable')
    st.to_parquet(DATA / f'{name}.parquet', index=False)
    print(name, len(st), st.market_id.nunique())
