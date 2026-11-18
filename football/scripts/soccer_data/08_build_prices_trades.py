"""Step 8: final `price_history`, `trades` and `markets` tables (league matches only; play-offs excluded).

price_history.parquet: one row per (token, minute point) from raw/prices/prices_raw.jsonl, with windows of re-dated
  matches replaced by raw/prices/prices_raw_fix.jsonl.
trades.parquet: all taker prints from data/soccer_data/trades_parts/<season>/<conditionId>.parquet, chronological per
  market, with derived columns (yes_price, taker_buys_yes, notional_usdc, t_rel_kickoff_s, seq_in_market).
markets.parquet: rewritten with the canonical kickoff and per-market print statistics.
Usage: .venv/bin/python -I scripts/soccer_data/08_build_prices_trades.py [--skip-trades]
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import DATA, RAW, log

PARTS = f'{DATA}/trades_parts'


def load_prices():
    recs = {}
    for name, basis in (('prices_raw.jsonl', 'pm_gameStartTime'), ('prices_raw_fix.jsonl', 'espn_scheduled')):
        p = f'{RAW}/prices/{name}'
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                r['window_basis'] = basis
                recs[r['token_id']] = r  # fix file read last -> overrides
    return recs


def build_prices(m, mk):
    recs = load_prices()
    kick = dict(zip(m.match_id, m.kickoff_ts.astype('int64')))
    keep = set(mk.token0_id)
    cols = {k: [] for k in ('match_id', 'role', 'condition_id', 'token_id', 'outcome', 'window_basis')}
    ts_l, px_l, qa = [], [], []
    for tok, r in recs.items():
        if tok not in keep or r['match_id'] not in kick:
            continue
        h = np.array(r['history'], dtype='float64').reshape(-1, 2)
        n_raw = len(h)
        h = h[(h[:, 0] >= r['start_ts']) & (h[:, 0] <= r['end_ts'])]  # CLOB occasionally returns points outside the window
        n = len(h)
        for k in cols:
            cols[k].append(np.repeat(np.array([r[k]], dtype=object), n))
        ts_l.append(h[:, 0].astype('int64'))
        px_l.append(h[:, 1])
        k0 = kick[r['match_id']]
        t = h[:, 0].astype('int64')
        gaps = np.diff(t) if n > 1 else np.array([0])
        qa.append(dict(token_id=tok, match_id=r['match_id'], role=r['role'], window_basis=r['window_basis'], n_points=n,
                       n_dropped_outside_window=n_raw - n,
                       first_rel_s=int(t[0] - k0) if n else None, last_rel_s=int(t[-1] - k0) if n else None,
                       max_gap_s=int(gaps.max()) if n > 1 else None,
                       n_inplay=int(((t >= k0) & (t <= k0 + 9000)).sum()), n_pre=int(((t >= k0 - 21600) & (t < k0)).sum())))
    ts = np.concatenate(ts_l)
    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df['ts'] = ts
    df['price'] = np.concatenate(px_l)
    df['ts_utc'] = pd.to_datetime(df.ts, unit='s', utc=True)
    df['t_rel_kickoff_s'] = (df.ts - df.match_id.map(kick)).astype('int32')
    df = df.sort_values(['match_id', 'role', 'ts']).reset_index(drop=True)
    for c in ('match_id', 'role', 'condition_id', 'token_id', 'outcome', 'window_basis'):
        df[c] = df[c].astype('category')
    df.to_parquet(f'{DATA}/price_history.parquet', index=False)
    q = pd.DataFrame(qa)
    log('price_history rows', len(df), 'tokens', df.token_id.nunique(), 'tokens with 0 points', int((q.n_points == 0).sum()))
    return q


def build_trades(m, mk):
    meta = mk[mk.role.isin(['home', 'draw', 'away'])][['condition_id', 'match_id', 'role']].merge(
        m[['match_id', 'season', 'league', 'kickoff_ts']], on='match_id')
    meta = meta.sort_values(['kickoff_ts', 'match_id', 'role'])
    writer, stats, total = None, [], 0
    out = f'{DATA}/trades.parquet'
    tmp = out + '.tmp'
    buf = []

    def flush():
        nonlocal writer, buf
        if not buf:
            return
        t = pa.Table.from_pandas(pd.concat(buf, ignore_index=True), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression='zstd')
        writer.write_table(t, row_group_size=1_000_000)
        buf = []

    nb = 0
    for r in meta.itertuples():
        p = f'{PARTS}/{r.season}/{r.condition_id}.parquet'
        if not os.path.exists(p):
            stats.append(dict(condition_id=r.condition_id, fetched=False))
            continue
        d = pd.read_parquet(p)
        if len(d) == 0:
            stats.append(dict(condition_id=r.condition_id, fetched=True, n_taker_prints=0, taker_size_sum=0.0))
            continue
        d = d.iloc[::-1].reset_index(drop=True)  # API is newest-first; reverse, then stable sort by time
        d = d.sort_values('timestamp', kind='stable').reset_index(drop=True)
        d = d.astype({'price': 'float64', 'size': 'float64', 'timestamp': 'int64', 'outcome_index': 'int8'})
        o = pd.DataFrame({
            'match_id': r.match_id, 'season': r.season, 'league': r.league, 'role': r.role,
            'condition_id': d.condition_id, 'token_id': d.token_id, 'outcome': d.outcome, 'outcome_index': d.outcome_index,
            'side': d.side, 'price': d.price, 'size': d['size'], 'ts': d.timestamp,
            'ts_utc': pd.to_datetime(d.timestamp, unit='s', utc=True),
            't_rel_kickoff_s': (d.timestamp - int(r.kickoff_ts)).astype('int64'),
            'yes_price': np.where(d.outcome_index == 0, d.price, 1.0 - d.price),
            'taker_buys_yes': ((d.side == 'BUY') & (d.outcome_index == 0)) | ((d.side == 'SELL') & (d.outcome_index == 1)),
            'notional_usdc': d.price * d['size'],
            'seq_in_market': np.arange(len(d), dtype='int32'),
            'transaction_hash': d.transaction_hash, 'proxy_wallet': d.proxy_wallet})
        stats.append(dict(condition_id=r.condition_id, fetched=True, n_taker_prints=len(o), taker_size_sum=float(o['size'].sum()),
                          taker_notional_sum=float(o.notional_usdc.sum()), first_print_ts=int(o.ts.iloc[0]), last_print_ts=int(o.ts.iloc[-1]),
                          n_prints_inplay=int(((o.t_rel_kickoff_s >= 0) & (o.t_rel_kickoff_s <= 9000)).sum())))
        buf.append(o)
        nb += len(o)
        total += len(o)
        if nb >= 1_000_000:
            flush()
            nb = 0
    flush()
    if writer is not None:
        writer.close()
        os.replace(tmp, out)
    log('trades rows', total)
    return pd.DataFrame(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-trades', action='store_true')
    a = ap.parse_args()
    m = pd.read_parquet(f'{DATA}/matches.parquet')
    # matches.parquet already holds league matches only
    mk = pd.read_parquet(f'{DATA}/markets.parquet')
    mk = mk.drop(columns=[c for c in mk.columns if c in ('kickoff_ts', 'season', 'league', 'n_taker_prints', 'taker_size_sum',
                                                          'taker_notional_sum', 'first_print_ts', 'last_print_ts', 'n_prints_inplay',
                                                          'fetched', 'taker_size_to_volume')])
    mk = mk.merge(m[['match_id', 'season', 'league', 'kickoff_ts']], on='match_id', how='inner')
    pq_ = build_prices(m, mk)
    pq_.to_parquet(f'{DATA}/qa_price_tokens.parquet', index=False)
    if not a.skip_trades:
        st = build_trades(m, mk)
        mk = mk.merge(st, on='condition_id', how='left')
        mk['taker_size_to_volume'] = mk.taker_size_sum / mk.volume
        for c in ('first_print_ts', 'last_print_ts', 'n_taker_prints', 'n_prints_inplay'):
            mk[c] = mk[c].astype('Int64')
    first = ['match_id', 'league', 'season', 'kickoff_ts', 'role']
    mk = mk[first + [c for c in mk.columns if c not in first]]
    mk.to_parquet(f'{DATA}/markets.parquet', index=False)
    log('markets', len(mk), mk.role.value_counts().to_dict())


if __name__ == '__main__':
    main()
