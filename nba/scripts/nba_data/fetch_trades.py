"""Step 3 - full taker trade prints from Data API v2 (/v2/trades?condition=..&taker_only=true, newest first, cursor paging).

Modes
  --mode probe : full-game spread/total markets of in-scope games whose Gamma `volume` is null (Gamma lost volume
                 for most markets of 2026-03-04..03-25 and for many untraded 2026-27 lines). Their trade volume is
                 needed to pick the main spread / total. Run gamma_tables.py again afterwards.
  --mode main  : the moneyline, main spread and main total of every in-scope game.
  --refresh-open : re-fetch markets that were not closed when last fetched (latest games).

One part file per market: data/nba_data/work/trades_parts/<market_id>.parquet, plus a manifest row in
data/nba_data/work/trades_manifest/<market_id>.json (rows, pages, share sum, ts range, Gamma closed flag). Resumable: a
market with a manifest whose Gamma closed flag was true is skipped. build_tables.py consolidates the parts.

Columns per print: ts (epoch s, UTC), token_index (0/1 = position in clobTokenIds, -1 = token not in the market),
side (BUY/SELL from the taker's view: BUY = taker bought token_index at price), price, size (shares),
outcome_index_api (the API's own outcome_index, kept to verify token_index), tx_hash, taker_wallet (proxy wallet).

Usage: .venv/bin/python scripts/nba_data/fetch_trades.py --mode main --threads 4
Threads: Data API only; keep the total of concurrently running jobs <= 6.
"""
import argparse
import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from common import DATA, DATA_API, STATS, WORK, get_json, log, write_parquet

PARTS = WORK / 'trades_parts'
MANI = WORK / 'trades_manifest'
MAX_PAGES = 5000            # 5M prints per market; far above anything seen (largest 2025-26 moneyline ~11k)


def fetch_market(row):
    mid, cond = row['market_id'], row['condition_id']
    toks = {row['token_0']: 0, row['token_1']: 1}
    t0 = time.time()
    recs, cur, pages, truncated = [], None, 0, False
    while True:
        params = dict(condition=cond, taker_only='true', limit=1000)
        if cur:
            params['cursor'] = cur
        js = get_json(f'{DATA_API}/trades', params)
        data = js.get('data') or []
        pages += 1
        for r in data:
            recs.append((r.get('timestamp'), toks.get(str(r.get('token_id')), -1), r.get('side'), r.get('price'),
                         r.get('size'), r.get('outcome_index'), r.get('transaction_hash'), r.get('proxy_wallet')))
        cur = (js.get('pagination') or {}).get('next_cursor')
        if not cur or not data:
            break
        if pages >= MAX_PAGES:
            truncated = True
            break
    df = pd.DataFrame(recs, columns=['ts', 'token_index', 'side', 'price', 'size', 'outcome_index_api', 'tx_hash',
                                     'taker_wallet'])
    df = df.astype({'ts': 'int64', 'token_index': 'int8', 'price': 'float64', 'size': 'float64'})
    df['outcome_index_api'] = pd.to_numeric(df['outcome_index_api'], errors='coerce').astype('Int8')
    df['side'] = df['side'].astype('string')
    df = df.sort_values(['ts', 'tx_hash'], kind='stable').reset_index(drop=True)
    write_parquet(df, PARTS / f'{mid}.parquet')
    man = dict(market_id=mid, condition_id=cond, n_rows=int(len(df)), n_pages=pages, truncated=truncated,
               sum_size=float(df['size'].sum()), sum_notional=float((df['size'] * df['price']).sum()),
               min_ts=int(df['ts'].min()) if len(df) else None, max_ts=int(df['ts'].max()) if len(df) else None,
               n_token_unmatched=int((df['token_index'] < 0).sum()),
               n_outcome_index_disagree=int(((df['outcome_index_api'].astype('float') != df['token_index'])
                                             & df['outcome_index_api'].notna()).sum()),
               n_exact_dup=int(df.duplicated().sum()),
               gamma_closed=bool(row.get('closed')), fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),
               secs=round(time.time() - t0, 2))
    (MANI / f'{mid}.json').write_text(json.dumps(man))
    return man


def targets(mode):
    mk = pd.read_parquet(DATA / 'markets.parquet')
    g = pd.read_parquet(WORK / 'games_gamma.parquet')
    mk = mk[mk.game_id.isin(g[g.in_scope].game_id)]
    if mode == 'probe':
        t = mk[mk.kind.isin(['spreads', 'totals']) & mk.event_role.isin(['main', 'v2']) & mk.volume.isna()]
    else:
        t = mk[mk.role.isin(['moneyline', 'main_spread', 'main_total'])]
    return t[t.condition_id.notna()].drop_duplicates('market_id')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['probe', 'main'], required=True)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--refresh-open', action='store_true')
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    MANI.mkdir(parents=True, exist_ok=True)
    t = targets(args.mode)
    todo = []
    for r in t.to_dict('records'):
        mf = MANI / f"{r['market_id']}.json"
        if mf.exists() and (PARTS / f"{r['market_id']}.parquet").exists():
            m = json.loads(mf.read_text())
            if m.get('gamma_closed') and not m.get('truncated'):
                continue          # complete: the market was closed (no more trading) when its prints were pulled
            if not args.refresh_open:
                continue          # pulled while still open; re-pull only with --refresh-open
        todo.append(r)
    log(f'mode={args.mode}: {len(t)} target markets, {len(todo)} to fetch (threads={args.threads})')
    done = rows = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.threads) as ex:
        futs = {ex.submit(fetch_market, r): r for r in todo}
        for fu in as_completed(futs):
            r = futs[fu]
            try:
                m = fu.result()
                rows += m['n_rows']
                if m['truncated'] or m['n_token_unmatched']:
                    log(f"  WARN {r['market_id']} {r['market_slug']}: {m}")
            except Exception as e:
                log(f"  FAILED {r['market_id']} {r['market_slug']}: {e}")
            done += 1
            if done % 200 == 0 or done == len(todo):
                el = time.time() - t0
                log(f'  {done}/{len(todo)} markets, {rows:,} prints, {el / 60:.1f} min, '
                    f'eta {el / done * (len(todo) - done) / 60:.1f} min, http {STATS}')
    log('done')


if __name__ == '__main__':
    main()
