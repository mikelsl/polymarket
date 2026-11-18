"""nba_pre fetcher (read-only): data the nba_data set does not have, for pre-game line selection and the live-rule replay.

Modes
  --mode prices : CLOB /prices-history (fidelity=1) of token 0 for every traded full-game spread / total line of the
                  2025-26 and 2026-27 core games that nba_data did not already pull (token 1 = 1 - token 0 on the CLOB
                  mid series; checked on the moneylines: |p0 + p1 - 1| <= 0.01 everywhere).
                  Window [min(pm start, espn sched) - 6h, max(...) + 4h], the same as nba_data.
  --mode trades [--kind totals] : Data API /v2/trades?condition=..&taker_only=true, full history, for every traded full-game SPREAD
                  line of those games that nba_data did not already pull (needed for the volume ranking at the
                  scheduled start used by the live bot, and for in-game exits).
Output: data/nba_pre/price_parts/<market_id>.parquet (t, p) and data/nba_pre/trades_parts/<market_id>.parquet
(ts, token_index, side, price, size), plus a JSON manifest per market. Resumable: finished markets are skipped.

Usage (background): nohup caffeinate -is .venv/bin/python -u scripts/nba_pre/f1_fetch.py --mode prices --threads 3 \
                      > results/nba_pre/fetch_prices.log 2>&1 &
Rates: CLOB 12 req/s, Data API 8 req/s from this worker (shared budgets: README).
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data' / 'nba_data'
OUT = ROOT / 'data' / 'nba_pre'
CLOB = 'https://clob.polymarket.com'
DATA_API = 'https://data-api.polymarket.com/v2'
RPS = {'clob.polymarket.com': 12.0, 'data-api.polymarket.com': 8.0}
STATS = {'requests': 0, 'retries': 0, 'http429': 0}
_lock = threading.Lock()
_next = {h: 0.0 for h in RPS}
_tls = threading.local()


def log(*a):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), *a, flush=True)


def _wait(host):
    with _lock:
        now = time.monotonic()
        t = max(now, _next[host])
        _next[host] = t + 1.0 / RPS[host]
    if t > now:
        time.sleep(t - now)


def get_json(url, params, tries=8):
    host = url.split('/')[2]
    if not hasattr(_tls, 's'):
        _tls.s = requests.Session()
    for i in range(tries):
        _wait(host)
        try:
            r = _tls.s.get(url, params=params, timeout=60)
            with _lock:
                STATS['requests'] += 1
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                with _lock:
                    STATS['http429'] += 1
                    _next[host] = max(_next[host], time.monotonic() + float(r.headers.get('Retry-After') or 5))
            elif r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        with _lock:
            STATS['retries'] += 1
        time.sleep(min(60, 2 ** i) + random.random())
    raise RuntimeError(f'failed {url} {params}')


def write_parquet(df, path):
    tmp = path.with_suffix(f'.tmp{threading.get_ident()}')
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def targets(mode, kind='spreads'):
    g = pd.read_parquet(SRC / 'games.parquet')
    g = g[g.is_core_game & g.season.isin(['2025-26', '2026-27'])][['game_id', 'game_start_utc', 'espn_sched_utc']]
    m = pd.read_parquet(SRC / 'markets.parquet')
    m = m[m.kind.isin(['spreads', 'totals']) & m.event_role.isin(['main', 'v2'])
          & (m.volume_effective.fillna(0) > 0)].merge(g, on='game_id')
    have = SRC / 'work' / ('price_parts' if mode == 'prices' else 'trades_parts')
    have = {p.stem for p in have.glob('*.parquet')}
    if mode == 'trades':
        m = m[m.kind == kind]
    m = m[~m.market_id.isin(have) & m.condition_id.notna()]
    return m.drop_duplicates('market_id')


def fetch_prices(r, parts, mani):
    gst = int(pd.Timestamp(r['game_start_utc']).timestamp())
    alt = [gst] + ([int(pd.Timestamp(r['espn_sched_utc']).timestamp())] if pd.notna(r['espn_sched_utc']) else [])
    a, b = min(alt) - 6 * 3600, max(alt) + 4 * 3600
    js = get_json(f'{CLOB}/prices-history', dict(market=r['token_0'], startTs=a, endTs=b, fidelity=1)) or {}
    h = js.get('history') or []
    df = pd.DataFrame(h, columns=['t', 'p']).astype({'t': 'int64', 'p': 'float64'})
    write_parquet(df, parts / f"{r['market_id']}.parquet")
    (mani / f"{r['market_id']}.json").write_text(json.dumps(dict(market_id=r['market_id'], n=len(df), a=a, b=b)))
    return len(df)


def fetch_trades(r, parts, mani):
    toks = {r['token_0']: 0, r['token_1']: 1}
    recs, cur, pages = [], None, 0
    while True:
        params = dict(condition=r['condition_id'], taker_only='true', limit=1000)
        if cur:
            params['cursor'] = cur
        js = get_json(f'{DATA_API}/trades', params) or {}
        data = js.get('data') or []
        pages += 1
        for x in data:
            recs.append((x.get('timestamp'), toks.get(str(x.get('token_id')), -1), x.get('side'), x.get('price'),
                         x.get('size')))
        cur = (js.get('pagination') or {}).get('next_cursor')
        if not cur or not data or pages >= 3000:
            break
    df = pd.DataFrame(recs, columns=['ts', 'token_index', 'side', 'price', 'size'])
    df = df.astype({'ts': 'int64', 'token_index': 'int8', 'price': 'float64', 'size': 'float64'})
    df = df.sort_values('ts', kind='stable').reset_index(drop=True)
    write_parquet(df, parts / f"{r['market_id']}.parquet")
    (mani / f"{r['market_id']}.json").write_text(json.dumps(dict(market_id=r['market_id'], n=len(df), pages=pages,
                                                                  sum_size=float(df['size'].sum()))))
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['prices', 'trades'], required=True)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--kind', choices=['spreads', 'totals'], default='spreads', help='trades mode only')
    args = ap.parse_args()
    parts = OUT / f'{args.mode}_parts'
    mani = OUT / f'{args.mode}_manifest'
    parts.mkdir(parents=True, exist_ok=True)
    mani.mkdir(parents=True, exist_ok=True)
    t = targets(args.mode, args.kind)
    done_ids = {p.stem for p in mani.glob('*.json')}
    todo = [r for r in t.to_dict('records') if r['market_id'] not in done_ids]
    # 2026-27 first (smaller, most relevant for the live rule), then 2025-26 by date
    todo.sort(key=lambda r: (-pd.Timestamp(r['game_start_utc']).value))
    log(f'mode={args.mode}: {len(t)} targets, {len(todo)} to fetch')
    fn = fetch_prices if args.mode == 'prices' else fetch_trades
    t0, done, rows = time.time(), 0, 0
    with ThreadPoolExecutor(args.threads) as ex:
        futs = {ex.submit(fn, r, parts, mani): r for r in todo}
        for fu in as_completed(futs):
            r = futs[fu]
            try:
                rows += fu.result()
            except Exception as e:  # noqa: BLE001
                log('FAILED', r['market_id'], r.get('market_slug'), e)
            done += 1
            if done % 250 == 0 or done == len(todo):
                el = time.time() - t0
                log(f'{done}/{len(todo)} markets, {rows:,} rows, {el / 60:.1f} min, '
                    f'eta {el / done * (len(todo) - done) / 60:.1f} min, http {STATS}')
    log('done')


if __name__ == '__main__':
    main()
