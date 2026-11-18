"""Background: taker prints for O/U 2.5 and BTTS markets (analysis-ready matches), read-only Data API v2.

/v2/trades?condition=<conditionId>&taker_only=true&limit=1000&cursor=... returns newest first; we stop paging once a
page reaches prints older than kickoff - 2 h (only in-play and the last pre-match hours are needed).
Resumable: one parquet per market in data/soccer_strat/ou_btts_trades/. Max 4 concurrent requests, backoff on errors,
Retry-After honoured on 429. Stop: touch data/soccer_strat/ou_btts_STOP.
Run: nohup caffeinate -is .venv/bin/python -I scripts/soccer_strat/09_fetch_ou_btts_trades.py > results/soccer_strat/09_fetch.log 2>&1 &
"""
import concurrent.futures as cf
import os
import sys
import threading
import time

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import pandas as pd
import requests

from lib import DS, SD, log

OUT = f'{DS}/ou_btts_trades'
STOP = f'{DS}/ou_btts_STOP'
API = 'https://data-api.polymarket.com/v2/trades'
COLS = ['condition_id', 'token_id', 'outcome', 'outcome_index', 'side', 'price', 'size', 'timestamp']
SEM = threading.Semaphore(4)
TL = threading.local()


def get(params):
    if not hasattr(TL, 's'):
        TL.s = requests.Session()
    for k in range(8):
        with SEM:
            try:
                r = TL.s.get(API, params=params, timeout=30)
            except requests.RequestException:
                r = None
        if r is not None and r.status_code == 200:
            return r.json()
        wait = 2 ** k
        if r is not None and r.status_code == 429:
            wait = float(r.headers.get('Retry-After', wait))
        time.sleep(min(wait, 60))
    raise RuntimeError(f'failed {params.get("condition")}')


def fetch(job):
    out = f'{OUT}/{job["condition_id"]}.parquet'
    if os.path.exists(out) or os.path.exists(STOP):
        return 0
    rows, cur = [], None
    stop_ts = job['kickoff_ts'] - 7200
    while True:
        p = dict(condition=job['condition_id'], taker_only='true', limit=1000)
        if cur:
            p['cursor'] = cur
        r = get(p)
        d = r.get('data') or []
        rows.extend({k: x.get(k) for k in COLS} for x in d)
        cur = (r.get('pagination') or {}).get('next_cursor')
        if not cur or not d or min(int(x.get('timestamp') or 0) for x in d) < stop_ts:
            break
    df = pd.DataFrame(rows, columns=COLS).drop_duplicates()
    df.insert(0, 'role', job['role'])
    df.insert(0, 'match_id', job['match_id'])
    df.to_parquet(out + '.tmp', index=False)
    os.replace(out + '.tmp', out)
    return len(df)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_parquet(f'{SD}/matches.parquet')
    ready = set(m.match_id[m.analysis_ready])
    mk = pd.read_parquet(f'{SD}/markets.parquet')
    mk = mk[mk.role.isin(['ou25', 'btts']) & mk.match_id.isin(ready)]
    jobs = mk[['condition_id', 'match_id', 'role', 'kickoff_ts']].to_dict('records')
    todo = [j for j in jobs if not os.path.exists(f'{OUT}/{j["condition_id"]}.parquet')]
    log('markets', len(jobs), 'todo', len(todo))
    t0, n, rows = time.time(), 0, 0
    with cf.ThreadPoolExecutor(4) as ex:
        for f in cf.as_completed([ex.submit(fetch, j) for j in todo]):
            try:
                rows += f.result()
            except Exception as e:
                log('ERR', repr(e)[:200])
            n += 1
            if n % 200 == 0:
                log(f'done {n}/{len(todo)} rows {rows} elapsed {(time.time() - t0) / 60:.1f} min')
    log('finished', n, rows)


if __name__ == '__main__':
    main()
