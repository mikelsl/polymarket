"""Step 3: CLOB minute price history (fidelity=1) for token0 of every in-scope market
(1X2: the YES token; O/U 2.5: 'Over'; BTTS: 'Yes'), window [kickoff - 6 h, kickoff + 2.5 h].

Appends one JSON line per token to data/soccer_data/raw/prices/prices_raw.jsonl; rerun skips tokens already present.
Run: nohup caffeinate -is .venv/bin/python -I scripts/soccer_data/03_prices.py --threads 2 > results/soccer_data/03_prices.log 2>&1 &
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import argparse
import concurrent.futures as cf
import json
import os
import threading
import time

import pandas as pd

from common import CLOB, DATA, RAW, STATS, get_json, log

OUT = f'{RAW}/prices/prices_raw.jsonl'
PRE, POST = 6 * 3600, int(2.5 * 3600)
LOCK = threading.Lock()


def fetch(j):
    s, e = int(j['kickoff_ts']) - PRE, int(j['kickoff_ts']) + POST
    r = get_json(f'{CLOB}/prices-history', dict(market=j['token0_id'], startTs=s, endTs=e, fidelity=1))
    h = [[int(p['t']), float(p['p'])] for p in (r.get('history') or [])]
    rec = dict(match_id=j['match_id'], role=j['role'], condition_id=j['condition_id'], token_id=j['token0_id'],
               outcome=j['outcome0'], start_ts=s, end_ts=e, n=len(h), history=h)
    with LOCK:
        with open(OUT, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    return len(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['token_id'])
                except Exception:
                    pass
    mk = pd.read_parquet(f'{DATA}/markets.parquet')
    mk['prio'] = (mk.season == '2025-26').astype(int)
    mk = mk.sort_values(['prio', 'kickoff_ts'], ascending=[True, False])
    jobs = [j for j in mk[['match_id', 'role', 'condition_id', 'token0_id', 'outcome0', 'kickoff_ts']].to_dict('records') if j['token0_id'] not in done]
    log(f'tokens to fetch {len(jobs)} (already {len(done)})')
    t0, n, pts, err, last = time.time(), 0, 0, 0, 0
    with cf.ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(fetch, j): j for j in jobs}
        for f in cf.as_completed(futs):
            try:
                pts += f.result()
                n += 1
            except Exception as e:
                err += 1
                log('ERR', futs[f]['match_id'], futs[f]['role'], repr(e)[:200])
            if time.time() - last > 30:
                last = time.time()
                log(f'done {n}/{len(jobs)} err {err} points {pts} {time.time()-t0:.0f}s http {STATS}')
    log(f'FINISHED done {n} err {err} points {pts} {time.time()-t0:.0f}s http {STATS}')


if __name__ == '__main__':
    main()
