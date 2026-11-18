"""Step 4 (background): full taker trade prints for the three 1X2 markets of every in-scope match.

Source: Data API v2 /v2/trades?condition=<conditionId>&taker_only=true&limit=1000&cursor=... (newest first, cursor paging,
whole market history). One parquet per market: data/soccer_data/trades_parts/<season>/<conditionId>.parquet.
Priority: 2026-27 (newest kickoff first), then 2025-26 (newest kickoff first).
Resumable: markets whose parquet exists are skipped. Clean stop: create data/soccer_data/trades_STOP, or pass
--max-hours H (no new market starts after H hours; in-flight markets finish).
Concurrency: at most 6 worker threads; the number of simultaneous HTTP requests is capped by the integer in
data/soccer_data/trades_concurrency.txt (re-read every 5 s, default 4, max 6), so other jobs can share the 6-request budget.
Run: nohup caffeinate -is .venv/bin/python -I scripts/soccer_data/04_trades.py --max-hours 4 > results/soccer_data/04_trades.log 2>&1 &
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

from common import DATA, DATA_API, STATS, get_json, log

PARTS = f'{DATA}/trades_parts'
STATE = f'{DATA}/trades_state.jsonl'
ERRS = f'{DATA}/trades_errors.jsonl'
CONC = f'{DATA}/trades_concurrency.txt'
STOP = f'{DATA}/trades_STOP'
COLS = ['condition_id', 'token_id', 'outcome', 'outcome_index', 'side', 'price', 'size', 'timestamp', 'transaction_hash', 'proxy_wallet']


class Gate:
    """Caps simultaneous HTTP requests at the integer stored in CONC (re-read every 5 s)."""

    def __init__(self, default=4):
        self.cv = threading.Condition()
        self.active, self.limit, self.default, self.t = 0, default, default, 0.0

    def _refresh(self):
        if time.time() - self.t > 5:
            self.t = time.time()
            try:
                self.limit = max(1, min(6, int(open(CONC).read().strip())))
            except Exception:
                self.limit = self.default

    def __enter__(self):
        with self.cv:
            while True:
                self._refresh()
                if self.active < self.limit:
                    self.active += 1
                    return self
                self.cv.wait(1.0)

    def __exit__(self, *a):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()


GATE = Gate()
LOCK = threading.Lock()
TOT = dict(done=0, skipped=0, err=0, rows=0, pages=0, shares=0.0)


def fetch(job, deadline):
    if time.time() > deadline or os.path.exists(STOP):
        return 'skipped'
    cid, season = job['condition_id'], job['season']
    out = f'{PARTS}/{season}/{cid}.parquet'
    if os.path.exists(out):
        return 'cached'
    t0 = time.time()
    rows, cur, pages = [], None, 0
    while True:
        params = dict(condition=cid, taker_only='true', limit=1000)
        if cur:
            params['cursor'] = cur
        with GATE:
            r = get_json(f'{DATA_API}/trades', params)
        d = r.get('data') or []
        rows.extend({k: x.get(k) for k in COLS} for x in d)
        pages += 1
        cur = (r.get('pagination') or {}).get('next_cursor')
        if not cur or not d:
            break
        if pages >= 5000:
            raise RuntimeError(f'{cid}: more than 5000 pages')
    df = pd.DataFrame(rows, columns=COLS)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.astype({'price': 'float64', 'size': 'float64', 'timestamp': 'int64', 'outcome_index': 'int8'}) if len(df) else df
    df.insert(0, 'role', job['role'])
    df.insert(0, 'match_id', job['match_id'])
    tmp = out + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    st = dict(condition_id=cid, match_id=job['match_id'], role=job['role'], season=season, rows=len(df), raw_rows=n_raw,
              pages=pages, size_sum=float(df['size'].sum()) if len(df) else 0.0,
              ts_min=int(df['timestamp'].min()) if len(df) else None, ts_max=int(df['timestamp'].max()) if len(df) else None,
              gamma_volume=job['volume'], secs=round(time.time() - t0, 2), done_at=int(time.time()))
    with LOCK:
        with open(STATE, 'a') as f:
            f.write(json.dumps(st) + '\n')
        TOT['rows'] += len(df)
        TOT['pages'] += pages
        TOT['shares'] += st['size_sum']
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-hours', type=float, default=4.0)
    a = ap.parse_args()
    deadline = time.time() + a.max_hours * 3600
    for s in ('2025-26', '2026-27'):
        os.makedirs(f'{PARTS}/{s}', exist_ok=True)
    mk = pd.read_parquet(f'{DATA}/markets.parquet')
    mk = mk[mk.role.isin(['home', 'draw', 'away'])].copy()
    mk['prio'] = (mk.season == '2025-26').astype(int)
    mk = mk.sort_values(['prio', 'kickoff_ts', 'match_id', 'role'], ascending=[True, False, True, True])
    jobs = mk[['condition_id', 'match_id', 'role', 'season', 'volume']].to_dict('records')
    todo = [j for j in jobs if not os.path.exists(f'{PARTS}/{j["season"]}/{j["condition_id"]}.parquet')]
    log(f'markets {len(jobs)}, already done {len(jobs) - len(todo)}, to fetch {len(todo)}; deadline in {a.max_hours} h')
    t0, last = time.time(), 0.0
    n_by_season = {s: sum(1 for j in todo if j['season'] == s) for s in ('2026-27', '2025-26')}
    done_by_season = {s: 0 for s in n_by_season}
    with cf.ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(fetch, j, deadline): j for j in todo}
        for f in cf.as_completed(futs):
            j = futs[f]
            try:
                res = f.result()
            except Exception as e:
                res = 'err'
                with open(ERRS, 'a') as fe:
                    fe.write(json.dumps(dict(condition_id=j['condition_id'], match_id=j['match_id'], err=repr(e)[:400], at=int(time.time()))) + '\n')
                log('ERR', j['match_id'], j['role'], repr(e)[:200])
            TOT['done' if res in ('ok', 'cached') else ('skipped' if res == 'skipped' else 'err')] += 1
            if res in ('ok', 'cached'):
                done_by_season[j['season']] += 1
            if time.time() - last > 30:
                last = time.time()
                el = time.time() - t0
                rate = TOT['done'] / el if el else 0
                left = len(todo) - TOT['done'] - TOT['skipped'] - TOT['err']
                log(f"done {TOT['done']}/{len(todo)} by season {done_by_season} of {n_by_season} err {TOT['err']} skipped {TOT['skipped']} "
                    f"rows {TOT['rows']} pages {TOT['pages']} shares {TOT['shares']:.0f} elapsed {el/60:.1f}m "
                    f"rate {rate*60:.1f} mk/min eta {left/rate/60 if rate else 0:.0f}m gate {GATE.limit} http {STATS}")
    log(f"FINISHED done {TOT['done']} err {TOT['err']} skipped {TOT['skipped']} rows {TOT['rows']} pages {TOT['pages']} "
        f"elapsed {(time.time()-t0)/60:.1f}m http {STATS} by season {done_by_season}")


if __name__ == '__main__':
    main()
