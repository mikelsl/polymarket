"""Step 5b: ESPN match summaries (key events with wallclock) for every linked Polymarket match.

GET .../soccer/<league>/summary?event=<id>; keeps header, keyEvents, commentary, gameInfo, format, wallclockAvailable,
pickcenter and odds in raw/espn/summary/<id>.json.gz. Resumable (skips existing files).
Run: nohup caffeinate -is .venv/bin/python -I scripts/soccer_data/05_espn_summaries.py --threads 2 > results/soccer_data/05_espn_summaries.log 2>&1 &
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import argparse
import concurrent.futures as cf
import os
import time

import pandas as pd

from common import DATA, ESPN, LEAGUES, RAW, STATS, get_json, log, write_json_gz

OUT = f'{RAW}/espn/summary'
KEEP = ['header', 'keyEvents', 'commentary', 'gameInfo', 'format', 'wallclockAvailable', 'pickcenter', 'odds']


def fetch(lg, eid):
    p = f'{OUT}/{eid}.json.gz'
    if os.path.exists(p):
        return 'cached'
    d = get_json(f'{ESPN}/{LEAGUES[lg]["espn"]}/summary', dict(event=eid))
    write_json_gz(p, {k: d.get(k) for k in KEEP})
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lk = pd.read_parquet(f'{DATA}/pm_espn_link.parquet')
    lk = lk[lk.espn_event_id.notna()].drop_duplicates('espn_event_id')
    jobs = [(r.league, r.espn_event_id) for r in lk.itertuples() if not os.path.exists(f'{OUT}/{r.espn_event_id}.json.gz')]
    log(f'summaries to fetch {len(jobs)} of {len(lk)}')
    t0, n, err, last = time.time(), 0, 0, 0
    with cf.ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(fetch, *j): j for j in jobs}
        for f in cf.as_completed(futs):
            try:
                f.result()
                n += 1
            except Exception as e:
                err += 1
                log('ERR', futs[f], repr(e)[:200])
            if time.time() - last > 30:
                last = time.time()
                log(f'done {n}/{len(jobs)} err {err} {time.time()-t0:.0f}s http {STATS}')
    log(f'FINISHED done {n} err {err} {time.time()-t0:.0f}s http {STATS}')


if __name__ == '__main__':
    main()
