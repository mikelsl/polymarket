"""Step 1: discover every Polymarket event in the five big-5 league series (Gamma series_id), 2025-07-01..2026-11-08.

Pages Gamma /events by 7-day endDate windows (offset paging only inside a window, never past a few hundred rows).
Saves, per match:
  raw/gamma/main/<slug>.json.gz          full main event (the three 1X2 markets)
  raw/gamma/more/<slug>.json.gz          full '-more-markets' sibling event (totals, BTTS, spreads, ...)
and raw/gamma_index.json: one row per event seen (any kind), used for the coverage report.
Idempotent: rerun overwrites. Usage: .venv/bin/python -I scripts/soccer_data/01_discover.py
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import collections
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import time

from common import GAMMA, LEAGUES, RAW, get_json, log, write_json_gz, STATS

MAIN = re.compile(r'^(epl|lal|sea|bun|fl1)-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})$')
SUB = re.compile(r'^(epl|lal|sea|bun|fl1)-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})-(.+)$')
START, END = dt.date(2025, 7, 1), dt.date(2026, 11, 8)

os.makedirs(f'{RAW}/gamma/main', exist_ok=True)
os.makedirs(f'{RAW}/gamma/more', exist_ok=True)


def classify(slug):
    if MAIN.match(slug):
        return 'main'
    m = SUB.match(slug)
    if m:
        return 'more' if m.group(5) == 'more-markets' else 'sub:' + m.group(5)
    return 'other'


def window(lg, a, b):
    sid = LEAGUES[lg]['series']
    out, off = [], 0
    while True:
        page = get_json(f'{GAMMA}/events', dict(series_id=sid, end_date_min=f'{a}T00:00:00Z', end_date_max=f'{b}T00:00:00Z',
                                                 limit=100, offset=off))
        for e in page:
            kind = classify(e.get('slug', ''))
            out.append(dict(event_id=str(e['id']), slug=e.get('slug'), series_league=lg, kind=kind, title=e.get('title'),
                            endDate=e.get('endDate'), startDate=e.get('startDate'), startTime=e.get('startTime'),
                            closed=e.get('closed'), archived=e.get('archived'), volume=e.get('volume'), n_markets=len(e.get('markets') or []),
                            market_types=dict(collections.Counter(str(m.get('sportsMarketType')) for m in e.get('markets') or [])),
                            series_slug=e.get('seriesSlug')))
            if kind == 'main':
                write_json_gz(f'{RAW}/gamma/main/{e["slug"]}.json.gz', e)
            elif kind == 'more':
                write_json_gz(f'{RAW}/gamma/more/{e["slug"]}.json.gz', e)
        if len(page) < 100:
            break
        off += 100
        if off > 2000:
            raise RuntimeError(f'window too large {lg} {a}')
    return out


def main():
    t0 = time.time()
    jobs = []
    d = START
    while d < END:
        e = min(d + dt.timedelta(days=7), END)
        for lg in LEAGUES:
            jobs.append((lg, d.isoformat(), e.isoformat()))
        d = e
    log(f'{len(jobs)} windows')
    index = {}
    done = 0
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(window, *j): j for j in jobs}
        for f in cf.as_completed(futs):
            for r in f.result():
                index[r['event_id']] = r
            done += 1
            if done % 25 == 0:
                log(f'windows {done}/{len(jobs)} events {len(index)} req {STATS} {time.time()-t0:.0f}s')
    rows = sorted(index.values(), key=lambda r: (r['series_league'], r['endDate'] or '', r['slug']))
    with open(f'{RAW}/gamma_index.json', 'w') as f:
        json.dump(rows, f)
    c = collections.Counter((r['series_league'], r['kind'] if not r['kind'].startswith('sub:') else 'sub') for r in rows)
    log('DONE', len(rows), 'events', dict(sorted(c.items())), f'{time.time()-t0:.0f}s', STATS)


if __name__ == '__main__':
    main()
