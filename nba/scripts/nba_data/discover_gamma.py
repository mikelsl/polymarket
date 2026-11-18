"""Step 1 - Gamma discovery of every NBA event (tag_id=745), paged by endDate windows.

Why windows: Gamma offset paging breaks past ~4000 rows, and limit is capped at 100. NBA game events have
endDate == game start, so 3-day endDate windows hold <100 events each; inside a window we page with a small offset.
2026-27 events carry ~500 markets (~1.7 MB each), so the page size is small there.

Every event returned (games and non-games, all fields) is saved per window to data/nba_data/raw/gamma_windows/.
Resumable: finished windows are skipped unless they end after --refresh-since (use this to re-pull the last days
once their markets have resolved).

Usage:  .venv/bin/python scripts/nba_data/discover_gamma.py [--refresh-since 2026-10-30] [--threads 3]
Threads: Gamma requests only; keep the total of concurrently running jobs <= 6.
"""
import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import GAMMA, RAW, get_json, log, read_json_gz, write_json_gz, STATS

WIN_DIR = RAW / 'gamma_windows'
FIRST = dt.date(2024, 10, 1)      # a bit before the scope start (2024-10-22)
LAST = dt.date(2026, 11, 10)      # a bit after the scope end (2026-11-03), catches late endDates
STEP = 3


def windows():
    d = FIRST
    while d < LAST:
        e = min(d + dt.timedelta(days=STEP), LAST)
        yield d, e
        d = e


def page_size(day):
    return 4 if day >= dt.date(2026, 8, 1) else 20


def fetch_window(a, b):
    out, off, lim = [], 0, page_size(a)
    while True:
        params = dict(tag_id=745, limit=lim, offset=off, order='endDate', ascending='true',
                      end_date_min=f'{a.isoformat()}T00:00:00Z', end_date_max=f'{b.isoformat()}T00:00:00Z')
        batch = get_json(f'{GAMMA}/events', params)
        if not isinstance(batch, list):
            raise RuntimeError(f'unexpected Gamma payload {str(batch)[:200]}')
        out.extend(batch)
        if len(batch) < lim:
            break
        off += lim
        if off > 3500:
            raise RuntimeError(f'window {a}..{b} too large for offset paging')
    # endDate_max is inclusive on Gamma; keep [a, b) so adjacent windows do not double count (dedupe later anyway)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh-since', default=None, help='re-fetch windows ending after this date (YYYY-MM-DD)')
    ap.add_argument('--threads', type=int, default=3)
    args = ap.parse_args()
    refresh = dt.date.fromisoformat(args.refresh_since) if args.refresh_since else None
    todo = []
    for a, b in windows():
        f = WIN_DIR / f'{a.isoformat()}_{b.isoformat()}.json.gz'
        if f.exists() and not (refresh and b > refresh):
            continue
        todo.append((a, b, f))
    log(f'gamma windows to fetch: {len(todo)} (threads={args.threads})')
    n_ev = 0
    with ThreadPoolExecutor(args.threads) as ex:
        futs = {ex.submit(fetch_window, a, b): (a, b, f) for a, b, f in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            a, b, f = futs[fu]
            evs = fu.result()
            write_json_gz(f, {'window': [a.isoformat(), b.isoformat()], 'fetched_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                              'events': evs})
            n_ev += len(evs)
            if i % 10 == 0 or i == len(todo):
                log(f'  {i}/{len(todo)} windows, last {a}..{b}: {len(evs)} events; total {n_ev}; http {STATS}')
    files = sorted(WIN_DIR.glob('*.json.gz'))
    ids = set()
    for f in files:
        ids.update(str(e['id']) for e in read_json_gz(f)['events'])
    log(f'done: {len(files)} window files, {len(ids)} distinct events')


if __name__ == '__main__':
    main()
