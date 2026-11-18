"""Step 2 - ESPN raw pull: scoreboard for every ET date, then summary (play-by-play, pickcenter, header) and the
core-API odds list for every completed game in the date range.

Raw JSON is kept gzipped so tables can be rebuilt without refetching:
  raw/espn_scoreboard/YYYYMMDD.json.gz, raw/espn_summary/<event>.json.gz, raw/espn_odds/<event>.json.gz
Resumable: existing files are skipped. --refresh-since YYYY-MM-DD re-pulls scoreboards (and summaries/odds of games)
on or after that date, for games that were not final at the first pull.

Usage:  .venv/bin/python scripts/nba_data/espn_fetch.py [--threads 2] [--refresh-since 2026-11-01]
Threads: ESPN only; keep the total of concurrently running jobs <= 6.
"""
import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import ESPN_CORE, ESPN_SITE, RAW, STATS, get_json, log, read_json_gz, write_json_gz

SB_DIR, SUM_DIR, ODDS_DIR = RAW / 'espn_scoreboard', RAW / 'espn_summary', RAW / 'espn_odds'
FIRST = dt.date(2024, 10, 1)
LAST = dt.date(2026, 11, 4)


def fetch_scoreboard(d):
    f = SB_DIR / f'{d:%Y%m%d}.json.gz'
    js = get_json(f'{ESPN_SITE}/scoreboard', dict(dates=f'{d:%Y%m%d}', limit=100))
    write_json_gz(f, js)
    return d, len(js.get('events') or [])


def fetch_summary(eid):
    js = get_json(f'{ESPN_SITE}/summary', dict(event=eid))
    write_json_gz(SUM_DIR / f'{eid}.json.gz', js)
    return eid


def fetch_odds(eid):
    js = get_json(f'{ESPN_CORE}/events/{eid}/competitions/{eid}/odds', dict(limit=50), allow_404=True)
    write_json_gz(ODDS_DIR / f'{eid}.json.gz', js if js is not None else {'_status': 404})
    return eid


def run_pool(fn, items, threads, label):
    done = 0
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fu in as_completed(futs):
            try:
                fu.result()
            except Exception as e:  # keep going, the item is retried on the next run
                log(f'  {label} FAILED {futs[fu]}: {e}')
            done += 1
            if done % 100 == 0 or done == len(items):
                log(f'  {label} {done}/{len(items)} http {STATS}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--refresh-since', default=None)
    args = ap.parse_args()
    refresh = dt.date.fromisoformat(args.refresh_since) if args.refresh_since else None

    days = [FIRST + dt.timedelta(n) for n in range((LAST - FIRST).days + 1)]
    todo = [d for d in days if not (SB_DIR / f'{d:%Y%m%d}.json.gz').exists() or (refresh and d >= refresh)]
    log(f'scoreboards to fetch: {len(todo)} of {len(days)}')
    run_pool(fetch_scoreboard, todo, args.threads, 'scoreboard')

    events = {}
    for d in days:
        f = SB_DIR / f'{d:%Y%m%d}.json.gz'
        if not f.exists():
            continue
        for e in read_json_gz(f).get('events') or []:
            st = (((e.get('competitions') or [{}])[0].get('status') or {}).get('type') or {})
            events[str(e['id'])] = (d, bool(st.get('completed')), st.get('name'))
    final = sorted(eid for eid, (d, comp, _) in events.items() if comp)
    log(f'ESPN events in range: {len(events)}, completed: {len(final)}; not completed: '
        f'{sorted((str(v[0]), k, v[2]) for k, v in events.items() if not v[1])[:40]}')

    def need(dirp, eid):
        return not (dirp / f'{eid}.json.gz').exists() or (refresh and events[eid][0] >= refresh)

    sums = [e for e in final if need(SUM_DIR, e)]
    log(f'summaries to fetch: {len(sums)}')
    run_pool(fetch_summary, sums, args.threads, 'summary')
    odds = [e for e in final if need(ODDS_DIR, e)]
    log(f'core odds to fetch: {len(odds)}')
    run_pool(fetch_odds, odds, args.threads, 'odds')
    log(f'done. http {STATS}')


if __name__ == '__main__':
    main()
