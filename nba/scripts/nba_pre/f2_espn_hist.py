"""Independent out-of-sample check for the heavy-favourite lead (read-only, ESPN only): sportsbook moneylines of the
2022-23 and 2023-24 NBA seasons (before Polymarket's NBA game markets), to test whether heavy book favourites
(de-vigged >= 0.90) win more often than priced, especially in March-April.

Pulls ESPN scoreboards per date and the core-API odds list per completed game into data/nba_pre/espn_hist/
(raw JSON per event, resumable), then writes data/nba_pre/espn_hist_odds.parquet (one row per event x provider).
Usage: nohup caffeinate -is .venv/bin/python -u scripts/nba_pre/f2_espn_hist.py > results/nba_pre/espn_hist.log 2>&1 &
Rate: <= 4 requests/s (scoreboards 2 threads, odds 5 threads).
"""
import datetime as dt
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from common_pre import DATA

OUT = DATA / 'espn_hist'
OUT.mkdir(parents=True, exist_ok=True)
SITE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
CORE = 'https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba'
SEASONS = [('2022-23', dt.date(2022, 10, 18), dt.date(2023, 6, 13)),
           ('2023-24', dt.date(2023, 10, 24), dt.date(2024, 6, 18))]
_lock, _next = threading.Lock(), [0.0]


def get(url, params=None):
    for i in range(6):
        with _lock:
            t = max(time.monotonic(), _next[0])
            _next[0] = t + 0.25
        time.sleep(max(0.0, t - time.monotonic()))
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    return None


def scoreboard(d):
    f = OUT / f'sb_{d:%Y%m%d}.json'
    if f.exists():
        return json.loads(f.read_text())
    js = get(f'{SITE}/scoreboard', dict(dates=f'{d:%Y%m%d}', limit=100)) or {}
    rows = []
    for e in js.get('events', []):
        c = e['competitions'][0]
        st = c.get('status', {}).get('type', {})
        teams = {x['homeAway']: x for x in c['competitors']}
        rows.append(dict(event=e['id'], date=f'{d:%Y-%m-%d}', season_type=e.get('season', {}).get('type'),
                         completed=bool(st.get('completed')), home=teams['home']['team'].get('abbreviation'),
                         away=teams['away']['team'].get('abbreviation'),
                         home_score=float(teams['home'].get('score') or 'nan'),
                         away_score=float(teams['away'].get('score') or 'nan'), neutral=c.get('neutralSite')))
    f.write_text(json.dumps(rows))
    return rows


def odds(ev):
    f = OUT / f'odds_{ev}.json'
    if f.exists():
        return json.loads(f.read_text())
    js = get(f'{CORE}/events/{ev}/competitions/{ev}/odds', dict(limit=50)) or {}
    rows = []
    for it in js.get('items', []):
        h, a = it.get('homeTeamOdds', {}) or {}, it.get('awayTeamOdds', {}) or {}
        rows.append(dict(event=ev, provider=(it.get('provider') or {}).get('name'),
                         ml_home=h.get('moneyLine'), ml_away=a.get('moneyLine'),     # final (closing) line
                         spread=it.get('spread'), over_under=it.get('overUnder')))
    f.write_text(json.dumps(rows))
    return rows


def main():
    games = []
    for season, a, b in SEASONS:
        days = [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]
        with ThreadPoolExecutor(2) as ex:
            for rows in ex.map(scoreboard, days):
                games += [dict(r, season=season) for r in rows]
        print(season, 'scoreboards done', len(games), flush=True)
    g = pd.DataFrame(games)
    g = g[g.completed & g.season_type.isin([2, 3])].drop_duplicates('event')
    out = []
    with ThreadPoolExecutor(5) as ex:
        for i, rows in enumerate(ex.map(odds, g.event)):
            out += rows
            if i % 250 == 0:
                print('odds', i, len(g), flush=True)
    o = pd.DataFrame(out)
    g.merge(o, on='event', how='left').to_parquet(DATA / 'espn_hist_odds.parquet', index=False)
    print('done', len(g), len(o), flush=True)


if __name__ == '__main__':
    main()
