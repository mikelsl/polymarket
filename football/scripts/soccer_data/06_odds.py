"""Step 6: bookmaker odds from football-data.co.uk (2025-26 '2526' and 2026-27 '2627'; E0, SP1, I1, D1, F1).

Raw CSVs -> data/soccer_data/raw/football_data/<season>/<div>.csv (downloaded data; read only with python -I).
Keeps match identifiers, results, Pinnacle opening/closing 1X2 (PSH/PSD/PSA, PSCH/PSCD/PSCA), Bet365, market
maxima/averages (Max*/Avg*, opening and closing), over/under 2.5 and Asian-handicap columns.
Column renames: '>2.5' -> '_over25', '<2.5' -> '_under25' (e.g. 'PC>2.5' -> 'PC_over25'); everything else verbatim.
Team names: football-data name -> ESPN displayName map learned from same-day fixtures
(data/soccer_data/team_map_fd_espn.csv, 'manual' column overrides), then joined to Polymarket matches on
(league, ESPN home, ESPN away, |date difference| <= 1 day).
Usage: .venv/bin/python -I scripts/soccer_data/06_odds.py [--refetch]
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import argparse
import collections
import datetime as dt
import io
import os
import re
import zoneinfo

import pandas as pd

from common import DATA, LEAGUES, RAW, get_text, log

FD = f'{RAW}/football_data'
SEAS = {'2526': '2025-26', '2627': '2026-27'}
DIV2LG = {v['fd']: k for k, v in LEAGUES.items()}
LON = zoneinfo.ZoneInfo('Europe/London')
ODDS_RE = re.compile(r'^(PS|P|Max|Avg|B365|BF|BFE)(C)?(H|D|A|>2\.5|<2\.5|AHH|AHA)$')
KEEP_ID = ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'AHh', 'AHCh']

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module  # noqa: E402
fx_mod = import_module('05_espn_fixtures')
sim = fx_mod.sim


def load(season, div, refetch):
    d = f'{FD}/{season}'
    os.makedirs(d, exist_ok=True)
    p = f'{d}/{div}.csv'
    if refetch or not os.path.exists(p):
        b = get_text(f'https://www.football-data.co.uk/mmz4281/{season}/{div}.csv')
        if b is None:
            return None
        with open(p, 'wb') as f:
            f.write(b)
    raw = open(p, 'rb').read()
    for enc in ('utf-8-sig', 'latin-1'):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(txt))
    df = df.dropna(how='all')
    df = df[df['HomeTeam'].notna()]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refetch', action='store_true')
    a = ap.parse_args()
    frames = []
    for s in SEAS:
        for div, lg in DIV2LG.items():
            df = load(s, div, a.refetch)
            if df is None:
                log('missing file', s, div)
                continue
            cols = [c for c in KEEP_ID if c in df.columns] + [c for c in df.columns if ODDS_RE.match(c)]
            df = df[cols].copy()
            df.insert(0, 'league', lg)
            df.insert(1, 'season', SEAS[s])
            frames.append(df)
            log(s, div, len(df), 'rows; Pinnacle cols:', [c for c in df.columns if c.startswith('PS')][:6])
    od = pd.concat(frames, ignore_index=True)
    od = od.rename(columns=lambda c: c.replace('>2.5', '_over25').replace('<2.5', '_under25'))
    od['date'] = pd.to_datetime(od['Date'], dayfirst=True, errors='coerce').dt.date
    def kick(r):
        try:
            hh, mm = str(r['Time']).split(':')[:2]
            t = dt.datetime.combine(r['date'], dt.time(int(hh), int(mm)), tzinfo=LON)
            return int(t.timestamp())
        except Exception:
            return None
    od['fd_kickoff_ts'] = od.apply(kick, axis=1).astype('Int64')
    od = od.rename(columns={'HomeTeam': 'fd_home', 'AwayTeam': 'fd_away', 'Date': 'fd_date_str', 'Time': 'fd_time_uk'})

    fx = pd.read_parquet(f'{DATA}/espn_fixtures.parquet')
    fx['d'] = pd.to_datetime(fx.espn_date_ts, unit='s', utc=True).dt.date
    byday = collections.defaultdict(list)
    for e in fx.to_dict('records'):
        byday[(e['league'], e['d'])].append(e)
    votes = collections.defaultdict(collections.Counter)
    for r in od.to_dict('records'):
        best = None
        for off in (0, -1, 1):
            for e in byday.get((r['league'], r['date'] + dt.timedelta(days=off)), []):
                s_ = sim(r['fd_home'], e['espn_home']) + sim(r['fd_away'], e['espn_away']) - 0.05 * abs(off)
                if best is None or s_ > best[0]:
                    best = (s_, e)
        if best and best[0] >= 1.0:
            votes[(r['league'], r['fd_home'])][best[1]['espn_home']] += 1
            votes[(r['league'], r['fd_away'])][best[1]['espn_away']] += 1
    mp_path = f'{DATA}/team_map_fd_espn.csv'
    manual = {}
    if os.path.exists(mp_path):
        old = pd.read_csv(mp_path, keep_default_na=False)
        manual = {(x.league, x.fd_name): x.manual for x in old.itertuples() if x.manual}
    names = sorted({(r['league'], r['fd_home']) for r in od.to_dict('records')} | {(r['league'], r['fd_away']) for r in od.to_dict('records')})
    rows = []
    for lg, nm in names:
        top = votes.get((lg, nm), collections.Counter()).most_common(2)
        auto = top[0][0] if top else ''
        rows.append(dict(league=lg, fd_name=nm, espn_name=manual.get((lg, nm)) or auto, auto=auto, votes=top[0][1] if top else 0,
                         runner_up=top[1][0] if len(top) > 1 else '', runner_votes=top[1][1] if len(top) > 1 else 0,
                         sim=round(sim(nm, auto), 3) if auto else 0, manual=manual.get((lg, nm), '')))
    tm = pd.DataFrame(rows)
    tm.to_csv(mp_path, index=False)
    mp = {(x.league, x.fd_name): x.espn_name for x in tm.itertuples()}
    od['home_team'] = [mp.get((l, n)) or None for l, n in zip(od.league, od.fd_home)]
    od['away_team'] = [mp.get((l, n)) or None for l, n in zip(od.league, od.fd_away)]

    # 1) football-data row -> ESPN fixture: same league, mapped names, |date diff| <= 1 day (swap fallback)
    fidx = collections.defaultdict(list)
    for e in fx.to_dict('records'):
        fidx[(e['league'], e['espn_home'], e['espn_away'])].append(e)
    eids, swaps = [], []
    for r in od.to_dict('records'):
        sw = False
        c = [e for e in fidx.get((r['league'], r['home_team'], r['away_team']), []) if abs((e['d'] - r['date']).days) <= 1]
        if not c:
            c = [e for e in fidx.get((r['league'], r['away_team'], r['home_team']), []) if abs((e['d'] - r['date']).days) <= 1]
            sw = bool(c)
        c.sort(key=lambda e: abs((e['d'] - r['date']).days))
        eids.append(c[0]['espn_event_id'] if c else None)
        swaps.append(sw)
    od['espn_event_id'] = eids
    od['fd_vs_espn_swapped'] = swaps
    # 2) ESPN fixture -> Polymarket event(s); match_id = primary event (closest gameStartTime to ESPN date, then volume)
    lk = pd.read_parquet(f'{DATA}/pm_espn_link.parquet')
    pmv = pd.read_parquet(f'{DATA}/pm_matches.parquet')[['match_id', 'vol_home', 'vol_draw', 'vol_away']]
    lk = lk.merge(pmv, on='match_id', how='left')
    lk['vol'] = lk[['vol_home', 'vol_draw', 'vol_away']].sum(axis=1, min_count=1).fillna(0)
    lk['off'] = lk.espn_minus_pm_kickoff_s.abs()
    prim = (lk[lk.espn_event_id.notna()].sort_values(['off', 'vol'], ascending=[True, False])
            .drop_duplicates('espn_event_id').set_index('espn_event_id').match_id.to_dict())
    od['match_id'] = od.espn_event_id.map(prim)
    swap_pm = dict(zip(lk.match_id, lk.home_away_swapped))
    # True when the football-data home team is the Polymarket AWAY team (then PSH etc. refer to the PM 'away' market)
    od['pm_orientation_swapped'] = [bool(s) != bool(swap_pm.get(m, False)) if m else None for s, m in zip(od.fd_vs_espn_swapped, od.match_id)]
    have = set(od.espn_event_id.dropna())
    res = []
    for m in lk.to_dict('records'):
        if m['espn_event_id'] in have:
            res.append((m['match_id'], 'ok' if prim.get(m['espn_event_id']) == m['match_id'] else 'ok_secondary_event'))
        else:
            res.append((m['match_id'], 'none'))
    od['date'] = pd.to_datetime(od['date'])
    od.to_parquet(f'{DATA}/odds.parquet', index=False)
    rs = pd.DataFrame(res, columns=['match_id', 'odds_join'])
    rs.to_parquet(f'{DATA}/odds_join.parquet', index=False)
    log('odds rows', len(od), 'with ESPN fixture', int(od.espn_event_id.notna().sum()), 'joined to PM', int(od.match_id.notna().sum()),
        'swapped vs ESPN', int(od.fd_vs_espn_swapped.sum()), '| PM events:', rs.odds_join.value_counts().to_dict(),
        'map empty', int((tm.espn_name == '').sum()))


if __name__ == '__main__':
    main()
