"""Step 5a: ESPN fixture lists (monthly scoreboards) for the big-5 leagues and the Polymarket <-> ESPN match link.

1. GET .../soccer/<league>/scoreboard?dates=YYYYMM&limit=1000 for 2025-07..2026-11 -> raw/espn/scoreboard/<lg>_<YYYYMM>.json.gz
2. Team-name map Polymarket title name -> ESPN displayName, learned from kickoff-time + fuzzy-name matches and written to
   data/soccer_data/team_map_pm_espn.csv (review it; a 'manual' column overrides).
3. Strict link: same league, mapped home AND away names, |ESPN date - Polymarket kickoff| <= 5 days.
Outputs data/soccer_data/espn_fixtures.parquet and data/soccer_data/pm_espn_link.parquet.
Usage: .venv/bin/python -I scripts/soccer_data/05_espn_fixtures.py [--threads 2] [--refetch]
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import argparse
import collections
import concurrent.futures as cf
import difflib
import os
import re
import unicodedata

import pandas as pd

from common import DATA, ESPN, LEAGUES, RAW, get_json, log, parse_ts, read_json_gz, season_of, write_json_gz

SB = f'{RAW}/espn/scoreboard'
MONTHS = [f'{y}{m:02d}' for y in (2025, 2026) for m in range(1, 13) if '202507' <= f'{y}{m:02d}' <= '202611']
STOP = r'\b(fc|afc|cf|ac|ss|ssc|as|rc|sc|us|vfb|vfl|tsg|sv|bsc|fsv|1|de|ud|ca|rcd|club|calcio|hsc|cfc|ogc|stade|olympique|sco|losc|aj|rcd|sd|cd|real|athletic|deportivo|borussia|bayer|hellas|fk)\b'


MANUAL_PM = {('lal', 'RC Deportivo A Coruña'): 'Deportivo'}  # explicit overrides (ESPN displayName)


def base(s):
    s = unicodedata.normalize('NFKD', s or '').encode('ascii', 'ignore').decode().lower()
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', s)).strip()


def norm(s):
    b = base(s)
    n = re.sub(r'\s+', ' ', re.sub(STOP, ' ', b)).strip()
    return n or b  # never empty ('Athletic Club' would vanish otherwise)


def sim(a, b):
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    r = max(difflib.SequenceMatcher(None, na, nb).ratio(), difflib.SequenceMatcher(None, base(a), base(b)).ratio())
    if na in nb or nb in na:
        r = max(r, 0.9)
    return r


def fetch(lg, ym, refetch):
    p = f'{SB}/{lg}_{ym}.json.gz'
    if os.path.exists(p) and not refetch:
        return read_json_gz(p)
    d = get_json(f'{ESPN}/{LEAGUES[lg]["espn"]}/scoreboard', dict(dates=ym, limit=1000))
    write_json_gz(p, d)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--refetch', action='store_true')
    a = ap.parse_args()
    os.makedirs(SB, exist_ok=True)
    rows = {}
    with cf.ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(fetch, lg, ym, a.refetch): (lg, ym) for lg in LEAGUES for ym in MONTHS}
        for f in cf.as_completed(futs):
            lg, ym = futs[f]
            for e in f.result().get('events') or []:
                c = e['competitions'][0]
                h = next(x for x in c['competitors'] if x['homeAway'] == 'home')
                w = next(x for x in c['competitors'] if x['homeAway'] == 'away')
                st = (e.get('status') or {}).get('type') or {}
                ts = parse_ts(e['date'])
                rows[e['id']] = dict(
                    espn_event_id=e['id'], league=lg, espn_league=LEAGUES[lg]['espn'], espn_date_ts=ts, season=season_of(ts),
                    espn_season=(e.get('season') or {}).get('slug') or (e.get('season') or {}).get('year'),
                    espn_name=e.get('name'), espn_home=h['team'].get('displayName'), espn_away=w['team'].get('displayName'),
                    espn_home_id=h['team'].get('id'), espn_away_id=w['team'].get('id'),
                    espn_home_abbr=h['team'].get('abbreviation'), espn_away_abbr=w['team'].get('abbreviation'),
                    espn_home_score=h.get('score'), espn_away_score=w.get('score'), espn_status=st.get('name'),
                    espn_status_detail=st.get('detail'), espn_completed=st.get('completed'), scoreboard_month=ym)
    fx = pd.DataFrame(rows.values())
    fx['espn_date_ts'] = fx['espn_date_ts'].astype('Int64')
    fx.to_parquet(f'{DATA}/espn_fixtures.parquet', index=False)
    log('espn fixtures', len(fx), fx.groupby(['season', 'league']).size().to_dict())
    log('status', fx.espn_status.value_counts().to_dict())

    pm = pd.read_parquet(f'{DATA}/pm_matches.parquet')
    pm = pm[pm.in_scope]
    byl = {lg: g.to_dict('records') for lg, g in fx.groupby('league')}
    # pass 1: fuzzy candidates -> name votes
    votes = collections.defaultdict(collections.Counter)
    for m in pm.to_dict('records'):
        best = None
        for e in byl.get(m['league'], []):
            dtd = abs(int(e['espn_date_ts']) - int(m['kickoff_ts']))
            if dtd > 2 * 86400:
                continue
            s = sim(m['home_team_pm'], e['espn_home']) + sim(m['away_team_pm'], e['espn_away']) - dtd / 86400 * 0.05
            if best is None or s > best[0]:
                best = (s, e)
        if best and best[0] >= 1.2:
            votes[(m['league'], m['home_team_pm'])][best[1]['espn_home']] += 1
            votes[(m['league'], m['away_team_pm'])][best[1]['espn_away']] += 1
    names = sorted({(m['league'], m['home_team_pm']) for m in pm.to_dict('records')} | {(m['league'], m['away_team_pm']) for m in pm.to_dict('records')})
    mp_path = f'{DATA}/team_map_pm_espn.csv'
    manual = {}
    if os.path.exists(mp_path):
        old = pd.read_csv(mp_path, keep_default_na=False)
        manual = {(r.league, r.pm_name): r.manual for r in old.itertuples() if r.manual}
    manual.update(MANUAL_PM)
    out = []
    for lg, nm in names:
        v = votes.get((lg, nm), collections.Counter())
        top = v.most_common(2)
        auto = top[0][0] if top else ''
        out.append(dict(league=lg, pm_name=nm, espn_name=manual.get((lg, nm)) or auto, auto=auto, votes=top[0][1] if top else 0,
                        runner_up=top[1][0] if len(top) > 1 else '', runner_votes=top[1][1] if len(top) > 1 else 0,
                        sim=round(sim(nm, auto), 3) if auto else 0, manual=manual.get((lg, nm), '')))
    tm = pd.DataFrame(out)
    tm.to_csv(mp_path, index=False)
    mp = {(r.league, r.pm_name): r.espn_name for r in tm.itertuples()}
    log('team map rows', len(tm), 'empty', int((tm.espn_name == '').sum()), 'with runner-up', int((tm.runner_votes > 0).sum()))
    # pass 2: strict link. Within one league season each ordered (home, away) pair is played once, so the key is
    # (league, season, home, away); this also catches postponed matches whose Polymarket gameStartTime kept the old date.
    idx = collections.defaultdict(list)
    for e in fx.to_dict('records'):
        idx[(e['league'], e['season'], e['espn_home'], e['espn_away'])].append(e)
    link = []
    for m in pm.to_dict('records'):
        h, w = mp.get((m['league'], m['home_team_pm'])), mp.get((m['league'], m['away_team_pm']))
        cands = list(idx.get((m['league'], m['season'], h, w), []))
        swapped = False
        if not cands:  # home/away swapped?
            cands = [e for e in idx.get((m['league'], m['season'], w, h), []) if abs(int(e['espn_date_ts']) - int(m['kickoff_ts'])) <= 5 * 86400]
            swapped = bool(cands)
        cands.sort(key=lambda e: abs(int(e['espn_date_ts']) - int(m['kickoff_ts'])))
        e = cands[0] if cands else None
        link.append(dict(match_id=m['match_id'], league=m['league'], season=m['season'], home_team_pm=m['home_team_pm'],
                         away_team_pm=m['away_team_pm'], home_team=h, away_team=w, kickoff_ts=m['kickoff_ts'],
                         espn_event_id=e['espn_event_id'] if e else None, home_away_swapped=swapped, n_candidates=len(cands),
                         espn_minus_pm_kickoff_s=(int(e['espn_date_ts']) - int(m['kickoff_ts'])) if e else None))
    lk = pd.DataFrame(link)
    lk['espn_minus_pm_kickoff_s'] = lk['espn_minus_pm_kickoff_s'].astype('Int64')
    lk.to_parquet(f'{DATA}/pm_espn_link.parquet', index=False)
    dup = lk[lk.espn_event_id.notna()].espn_event_id.duplicated(keep=False)
    log('linked', int(lk.espn_event_id.notna().sum()), 'of', len(lk), 'unlinked', int(lk.espn_event_id.isna().sum()),
        'swapped', int(lk.home_away_swapped.sum()), 'dup espn ids', int(dup.sum()))
    for r in lk[lk.espn_event_id.isna()].itertuples():
        log('UNLINKED', r.match_id, r.home_team_pm, '->', r.home_team, '|', r.away_team_pm, '->', r.away_team)


if __name__ == '__main__':
    main()
