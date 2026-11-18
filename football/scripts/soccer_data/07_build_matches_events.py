"""Step 7: final `matches` and `events` tables plus the resolution check.

Inputs: pm_matches.parquet, markets.parquet, pm_espn_link.parquet, espn_fixtures.parquet, raw/espn/summary/*.json.gz,
odds_join.parquet (optional). Outputs (data/soccer_data/): matches.parquet, events.parquet,
results/soccer_data/resolution_mismatches.csv.
Usage: .venv/bin/python -I scripts/soccer_data/07_build_matches_events.py
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import os
import re

import numpy as np
import pandas as pd

from common import DATA, LEAGUES, RAW, RES, log, parse_ts, read_json_gz

PLAYOFFS = {'bun-pad-wol-2026-05-25': 'Bundesliga relegation play-off', 'fl1-se-nic-2026-05-26': 'Ligue 1 promotion/relegation play-off',
            'fl1-nic-se-2026-05-29': 'Ligue 1 promotion/relegation play-off'}
GOAL_KIND = {'goal': 'goal', 'goal---header': 'header', 'goal---volley': 'volley', 'goal---free-kick': 'free-kick',
             'penalty---scored': 'penalty', 'own-goal': 'own-goal'}
RED_TYPES = {'red-card', 'yellow-red-card', 'second-yellow-card', 'yellow-card---red-card'}
MARKERS = {'kickoff', 'halftime', 'start-2nd-half', 'end-regular-time', 'start-extra-time', 'end-extra-time', 'final-whistle'}
SCORE_RE = re.compile(r'(?<=\s)(\d+), (.+?) (\d+)\.(?:\s|$)')


def minute_parts(disp):
    m = re.match(r"^(\d+)'?(?:\+(\d+)'?)?", disp or '')
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (None, None)


def parse_summary(match_id, eid, d):
    comp = d['header']['competitions'][0]
    cs = {c['homeAway']: c for c in comp['competitors']}
    side_of = {c['team']['id']: c['homeAway'] for c in comp['competitors']}
    st = (comp.get('status') or {}).get('type') or {}
    info = dict(espn_status=st.get('name'), espn_status_detail=st.get('detail'),
                espn_home_score=pd.to_numeric(cs['home'].get('score'), errors='coerce'),
                espn_away_score=pd.to_numeric(cs['away'].get('score'), errors='coerce'),
                wallclock_available=d.get('wallclockAvailable'),
                venue=((d.get('gameInfo') or {}).get('venue') or {}).get('fullName'),
                attendance=(d.get('gameInfo') or {}).get('attendance'))
    for side in ('home', 'away'):
        ls = cs[side].get('linescores') or []
        info[f'espn_{side}_ht_score'] = pd.to_numeric(ls[0].get('displayValue', ls[0].get('value')), errors='coerce') if ls else np.nan
    rows, h, a = [], 0, 0
    for i, q in enumerate(d.get('keyEvents') or []):
        t = (q.get('type') or {}).get('type')
        team = q.get('team') or {}
        side = side_of.get(team.get('id'))
        clock = q.get('clock') or {}
        mi, add = minute_parts(clock.get('displayValue'))
        scoring = bool(q.get('scoringPlay'))
        text = q.get('text') or ''
        hs = as_ = None
        src = None
        if scoring:
            if side == 'home':
                h += 1
            elif side == 'away':
                a += 1
            m = SCORE_RE.search(text)
            if m:
                hs, as_, src = int(m.group(1)), int(m.group(3)), 'text'
            else:
                hs, as_, src = h, a, 'count'
        rows.append(dict(match_id=match_id, espn_event_id=eid, seq=i, espn_play_id=str(q.get('id')), type=t,
                         type_text=(q.get('type') or {}).get('text'), period=(q.get('period') or {}).get('number'),
                         clock_display=clock.get('displayValue'), clock_value_s=clock.get('value'), minute=mi, added_minute=add,
                         wallclock_ts=parse_ts(q.get('wallclock')), team_side=side, team_name=team.get('displayName'),
                         scoring_play=scoring, goal_kind=GOAL_KIND.get(t) if scoring else None,
                         is_red_card=t in RED_TYPES, is_period_marker=t in MARKERS,
                         home_score_after=hs, away_score_after=as_, score_source=src,
                         goal_count_home_after=h if scoring else None, goal_count_away_after=a if scoring else None, text=text))
    for r in rows:
        r['source'] = 'keyEvents'
    # VAR decisions are only in the commentary feed (no keyEvent); keep them, they move prices without a goal event
    for c in d.get('commentary') or []:
        text = c.get('text') or ''
        if 'VAR' not in text:
            continue
        low = text.lower()
        if 'overturned by var' in low:
            t = 'var-goal-overturned'
        elif low.startswith('var decision: no goal'):
            t = 'var-decision-no-goal'
        elif low.startswith('var decision: no penalty'):
            t = 'var-decision-no-penalty'
        elif low.startswith('var decision: penalty'):
            t = 'var-decision-penalty'
        elif low.startswith('var decision: goal'):
            t = 'var-decision-goal'
        elif 'red card' in low or 'no red card' in low:
            t = 'var-decision-red-card' if 'no red' not in low else 'var-decision-no-red-card'
        else:
            t = 'var-other'
        play = c.get('play') or {}
        tm = c.get('time') or {}
        mi, add = minute_parts(tm.get('displayValue'))
        team = play.get('team') or {}
        rows.append(dict(match_id=match_id, espn_event_id=eid, seq=1000 + int(c.get('sequence') or 0), espn_play_id=str(play.get('id')) if play.get('id') else None,
                         type=t, type_text='VAR (commentary)', period=(play.get('period') or {}).get('number'),
                         clock_display=tm.get('displayValue'), clock_value_s=tm.get('value'), minute=mi, added_minute=add,
                         wallclock_ts=parse_ts(play.get('wallclock')), team_side=side_of.get(team.get('id')), team_name=team.get('displayName'),
                         scoring_play=False, goal_kind=None, is_red_card=False, is_period_marker=False, home_score_after=None,
                         away_score_after=None, score_source=None, goal_count_home_after=None, goal_count_away_after=None,
                         text=text, source='commentary'))
    wall = {}
    for r in rows:
        if r['is_period_marker'] and r['type'] not in wall:
            wall[r['type']] = r['wallclock_ts']
    info.update(kickoff_wall_ts=wall.get('kickoff'), halftime_wall_ts=wall.get('halftime'),
                second_half_wall_ts=wall.get('start-2nd-half'), fulltime_wall_ts=wall.get('end-regular-time'),
                ke_goals_home=h, ke_goals_away=a, ke_n_events=len(rows),
                red_cards_home=sum(r['is_red_card'] and r['team_side'] == 'home' for r in rows),
                red_cards_away=sum(r['is_red_card'] and r['team_side'] == 'away' for r in rows),
                n_goal_wallclock_missing=sum(r['scoring_play'] and r['wallclock_ts'] is None for r in rows))
    return info, rows


def res3(h, a):
    if pd.isna(h) or pd.isna(a):
        return None
    return 'H' if h > a else ('A' if a > h else 'D')


def main():
    pm = pd.read_parquet(f'{DATA}/pm_matches.parquet')
    pm = pm[pm.in_scope].copy()
    lk = pd.read_parquet(f'{DATA}/pm_espn_link.parquet')
    fx = pd.read_parquet(f'{DATA}/espn_fixtures.parquet')
    m = pm.merge(lk[['match_id', 'home_team', 'away_team', 'espn_event_id', 'home_away_swapped']], on='match_id', how='left')
    m = m.merge(fx[['espn_event_id', 'espn_date_ts', 'espn_home', 'espn_away', 'espn_home_id', 'espn_away_id']], on='espn_event_id', how='left')
    m['competition'] = m.match_id.map(PLAYOFFS).fillna('league')
    infos, events = {}, []
    for r in m[m.espn_event_id.notna()].itertuples():
        p = f'{RAW}/espn/summary/{r.espn_event_id}.json.gz'
        if not os.path.exists(p):
            continue
        info, rows = parse_summary(r.match_id, r.espn_event_id, read_json_gz(p))
        infos[r.match_id] = info
        events += rows
    inf = pd.DataFrame.from_dict(infos, orient='index')
    inf.index.name = 'match_id'
    m = m.merge(inf.reset_index(), on='match_id', how='left')
    # canonical kickoff: ESPN scheduled date when linked (tracks postponements), else Polymarket gameStartTime
    m['pm_game_start_ts'] = m['kickoff_ts']
    m['kickoff_ts'] = m['espn_date_ts'].fillna(m['pm_game_start_ts']).astype('Int64')
    m['kickoff_source'] = np.where(m.espn_date_ts.notna(), 'espn_scheduled', 'pm_gameStartTime')
    m['pm_minus_espn_kickoff_s'] = (m.pm_game_start_ts - m.espn_date_ts).astype('Int64')
    m['kickoff_wall_minus_sched_s'] = (m.kickoff_wall_ts - m.kickoff_ts).astype('Int64')
    # duplicates: two Polymarket events for one ESPN match
    dup = m.espn_event_id.notna() & m.espn_event_id.duplicated(keep=False)
    m['is_duplicate_event'] = dup
    m['abs_pm_off'] = m.pm_minus_espn_kickoff_s.abs()
    m['vol_1x2'] = m[['vol_home', 'vol_draw', 'vol_away']].sum(axis=1, min_count=1)
    m['is_primary_event'] = True
    for eid, g in m[dup].groupby('espn_event_id'):
        best = g.sort_values(['abs_pm_off', 'vol_1x2'], ascending=[True, False]).index[0]
        m.loc[g.index, 'is_primary_event'] = False
        m.loc[best, 'is_primary_event'] = True
    # results in Polymarket orientation (roles come from the Polymarket slug: home = first-listed team)
    sw = m.home_away_swapped.fillna(False).astype(bool)
    m['home_score'] = np.where(sw, m.espn_away_score, m.espn_home_score)
    m['away_score'] = np.where(sw, m.espn_home_score, m.espn_away_score)
    m['home_ht_score'] = np.where(sw, m.espn_away_ht_score, m.espn_home_ht_score)
    m['away_ht_score'] = np.where(sw, m.espn_home_ht_score, m.espn_away_ht_score)
    m['espn_result'] = [res3(h, a) for h, a in zip(m.home_score, m.away_score)]
    m['ke_goals_consistent'] = (m.ke_goals_home == m.espn_home_score) & (m.ke_goals_away == m.espn_away_score)
    m['resolution_check'] = np.select(
        [m.pm_result.isin(['UNRESOLVED', 'NONE', 'MULTI']), m.espn_status.eq('STATUS_ABANDONED'), m.espn_result.isna(),
         m.pm_result == m.espn_result],
        ['pm_unresolved', 'espn_abandoned', 'no_espn_result', 'match'], 'MISMATCH')
    m['season_month'] = pd.to_datetime(m.kickoff_ts, unit='s', utc=True).dt.strftime('%Y-%m')
    if os.path.exists(f'{DATA}/odds_join.parquet'):
        oj = pd.read_parquet(f'{DATA}/odds_join.parquet')
        m = m.merge(oj, on='match_id', how='left')
        m['has_odds'] = m.odds_join.eq('ok')
    for c in ['kickoff_wall_ts', 'halftime_wall_ts', 'second_half_wall_ts', 'fulltime_wall_ts', 'espn_date_ts', 'pm_game_start_ts']:
        m[c] = m[c].astype('Int64')
    for c in ['kickoff', 'kickoff_wall', 'halftime_wall', 'second_half_wall', 'fulltime_wall']:
        m[c + '_utc'] = pd.to_datetime(m[c + '_ts'], unit='s', utc=True)
    m = m.drop(columns=['abs_pm_off', 'in_scope', 'kickoff_utc_x', 'kickoff_utc_y'], errors='ignore')
    first = ['match_id', 'league', 'league_name', 'season', 'competition', 'season_month', 'kickoff_ts', 'kickoff_utc', 'kickoff_source',
             'home_team_pm', 'away_team_pm', 'home_team', 'away_team', 'home_score', 'away_score', 'home_ht_score', 'away_ht_score',
             'pm_result', 'espn_result', 'resolution_check', 'espn_event_id', 'is_duplicate_event', 'is_primary_event']
    m = m[first + [c for c in m.columns if c not in first]].sort_values(['kickoff_ts', 'match_id']).reset_index(drop=True)
    po = m[m.competition != 'league']
    po[['match_id', 'competition', 'kickoff_utc', 'home_team_pm', 'away_team_pm', 'pm_result', 'vol_home', 'vol_draw', 'vol_away']].to_csv(
        f'{RES}/excluded_playoffs.csv', index=False)
    m = m[m.competition == 'league'].drop(columns=['competition']).reset_index(drop=True)
    m['analysis_ready'] = m.is_primary_event & (m.resolution_check == 'match') & (m.espn_status == 'STATUS_FULL_TIME')
    m.to_parquet(f'{DATA}/matches.parquet', index=False)
    keep_ids = set(m.match_id)
    events = [e for e in events if e['match_id'] in keep_ids]
    ev = pd.DataFrame(events)
    ev = ev.merge(m[['match_id', 'kickoff_ts', 'kickoff_wall_ts', 'home_away_swapped']], on='match_id', how='left')
    ev['wallclock_ts'] = ev['wallclock_ts'].astype('Int64')
    ev['wallclock_utc'] = pd.to_datetime(ev.wallclock_ts, unit='s', utc=True)
    ev['t_from_kickoff_wall_s'] = (ev.wallclock_ts - ev.kickoff_wall_ts).astype('Int64')
    ev['t_from_sched_kickoff_s'] = (ev.wallclock_ts - ev.kickoff_ts).astype('Int64')
    sw = ev.home_away_swapped.fillna(False).astype(bool)  # express sides in Polymarket orientation
    flip = {'home': 'away', 'away': 'home'}
    ev['team_side_pm'] = np.where(sw, ev.team_side.map(flip), ev.team_side)
    ev = ev.drop(columns=['kickoff_ts', 'kickoff_wall_ts', 'home_away_swapped'])
    for c in ['home_score_after', 'away_score_after', 'goal_count_home_after', 'goal_count_away_after', 'minute', 'added_minute', 'period']:
        ev[c] = ev[c].astype('Int64')
    ev.to_parquet(f'{DATA}/events.parquet', index=False)
    mm = m[m.resolution_check != 'match']
    mm[['match_id', 'kickoff_utc', 'home_team_pm', 'away_team_pm', 'home_score', 'away_score', 'pm_result', 'espn_result',
        'resolution_check', 'espn_status', 'is_duplicate_event', 'is_primary_event', 'event_score']].to_csv(f'{RES}/resolution_mismatches.csv', index=False)
    log('matches', len(m), 'events', len(ev), 'resolution', m.resolution_check.value_counts().to_dict())
    log('ke goals consistent', m.ke_goals_consistent.value_counts(dropna=False).to_dict())


if __name__ == '__main__':
    main()
