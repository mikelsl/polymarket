"""Step 2b - ESPN tables from the raw pulls (no network).

Outputs (data/nba_data/work/, joined to Polymarket games in build_tables.py)
  espn_games.parquet   one row per ESPN event in 2024-10-01..2026-11-04 (scoreboard + summary header + pbp QA)
  espn_pbp.parquet     compact play-by-play, one row per play
  espn_odds.parquet    pickcenter (summary) + core-API odds, one row per (event, provider, source)
  espn_score_reversals.parquet  plays where the home or away score is lower than on the previous play, in game-clock
                                 order (order='clock') and among scoring plays in wallclock order ('scoring_wallclock')

Usage: .venv/bin/python scripts/nba_data/espn_tables.py
"""
import json
import re
import zoneinfo

import numpy as np
import pandas as pd

from common import RAW, WORK, log, parse_ts, read_json_gz, write_parquet

ET = zoneinfo.ZoneInfo('America/New_York')
SB_DIR, SUM_DIR, ODDS_DIR = RAW / 'espn_scoreboard', RAW / 'espn_summary', RAW / 'espn_odds'


def clock_seconds(s):
    """'11:48' -> 708.0, '45.3' -> 45.3, '0.0' -> 0.0."""
    if s is None:
        return None
    s = str(s).strip()
    m = re.match(r'^(\d+):(\d+(?:\.\d+)?)$', s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(s)
    except ValueError:
        return None


def american(x):
    """'-198' / '+164' / 'EVEN' / -110.0 -> float american odds (EVEN -> 100)."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().upper()
    if s in ('EVEN', 'EV', 'PK'):
        return 100.0
    s = s.lstrip('OU')
    try:
        return float(s)
    except ValueError:
        return None


def line_num(x):
    """'o241.5' / 'u241.5' / '-4.5' / '+4.5' / 'PK' -> float."""
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ('pk', 'even'):
        return 0.0
    s = s.lstrip('ou')
    try:
        return float(s)
    except ValueError:
        return None


def g(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def scoreboard_events():
    rows = {}
    for f in sorted(SB_DIR.glob('*.json.gz')):
        for e in read_json_gz(f).get('events') or []:
            c = (e.get('competitions') or [{}])[0]
            comp = {x.get('homeAway'): x for x in c.get('competitors') or []}
            st = g(c, 'status', 'type') or {}
            d = parse_ts(e.get('date'))
            row = dict(espn_event_id=str(e['id']), scoreboard_date=f.name[:8], espn_sched_utc=d,
                       espn_date_et=d.astimezone(ET).date().isoformat() if d else None,
                       season_year=g(e, 'season', 'year'), season_type=g(e, 'season', 'type'),
                       season_slug=g(e, 'season', 'slug'), comp_type=g(c, 'type', 'abbreviation'),
                       notes='; '.join(n.get('headline') or '' for n in c.get('notes') or []) or None,
                       neutral_site=c.get('neutralSite'), status_name=st.get('name'), completed=st.get('completed'),
                       status_detail=st.get('detail'), short_name=e.get('shortName'))
            for side in ('home', 'away'):
                t = comp.get(side) or {}
                row.update({f'{side}_team_id': g(t, 'team', 'id'), f'{side}_abbr': g(t, 'team', 'abbreviation'),
                            f'{side}_name': g(t, 'team', 'name'), f'{side}_display': g(t, 'team', 'displayName'),
                            f'{side}_score_sb': pd.to_numeric(t.get('score'), errors='coerce')})
            rows[row['espn_event_id']] = row
    return rows


def odds_rows_summary(eid, pcs, home_abbr):
    out = []
    for p in pcs or []:
        ao, ho = p.get('awayTeamOdds') or {}, p.get('homeTeamOdds') or {}
        ml, ps, tot = p.get('moneyline') or {}, p.get('pointSpread') or {}, p.get('total') or {}
        raw = {k: v for k, v in p.items() if k not in ('links', 'link', 'header', 'footer')}
        for k in ('awayTeamOdds', 'homeTeamOdds'):
            if isinstance(raw.get(k), dict):
                raw[k] = {kk: vv for kk, vv in raw[k].items() if kk != 'team'}
        if isinstance(raw.get('provider'), dict):
            raw['provider'] = {kk: raw['provider'].get(kk) for kk in ('id', 'name', 'priority')}
        out.append(dict(
            espn_event_id=eid, source='summary_pickcenter', provider_id=str(g(p, 'provider', 'id')),
            provider_name=g(p, 'provider', 'name'), details=p.get('details'), spread=p.get('spread'),
            over_under=p.get('overUnder'), over_odds=american(p.get('overOdds')), under_odds=american(p.get('underOdds')),
            home_ml=american(ho.get('moneyLine')), away_ml=american(ao.get('moneyLine')),
            home_spread_odds=american(ho.get('spreadOdds')), away_spread_odds=american(ao.get('spreadOdds')),
            home_favorite=ho.get('favorite'), away_favorite=ao.get('favorite'),
            home_favorite_at_open=ho.get('favoriteAtOpen'), away_favorite_at_open=ao.get('favoriteAtOpen'),
            ml_home_open=american(g(ml, 'home', 'open', 'odds')), ml_home_close=american(g(ml, 'home', 'close', 'odds')),
            ml_away_open=american(g(ml, 'away', 'open', 'odds')), ml_away_close=american(g(ml, 'away', 'close', 'odds')),
            spread_home_open=line_num(g(ps, 'home', 'open', 'line')),
            spread_home_close=line_num(g(ps, 'home', 'close', 'line')),
            spread_home_open_odds=american(g(ps, 'home', 'open', 'odds')),
            spread_home_close_odds=american(g(ps, 'home', 'close', 'odds')),
            spread_away_open_odds=american(g(ps, 'away', 'open', 'odds')),
            spread_away_close_odds=american(g(ps, 'away', 'close', 'odds')),
            total_open=line_num(g(tot, 'over', 'open', 'line')), total_close=line_num(g(tot, 'over', 'close', 'line')),
            over_open_odds=american(g(tot, 'over', 'open', 'odds')), over_close_odds=american(g(tot, 'over', 'close', 'odds')),
            under_open_odds=american(g(tot, 'under', 'open', 'odds')),
            under_close_odds=american(g(tot, 'under', 'close', 'odds')),
            moneyline_winner=None, spread_winner=None, raw_json=json.dumps(raw, separators=(',', ':'))))
    return out


def odds_rows_core(eid, js):
    out = []
    for p in (js or {}).get('items') or []:
        ao, ho = p.get('awayTeamOdds') or {}, p.get('homeTeamOdds') or {}
        raw = {k: v for k, v in p.items() if k not in ('links', '$ref')}
        for k in ('awayTeamOdds', 'homeTeamOdds'):
            if isinstance(raw.get(k), dict):
                raw[k] = {kk: vv for kk, vv in raw[k].items() if kk != 'team'}
        if isinstance(raw.get('provider'), dict):
            raw['provider'] = {kk: raw['provider'].get(kk) for kk in ('id', 'name', 'priority')}
        am = lambda d, *pth: american(g(d, *pth, 'american'))
        out.append(dict(
            espn_event_id=eid, source='core_odds', provider_id=str(g(p, 'provider', 'id')),
            provider_name=g(p, 'provider', 'name'), details=p.get('details'), spread=p.get('spread'),
            over_under=p.get('overUnder'), over_odds=american(p.get('overOdds')), under_odds=american(p.get('underOdds')),
            home_ml=american(ho.get('moneyLine')), away_ml=american(ao.get('moneyLine')),
            home_spread_odds=american(ho.get('spreadOdds')), away_spread_odds=american(ao.get('spreadOdds')),
            home_favorite=ho.get('favorite'), away_favorite=ao.get('favorite'),
            home_favorite_at_open=g(ho, 'open', 'favorite'), away_favorite_at_open=g(ao, 'open', 'favorite'),
            ml_home_open=am(ho, 'open', 'moneyLine'), ml_home_close=am(ho, 'close', 'moneyLine'),
            ml_away_open=am(ao, 'open', 'moneyLine'), ml_away_close=am(ao, 'close', 'moneyLine'),
            spread_home_open=line_num(g(ho, 'open', 'pointSpread', 'american')),
            spread_home_close=line_num(g(ho, 'close', 'pointSpread', 'american')),
            spread_home_open_odds=am(ho, 'open', 'spread'), spread_home_close_odds=am(ho, 'close', 'spread'),
            spread_away_open_odds=am(ao, 'open', 'spread'), spread_away_close_odds=am(ao, 'close', 'spread'),
            total_open=line_num(g(p, 'open', 'total', 'american')), total_close=line_num(g(p, 'close', 'total', 'american')),
            over_open_odds=am(p, 'open', 'over'), over_close_odds=am(p, 'close', 'over'),
            under_open_odds=am(p, 'open', 'under'), under_close_odds=am(p, 'close', 'under'),
            moneyline_winner=p.get('moneylineWinner'), spread_winner=p.get('spreadWinner'),
            raw_json=json.dumps(raw, separators=(',', ':'))))
    return out


def main():
    sb = scoreboard_events()
    log(f'scoreboard events: {len(sb)}')
    pbp_frames, odds, revs = [], [], []
    for eid, row in sb.items():
        f = SUM_DIR / f'{eid}.json.gz'
        row['has_summary'] = f.exists()
        if not f.exists():
            continue
        d = read_json_gz(f)
        h = (g(d, 'header', 'competitions') or [{}])[0]
        comp = {x.get('homeAway'): x for x in h.get('competitors') or []}
        side_of = {str(g(comp.get(s) or {}, 'team', 'id')): s for s in ('home', 'away')}
        for s in ('home', 'away'):
            t = comp.get(s) or {}
            row[f'{s}_score'] = pd.to_numeric(t.get('score'), errors='coerce')
            row[f'{s}_winner'] = t.get('winner')
            row[f'{s}_linescores'] = json.dumps([line_num(x.get('displayValue')) for x in t.get('linescores') or []])
        row['header_status'] = g(h, 'status', 'type', 'name')
        row['summary_source'] = d.get('_fallback_source', 'site_summary')
        row['wallclock_available'] = d.get('wallclockAvailable')
        row['n_pickcenter'] = len(d.get('pickcenter') or [])
        odds += odds_rows_summary(eid, d.get('pickcenter'), row.get('home_abbr'))
        of = ODDS_DIR / f'{eid}.json.gz'
        if of.exists():
            odds += odds_rows_core(eid, read_json_gz(of))
        plays = d.get('plays') or []
        row['n_plays'] = len(plays)
        if not plays:
            continue
        rec = []
        for p in plays:
            rec.append((int(p.get('sequenceNumber') or 0), str(p.get('id')), parse_ts(p.get('wallclock')),
                        g(p, 'period', 'number'), clock_seconds(g(p, 'clock', 'displayValue')),
                        g(p, 'clock', 'displayValue'), p.get('homeScore'), p.get('awayScore'), bool(p.get('scoringPlay')),
                        p.get('scoreValue'), str(g(p, 'type', 'id')), g(p, 'type', 'text'),
                        side_of.get(str(g(p, 'team', 'id'))) if g(p, 'team', 'id') else None))
        pb = pd.DataFrame(rec, columns=['seq', 'play_id', 'wallclock_utc', 'period', 'clock_s', 'clock_display',
                                        'home_score', 'away_score', 'scoring_play', 'score_value', 'play_type_id',
                                        'play_type', 'team_side'])
        pb.insert(0, 'espn_event_id', eid)
        pb = pb.sort_values('seq', kind='stable').reset_index(drop=True)   # API order == sequence order
        pb['play_index'] = np.arange(len(pb), dtype='int32')
        per = pb['period'].astype(float)
        pb['elapsed_s'] = np.where(per <= 4, (per - 1) * 720 + 720, 2880 + (per - 4) * 300) - pb['clock_s']
        hs, as_ = pb['home_score'].astype(float), pb['away_score'].astype(float)
        pb['score_backstep'] = ((hs < hs.shift()) | (as_ < as_.shift())).fillna(False)
        wsec = pb['wallclock_utc'].map(lambda x: x.timestamp() if x is not None else np.nan)
        pb['wallclock_backstep'] = (wsec < wsec.shift()).fillna(False)
        # Suggested wallclock fixes (not applied). Per play: +1 day when the wallclock is > 12 h after ESPN's scheduled
        # start (2025-10-28..31: whole or partial games). Per game: +-1 h when the (day-corrected) first play is
        # 40-75 min early or 55-85 min late (DST days 2026-03-08 / 2026-11-01). Normal tip-off: sched + 7..15 min.
        sched = row['espn_sched_utc'].timestamp() if row.get('espn_sched_utc') is not None else np.nan
        d = wsec - sched
        shift = np.where(d > 12 * 3600, 86400.0, 0.0)
        first_ok = (pb['period'] == 1) & wsec.notna()
        tip_d = ((d - shift)[first_ok].iloc[0] / 60) if first_ok.any() else np.nan
        game_shift = 3600.0 if 55 <= tip_d <= 85 else (-3600.0 if -75 <= tip_d <= -40 else 0.0)
        shift = np.where(shift == 0, game_shift, shift)
        pb['wallclock_shift_suspect_s'] = shift
        dfix = d - shift
        pb['wallclock_out_of_window'] = ((dfix < -20 * 60) | (dfix > 5 * 3600)).fillna(False)
        row['n_plays_wallclock_shift'] = int((shift != 0).sum())
        row['n_plays_wallclock_out_of_window'] = int(pb['wallclock_out_of_window'].sum())
        row['tipoff_utc_fixed'] = (pd.Timestamp(sched + tip_d * 60 - game_shift, unit='s', tz='UTC')
                                   if not np.isnan(tip_d) else None)
        pbp_frames.append(pb)
        # QA per game -----------------------------------------------------------------------------------------
        wc = pb['wallclock_utc']
        p1 = pb[(pb.period == 1) & wc.notna()]
        first = p1.iloc[0] if len(p1) else None
        row['tipoff_utc'] = first['wallclock_utc'] if first is not None else None
        row['first_play_type'] = first['play_type'] if first is not None else None
        row['first_play_clock'] = first['clock_display'] if first is not None else None
        row['last_play_utc'] = wc.dropna().iloc[-1] if wc.notna().any() else None
        row['n_wallclock_missing'] = int(wc.isna().sum())
        dif = np.diff(wsec.dropna().to_numpy())
        row['n_wallclock_backsteps'] = int((dif < 0).sum())
        row['max_wallclock_backstep_s'] = float(-dif.min()) if len(dif) and dif.min() < 0 else 0.0
        row['max_wallclock_gap_s'] = float(dif.max()) if len(dif) else None
        row['max_period'] = int(pb['period'].max())
        row['n_score_backsteps_seq'] = int(pb['score_backstep'].sum())
        # The same check in two chronological orders. Game-clock order (elapsed_s, seq) isolates real score
        # rewrites (a scoring play inserted late whose effect never reached the plays logged before it, or a
        # corrected basket); wallclock order is what a user aligning scores to prices by wallclock will see.
        # wallclock orders use the suggested-fixed wallclock, so that known day/hour shifts do not count as score
        # problems (they are reported separately)
        pbx = pb.assign(_wfix=wsec - pb['wallclock_shift_suspect_s'])
        for name, order, sub in (('clock', ['elapsed_s', 'seq'], pbx),
                                 ('wallclock', ['_wfix', 'seq'], pbx[wc.notna()]),
                                 ('scoring_wallclock', ['_wfix', 'seq'], pbx[wc.notna() & pbx.scoring_play])):
            d = sub.sort_values(order, kind='stable')
            h2, a2 = d['home_score'].astype(float), d['away_score'].astype(float)
            b = (h2 < h2.shift()) | (a2 < a2.shift())
            row[f'n_score_backsteps_{name}'] = int(b.sum())
            if name != 'wallclock':
                for i in np.flatnonzero(b.to_numpy()):
                    revs.append(dict(espn_event_id=eid, order=name, seq=int(d.seq.iloc[i]),
                                     prev_seq=int(d.seq.iloc[i - 1]), period=int(d.period.iloc[i]),
                                     clock_display=d.clock_display.iloc[i], wallclock_utc=d.wallclock_utc.iloc[i],
                                     play_type=d.play_type.iloc[i], home_before=h2.iloc[i - 1],
                                     away_before=a2.iloc[i - 1], home_after=h2.iloc[i], away_after=a2.iloc[i]))
        # final score from the last play in game-clock order: ESPN appends late-entered plays after 'End Game'
        # in sequence order, so the last play by sequence is not the end of the game
        last = pb.sort_values(['elapsed_s', 'seq'], kind='stable').iloc[-1]
        row['final_home_pbp'], row['final_away_pbp'] = float(last['home_score']), float(last['away_score'])
        row['last_play_type'] = last['play_type']
        row['max_home_pbp'], row['max_away_pbp'] = hs.max(), as_.max()
    eg = pd.DataFrame(list(sb.values()))
    # final score: summary header, else the scoreboard (games whose /summary 404s)
    eg['score_source'] = np.where(eg['home_score'].notna(), 'summary_header',
                                  np.where(eg['home_score_sb'].notna() & eg['completed'].fillna(False), 'scoreboard', None))
    for side in ('home', 'away'):
        eg[f'{side}_score'] = eg[f'{side}_score'].where(eg[f'{side}_score'].notna(),
                                                        eg[f'{side}_score_sb'].where(eg['completed'].fillna(False)))
    for c in ('tipoff_utc', 'last_play_utc', 'tipoff_utc_fixed'):
        if c in eg:
            eg[c] = pd.to_datetime(eg[c], utc=True)
    pbp = pd.concat(pbp_frames, ignore_index=True)
    pbp = pbp.astype({'period': 'int16', 'clock_s': 'float32', 'elapsed_s': 'float32', 'home_score': 'int16',
                      'away_score': 'int16', 'seq': 'int32', 'score_backstep': 'bool', 'wallclock_backstep': 'bool',
                      'wallclock_shift_suspect_s': 'float32', 'wallclock_out_of_window': 'bool'})
    pbp['score_value'] = pd.to_numeric(pbp['score_value'], errors='coerce').astype('Int16')
    od = pd.DataFrame(odds)
    write_parquet(eg, WORK / 'espn_games.parquet')
    write_parquet(pbp, WORK / 'espn_pbp.parquet')
    write_parquet(od, WORK / 'espn_odds.parquet')
    write_parquet(pd.DataFrame(revs), WORK / 'espn_score_reversals.parquet')
    log(f'espn_games {len(eg)} (with summary {int(eg.has_summary.sum())}), pbp rows {len(pbp):,}, odds rows {len(od)}, '
        f'score reversals {len(revs)}')
    log('odds by source/provider: ' + str(od.groupby(['source', 'provider_name']).size().to_dict() if len(od) else {}))


if __name__ == '__main__':
    main()
