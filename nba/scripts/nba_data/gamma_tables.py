"""Step 1b - build the Polymarket-side tables from the raw Gamma windows.

Outputs
  data/nba_data/work/games_gamma.parquet  one row per game event (main slug nba-<away>-<home>-YYYY-MM-DD)
  data/nba_data/markets.parquet           every market of the in-scope game events and of their sibling events
                                          (<slug>-v2, <slug>-moneyline, <slug>-exact-margin)
  data/nba_data/work/markets_all.parquet  the same for every discovered game event (incl. out-of-scope dates)

Market roles: the moneyline is the main event's market whose slug equals the event slug (some 2024-25 moneylines are
labelled sportsMarketType='totals'); main spread / main total = the full-game spread / total market with the highest
final volume (> 0) across the main event and its -v2 sibling; volume = Gamma `volume`, or the sum of taker print
sizes from fetch_trades.py --mode probe where Gamma's volume is null (most markets of 2026-03-04..03-25). NOTE: ranking by final volume is look-ahead; the
markets table also carries the creation-order rank so analysts can pick a pre-game rule instead.

Usage: .venv/bin/python scripts/nba_data/gamma_tables.py
"""
import json
import re
import zoneinfo

import pandas as pd

from common import DATA, RAW, SCOPE_FIRST_DATE, SCOPE_LAST_DATE, WORK, jload, log, parse_ts, read_json_gz, season_of, \
    write_parquet

ET = zoneinfo.ZoneInfo('America/New_York')
MAIN_RE = re.compile(r'^nba-([a-z]{2,8})-([a-z]{2,8})-(\d{4}-\d{2}-\d{2})$')
SIBLING_SUFFIX = {'-v2': 'v2', '-moneyline': 'moneyline_dup', '-exact-margin': 'exact_margin'}
# Real single-game events with a non-standard slug (found by scanning in-scope events for markets whose two outcomes
# are NBA team nicknames; everything else of that kind is a series-winner market). slug -> (away_code, home_code).
SPECIAL_GAMES = {'nba-cup-final-thunder-vs-bucks': ('okc', 'mil')}      # 2024 NBA Cup final, Las Vegas
BEAT_Q = re.compile(r'^Will the (.+?) beat the (.+?) by (\d+) or more points\?$')
COMBINE_Q = re.compile(r'combine for (\d+) or more points')
SPREAD_Q = re.compile(r'^(?:1H )?Spread:\s*(.+?)\s*\(([+-]?\d+(?:\.\d+)?)\)\s*$')
FAV_Q = re.compile(r'^Favorite\((.+?)\) vs Underdog\((.+?)\)\s*Line:\s*([+-]?\d+(?:\.\d+)?)')
TOTAL_Q = re.compile(r'(?:O/U|Line:|^Over)\s*(\d+(?:\.\d+)?)')


def load_events():
    evs = {}
    for f in sorted((RAW / 'gamma_windows').glob('*.json.gz')):
        js = read_json_gz(f)
        for e in js['events']:
            k = str(e['id'])
            if k not in evs or (e.get('updatedAt') or '') >= (evs[k].get('updatedAt') or ''):
                evs[k] = e
    return evs


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def winner_index(prices):
    """outcomePrices ['1','0'] -> 0, ['0','1'] -> 1, ['0.5','0.5'] -> -1 (50-50), anything else -> None."""
    p = [fnum(x) for x in (prices or [])]
    if len(p) != 2 or None in p:
        return None
    if p == [1.0, 0.0]:
        return 0
    if p == [0.0, 1.0]:
        return 1
    if p == [0.5, 0.5]:
        return -1
    return None


def norm_kind(m, ev_slug, role):
    """Normalised market kind + a note when Gamma's sportsMarketType disagrees with the market's structure."""
    smt = m.get('sportsMarketType')
    q = (m.get('question') or '').strip()
    slug = m.get('slug') or ''
    if role == 'main' and slug == ev_slug:
        return 'moneyline', (None if smt == 'moneyline' else f'sportsMarketType={smt} but market slug == event slug')
    if role == 'exact_margin':
        return 'exact_margin', None
    kind = smt
    if kind is None:
        if slug.endswith('-moneyline') and '-1h-' not in slug:
            kind = 'moneyline'
        elif '-spread-' in slug and '-1h-' not in slug:
            kind = 'spreads'
        elif '-total-' in slug and '-1h-' not in slug:
            kind = 'totals'
        else:
            kind = 'unknown'
    note = None
    if kind == 'spreads' and not (q.startswith('Spread:') or q.startswith('Favorite(')):
        note = f'spreads but question {q[:60]!r}'
    if kind == 'totals' and not ('O/U' in q or 'Over vs Under' in q or q.startswith('Over ')):
        note = f'totals but question {q[:60]!r}'
    if kind in ('spreads', 'totals') and ('1H ' in q or '-1h-' in slug):
        note = f'{kind} label on a first-half market'
        kind = 'first_half_' + kind
    return kind, note


def parse_lines(m, kind, outcomes, ml_teams):
    """-> dict(handicap0=, team0=, team1=, total_line=) parsed from the question text (Gamma `line` is kept separately:
    its sign is not consistent across seasons). handicap0 is the points handicap of outcome 0, so outcome 0 wins iff
    score(team0) + handicap0 > score(team1). team0/team1 are team nicknames as in the moneyline outcomes.
    Formats seen: 'Spread: Knicks (-2.5)' with outcomes [team, team]; 'Spread: Pacers (-10.5)' with outcomes
    ['Favorite','Underdog'] (the named team is the favourite); 'Favorite(Celtics) vs Underdog(Heat) Line: 8.5';
    totals 'X vs. Y: O/U 230.5', 'Over 217.5', 'Over vs Under Line: 215.5'."""
    q = (m.get('question') or '').strip()
    r = dict(handicap0=None, team0=None, team1=None, total_line=None)
    if kind in ('spreads', 'first_half_spreads'):
        mm = SPREAD_Q.match(q)
        if mm and len(outcomes) == 2:
            team, h = mm.group(1), float(mm.group(2))
            if team == outcomes[0]:
                r.update(handicap0=h, team0=outcomes[0], team1=outcomes[1])
            elif team == outcomes[1]:
                r.update(handicap0=-h, team0=outcomes[0], team1=outcomes[1])
            elif outcomes == ['Favorite', 'Underdog'] and team in ml_teams:
                other = [t for t in ml_teams if t != team]
                r.update(handicap0=-abs(h), team0=team, team1=other[0] if other else None)
        mm = BEAT_Q.match(q)          # 2024 Cup final: 'Will the Thunder beat the Bucks by 5 or more points?'
        if mm and len(outcomes) == 2:
            r.update(handicap0=-(int(mm.group(3)) - 0.5), team0=mm.group(1), team1=mm.group(2))
        mm = FAV_Q.match(q)
        if mm and outcomes == ['Favorite', 'Underdog']:
            r.update(handicap0=-abs(float(mm.group(3))), team0=mm.group(1), team1=mm.group(2))
    elif kind in ('totals', 'first_half_totals'):
        mm = TOTAL_Q.search(q)
        if mm:
            r['total_line'] = float(mm.group(1))
        mm = COMBINE_Q.search(q)      # 'Will the Thunder and Bucks combine for 215 or more points?' -> Over 214.5
        if mm:
            r['total_line'] = int(mm.group(1)) - 0.5
    return r


def main():
    evs = load_events()
    by_slug = {}
    for e in evs.values():
        by_slug.setdefault(e['slug'], []).append(e)
    dup_slugs = {s: [x['id'] for x in v] for s, v in by_slug.items() if len(v) > 1}
    if dup_slugs:
        log(f'WARNING duplicate slugs: {dup_slugs}')
    mains = [e for e in evs.values() if MAIN_RE.match(e['slug']) or e['slug'] in SPECIAL_GAMES]
    log(f'events {len(evs)}, game-slug events {len(mains)}')
    # trade-based volume (sum of taker print sizes) for markets whose trades were pulled (fetch_trades.py manifests);
    # used where Gamma `volume` is null. Gamma volume == sum of taker sizes wherever both exist.
    tvol = {}
    for f in (WORK / 'trades_manifest').glob('*.json'):
        mm = json.loads(f.read_text())
        tvol[str(mm['market_id'])] = (mm['sum_size'], mm['n_rows'])
    log(f'trade manifests available: {len(tvol)}')

    game_rows, mkt_rows = [], []
    for e in mains:
        if e['slug'] in SPECIAL_GAMES:
            (away_code, home_code), slug_date = SPECIAL_GAMES[e['slug']], None
        else:
            away_code, home_code, slug_date = MAIN_RE.match(e['slug']).groups()
        ml_teams = next(([o for o in (jload(m.get('outcomes'), []) or [])] for m in e.get('markets') or []
                         if m.get('slug') == e['slug']), [])
        fam = [('main', e)]
        for suf, role in SIBLING_SUFFIX.items():
            for s in by_slug.get(e['slug'] + suf, []):
                fam.append((role, s))
        ev_mkts = []
        for role, ev in fam:
            for m in ev.get('markets') or []:
                outs = jload(m.get('outcomes'), []) or []
                toks = jload(m.get('clobTokenIds'), []) or []
                prices = jload(m.get('outcomePrices'), []) or []
                kind, note = norm_kind(m, e['slug'], role)
                fs = m.get('feeSchedule') or {}
                row = dict(
                    game_id=str(e['id']), event_id=str(ev['id']), event_slug=ev['slug'], event_role=role,
                    market_id=str(m['id']), condition_id=m.get('conditionId'), market_slug=m.get('slug'),
                    question=m.get('question'), sports_market_type=m.get('sportsMarketType'), kind=kind, kind_note=note,
                    line_gamma=fnum(m.get('line')),
                    outcome_0=outs[0] if len(outs) > 0 else None, outcome_1=outs[1] if len(outs) > 1 else None,
                    n_outcomes=len(outs),
                    token_0=str(toks[0]) if len(toks) > 0 else None, token_1=str(toks[1]) if len(toks) > 1 else None,
                    outcome_price_0=fnum(prices[0]) if len(prices) > 0 else None,
                    outcome_price_1=fnum(prices[1]) if len(prices) > 1 else None,
                    winner_index=winner_index(prices),
                    volume=fnum(m.get('volume')), volume_clob=fnum(m.get('volumeClob')),
                    closed=bool(m.get('closed')), active=bool(m.get('active')),
                    uma_resolution_status=m.get('umaResolutionStatus'),
                    game_start_time=parse_ts(m.get('gameStartTime')), created_at=parse_ts(m.get('createdAt')),
                    accepting_orders_at=parse_ts(m.get('acceptingOrdersTimestamp')),
                    closed_time=parse_ts(m.get('closedTime')), start_date=parse_ts(m.get('startDate')),
                    end_date=parse_ts(m.get('endDate')),
                    fees_enabled=m.get('feesEnabled'), fee_type=m.get('feeType'),
                    fee_rate=fnum(fs.get('rate')), fee_exponent=fnum(fs.get('exponent')),
                    fee_taker_only=fs.get('takerOnly'), fee_rebate_rate=fnum(fs.get('rebateRate')),
                    fee_schedule_json=json.dumps(m.get('feeSchedule')) if m.get('feeSchedule') is not None else None,
                    tick_size=fnum(m.get('orderPriceMinTickSize')), min_order_size=fnum(m.get('orderMinSize')),
                    neg_risk=m.get('negRisk'),
                )
                row.update(parse_lines(m, kind, outs, ml_teams))
                tv = tvol.get(row['market_id'])
                row['volume_trades'] = tv[0] if tv else None
                row['n_prints'] = tv[1] if tv else None
                if row['volume'] is not None:
                    row['volume_effective'], row['volume_source'] = row['volume'], 'gamma'
                elif tv:
                    row['volume_effective'], row['volume_source'] = tv[0], 'trades'
                else:
                    row['volume_effective'], row['volume_source'] = None, None
                ev_mkts.append(row)
        # roles -------------------------------------------------------------------------------------------------
        ml = [r for r in ev_mkts if r['event_role'] == 'main' and r['kind'] == 'moneyline' and r['market_slug'] == e['slug']]
        if not ml:
            ml = sorted([r for r in ev_mkts if r['event_role'] == 'main' and r['kind'] == 'moneyline'],
                        key=lambda r: -(r['volume'] or 0))[:1]
        for kind, role in (('spreads', 'main_spread'), ('totals', 'main_total')):
            cand = [r for r in ev_mkts if r['kind'] == kind and r['event_role'] in ('main', 'v2')]
            cts = lambda r: r['created_at'].timestamp() if r['created_at'] else float('inf')
            by_created = sorted(cand, key=lambda r: (cts(r), int(r['market_id'])))
            for i, r in enumerate(by_created, 1):
                r['rank_created_in_kind'] = i
            by_vol = sorted(cand, key=lambda r: (-(r['volume_effective'] or 0), cts(r), int(r['market_id'])))
            for i, r in enumerate(by_vol, 1):
                r['rank_volume_in_kind'] = i
            if by_vol and (by_vol[0]['volume_effective'] or 0) > 0:
                by_vol[0]['role'] = role
        for r in ml:
            r['role'] = 'moneyline'
        for r in ev_mkts:
            r.setdefault('role', None)
            r.setdefault('rank_created_in_kind', None)
            r.setdefault('rank_volume_in_kind', None)
        mkt_rows.extend(ev_mkts)

        ml = ml[0] if ml else None
        gst = (ml or {}).get('game_start_time') or parse_ts(e.get('startTime')) or parse_ts(e.get('endDate'))
        et_date = gst.astimezone(ET).date().isoformat() if gst else slug_date
        slug_nonstandard = slug_date is None
        slug_date = slug_date or et_date
        teams = {t.get('ordering'): t for t in (e.get('teams') or []) if t.get('ordering') in ('home', 'away')}
        sp = next((r for r in ev_mkts if r['role'] == 'main_spread'), None)
        tt = next((r for r in ev_mkts if r['role'] == 'main_total'), None)
        n_kind = lambda k, traded=False: sum(1 for r in ev_mkts if r['kind'] == k and r['event_role'] in ('main', 'v2')
                                             and (not traded or (r['volume_effective'] or 0) > 0))
        game_rows.append(dict(
            game_id=str(e['id']), slug=e['slug'], title=e.get('title'), away_code=away_code, home_code=home_code,
            slug_date=slug_date, slug_nonstandard=slug_nonstandard, game_start_utc=gst, game_date_et=et_date,
            season=season_of(et_date),
            slug_date_matches_et=(slug_date == et_date),
            in_scope=SCOPE_FIRST_DATE <= et_date <= SCOPE_LAST_DATE,
            event_start_time=parse_ts(e.get('startTime')), event_end_date=parse_ts(e.get('endDate')),
            event_created_at=parse_ts(e.get('createdAt')), event_closed=bool(e.get('closed')),
            event_closed_time=parse_ts(e.get('closedTime')), event_volume=fnum(e.get('volume')),
            pm_score=e.get('score'), pm_period=e.get('period'),
            pm_team_away=(teams.get('away') or {}).get('name'), pm_team_home=(teams.get('home') or {}).get('name'),
            pm_team_away_abbr=(teams.get('away') or {}).get('abbreviation'),
            pm_team_home_abbr=(teams.get('home') or {}).get('abbreviation'),
            series_slug=','.join(s.get('slug') or '' for s in e.get('series') or []),
            n_markets_main=len(e.get('markets') or []), n_markets_family=len(ev_mkts),
            sibling_roles=','.join(sorted({r for r, _ in fam if r != 'main'})),
            n_spread_markets=n_kind('spreads'), n_spread_markets_traded=n_kind('spreads', True),
            n_total_markets=n_kind('totals'), n_total_markets_traded=n_kind('totals', True),
            ml_market_id=(ml or {}).get('market_id'), ml_condition_id=(ml or {}).get('condition_id'),
            ml_outcome_0=(ml or {}).get('outcome_0'), ml_outcome_1=(ml or {}).get('outcome_1'),
            ml_token_0=(ml or {}).get('token_0'), ml_token_1=(ml or {}).get('token_1'),
            ml_volume=(ml or {}).get('volume'), ml_volume_trades=(ml or {}).get('volume_trades'),
            ml_closed=(ml or {}).get('closed'),
            ml_winner_index=(ml or {}).get('winner_index'), ml_kind_note=(ml or {}).get('kind_note'),
            ml_created_at=(ml or {}).get('created_at'), ml_closed_time=(ml or {}).get('closed_time'),
            ml_fees_enabled=(ml or {}).get('fees_enabled'), ml_fee_type=(ml or {}).get('fee_type'),
            main_spread_market_id=(sp or {}).get('market_id'), main_spread_event_role=(sp or {}).get('event_role'),
            main_spread_team0=(sp or {}).get('team0'), main_spread_handicap0=(sp or {}).get('handicap0'),
            main_spread_volume=(sp or {}).get('volume_effective'),
            main_spread_volume_source=(sp or {}).get('volume_source'),
            main_total_market_id=(tt or {}).get('market_id'), main_total_event_role=(tt or {}).get('event_role'),
            main_total_line=(tt or {}).get('total_line'), main_total_volume=(tt or {}).get('volume_effective'),
            main_total_volume_source=(tt or {}).get('volume_source'),
        ))

    games = pd.DataFrame(game_rows).sort_values(['game_start_utc', 'slug']).reset_index(drop=True)
    mk = pd.DataFrame(mkt_rows)
    for c in ('winner_index', 'rank_created_in_kind', 'rank_volume_in_kind', 'n_prints'):
        mk[c] = mk[c].astype('Int64')
    games['ml_winner_index'] = games['ml_winner_index'].astype('Int64')
    for c in ('fees_enabled', 'fee_taker_only', 'neg_risk'):
        mk[c] = mk[c].astype('boolean')
    write_parquet(games, WORK / 'games_gamma.parquet')
    write_parquet(mk, WORK / 'markets_all.parquet')                       # every discovered game event
    write_parquet(mk[mk.game_id.isin(games[games.in_scope].game_id)].reset_index(drop=True),
                  DATA / 'markets.parquet')                             # in-scope games only
    sc = games[games.in_scope]
    log(f'games_gamma: {len(games)} rows ({len(sc)} in scope); markets: {len(mk)} rows')
    log('in-scope games by season: ' + str(sc.groupby('season').size().to_dict()))
    log(f'in scope with ML {sc.ml_market_id.notna().sum()}, main spread {sc.main_spread_market_id.notna().sum()}, '
        f'main total {sc.main_total_market_id.notna().sum()}; slug date != ET date: {(~sc.slug_date_matches_et).sum()}; '
        f'ML mislabelled: {sc.ml_kind_note.notna().sum()}')
    notes = mk[mk.kind_note.notna()]
    log(f'markets with kind notes: {len(notes)}: ' + str(notes.kind_note.str[:40].value_counts().head(10).to_dict()))


if __name__ == '__main__':
    main()
