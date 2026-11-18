"""Step 5 - final analysis-ready tables + QA (no network).

Inputs: work/games_gamma.parquet, markets.parquet, work/espn_*.parquet, work/trades_parts + manifests,
        work/price_parts + manifests.
Outputs (data/nba_data/):
  games.parquet                      one row per in-scope Polymarket game event, with ESPN match, final score,
                                     tip-off, resolution checks and coverage counters
  markets.parquet                    (from gamma_tables.py, left as is)
  price_history/season=<s>/part-0.parquet
  trades/season=<s>/month=<YYYY-MM>/part-0.parquet
  espn_pbp.parquet                   play-by-play of matched games
  pickcenter.parquet                 ESPN pickcenter + core odds of matched games
QA outputs: results/nba_data/qa/*.csv and results/nba_data/qa_summary.json (read by make_report.py).

Usage: .venv/bin/python scripts/nba_data/build_tables.py
"""
import datetime as dt
import json
import shutil
import zoneinfo

import numpy as np
import pandas as pd

from common import DATA, RES, WORK, log, write_parquet

ET = zoneinfo.ZoneInfo('America/New_York')
QA = RES / 'qa'
QA.mkdir(parents=True, exist_ok=True)

# Polymarket slug code -> ESPN abbreviation. Older/alternative Polymarket codes: wsh (2024), pho (once).
PM2ESPN = {'atl': 'ATL', 'bos': 'BOS', 'bkn': 'BKN', 'cha': 'CHA', 'chi': 'CHI', 'cle': 'CLE', 'dal': 'DAL',
           'den': 'DEN', 'det': 'DET', 'gsw': 'GS', 'hou': 'HOU', 'ind': 'IND', 'lac': 'LAC', 'lal': 'LAL',
           'mem': 'MEM', 'mia': 'MIA', 'mil': 'MIL', 'min': 'MIN', 'nop': 'NO', 'nyk': 'NY', 'okc': 'OKC',
           'orl': 'ORL', 'phi': 'PHI', 'phx': 'PHX', 'pho': 'PHX', 'por': 'POR', 'sac': 'SAC', 'sas': 'SA',
           'tor': 'TOR', 'uta': 'UTAH', 'was': 'WSH', 'wsh': 'WSH',
           # All-Star: 2025 mini-tournament (Chuck's Global Stars, Shaq's OGs, Kenny's Young Stars, Candace's
           # Rising Stars), 2026 USA Stars / USA Stripes / World
           'cgs': 'CHK', 'sog': 'SHQ', 'kys': 'KEN', 'crs': 'CAN', 'stars': 'STARS', 'stripes': 'STRIPES',
           'world': 'WORLD'}
ROLES = ('moneyline', 'main_spread', 'main_total')


def season_type_label(r):
    st, ct = r.get('season_type'), r.get('comp_type')
    if pd.isna(st):
        return None
    st = int(st)
    if ct == 'ALLSTAR':
        return 'allstar'
    if ct == 'CC':
        return 'cup_final'
    return {1: 'preseason', 2: 'regular', 3: 'playoffs', 5: 'play_in'}.get(st, f'season_type_{st}')


def match_espn(games, eg):
    """Match each Polymarket game to one ESPN event by unordered team pair and ET date (game date from Gamma
    gameStartTime, then the slug date, then +-1 day)."""
    idx = {}
    for r in eg.itertuples():
        idx.setdefault((r.espn_date_et, frozenset((r.away_abbr, r.home_abbr))), []).append(
            (r.espn_event_id, r.away_abbr, r.home_abbr))
    out = []
    for r in games.itertuples():
        a, h = PM2ESPN.get(r.away_code), PM2ESPN.get(r.home_code)
        res = dict(game_id=r.game_id, espn_event_id=None, match_method=None, n_candidates=0)
        if a and h:
            pair = frozenset((a, h))
            d0 = dt.date.fromisoformat(r.game_date_et)
            # same ET date (either orientation: neutral-site games), then +-1 day with the same away/home only
            # (a +-1 day match with flipped teams is usually the reverse fixture, e.g. CHI@MIA vs MIA@CHI)
            tries = [('et_date', r.game_date_et, False), ('slug_date', r.slug_date, False),
                     ('et_date+1', (d0 + dt.timedelta(1)).isoformat(), True),
                     ('et_date-1', (d0 - dt.timedelta(1)).isoformat(), True)]
            for meth, d, strict in tries:
                c = [x for x in idx.get((d, pair), []) if not strict or (x[1], x[2]) == (a, h)]
                if c:
                    res.update(espn_event_id=c[0][0] if len(c) == 1 else None, match_method=meth, n_candidates=len(c))
                    break
        out.append(res)
    return pd.DataFrame(out)


def spread_result(s0, s1, h0):
    """Outcome-0 result of a spread: 0 if team0 covers (score0 + handicap0 > score1), 1 if not, -1 push."""
    if pd.isna(s0) or pd.isna(s1) or pd.isna(h0):
        return None
    v = s0 + h0 - s1
    return 0 if v > 0 else (1 if v < 0 else -1)


def consolidate_parts(kind, keep, games):
    """Concatenate per-market part files for the markets in `keep` (DataFrame market_id, game_id, role) and write
    them partitioned. kind = 'trades' | 'price_history'."""
    parts_dir = WORK / ('trades_parts' if kind == 'trades' else 'price_parts')
    out_dir = DATA / kind
    if out_dir.exists():
        shutil.rmtree(out_dir)
    meta = keep.merge(games[['game_id', 'season', 'game_date_et']], on='game_id')
    meta['month'] = meta.game_date_et.str[:7]
    groups = meta.groupby(['season'] + (['month'] if kind == 'trades' else []))
    total = 0
    for key, grp in groups:
        frames = []
        for r in grp.itertuples():
            f = parts_dir / f'{r.market_id}.parquet'
            if not f.exists():
                continue
            df = pd.read_parquet(f)
            if not len(df):
                continue
            df.insert(0, 'game_id', r.game_id)
            df.insert(1, 'market_id', r.market_id)
            df.insert(2, 'market_role', r.role)
            frames.append(df)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        for c in ('game_id', 'market_id', 'market_role'):
            df[c] = df[c].astype('string')
        if kind == 'trades':
            df['side'] = df['side'].astype('string')
            df.insert(3, 'time_utc', pd.to_datetime(df['ts'], unit='s', utc=True))
            path = out_dir / f'season={key[0]}' / f'month={key[1]}' / 'part-0.parquet'
        else:
            df.insert(3, 'time_utc', pd.to_datetime(df['t'], unit='s', utc=True))
            path = out_dir / f'season={key if isinstance(key, str) else key[0]}' / 'part-0.parquet'
        write_parquet(df, path)
        total += len(df)
    return total


def main():
    games = pd.read_parquet(WORK / 'games_gamma.parquet')
    games = games[games.in_scope].reset_index(drop=True)
    mk = pd.read_parquet(DATA / 'markets.parquet')
    eg = pd.read_parquet(WORK / 'espn_games.parquet')
    pbp = pd.read_parquet(WORK / 'espn_pbp.parquet')
    od = pd.read_parquet(WORK / 'espn_odds.parquet')
    qa = {}

    # ---- ESPN match ---------------------------------------------------------------------------------------------
    m = match_espn(games, eg)
    games = games.merge(m, on='game_id', how='left')
    ecols = ['espn_event_id', 'espn_sched_utc', 'espn_date_et', 'season_year', 'season_type', 'season_slug',
             'comp_type', 'notes', 'neutral_site', 'status_name', 'home_abbr', 'away_abbr', 'home_name', 'away_name',
             'home_score', 'away_score', 'home_linescores', 'away_linescores', 'max_period', 'n_plays',
             'n_pickcenter', 'tipoff_utc', 'first_play_type', 'first_play_clock', 'last_play_utc', 'last_play_type',
             'n_wallclock_missing', 'n_wallclock_backsteps', 'max_wallclock_backstep_s', 'max_wallclock_gap_s',
             'n_score_backsteps_seq', 'n_score_backsteps_clock', 'n_score_backsteps_wallclock',
             'n_score_backsteps_scoring_wallclock', 'final_home_pbp', 'final_away_pbp', 'wallclock_available',
             'summary_source', 'score_source', 'n_plays_wallclock_shift', 'n_plays_wallclock_out_of_window',
             'tipoff_utc_fixed']
    e2 = eg[ecols].rename(columns={c: 'espn_' + c for c in ecols if c != 'espn_event_id' and not c.startswith('espn_')})
    games = games.merge(e2, on='espn_event_id', how='left')
    for c in ('espn_season_year', 'espn_season_type', 'espn_max_period', 'espn_n_plays', 'espn_n_pickcenter',
              'espn_n_wallclock_missing', 'espn_n_wallclock_backsteps', 'espn_n_score_backsteps_seq',
              'espn_n_score_backsteps_clock', 'espn_n_score_backsteps_wallclock', 'espn_n_score_backsteps_scoring_wallclock',
              'espn_n_plays_wallclock_shift', 'espn_n_plays_wallclock_out_of_window'):
        games[c] = games[c].round().astype('Int64')
    games['game_type'] = [season_type_label(dict(season_type=st, comp_type=ct)) for st, ct in
                          zip(games.espn_season_type, games.espn_comp_type)]
    summer = games.espn_event_id.isna() & games.game_date_et.str[5:7].isin(['07'])
    games.loc[summer, 'game_type'] = 'summer_league'
    allstar_codes = {'cgs', 'sog', 'kys', 'crs', 'stars', 'stripes', 'world'}
    games.loc[games.game_type.isna() & (games.away_code.isin(allstar_codes) | games.home_code.isin(allstar_codes)),
              'game_type'] = 'allstar'
    games.loc[games.game_type.isna(), 'game_type'] = 'unmatched'
    # Postponed games: the Polymarket event matches ESPN's postponed event (no score). 2024-25 markets resolved
    # 50-50; later ones stayed open and resolved with the make-up game (a separate Polymarket event, or the same event
    # with gameStartTime moved to the make-up date, which then matches the played game directly).
    post = games.espn_status_name.isin(['STATUS_POSTPONED', 'STATUS_CANCELED'])
    games['flag_postponed'] = post
    games.loc[post, 'game_type'] = 'postponed'
    # never-played 'if necessary' playoff games: no ESPN event, placeholder start time, resolved 50-50
    games.loc[(games.game_type == 'unmatched') & (games.ml_winner_index == -1), 'game_type'] = 'not_played'
    vol = games.ml_volume.astype(float).fillna(games.ml_volume_trades.astype(float)).fillna(0.0)
    games['dup_group_size'] = games.groupby('espn_event_id').game_id.transform('size').where(
        games.espn_event_id.notna(), 1).astype(int)
    rank = vol.groupby(games.espn_event_id.fillna(games.game_id)).rank(method='first', ascending=False)
    games['is_primary_event'] = rank == 1
    games['is_core_game'] = games.game_type.isin(['regular', 'play_in', 'playoffs', 'cup_final']) & games.is_primary_event
    games['makeup_game_id'] = None
    for i in games.index[post]:
        r = games.loc[i]
        later = games[(games.away_code == r.away_code) & (games.home_code == r.home_code) &
                      (games.game_date_et > r.game_date_et) & (games.espn_status_name == 'STATUS_FINAL')]
        later = later[pd.to_datetime(later.game_date_et) <= pd.to_datetime(r.game_date_et) + pd.Timedelta(days=90)]
        if len(later):
            games.loc[i, 'makeup_game_id'] = later.sort_values('game_date_et').game_id.iloc[0]
    # orientation and names
    games['pm_away_espn'] = games.away_code.map(PM2ESPN)
    games['pm_home_espn'] = games.home_code.map(PM2ESPN)
    games['home_away_swapped_vs_espn'] = (games.espn_event_id.notna() & (games.pm_home_espn != games.espn_home_abbr))
    # final scores keyed by team nickname (Polymarket outcome names are nicknames)
    # Final score per team of each matched game, keyed by ESPN nickname (lower case) AND by Polymarket outcome name:
    # outcome names are nicknames, with variants ('Twolves', 'T-Wolves', 'PHO', All-Star team names). Outcome i of
    # the moneyline belongs to slug code i (0 = away code, 1 = home code); the code -> ESPN abbreviation map then
    # finds the score even when the name differs or ESPN's home/away orientation differs (neutral sites).
    name_score = {}
    for r in games.itertuples():
        if pd.isna(r.espn_event_id):
            continue
        d = {str(r.espn_home_name).lower(): r.espn_home_score, str(r.espn_away_name).lower(): r.espn_away_score}
        by_abbr = {r.espn_home_abbr: r.espn_home_score, r.espn_away_abbr: r.espn_away_score}
        for o, code in ((r.ml_outcome_0, r.away_code), (r.ml_outcome_1, r.home_code)):
            if isinstance(o, str) and o.lower() not in d and PM2ESPN.get(code) in by_abbr:
                d[o.lower()] = by_abbr[PM2ESPN[code]]
        name_score[r.game_id] = d
    # ---- timing QA ----------------------------------------------------------------------------------------------
    gst = pd.to_datetime(games.game_start_utc, utc=True)
    games['pm_start_minus_espn_sched_min'] = (gst - pd.to_datetime(games.espn_sched_utc, utc=True)).dt.total_seconds() / 60
    games['tipoff_minus_pm_start_min'] = (pd.to_datetime(games.espn_tipoff_utc, utc=True) - gst).dt.total_seconds() / 60
    games['tipoff_minus_espn_sched_min'] = (pd.to_datetime(games.espn_tipoff_utc, utc=True)
                                            - pd.to_datetime(games.espn_sched_utc, utc=True)).dt.total_seconds() / 60
    # Suspected whole-offset wallclock errors (normal tip-off is 7-15 min after the scheduled time, median 11):
    # +1 day (seen 2025-10-28..30), +1 h (2026-03-08, DST start), -1 h (2026-11-01, DST end). Not applied to the data;
    # wallclock_corrected = wallclock_utc - wallclock_shift_suspect_s if a user accepts the suggestion.
    dmin = games.tipoff_minus_espn_sched_min
    games['wallclock_shift_suspect_s'] = np.select(
        [dmin.between(1380, 1500), dmin.between(55, 85), dmin.between(-75, -40)], [86400.0, 3600.0, -3600.0],
        default=np.nan)
    games['tipoff_fixed_minus_espn_sched_min'] = (pd.to_datetime(games.espn_tipoff_utc_fixed, utc=True)
                                                  - pd.to_datetime(games.espn_sched_utc, utc=True)).dt.total_seconds() / 60
    games['flag_wallclock_plays_shifted'] = games.espn_n_plays_wallclock_shift.fillna(0) > 0
    games['flag_wallclock_plays_out_of_window'] = games.espn_n_plays_wallclock_out_of_window.fillna(0) > 0
    games['flag_wallclock_early'] = games.tipoff_minus_espn_sched_min < -5
    games['flag_wallclock_late'] = games.tipoff_minus_espn_sched_min > 60
    games['flag_pm_start_differs'] = games.pm_start_minus_espn_sched_min.abs() > 5
    games['flag_first_play_not_jumpball'] = games.espn_event_id.notna() & games.espn_first_play_type.notna() & \
        ~games.espn_first_play_type.fillna('').str.contains('Jumpball|Jump Ball', case=False)
    games['flag_final_score_pbp_mismatch'] = games.espn_home_score.notna() & (
        (games.espn_final_home_pbp != games.espn_home_score) | (games.espn_final_away_pbp != games.espn_away_score))
    # ---- resolution checks --------------------------------------------------------------------------------------
    mkm = mk.set_index('market_id')
    rows = []
    gidx = games.set_index('game_id')
    for r in games.itertuples():
        src = r.makeup_game_id if r.flag_postponed and isinstance(r.makeup_game_id, str) else r.game_id
        ns = name_score.get(src)
        if r.flag_postponed and r.ml_winner_index == -1:
            ns = None      # postponed game resolved 50-50 by rule (2024-25): nothing to compare
        rec = dict(game_id=r.game_id, resolution_checked_against=('makeup_game' if src != r.game_id else 'same_game')
                   if ns else None)
        sx = gidx.loc[src]
        # moneyline
        if isinstance(r.ml_market_id, str) and ns:
            o0, o1 = str(r.ml_outcome_0).lower(), str(r.ml_outcome_1).lower()
            s0, s1 = ns.get(o0), ns.get(o1)
            rec['ml_names_matched'] = s0 is not None and s1 is not None
            if rec['ml_names_matched'] and not (pd.isna(s0) or pd.isna(s1)):
                rec['ml_winner_espn'] = 0 if s0 > s1 else (1 if s1 > s0 else -1)
        # spread
        if isinstance(r.main_spread_market_id, str) and ns:
            x = mkm.loc[r.main_spread_market_id]
            s0, s1 = ns.get(str(x.team0).lower()), ns.get(str(x.team1).lower())
            rec['spread_names_matched'] = s0 is not None and s1 is not None
            if rec['spread_names_matched']:
                res = spread_result(s0, s1, x.handicap0)
                # outcome order: index 0 = team0 in both the [team, team] and the [Favorite, Underdog] formats
                rec['spread_winner_espn'] = res
            rec['spread_winner_pm'] = x.winner_index
        # total
        if isinstance(r.main_total_market_id, str) and ns:
            x = mkm.loc[r.main_total_market_id]
            tot = sx.espn_home_score + sx.espn_away_score
            if not pd.isna(tot) and not pd.isna(x.total_line):
                over_idx = 0 if str(x.outcome_0).lower() == 'over' else 1
                rec['total_points_espn'] = tot
                rec['total_winner_espn'] = over_idx if tot > x.total_line else (1 - over_idx if tot < x.total_line else -1)
            rec['total_winner_pm'] = x.winner_index
        rows.append(rec)
    rc = pd.DataFrame(rows)
    games = games.merge(rc, on='game_id', how='left')
    for c in ('ml_winner_espn', 'spread_winner_espn', 'spread_winner_pm', 'total_winner_espn', 'total_winner_pm'):
        games[c] = games[c].astype('Int64')
    games['ml_resolution_ok'] = np.where(games.ml_winner_espn.isna() | games.ml_winner_index.isna(), None,
                                         games.ml_winner_espn == games.ml_winner_index)
    games['spread_resolution_ok'] = np.where(games.spread_winner_espn.isna() | games.spread_winner_pm.isna(), None,
                                             games.spread_winner_espn == games.spread_winner_pm)
    games['total_resolution_ok'] = np.where(games.total_winner_espn.isna() | games.total_winner_pm.isna(), None,
                                            games.total_winner_espn == games.total_winner_pm)
    # convenience: main spread expressed as the Polymarket home team's handicap (home = slug home = ml outcome 1)
    h0 = games.main_spread_handicap0.astype(float)
    games['main_spread_home_handicap'] = np.where(games.main_spread_team0 == games.ml_outcome_1, h0,
                                                  np.where(games.main_spread_team0 == games.ml_outcome_0, -h0, np.nan))
    # main lines vs the sportsbook closing line (ESPN; summary pickcenter first, else core odds; live rows excluded).
    # The highest-volume Polymarket line is sometimes a far or reverse line (e.g. 'Spread: Nets (-13.5)' when BOS -13.5).
    book = od[~od.provider_name.fillna('').str.contains('Live')].sort_values('source', ascending=False) \
        .drop_duplicates('espn_event_id').set_index('espn_event_id')
    bh = games.espn_event_id.map(book.spread_home_close).astype(float)
    bt = games.espn_event_id.map(book.total_close).astype(float)
    games['book_spread_home_close'] = np.where(games.home_away_swapped_vs_espn, -bh, bh)   # Polymarket-home view
    games['book_total_close'] = bt
    games['book_provider'] = games.espn_event_id.map(book.provider_name)
    games['main_spread_minus_book'] = games.main_spread_home_handicap - games.book_spread_home_close
    games['main_total_minus_book'] = games.main_total_line - games.book_total_close
    # moneyline outcome 0 should be the away team (slug order); check against ESPN team nicknames
    games['ml_outcome0_is_espn_away'] = np.where(
        games.espn_away_name.isna(), None,
        games.ml_outcome_0.str.lower() == games.espn_away_name.str.lower())
    # Polymarket's own event score string ('away-home') vs ESPN
    sc = games.pm_score.fillna('').str.extract(r'^(\d+)-(\d+)$').astype(float)
    games['pm_score_away'], games['pm_score_home'] = sc[0], sc[1]
    games['flag_pm_score_differs_espn'] = games.espn_event_id.notna() & sc[0].notna() & (
        (np.where(games.home_away_swapped_vs_espn, sc[1], sc[0]) != games.espn_away_score) |
        (np.where(games.home_away_swapped_vs_espn, sc[0], sc[1]) != games.espn_home_score))

    # ---- trades / prices coverage -------------------------------------------------------------------------------
    keep = mk[mk.role.isin(ROLES) & mk.game_id.isin(games.game_id)][['market_id', 'game_id', 'role', 'volume', 'closed',
                                                                         'kind', 'market_slug', 'volume_effective',
                                                                         'fee_type', 'fee_rate']]
    tman = {}
    for f in (WORK / 'trades_manifest').glob('*.json'):
        j = json.loads(f.read_text())
        tman[str(j['market_id'])] = j
    pman = {}
    for f in (WORK / 'price_manifest').glob('*.json'):
        j = json.loads(f.read_text())
        pman[str(j['market_id'])] = j
    cov = []
    for r in keep.itertuples():
        t, p = tman.get(r.market_id), pman.get(r.market_id)
        cov.append(dict(market_id=r.market_id, game_id=r.game_id, role=r.role, market_slug=r.market_slug,
                        fee_type=r.fee_type, fee_rate=r.fee_rate,
                        gamma_volume=r.volume, gamma_closed=r.closed,
                        trades_fetched=t is not None, n_prints=(t or {}).get('n_rows'),
                        prints_sum_size=(t or {}).get('sum_size'), prints_sum_notional=(t or {}).get('sum_notional'),
                        prints_first_ts=(t or {}).get('min_ts'), prints_last_ts=(t or {}).get('max_ts'),
                        prints_truncated=(t or {}).get('truncated'), n_token_unmatched=(t or {}).get('n_token_unmatched'),
                        n_outcome_index_disagree=(t or {}).get('n_outcome_index_disagree'),
                        n_exact_dup=(t or {}).get('n_exact_dup'),
                        prices_fetched=p is not None, n_price_0=(p or {}).get('n_0'), n_price_1=(p or {}).get('n_1'),
                        price_first_t_0=(p or {}).get('first_t_0'), price_last_t_0=(p or {}).get('last_t_0'),
                        price_window_start=(p or {}).get('window_start'), price_window_end=(p or {}).get('window_end')))
    cov = pd.DataFrame(cov)
    # Empirical: for games 2026-03-30..2026-04-28 (first weeks of the sports_fees_v2 schedule) taker BUY sizes are net of
    # the fee (never round numbers; print sum ~2.6% below Gamma volume). From 2026-04-29 sizes are gross again.
    gd = cov.game_id.map(games.set_index('game_id').game_date_et)
    cov['net_of_fee_sizes'] = (cov.fee_type == 'sports_fees_v2') & gd.between('2026-03-30', '2026-04-28')
    cov['volume_gap'] = cov.prints_sum_size - cov.gamma_volume
    cov['volume_rel_gap'] = cov.volume_gap / cov.gamma_volume.where(cov.gamma_volume > 0)
    cov.to_csv(QA / 'market_coverage.csv', index=False)
    for role in ROLES:
        c = cov[cov.role == role].set_index('game_id')
        pre = {'moneyline': 'ml', 'main_spread': 'spread', 'main_total': 'total'}[role]
        games[f'{pre}_n_prints'] = games.game_id.map(c.n_prints).astype('Int64')
        games[f'{pre}_prints_sum_size'] = games.game_id.map(c.prints_sum_size)
        games[f'{pre}_n_price_points_0'] = games.game_id.map(c.n_price_0).astype('Int64')
        games[f'{pre}_n_price_points_1'] = games.game_id.map(c.n_price_1).astype('Int64')

    # ---- write final tables -------------------------------------------------------------------------------------
    n_trades = consolidate_parts('trades', keep, games)
    n_prices = consolidate_parts('price_history', keep, games)
    # pbp and odds are keyed by espn_event_id (several Polymarket events can point to one ESPN game)
    used = set(games.espn_event_id.dropna())
    pb = pbp[pbp.espn_event_id.isin(used)].reset_index(drop=True)
    write_parquet(pb, DATA / 'espn_pbp.parquet')
    o = od[od.espn_event_id.isin(used)].reset_index(drop=True)
    write_parquet(o, DATA / 'pickcenter.parquet')
    for c in ('ml_resolution_ok', 'spread_resolution_ok', 'total_resolution_ok', 'ml_outcome0_is_espn_away'):
        games[c] = games[c].astype('boolean')
    write_parquet(games, DATA / 'games.parquet')

    # ---- QA lists -----------------------------------------------------------------------------------------------
    base = ['game_id', 'slug', 'game_type', 'game_start_utc', 'espn_event_id', 'espn_sched_utc']
    lists = {
        'ml_resolution_mismatch': games[games.ml_resolution_ok == False][base + [
            'ml_outcome_0', 'ml_outcome_1', 'ml_winner_index', 'ml_winner_espn', 'espn_away_score', 'espn_home_score',
            'pm_score']],
        'ml_unresolved_or_5050': games[games.ml_winner_index.isna() | (games.ml_winner_index == -1)][base + [
            'ml_market_id', 'ml_closed', 'ml_winner_index', 'espn_status_name']],
        'spread_resolution_mismatch': games[games.spread_resolution_ok == False][base + [
            'main_spread_market_id', 'main_spread_team0', 'main_spread_handicap0', 'spread_winner_pm',
            'spread_winner_espn', 'espn_away_score', 'espn_home_score']],
        'total_resolution_mismatch': games[games.total_resolution_ok == False][base + [
            'main_total_market_id', 'main_total_line', 'total_winner_pm', 'total_winner_espn', 'total_points_espn']],
        'unmatched_espn': games[games.espn_event_id.isna()][base + ['away_code', 'home_code', 'game_date_et',
                                                                     'slug_date', 'n_candidates', 'ml_winner_index']],
        'wallclock_anomalies': games[games.flag_wallclock_early | games.flag_wallclock_late |
                                     games.flag_wallclock_plays_shifted | games.flag_wallclock_plays_out_of_window][
            base + ['espn_tipoff_utc', 'tipoff_minus_espn_sched_min', 'tipoff_minus_pm_start_min', 'wallclock_shift_suspect_s',
                    'espn_n_plays', 'espn_n_plays_wallclock_shift', 'espn_n_plays_wallclock_out_of_window',
                    'tipoff_fixed_minus_espn_sched_min', 'espn_first_play_type']],
        'pm_start_vs_espn_sched': games[games.flag_pm_start_differs][base + ['pm_start_minus_espn_sched_min']],
        'first_play_not_jumpball': games[games.flag_first_play_not_jumpball][base + [
            'espn_first_play_type', 'espn_first_play_clock', 'espn_tipoff_utc']],
        'final_score_pbp_mismatch': games[games.flag_final_score_pbp_mismatch & games.espn_home_score.notna()][base + [
            'espn_away_score', 'espn_home_score', 'espn_final_away_pbp', 'espn_final_home_pbp']],
        'pm_score_differs_espn': games[games.flag_pm_score_differs_espn][base + [
            'pm_score', 'espn_away_score', 'espn_home_score', 'home_away_swapped_vs_espn']],
        'ml_names_not_matched': games[games.espn_home_score.notna() & (games.ml_names_matched != True)][base + [
            'ml_outcome_0', 'ml_outcome_1', 'espn_away_name', 'espn_home_name']],
        'ml_outcome0_not_away': games[games.ml_outcome0_is_espn_away == False][base + [
            'ml_outcome_0', 'ml_outcome_1', 'espn_away_name', 'espn_home_name', 'espn_neutral_site']],
        'home_away_swapped': games[games.home_away_swapped_vs_espn][base + [
            'away_code', 'home_code', 'espn_away_abbr', 'espn_home_abbr', 'espn_notes', 'espn_neutral_site']],
        'score_backsteps_clock_order': games[games.espn_n_score_backsteps_clock > 0][base + [
            'espn_n_score_backsteps_seq', 'espn_n_score_backsteps_clock', 'espn_n_score_backsteps_wallclock',
            'espn_n_score_backsteps_scoring_wallclock']],
        'slug_date_not_et_date': games[~games.slug_date_matches_et][base + ['slug_date', 'game_date_et']],
        'main_line_far_from_book': games[(games.main_spread_minus_book.abs() >= 3) | (games.main_total_minus_book.abs() >= 6)][
            base + ['main_spread_team0', 'main_spread_handicap0', 'main_spread_home_handicap', 'book_spread_home_close',
                    'main_spread_minus_book', 'main_total_line', 'book_total_close', 'main_total_minus_book',
                    'main_spread_volume', 'main_total_volume']],
        'duplicate_pm_events': games[games.dup_group_size > 1].sort_values(['espn_event_id', 'is_primary_event'])[
            base + ['dup_group_size', 'is_primary_event', 'ml_volume', 'ml_volume_trades', 'ml_winner_index',
                    'match_method', 'event_created_at']],
        'postponed': games[games.flag_postponed][base + ['espn_status_name', 'makeup_game_id', 'ml_winner_index',
                                                          'ml_closed_time']],
        # markets with net-of-fee BUY sizes (games 2026-03-30..04-28) are ~2.6% below Gamma volume by construction and
        # are left out of this completeness list.
        'trades_volume_gap': cov[cov.trades_fetched & cov.gamma_volume.notna() & (cov.volume_rel_gap.abs() > 1e-4)
                                 & ~cov.net_of_fee_sizes],
        'trades_missing': cov[~cov.trades_fetched],
        'prices_missing_or_empty': cov[~cov.prices_fetched | (cov.n_price_0.fillna(0) == 0) | (cov.n_price_1.fillna(0) == 0)],
    }
    for k, v in lists.items():
        v.to_csv(QA / f'{k}.csv', index=False)
        qa[f'n_{k}'] = int(len(v))
    vg = cov[cov.trades_fetched & cov.gamma_volume.notna() & (cov.gamma_volume > 0)].copy()
    vg['fee'] = vg.fee_type.fillna('none').where(~vg.net_of_fee_sizes, 'sports_fees_v2 (games to 2026-04-28)')
    qa['print_vs_gamma_volume_by_fee_type'] = {
        k: dict(markets=int(len(v)), within_1e4=int((v.volume_rel_gap.abs() <= 1e-4).sum()),
                median_rel_gap=round(float(v.volume_rel_gap.median()), 5), min_rel_gap=round(float(v.volume_rel_gap.min()), 5),
                max_rel_gap=round(float(v.volume_rel_gap.max()), 5)) for k, v in vg.groupby('fee')}
    # sanity of the consolidated prints and prices
    import pyarrow.dataset as pads
    tt = pads.dataset(DATA / 'trades', format='parquet', partitioning='hive').to_table(
        columns=['market_id', 'ts', 'token_index', 'side', 'price', 'size']).to_pandas()
    mt = mk.set_index('market_id')[['created_at', 'closed_time']]
    tt = tt.join(mt, on='market_id')
    qa['trades_price_outside_0_1'] = int(((tt.price <= 0) | (tt.price > 1)).sum())
    qa['trades_size_nonpositive'] = int((tt['size'] <= 0).sum())
    qa['trades_token_unmatched'] = int((tt.token_index < 0).sum())
    qa['trades_side_values'] = sorted(tt.side.astype(str).unique().tolist())
    qa['trades_before_market_created'] = int((tt.ts < tt.created_at.map(lambda x: x.timestamp() if pd.notna(x) else -1) - 60).sum())
    after = tt.ts > tt.closed_time.map(lambda x: x.timestamp() if pd.notna(x) else 1e12) + 60
    qa['trades_after_market_closed'] = int(after.sum())
    if after.any():
        ac = tt[after].assign(after_min=lambda d: (d.ts - d.closed_time.map(lambda x: x.timestamp())) / 60)
        ac.groupby('market_id').agg(n_prints=('ts', 'size'), max_minutes_after_close=('after_min', 'max'),
                                    min_price=('price', 'min'), max_price=('price', 'max'), shares=('size', 'sum')) \
            .reset_index().merge(mk[['market_id', 'game_id', 'market_slug', 'role', 'closed_time']], on='market_id') \
            .sort_values('n_prints', ascending=False).to_csv(QA / 'trades_after_close.csv', index=False)
    # Print-feed outages: per game night, gaps in the union of all fetched prints (moneyline + main lines) of the core
    # games that night while >= 2 of them are in play [tip+10 min, tip+2.5 h]. A gap counts as a suspected feed outage
    # when the CLOB minute price of an in-play moneyline still moved during it (>= 3 distinct values): trades happened
    # but the Data API does not return them. Gamma volume misses them as well, so the volume check cannot see these.
    core_ids = set(games.game_id[games.is_core_game])
    tg = pads.dataset(DATA / 'trades', format='parquet', partitioning='hive').to_table(columns=['game_id', 'ts']).to_pandas()
    tg = tg[tg.game_id.isin(core_ids)].merge(games[['game_id', 'game_date_et']], on='game_id')
    ph_ml = pads.dataset(DATA / 'price_history', format='parquet', partitioning='hive').to_table(
        columns=['game_id', 'market_role', 'token_index', 't', 'p']).to_pandas()
    ph_ml = ph_ml[(ph_ml.market_role == 'moneyline') & (ph_ml.token_index == 0)]
    ph_by_game = {k: v[['t', 'p']].to_numpy() for k, v in ph_ml.groupby('game_id')}
    gs = games[games.is_core_game][['game_id', 'game_date_et', 'game_start_utc']].copy()
    gs['gst'] = gs.game_start_utc.map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
    gaps = []
    for d, grp in tg.groupby('game_date_et'):
        ts = np.unique(grp.ts.to_numpy())
        night = gs[gs.game_date_et == d]
        lo, hi = night.gst.to_numpy() + 600, night.gst.to_numpy() + 2.5 * 3600
        dif = np.diff(ts)
        for i in np.flatnonzero(dif >= 300):
            a, b = ts[i], ts[i + 1]
            live = (lo <= a) & (hi >= b)
            if live.sum() < 2:
                continue
            moves = []
            for gid in night.game_id[live]:
                arr = ph_by_game.get(gid)
                if arr is not None:
                    w = arr[(arr[:, 0] > a) & (arr[:, 0] < b), 1]
                    moves.append(len(np.unique(w)))
            mx = max(moves) if moves else None
            # prints_missing: CLOB minute prices kept moving -> trades happened, the Data API lacks them.
            # both_missing: no CLOB minute point either for any in-play game -> hole in both sources.
            kind = 'prints_missing' if mx is not None and mx >= 3 else ('both_missing' if mx == 0 else 'quiet')
            gaps.append(dict(game_date_et=d, gap_start_utc=pd.to_datetime(a, unit='s', utc=True),
                             gap_end_utc=pd.to_datetime(b, unit='s', utc=True), minutes=round((b - a) / 60, 1),
                             games_in_play=int(live.sum()), core_games_that_night=len(night),
                             max_distinct_ml_prices_in_gap=mx, gap_kind=kind,
                             suspected_outage=kind in ('prints_missing', 'both_missing'),
                             game_ids=','.join(night.game_id[live])))
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(QA / 'trades_print_gaps.csv', index=False)
    sus = gaps[gaps.suspected_outage] if len(gaps) else gaps
    hit = set(','.join(sus.game_ids).split(',')) if len(sus) else set()
    games['flag_print_gap_in_play'] = games.game_id.isin(hit)
    qa['n_print_gap_windows'] = int(len(gaps))
    qa['n_suspected_print_outages'] = int(len(sus))
    qa['n_games_in_print_outages'] = int(games.flag_print_gap_in_play.sum())
    qa['print_outage_minutes_total'] = float(sus.minutes.sum()) if len(sus) else 0.0
    pp = pads.dataset(DATA / 'price_history', format='parquet', partitioning='hive').to_table(columns=['market_id', 'p']).to_pandas()
    qa['prices_outside_0_1'] = int(((pp.p < 0) | (pp.p > 1)).sum())
    qa['prices_outside_0_1_markets'] = sorted(pp[(pp.p < 0) | (pp.p > 1)].market_id.astype(str).unique().tolist())
    del tt, pp
    # ESPN completed games in range that have no Polymarket game event
    lo, hi = games.game_date_et.min(), games.game_date_et.max()
    miss = eg[(eg.espn_date_et >= lo) & (eg.espn_date_et <= hi) & ~eg.espn_event_id.isin(used)]
    miss[['espn_event_id', 'espn_date_et', 'short_name', 'status_name', 'season_type', 'comp_type', 'notes']].to_csv(
        QA / 'espn_games_without_pm_event.csv', index=False)
    qa['n_espn_games_without_pm_event'] = int(len(miss))
    qa.update(n_games=int(len(games)), n_trades_rows=int(n_trades), n_price_rows=int(n_prices),
              n_pbp_rows=int(len(pb)), n_pickcenter_rows=int(len(o)), built_at=dt.datetime.now(dt.timezone.utc).isoformat())
    write_parquet(games, DATA / 'games.parquet')          # again, now with flag_print_gap_in_play
    (RES / 'qa_summary.json').write_text(json.dumps(qa, indent=1))
    log(json.dumps(qa))
    log('games by season/type: ' + str(games.groupby(['season', 'game_type']).size().to_dict()))


if __name__ == '__main__':
    main()
