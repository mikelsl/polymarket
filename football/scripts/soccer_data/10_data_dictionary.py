"""Step 10: render results/soccer_data/DATA_DICTIONARY.md from the parquet schemas + the descriptions below.
Fails if a column has no description, so the dictionary cannot drift from the tables.
Usage: .venv/bin/python -I scripts/soccer_data/10_data_dictionary.py
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import re

import pyarrow.parquet as pq

from common import DATA, RES

TS = 'epoch seconds, UTC'
D = {}
D['matches'] = {
    'match_id': 'PK. Polymarket main event slug `<lg>-<home>-<away>-<YYYY-MM-DD>` (slug date = date Polymarket first listed the match; can differ from the played date).',
    'league': 'Polymarket slug prefix: epl, lal (LaLiga), sea (Serie A), bun (Bundesliga), fl1 (Ligue 1).',
    'league_name': 'League name.',
    'season': "'2025-26' (kickoff 2025-07-01..2026-06-30) or '2026-27' (kickoff 2026-07-01..2026-11-03, UTC).",
    'season_month': 'YYYY-MM of the canonical kickoff (UTC).',
    'kickoff_ts': f'Canonical scheduled kickoff ({TS}): ESPN scheduled date when linked (tracks postponements), else Polymarket gameStartTime.',
    'kickoff_utc': 'kickoff_ts as UTC timestamp.',
    'kickoff_source': "'espn_scheduled' or 'pm_gameStartTime'.",
    'home_team_pm': 'Home team name from the Polymarket event title (first-listed team; spelling changes between seasons).',
    'away_team_pm': 'Away team name from the Polymarket event title.',
    'home_team': 'Canonical home team name = ESPN displayName (via team_map_pm_espn.csv); same team in Polymarket orientation.',
    'away_team': 'Canonical away team name = ESPN displayName.',
    'home_score': 'Full-time goals of home_team (ESPN, Polymarket orientation).',
    'away_score': 'Full-time goals of away_team (ESPN).',
    'home_ht_score': 'Half-time goals of home_team (ESPN linescore period 1).',
    'away_ht_score': 'Half-time goals of away_team.',
    'pm_result': "Polymarket 1X2 resolution from closed-market outcomePrices: H, D, A; UNRESOLVED if any 1X2 market not closed/0-1.",
    'espn_result': 'H/D/A from ESPN final score (Polymarket orientation).',
    'resolution_check': "'match' (pm_result == espn_result), 'MISMATCH', 'pm_unresolved', 'no_espn_result'.",
    'espn_event_id': 'ESPN event id (summary?event=<id>); null if not linked.',
    'is_duplicate_event': 'True if two Polymarket events point to the same ESPN match (match moved by a day; both traded and resolved).',
    'is_primary_event': 'For duplicates: the event whose gameStartTime is closest to the ESPN date (ties: larger 1X2 volume). True for all others.',
    'pm_event_id': 'Gamma event id.',
    'pm_title': 'Gamma event title.',
    'home_code': 'Polymarket slug code of the home team (also the suffix of the home-win market slug).',
    'away_code': 'Polymarket slug code of the away team.',
    'slug_date': 'Date in the slug (YYYY-MM-DD).',
    'n_gamestart_values': 'Distinct gameStartTime values across the three 1X2 markets (always 1).',
    'event_end_date_ts': f'Gamma event endDate ({TS}); equals the original gameStartTime.',
    'event_start_date_ts': f'Gamma event startDate ({TS}) = listing time.',
    'event_created_at_ts': f'Gamma event createdAt ({TS}).',
    'event_closed': 'Gamma event closed flag at download time (2026-11-04).',
    'event_closed_time_ts': f'Gamma event closedTime ({TS}).',
    'event_volume': 'Gamma event volume (sum over the event markets).',
    'event_score': "Gamma sports score string (e.g. '1-0'); often null for 2025 events.",
    'event_period': 'Gamma sports period string at download time.',
    'event_finished_ts': f'Gamma finishedTimestamp ({TS}); mostly null.',
    'pm_game_id': 'Polymarket sports gameId.',
    'neg_risk': 'Gamma negRisk flag of the event.',
    'series_slug': 'Gamma series slug.',
    'n_1x2': 'Number of 1X2 markets found (always 3).',
    'unknown_roles': '1 if a market could not be assigned home/draw/away (always 0).',
    'vol_home': 'Gamma volume of the home-win market (null when Gamma has no volume).',
    'vol_draw': 'Gamma volume of the draw market.',
    'vol_away': 'Gamma volume of the away-win market.',
    'vol_1x2': 'vol_home + vol_draw + vol_away (Gamma).',
    'has_more_markets': "A '-more-markets' sibling event exists.",
    'has_ou25': 'Main full-match Over/Under 2.5 goals market exists (role ou25 in markets).',
    'has_btts': 'Full-match both-teams-to-score market exists (role btts in markets).',
    'fees_enabled_1x2': 'Any 1X2 market has feesEnabled.',
    'fee_rate_1x2': 'Max feeSchedule.rate over the 1X2 markets (0 = no fee). Taker fee = shares x rate x p x (1-p).',
    'home_away_swapped': 'True if ESPN lists the teams the other way round (1 match: fl1-psg-ren-2026-08-23). Scores here are in Polymarket orientation.',
    'espn_date_ts': f'ESPN scheduled kickoff ({TS}).',
    'espn_home': 'ESPN home team displayName (ESPN orientation).',
    'espn_away': 'ESPN away team displayName.',
    'espn_home_id': 'ESPN team id (home, ESPN orientation).',
    'espn_away_id': 'ESPN team id (away).',
    'espn_status': 'ESPN status at download (STATUS_FULL_TIME for all linked played matches).',
    'espn_status_detail': 'ESPN status detail text.',
    'espn_home_score': 'ESPN final home score (ESPN orientation).',
    'espn_away_score': 'ESPN final away score (ESPN orientation).',
    'espn_home_ht_score': 'ESPN half-time home score (ESPN orientation).',
    'espn_away_ht_score': 'ESPN half-time away score (ESPN orientation).',
    'wallclock_available': 'ESPN summary flag wallclockAvailable.',
    'venue': 'ESPN venue name.',
    'attendance': 'ESPN attendance.',
    'kickoff_wall_ts': f"ESPN key event 'kickoff' wallclock ({TS}) = actual first-half start, second precision.",
    'halftime_wall_ts': f"ESPN 'halftime' wallclock ({TS}) = end of first half.",
    'second_half_wall_ts': f"ESPN 'start-2nd-half' wallclock ({TS}).",
    'fulltime_wall_ts': f"ESPN 'end-regular-time' wallclock ({TS}) = final whistle.",
    'kickoff_wall_utc': 'kickoff_wall_ts as UTC timestamp.',
    'halftime_wall_utc': 'halftime_wall_ts as UTC timestamp.',
    'second_half_wall_utc': 'second_half_wall_ts as UTC timestamp.',
    'fulltime_wall_utc': 'fulltime_wall_ts as UTC timestamp.',
    'ke_goals_home': 'Goals credited to the ESPN home side in key events (own goals count for the benefiting side).',
    'ke_goals_away': 'Goals credited to the ESPN away side in key events.',
    'ke_n_events': 'Number of ESPN key events.',
    'red_cards_home': 'Red cards (key events, ESPN home side).',
    'red_cards_away': 'Red cards (key events, ESPN away side).',
    'n_goal_wallclock_missing': 'Goals without a wallclock (0 everywhere).',
    'pm_game_start_ts': f'Polymarket gameStartTime ({TS}); NOT updated for postponed matches.',
    'pm_minus_espn_kickoff_s': 'pm_game_start_ts - espn_date_ts in seconds (nonzero for 23 re-dated matches, 16 by more than 30 min).',
    'kickoff_wall_minus_sched_s': 'kickoff_wall_ts - kickoff_ts in seconds (actual vs scheduled kickoff).',
    'ke_goals_consistent': 'Key-event goal counts equal the ESPN final score.',
    'odds_join': "football-data.co.uk join status: 'ok' (odds.match_id = this event), 'ok_secondary_event' (duplicate event; reach the row via espn_event_id), 'none'.",
    'has_odds': "odds_join == 'ok'.",
    'inplay_prints': 'Taker prints on the three 1X2 markets between kickoff_wall_ts and fulltime_wall_ts.',
    'inplay_max_gap_s': 'Longest interval (s) without any 1X2 print between kickoff and full-time wallclocks. >= 900 s in a liquid match = trading halt (see REPORT / qa_inplay_halts.csv).',
    'inplay_max_gap_start_ts': f'Start of that interval ({TS}).',
    'analysis_ready': 'is_primary_event AND resolution_check == match AND espn_status == STATUS_FULL_TIME. Recommended filter.',
}
D['markets'] = {
    'match_id': 'FK -> matches.match_id.', 'league': 'League code.', 'season': 'Season label.',
    'kickoff_ts': f'Canonical kickoff from matches ({TS}).',
    'role': "'home' (home team wins, YES), 'draw', 'away' (away team wins), 'ou25' (Over 2.5 goals: outcome0='Over'), 'btts' (both teams score: outcome0='Yes').",
    'market_id': 'Gamma market id.', 'condition_id': 'PK. CTF condition id (Data API `condition`).',
    'question': 'Market question.', 'group_item_title': 'Gamma groupItemTitle.', 'market_slug': 'Market slug.',
    'sports_market_type': "Gamma sportsMarketType ('moneyline', 'totals', 'both_teams_to_score'); null for 1X2 markets before Dec 2025.",
    'line': 'Totals line (2.5 for ou25), else null.',
    'outcomes': 'JSON list of outcome labels.', 'outcome0': "Label of token0 ('Yes' or 'Over').", 'outcome1': "Label of token1 ('No' or 'Under').",
    'clob_token_ids': 'JSON list of CLOB token ids [token0, token1].',
    'token0_id': 'Token id of outcome0 (YES for 1X2/BTTS, Over for ou25). price_history is for this token.',
    'token1_id': 'Token id of outcome1 (NO / Under).',
    'volume': 'Gamma volume. For 1X2 markets it equals the sum of taker print sizes in shares (see REPORT ratio); null for a few markets.',
    'volume_clob': 'Gamma volumeClob.', 'closed': 'Market closed at download.', 'active': 'Gamma active flag.', 'archived': 'Gamma archived flag.',
    'outcome_prices': 'JSON list of final outcomePrices (floats).',
    'outcome0_final_price': 'Final outcomePrices[0] (1 = outcome0 won, 0 = lost).',
    'resolved_outcome0': 'True/False if closed with 0/1 prices, null otherwise.',
    'uma_resolution_status': 'Gamma umaResolutionStatus.', 'resolved_by': 'Gamma resolvedBy address.',
    'automatically_resolved': 'Gamma automaticallyResolved.',
    'fees_enabled': 'Gamma feesEnabled.', 'fee_schedule': 'JSON feeSchedule {exponent, rate, rebateRate, takerOnly}; null when no fees.',
    'fee_rate': 'feeSchedule.rate (taker fee = shares x rate x p x (1-p)); null = no fee schedule.',
    'fee_exponent': 'feeSchedule.exponent.', 'fee_taker_only': 'feeSchedule.takerOnly.', 'fee_rebate_rate': 'feeSchedule.rebateRate (maker rebate share).',
    'fee_type': 'Gamma feeType.', 'taker_base_fee': 'Gamma takerBaseFee (legacy field, bps-like; not the sports fee formula).',
    'maker_base_fee': 'Gamma makerBaseFee (legacy).',
    'created_at_ts': f'Market createdAt ({TS}).', 'start_date_ts': f'Market startDate ({TS}).', 'end_date_ts': f'Market endDate ({TS}).',
    'closed_time_ts': f'Market closedTime ({TS}).', 'game_start_time_ts': f'Polymarket gameStartTime ({TS}); stale for postponed matches.',
    'neg_risk': 'negRisk flag.', 'neg_risk_market_id': 'negRiskMarketID.', 'order_price_min_tick_size': 'Tick size.',
    'order_min_size': 'Minimum order size (shares).', 'seconds_delay': 'Gamma secondsDelay (order matching delay for in-play markets).',
    'last_trade_price': 'Gamma lastTradePrice at download.',
    'fetched': '1X2 only: trade prints were downloaded for this market.',
    'n_taker_prints': '1X2 only: number of taker prints in trades.', 'taker_size_sum': '1X2 only: sum of print sizes (shares).',
    'taker_notional_sum': '1X2 only: sum of price x size (USDC).', 'first_print_ts': f'1X2 only: first print ({TS}).',
    'last_print_ts': f'1X2 only: last print ({TS}).', 'n_prints_inplay': '1X2 only: prints with 0 <= t_rel_kickoff_s <= 9000.',
    'taker_size_to_volume': '1X2 only: taker_size_sum / volume (completeness check; ~1.0).',
}
D['price_history'] = {
    'match_id': 'FK -> matches.', 'role': 'Market role (see markets.role).', 'condition_id': 'FK -> markets.condition_id.',
    'token_id': 'token0 of the market (YES / Over).', 'outcome': "outcome0 label ('Yes' / 'Over').",
    'window_basis': "Kickoff used for the window: 'pm_gameStartTime' or 'espn_scheduled' (re-fetched for re-dated matches).",
    'ts': f'Point time ({TS}); roughly one point per minute.', 'price': 'CLOB prices-history price (0-1, USDC per share); a minute snapshot, NOT a tradable quote; lags around goals.',
    'ts_utc': 'ts as UTC timestamp.', 't_rel_kickoff_s': 'ts - matches.kickoff_ts (seconds; window is -21600..+9000).',
}
D['trades'] = {
    'match_id': 'FK -> matches.', 'season': 'Season.', 'league': 'League code.', 'role': "'home', 'draw' or 'away' market.",
    'condition_id': 'FK -> markets.condition_id.', 'token_id': 'Token traded (YES or NO token of that market).',
    'outcome': "'Yes' or 'No' (token traded).", 'outcome_index': '0 = Yes token, 1 = No token.',
    'side': "Taker side: 'BUY' = taker bought `token_id` at `price`; 'SELL' = taker sold it.",
    'price': 'Fill price of token_id (USDC per share, 0-1).', 'size': 'Fill size in shares (outcome tokens).',
    'ts': f'Fill time ({TS}), one-second resolution; several fills can share a second.',
    'ts_utc': 'ts as UTC timestamp.', 't_rel_kickoff_s': 'ts - matches.kickoff_ts (seconds).',
    'yes_price': 'Price on the YES scale: price for Yes-token fills, 1 - price for No-token fills.',
    'taker_buys_yes': 'True if the taker gained YES exposure (BUY Yes or SELL No).',
    'notional_usdc': 'price x size (USDC paid/received for token_id, before fees).',
    'seq_in_market': 'Chronological index within the market (ties in ts keep API order, oldest first).',
    'transaction_hash': 'Polygon transaction hash (not unique per row: one tx can contain several fills).',
    'proxy_wallet': 'Taker proxy wallet address.',
}
D['events'] = {
    'match_id': 'FK -> matches.', 'espn_event_id': 'ESPN event id.',
    'seq': 'Order within the ESPN keyEvents list; commentary-derived VAR rows use 1000 + commentary sequence.',
    'espn_play_id': 'ESPN play id.', 'type': "ESPN key event type (kickoff, goal, goal---header, goal---volley, goal---free-kick, penalty---scored, own-goal, penalty---missed/saved, red-card, yellow-card, substitution, halftime, start-2nd-half, end-regular-time, start-delay, end-delay, ...) or var-* for VAR rows.",
    'type_text': 'ESPN type text.', 'period': '1 = first half, 2 = second half.',
    'clock_display': "ESPN match clock text, e.g. \"37'\" or \"45'+4'\".",
    'clock_value_s': 'ESPN clock value in seconds of match time (capped at 2700 / 5400 during added time).',
    'minute': 'Match minute parsed from clock_display (45 or 90 during added time).', 'added_minute': 'Added-time minute (\"+4\" -> 4), else 0.',
    'wallclock_ts': f'ESPN wallclock of the event ({TS}), one-second resolution. See REPORT for its accuracy vs trade prints.',
    'wallclock_utc': 'wallclock_ts as UTC timestamp.',
    'team_side': 'home/away in ESPN orientation (null for period markers).',
    'team_side_pm': 'home/away in Polymarket orientation (use this to pick the 1X2 market).',
    'team_name': 'ESPN team displayName. For own goals ESPN reports the BENEFITING team.',
    'scoring_play': 'ESPN scoringPlay (goals incl. penalties and own goals).',
    'goal_kind': 'goal, header, volley, free-kick, penalty, own-goal (scoring rows only).',
    'is_red_card': "type == 'red-card' (direct or second yellow as reported by ESPN).",
    'is_period_marker': 'kickoff / halftime / start-2nd-half / end-regular-time.',
    'home_score_after': 'ESPN-orientation home score after this goal (parsed from the text; falls back to counting).',
    'away_score_after': 'ESPN-orientation away score after this goal.',
    'score_source': "'text' or 'count' (how *_score_after was obtained).",
    'goal_count_home_after': 'Running count of goals credited to the ESPN home side.',
    'goal_count_away_after': 'Running count of goals credited to the ESPN away side.',
    'text': 'ESPN event text.', 'source': "'keyEvents' or 'commentary' (VAR decisions).",
    't_from_kickoff_wall_s': 'wallclock_ts - matches.kickoff_wall_ts (seconds since actual kickoff).',
    't_from_sched_kickoff_s': 'wallclock_ts - matches.kickoff_ts (seconds since scheduled kickoff).',
}
D['odds'] = {
    'league': 'League code.', 'season': 'Season.', 'Div': 'football-data division (E0, SP1, I1, D1, F1).',
    'fd_date_str': 'football-data Date (dd/mm/yyyy).', 'fd_time_uk': 'football-data kickoff time, UK local time (Europe/London).',
    'fd_home': 'football-data HomeTeam.', 'fd_away': 'football-data AwayTeam.',
    'FTHG': 'Full-time home goals.', 'FTAG': 'Full-time away goals.', 'FTR': 'Full-time result H/D/A.',
    'HTHG': 'Half-time home goals.', 'HTAG': 'Half-time away goals.', 'HTR': 'Half-time result.',
    'AHh': 'Asian handicap line (home), opening/pre-closing collection.', 'AHCh': 'Asian handicap line (home), closing.',
    'date': 'Match date (from fd_date_str).', 'fd_kickoff_ts': f'Kickoff from Date + Time interpreted as Europe/London ({TS}).',
    'home_team': 'Canonical home team (ESPN displayName via team_map_fd_espn.csv).', 'away_team': 'Canonical away team.',
    'match_id': 'FK -> matches.match_id of the primary Polymarket event (null for the 3 fixtures Polymarket did not list).',
    'espn_event_id': 'ESPN event id of the same fixture (join key that also reaches the secondary event of a duplicate pair).',
    'fd_vs_espn_swapped': 'True if football-data home/away is reversed relative to ESPN (never observed).',
    'pm_orientation_swapped': "True if the football-data home team is the Polymarket 'away' team (1 match: fl1-psg-ren-2026-08-23); then PSH/AvgH etc. price the Polymarket 'away' market.",
}
ODDS_PAT = [
    (r'^PS(C)?([HDA])$', 'Pinnacle 1X2 decimal odds ({c}; H/D/A). 2025-26 only.'),
    (r'^P(C)?_(over|under)25$', 'Pinnacle over/under 2.5 goals decimal odds ({c}).'),
    (r'^P(C)?AH([HA])$', 'Pinnacle Asian handicap decimal odds ({c}; H = home, A = away) at line AHh/AHCh.'),
    (r'^B365(C)?([HDA])$', 'Bet365 1X2 decimal odds ({c}).'),
    (r'^B365(C)?_(over|under)25$', 'Bet365 over/under 2.5 decimal odds ({c}).'),
    (r'^B365(C)?AH([HA])$', 'Bet365 Asian handicap decimal odds ({c}).'),
    (r'^Max(C)?([HDA])$', 'Market maximum 1X2 decimal odds ({c}).'),
    (r'^Avg(C)?([HDA])$', 'Market average 1X2 decimal odds ({c}).'),
    (r'^Max(C)?_(over|under)25$', 'Market maximum over/under 2.5 odds ({c}).'),
    (r'^Avg(C)?_(over|under)25$', 'Market average over/under 2.5 odds ({c}).'),
    (r'^Max(C)?AH([HA])$', 'Market maximum Asian handicap odds ({c}).'),
    (r'^Avg(C)?AH([HA])$', 'Market average Asian handicap odds ({c}).'),
    (r'^BFE?(C)?([HDA])$', 'Betfair (exchange) 1X2 odds ({c}).'),
    (r'^BFE?(C)?_(over|under)25$', 'Betfair over/under 2.5 odds ({c}).'),
    (r'^BFE?(C)?AH([HA])$', 'Betfair Asian handicap odds ({c}).'),
]


def describe(table, col):
    if col in D[table]:
        return D[table][col]
    if table == 'odds':
        for pat, txt in ODDS_PAT:
            mm = re.match(pat, col)
            if mm:
                return txt.format(c='closing' if mm.group(1) else 'opening / early-week collection')
    raise KeyError(f'{table}.{col} has no description')


def main():
    head = open('polymarket/research/auto-20261104/scripts/soccer_data/DATA_DICTIONARY.head.md').read()
    out = [head]
    for t in ['matches', 'markets', 'price_history', 'trades', 'events', 'odds']:
        f = pq.ParquetFile(f'{DATA}/{t}.parquet')
        out.append(f'\n## `{t}`  ({f.metadata.num_rows:,} rows, `data/soccer_data/{t}.parquet`)\n')
        out.append('| column | type | description |\n|---|---|---|')
        for fld in f.schema_arrow:
            out.append(f'| `{fld.name}` | {fld.type} | {describe(t, fld.name)} |')
    open(f'{RES}/DATA_DICTIONARY.md', 'w').write('\n'.join(out) + '\n')
    print('written', f'{RES}/DATA_DICTIONARY.md')


if __name__ == '__main__':
    main()
