# NBA data (Polymarket + ESPN): build and QA report

Workstream `nba_data`. Generated 2026-11-04 08:21 UTC by `scripts/nba_data/make_report.py` from the final tables.
Data only, no strategy analysis. Column definitions: `DATA_DICTIONARY.md`. QA lists: `results/nba_data/qa/*.csv`.

## 1. What was built
Every Polymarket NBA game event (Gamma `tag_id=745`, slug `nba-<away>-<home>-YYYY-MM-DD`, plus the 2024 NBA Cup final
listed as `nba-cup-final-thunder-vs-bucks`) whose game date (ET, from
`gameStartTime`) is 2024-10-22 .. 2026-11-03: **2,994 events**, of which **2,750 core games**
(`is_core_game`: regular season, play-in, playoffs, NBA Cup final; one primary event per real game). There were no NBA
games on 2026-11-03, so the last slate is 2026-11-02 (ET). The other events are kept and labelled in `game_type`:
preseason, All-Star, NBA Summer League (July 2025), postponed games, never-played "if necessary" playoff games;
duplicate Polymarket events for one game have `is_primary_event=False`.

## 2. Files
| path | rows | size |
|---|---|---|
| data/nba_data/games.parquet | 2,994 | 1.1 MB |
| data/nba_data/markets.parquet | 107,263 | 17.3 MB |
| data/nba_data/trades/ | 12,601,950 | 647.4 MB |
| data/nba_data/price_history/ | 7,190,986 | 26.9 MB |
| data/nba_data/espn_pbp.parquet | 1,387,738 | 19.7 MB |
| data/nba_data/pickcenter.parquet | 5,858 | 1.0 MB |

Raw pulls (gzipped JSON, rebuilds need no network): 167.5 MB; per-market parts and manifests: 851.4 MB;
whole `data/nba_data/`: 1,732.2 MB.

## 3. Coverage
`games` events; `espn_matched` matched to an ESPN event; `ml_resolved` moneyline resolved to a team; `ml_prices`
minute history for both moneyline tokens; `ml_trades` >= 1 moneyline print; `main_spread` / `main_total` a traded
full-game spread / total exists; `pbp` ESPN play-by-play; `pickcenter` ESPN summary pickcenter; `any_odds` pickcenter
or ESPN core-API odds.

### 3.1 Core games by season
| season | games | espn_matched | ml_resolved | ml_prices | ml_trades | main_spread | main_total | pbp | pickcenter | any_odds |
|---|---|---|---|---|---|---|---|---|---|---|
| 2024-25 | 1,321 | 1,321 | 1,321 | 1,321 | 1,321 | 46 | 43 | 1,321 | 0 | 1,321 |
| 2025-26 | 1,322 | 1,322 | 1,322 | 1,322 | 1,322 | 1,314 | 1,314 | 1,322 | 1,067 | 1,322 |
| 2026-27 | 107 | 107 | 107 | 107 | 107 | 107 | 107 | 107 | 107 | 107 |
| total | 2,750 | 2,750 | 2,750 | 2,750 | 2,750 | 1,467 | 1,464 | 2,750 | 1,174 | 2,750 |

### 3.2 All in-scope events by season and type
| season | game_type | games | espn_matched | ml_resolved | ml_prices | ml_trades | main_spread | main_total | pbp | pickcenter | any_odds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2024-25 | allstar | 2 | 2 | 2 | 2 | 2 | 0 | 0 | 2 | 0 | 2 |
| 2024-25 | cup_final | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| 2024-25 | play_in | 6 | 6 | 6 | 6 | 6 | 2 | 1 | 6 | 0 | 6 |
| 2024-25 | playoffs | 84 | 84 | 84 | 84 | 84 | 36 | 36 | 84 | 0 | 84 |
| 2024-25 | postponed | 4 | 4 | 0 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| 2024-25 | regular | 1,230 | 1,230 | 1,230 | 1,230 | 1,230 | 7 | 5 | 1,230 | 0 | 1,230 |
| 2024-25 | summer_league | 77 | 0 | 77 | 77 | 77 | 0 | 0 | 0 | 0 | 0 |
| 2025-26 | allstar | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 2 | 0 | 0 |
| 2025-26 | cup_final | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2025-26 | not_played | 19 | 0 | 0 | 2 | 19 | 0 | 0 | 0 | 0 | 0 |
| 2025-26 | play_in | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| 2025-26 | playoffs | 85 | 85 | 85 | 85 | 85 | 85 | 85 | 85 | 81 | 85 |
| 2025-26 | postponed | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 0 | 0 | 0 |
| 2025-26 | preseason | 63 | 63 | 63 | 63 | 63 | 40 | 42 | 63 | 0 | 63 |
| 2025-26 | regular | 1,238 | 1,238 | 1,238 | 1,238 | 1,238 | 1,225 | 1,225 | 1,238 | 982 | 1,238 |
| 2026-27 | preseason | 66 | 66 | 66 | 66 | 66 | 66 | 66 | 66 | 63 | 63 |
| 2026-27 | regular | 107 | 107 | 107 | 107 | 107 | 107 | 107 | 107 | 107 | 107 |

### 3.3 Core games by month (ET)
| month | games | espn_matched | ml_resolved | ml_prices | ml_trades | main_spread | main_total | pbp | pickcenter | any_odds |
|---|---|---|---|---|---|---|---|---|---|---|
| 2024-10 | 71 | 71 | 71 | 71 | 71 | 0 | 0 | 71 | 0 | 71 |
| 2024-11 | 222 | 222 | 222 | 222 | 222 | 0 | 0 | 222 | 0 | 222 |
| 2024-12 | 192 | 192 | 192 | 192 | 192 | 1 | 1 | 192 | 0 | 192 |
| 2025-01 | 227 | 227 | 227 | 227 | 227 | 0 | 0 | 227 | 0 | 227 |
| 2025-02 | 176 | 176 | 176 | 176 | 176 | 0 | 0 | 176 | 0 | 176 |
| 2025-03 | 238 | 238 | 238 | 238 | 238 | 7 | 5 | 238 | 0 | 238 |
| 2025-04 | 149 | 149 | 149 | 149 | 149 | 3 | 2 | 149 | 0 | 149 |
| 2025-05 | 39 | 39 | 39 | 39 | 39 | 28 | 28 | 39 | 0 | 39 |
| 2025-06 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 0 | 7 |
| 2025-10 | 80 | 80 | 80 | 80 | 80 | 77 | 77 | 80 | 0 | 80 |
| 2025-11 | 219 | 219 | 219 | 219 | 219 | 215 | 215 | 219 | 48 | 219 |
| 2025-12 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 | 198 |
| 2026-01 | 233 | 233 | 233 | 233 | 233 | 233 | 233 | 233 | 233 | 233 |
| 2026-02 | 166 | 166 | 166 | 166 | 166 | 166 | 166 | 166 | 166 | 166 |
| 2026-03 | 239 | 239 | 239 | 239 | 239 | 238 | 238 | 239 | 239 | 239 |
| 2026-04 | 144 | 144 | 144 | 144 | 144 | 144 | 144 | 144 | 143 | 144 |
| 2026-05 | 38 | 38 | 38 | 38 | 38 | 38 | 38 | 38 | 35 | 38 |
| 2026-06 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| 2026-10 | 88 | 88 | 88 | 88 | 88 | 88 | 88 | 88 | 88 | 88 |
| 2026-11 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 |

## 4. Trades (taker prints)
**12,601,950 rows, 647.4 MB on disk** (zstd parquet, `data/nba_data/trades/season=/month=`).

| season | market_role | prints | shares |
|---|---|---|---|
| 2024-25 | main_spread | 3,359 | 2,255,439 |
| 2024-25 | main_total | 2,519 | 981,014 |
| 2024-25 | moneyline | 1,936,594 | 617,592,424 |
| 2025-26 | main_spread | 930,390 | 623,503,376 |
| 2025-26 | main_total | 616,896 | 306,825,618 |
| 2025-26 | moneyline | 8,484,226 | 3,566,317,315 |
| 2026-27 | main_spread | 54,376 | 58,080,686 |
| 2026-27 | main_total | 45,400 | 23,134,633 |
| 2026-27 | moneyline | 528,190 | 265,276,361 |
| all | all | 12,601,950 | 5,463,966,866 |

Completeness: every market's feed was walked to its first print (no truncation). For 5,137 of 5,201
fetched markets that carry a Gamma volume, the sum of print sizes equals Gamma `volume` (relative gap <= 1e-4). Left out:
the markets of games 2026-03-30..04-28, whose taker BUY sizes are net of the fee (section 7). Data API holes are not
visible in this check (section 7).

## 5. Minute price history
**7,190,986 rows, 26.9 MB on disk** (`data/nba_data/price_history/season=`).

## 6. ESPN
- Matched 2,897 of 2,994 events to an ESPN event ({'et_date': 2894, 'et_date+1': 3, 'slug_date': 1}); unmatched = Summer League, All-Star specials and never-played playoff games (`qa/unmatched_espn.csv`). ESPN games in range with no Polymarket event: 15 (`qa/espn_games_without_pm_event.csv`: preseason x9, postponed shell x3, All-Star x3).
- Play-by-play: 1,387,738 plays for 2,883 games; tip-off = first period-1 wallclock (median 11.1 min after the scheduled time, p5 7.0, p95 14.4).
- Final scores: 2,891 events have an ESPN final; 139 went to overtime.
- Odds rows by source/provider (games): core_odds/DraftKings: 1,241, core_odds/DraftKings - Live Odds: 12, core_odds/ESPN BET: 1,685, core_odds/ESPN Bet - Live Odds: 1,683, summary_pickcenter/DraftKings: 1,237.
- Core games with a summary pickcenter / any odds, by season: 2024-25 0 / 1,321; 2025-26 1,067 / 1,322; 2026-27 107 / 107. ESPN has no summary pickcenter for 2024-25; the core-API odds (ESPN BET, with open/close) fill that season.
- Score step-backs in game-clock order: 609 plays in 379 games; among scoring plays in wallclock order: 354 plays in 253 games (`work/espn_score_reversals.parquet`). In ESPN sequence order 2,616 games show a step-back, almost all from late-inserted plays (not score changes).

## 7. Data-quality problems found
1. Gamma `volume` is null for most markets of 2026-03-04..03-25 (170 moneylines in scope, all spread/total ladders of those games). Those markets did trade: 2,318 of 2,348 null-volume spread/total lines of 2026-03 have prints. Main spread/total there were chosen by the sum of print sizes (`volume_source='trades'`); `markets.volume_effective` carries the filled value.
2. Moneyline mislabelled `sportsMarketType='totals'` on 55 events (every one an event listing the Thunder first, 2024-25); identified by structure (market slug = event slug).
3. The 2024 NBA Cup final (OKC vs MIL, 2024-12-17) is listed under the non-standard slug `nba-cup-final-thunder-vs-bucks` with question-style spread ("Will the Thunder beat the Bucks by 5 or more points?") and total markets; it is included explicitly (`games.slug_nonstandard`). A scan of all in-scope events whose markets have two NBA nicknames as outcomes found no other single-game event outside the slug pattern (the rest are series-winner markets).
4. Slug date is the UTC date instead of the ET date on 97 events (mostly 2025-10/11 evening games); the ET date from gameStartTime is used for scope/season.
5. Duplicate Polymarket events for one real game: 16 events in 8 groups (both resolved; secondary flagged `is_primary_event=False`, excluded from `is_core_game`). Pairs: UTC-date and ET-date slugs for the same game (2025-10-25/26, 11-07/08, 11-08/09), and wrong-date events (LAC@OKC and WAS@SAS 2025-12-17 for 12-18 games, CHI@MIA 2026-01-30 for the 01-31 game).
6. Polymarket gameStartTime differs from ESPN's schedule by >5 min on 44 events (+-60 min DST/timezone slips around 2025-11-01, +-17..25 min TV slots, a few hours to one day off on make-up/duplicate events). Price windows were widened to cover ESPN's scheduled time + 4 h.
7. Polymarket event `score` field disagrees with ESPN on 19 played games (2024-25 mostly); never use it - the moneyline resolution itself agrees with ESPN everywhere.
8. Spread/total ladder of HOU@GSW 2025-05-02 (playoffs G6, final 115-107) resolved 50-50 by Polymarket (and two lines closed at non-binary prices); listed as the 1 spread and 1 total resolution mismatch. Volume there was tiny (5-20 shares).
9. Postponed games: 6 events. The four 2024-25 ones resolved 50-50 by rule; 2025-26 ones stayed open and resolved with the make-up game (MIA@CHI 01-08 -> 01-29, GSW@MIN 01-24 -> 01-25); two others moved `gameStartTime` to the make-up date (DEN@MEM -> 03-18, DAL@MIL -> 03-31).
10. 19 never-played 'if necessary' playoff games (2026-04..06) resolved 50-50 after real trading (up to ~200k shares). Placeholder start time 04:00Z.
11. ESPN wallclock anomalies (45 games in `qa/wallclock_anomalies.csv`; rule: first play < sched-5 min or > sched+60 min, or any play needing a fix / outside the game window): 19 games of 2025-10-28..31 carry wallclocks exactly 1 day late (9 entirely, 10 only after some point in the game, so a whole-game shift is wrong); 5 games on 2026-03-08 are +1 h (DST start); 2 games on 2026-11-01 are -1 h (DST end; the later 11-01 games are fine); 15 preseason games of 2026-10 tip 5-14 min before the scheduled time (offset unknown, not fixed); 4 games have a few plays 5-7 h after the scheduled start. The per-play suggested fix is `espn_pbp.wallclock_shift_suspect_s` (not applied to `wallclock_utc`). Market close times confirm the games were played on the scheduled day.
12. ESPN site-API `/summary` returns 404 for 4 completed 2026 playoff games (401869402, 401871334, 401871337, 401873343); their play-by-play was rebuilt from the ESPN core-API play list (same fields incl. wallclock) and the final score from the scoreboard; no summary pickcenter for them (core-API odds exist).
13. ESPN pbp final score != header final on 27 games (missing late scoring plays); first period-1 play is not the jump ball on 14 games (tip-off then 20-80 s late).
14. Prints vs Gamma volume (markets without net-of-fee sizes): 27 markets have FEWER print shares than Gamma volume (-41.72%..-0.01%; games 2026-01-22, 2026-01-26, 2026-02-19, 2026-02-20, 2026-04-29, 2026-04-30, 2026-05-01, 2026-05-02, 2026-05-03): on 2026-01-22..02-20 a few prints are missing from the Data API (no night-wide gap; cause unknown); the 2026-04-29..05-03 markets straddle the end of the net-of-fee period (gaps <= 0.1% on core games, larger only on never-played games). 37 markets have MORE print shares than Gamma (0.01%..58.28%; games 2024-11-17, 2024-11-27, 2024-12-04, 2024-12-05, 2025-12-16, 2026-01-03, 2026-02-26, 2026-03-01, 2026-03-02, 2026-03-03, 2026-04-30, 2026-05-01, 2026-05-03, 2026-10-03, 2026-10-08, 2026-10-10, 2026-10-11, 2026-10-12, 2026-10-13, 2026-10-16): Gamma volume undercounts, by a single print in 2024-11/12 and heavily on 2026-03-01..03 when Gamma volume went stale just before turning null, and by up to 6.9% on thinly traded 2026-10 preseason lines. No exact duplicate prints, no token mismatches, the API outcome_index always equals the token position.
15. For games 2026-03-30..2026-04-28 (the first weeks of the `sports_fees_v2` fee schedule) taker BUY print sizes are net of the taker fee: never round numbers (0% integer sizes vs ~24-53% otherwise) and the print sum is ~2.6% (median) below Gamma volume. From 2026-04-29 (still v2), and under `sports_fees_v3` and `zero_fees`, prints match Gamma volume again. By group: none: 4516 markets, median gap 0.00% (-2.29%..58.28%); sports_fees_v2: 166 markets, median gap 0.00% (-41.72%..1.09%); sports_fees_v2 (games to 2026-04-28): 456 markets, median gap -2.56% (-3.86%..0.00%); sports_fees_v3: 255 markets, median gap 0.00% (-0.00%..6.89%); zero_fees: 264 markets, median gap 0.00% (-0.00%..0.00%). For the affected games treat `size` as the shares the taker received (`qa/market_coverage.csv` column `net_of_fee_sizes`).
16. Data holes during live games: 4 windows (total 137 min) with no print in any market of the night while >= 2 core games were in play: 2026-03-20 01:21..02:26Z (65 min, 5 games in play, prices kept moving); 2025-12-13 01:26..02:10Z (44 min, 5 games in play, prices kept moving); 2026-02-23 01:00..01:22Z (22 min, 3 games in play, no minute prices either); 2024-11-04 02:09..02:15Z (6 min, 2 games in play, prices kept moving). 15 core games are affected (`games.flag_print_gap_in_play`, `qa/trades_print_gaps.csv`). Where minute prices kept moving, trades happened but the Data API does not return them (Gamma volume misses them too, so volume checks cannot reveal it). Do not read these windows as "no trading".
17. 677 prints on 243 markets are stamped more than 60 s after Gamma `closedTime`. On 220 markets these are settlement prints at 0.001/0.999 within 5 minutes; 5 markets kept trading for 1.5-3 h after `closedTime` at non-settlement prices: nba-sas-hou-2026-01-28 (132 prints, up to 109 min after, 0.180-0.840), nba-gsw-uta-2026-01-28 (107 prints, up to 131 min after, 0.003-0.998), nba-min-dal-2026-01-28 (68 prints, up to 182 min after, 0.002-0.999), nba-nyk-lal-2025-03-06 (21 prints, up to 86 min after, 0.170-0.790), nba-gsw-uta-2026-01-28-spread-away-9pt5 (15 prints, up to 130 min after, 0.110-0.880). For the 2026-01-28 SAS@HOU moneyline the winning side was bought at 0.18-0.40 after the result, consistent with a reopened or disputed resolution; final `outcomePrices` match ESPN. List: `qa/trades_after_close.csv`.
18. CLOB minute history has 238 points outside [0, 1] (-0.0035 / 1.0035), all in market(s) 517327 (SAC@GSW 2025-01-05 moneyline, after the result was clear); clip to [0, 1].
19. 'Main' (highest final volume) is not always the market line: vs the sportsbook close, the main spread is within 1 point on 85% of core games but 3+ points away on 121 (6+ on 19); the main total within 1 point on 64%, 3+ away on 265. Some are reverse lines (e.g. 'Spread: Nets (-13.5)' traded 770k shares while the book had BOS -13.5). Columns `main_spread_minus_book`, `main_total_minus_book`; list in `qa/main_line_far_from_book.csv`. Selecting by final volume is also look-ahead.
20. 2024-25 games have only a moneyline in the main event. Spread/total ladders exist for 2025-03/04 games in separate `-v2` events (11 lines each, almost no volume) and from 2025-05 in the main event; `main_spread_event_role` tells which.
21. Gamma `line` sign is inconsistent (e.g. +16.5 for "Spread: Thunder (-16.5)" in 2025-05); handicaps are parsed from the question text instead (`markets.handicap0`, `total_line`), 0 unparsed.

## 8. Method notes
- Fee regimes on the moneyline / main lines (from `feeType` / `feeSchedule`, fee = shares x rate x p x (1-p), taker only): `none` rate -: games 2024-10-22..2026-03-31 (5014 markets); `sports_fees_v2` rate 0.03: games 2026-03-30..2026-06-19 (622 markets); `sports_fees_v3` rate 0.05: games 2026-10-03..2026-11-02 (255 markets); `zero_fees` rate 0.0: games 2026-10-20..2026-10-31 (264 markets). So: no fees before 2026-03-30, 3% schedule (v2) through June 2026, v3 (5%) for the 2026 preseason and from 2026-11-01, and zero fees for the regular-season games of 2026-10-20..31.
- Discovery: Gamma `/events?tag_id=745` in 3-day `end_date` windows (game events have endDate = start), 2024-10-01 to 2026-11-10, small pages (limit 20, 4 for 2026-27 events with ~500 markets). Every event saved raw. Game events match `^nba-<code>-<code>-YYYY-MM-DD$`; siblings `-v2`, `-moneyline`, `-exact-margin` are attached to the game.
- Moneyline = the main event market whose slug equals the event slug. Main spread / total = highest final volume (> 0) among full-game spreads / totals of the main and `-v2` events. **This ranking uses final volume, i.e. look-ahead**; use `markets.rank_created_in_kind` or a pre-game rule for strategy work.
- Trades: `/v2/trades?condition=..&taker_only=true&limit=1000`, cursor paging to the end of the feed for each market, stored per market then partitioned by season/month. Token index matched against `clobTokenIds`.
- Prices: `/prices-history?market=<token>&startTs&endTs&fidelity=1` for both tokens; window gameStart-6h .. +4h (widened for wrong start times). Median points per token: 600 (min 0, max 2040).
- ESPN: scoreboard per ET date (2024-10-01..2026-11-04), summary + core odds per completed game. Matching by unordered team pair (slug codes mapped: nyk->NY, gsw->GS, sas->SA, nop->NO, uta->UTAH, was/wsh->WSH, pho->PHX, All-Star codes) on the ET date, then slug date, then +-1 day with the same orientation.
- Resolution checks: moneyline winner from `outcomePrices` vs ESPN final by team nickname; spread outcome 0 wins iff score(team0)+handicap0 > score(team1); totals Over iff home+away > line (OT included).

## 9. Runtime and API usage
| job log | start | end | duration | last http stats |
|---|---|---|---|---|
| discover_gamma.log | 2026-11-04 05:32Z | 05:33Z | 1 min | {'requests': 466, 'retries': 0, 'http429': 0, 'errors': 0} |
| espn_fetch.log | 2026-11-04 05:33Z | 06:06Z | 33 min | {'requests': 6729, 'retries': 32, 'http429': 0, 'errors': 4} |
| espn_fallback.log | 2026-11-04 06:06Z | 06:07Z | 0 min |  |
| trades_probe.log | 2026-11-04 05:40Z | 05:56Z | 16 min | {'requests': 8231, 'retries': 0, 'http429': 0, 'errors': 0} |
| trades_main.log | 2026-11-04 05:56Z | 06:44Z | 48 min | {'requests': 15806, 'retries': 0, 'http429': 0, 'errors': 0} |
| trades_main.log | 2026-11-04 08:12Z | 08:12Z | 0 min | {'requests': 3, 'retries': 0, 'http429': 0, 'errors': 0} |
| prices.log | 2026-11-04 06:08Z | 06:49Z | 41 min | {'requests': 12308, 'retries': 4, 'http429': 0, 'errors': 0} |
| prices.log | 2026-11-04 08:12Z | 08:12Z | 0 min | {'requests': 6, 'retries': 0, 'http429': 0, 'errors': 0} |

No HTTP 429 from any API at the chosen rates (Gamma 8/s, Data API 12/s, CLOB 15/s, ESPN 6/s per host, <= 6 concurrent requests in total). Retries were network/5xx only, except the four ESPN summary 404s.

## 10. How to re-run
All steps are resumable (finished items are skipped) and read-only. From `polymarket/research/auto-20261104/`:
```
.venv/bin/python scripts/nba_data/discover_gamma.py --threads 3              # Gamma windows -> raw/gamma_windows
.venv/bin/python scripts/nba_data/espn_fetch.py --threads 2                  # ESPN scoreboards, summaries, core odds
.venv/bin/python scripts/nba_data/espn_fallback.py                           # summaries that 404 on the site API
.venv/bin/python scripts/nba_data/fetch_trades.py --mode probe --threads 4   # prints of null-volume spread/total lines
.venv/bin/python scripts/nba_data/gamma_tables.py                            # games_gamma + markets (probe volumes)
.venv/bin/python scripts/nba_data/fetch_trades.py --mode main --threads 4    # moneyline / main spread / main total
.venv/bin/python scripts/nba_data/espn_tables.py                             # ESPN games, pbp, odds (no network)
.venv/bin/python scripts/nba_data/build_tables.py                            # first pass: ESPN times for price windows
.venv/bin/python scripts/nba_data/fetch_prices.py --threads 2                # CLOB minute history
.venv/bin/python scripts/nba_data/gamma_tables.py && .venv/bin/python scripts/nba_data/build_tables.py
.venv/bin/python scripts/nba_data/make_report.py                             # this report
```
Run long steps in the background: `nohup caffeinate -is .venv/bin/python scripts/nba_data/<step>.py ... >
results/nba_data/<step>.log 2>&1 &`; thread counts of jobs running at the same time must add up to <= 6.
To extend the range: change `SCOPE_LAST_DATE` in `common.py` and `LAST` in `discover_gamma.py` / `espn_fetch.py`, then
re-run the fetchers with `--refresh-since <date>`.
