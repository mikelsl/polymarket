# NBA dataset (Polymarket + ESPN) - data dictionary

Workstream `nba_data`. All files under `data/nba_data/` (relative to `polymarket/research/auto-20261104/`).
Built by the scripts in `scripts/nba_data/` (see REPORT.md "How to re-run"). Read-only research data.

## Conventions
- **Time**: every timestamp column is UTC (`timestamp[us, tz=UTC]`). Integer epoch columns (`ts`, `t`) are UTC
  seconds. `*_date_et` / `slug_date` are US Eastern calendar dates (`America/New_York`, DST-aware) as `YYYY-MM-DD`.
- **Prices** are probabilities in [0, 1] (USDC per share). **Sizes / volumes** are shares (1 share pays 1 USDC if it
  wins). Gamma `volume` equals the sum of taker print sizes (within 1e-4 on 99% of markets; for games 2026-03-30..04-28
  prints are ~2.6% lower because BUY sizes are net of the fee), so it is a share count, not
  USDC notional. Notional = `size * price`.
- **Token index**: 0/1 = position in the market's `clobTokenIds` (= position in `outcomes`). Moneyline outcome 0 is the
  away team, outcome 1 the home team (slug order `nba-<away>-<home>-date`) for every game except neutral-site
  All-Star games - always use the outcome names, not the position, when it matters.
- **Season** label `YYYY-YY` from the ET game date: Aug-Dec -> that year's season, Jan-Jul -> previous year's.
  2024-25 includes the July 2025 Summer League events (flagged `game_type='summer_league'`).
- **Scope**: Polymarket game events whose game date (ET, from `gameStartTime`) is 2024-10-22 .. 2026-11-03.
  There were no NBA games on 2026-11-03, so the last games are the 2026-11-02 (ET) slate (UTC start up to 2026-11-03T03:00Z).

## Join keys
```
games.game_id (Gamma event id of the main game event)
  1-n markets.game_id            (markets of the game event and of its sibling events)
  1-n trades.game_id             (moneyline, main spread, main total prints)
  1-n price_history.game_id
games.espn_event_id (ESPN event id; several Polymarket events can map to one ESPN game, see dup_group_size)
  1-n espn_pbp.espn_event_id
  1-n pickcenter.espn_event_id
markets.market_id = trades.market_id = price_history.market_id;  markets.token_0/1 <-> token_index 0/1
```
For "one row per real game" use `games.is_core_game` (regular season, play-in, playoffs, NBA Cup final; primary
Polymarket event only). `game_type` tells what the other rows are.

---------------------------------------------------------------------------------------------------------------------
## games.parquet  (one row per in-scope Polymarket game event)

**Identity / Polymarket event**
| column | meaning |
|---|---|
| game_id | Gamma event id of the main game event (string) |
| slug | event slug `nba-<away>-<home>-YYYY-MM-DD` (one exception: the 2024 NBA Cup final `nba-cup-final-thunder-vs-bucks`) |
| slug_nonstandard | True for that exception (its codes `okc`/`mil` are assigned in `gamma_tables.py`, `slug_date` = ET game date) |
| title | Gamma title, e.g. "Knicks vs. Celtics" |
| away_code, home_code | Polymarket team codes from the slug (2-7 letters; `wsh`/`pho` old codes; All-Star codes `cgs sog kys crs stars stripes world`) |
| slug_date | date in the slug. Usually the ET game date, but 97 events use the UTC date (`slug_date_matches_et`=False) |
| game_start_utc | moneyline `gameStartTime` (fallback event `startTime`, then `endDate`). Wrong on ~45 events, see `pm_start_minus_espn_sched_min` |
| game_date_et | ET date of `game_start_utc` - the scope and season key |
| season | `2024-25`, `2025-26`, `2026-27` |
| in_scope | always True in this file |
| event_start_time, event_end_date, event_created_at, event_closed, event_closed_time | Gamma event fields |
| event_volume | Gamma event volume (shares, all markets of the event; unreliable where market volume is null) |
| pm_score, pm_period | Gamma event `score` ("away-home") and `period`. **Unreliable**: differs from ESPN on 19 played games; use ESPN |
| pm_score_away, pm_score_home | parsed `pm_score` |
| pm_team_away, pm_team_home, pm_team_away_abbr, pm_team_home_abbr | Gamma `teams[]` (missing on many 2024-25 events) |
| series_slug | Gamma series (`nba`, `nba-2026`) |
| n_markets_main | markets in the main event |
| n_markets_family | markets in main + sibling events (`-v2`, `-moneyline`, `-exact-margin`) |
| sibling_roles | which sibling events exist |
| n_spread_markets, n_spread_markets_traded, n_total_markets, n_total_markets_traded | full-game spread / total markets in main + `-v2` events (traded = volume > 0) |

**Moneyline** (`ml_*`): `ml_market_id`, `ml_condition_id`, `ml_outcome_0/1` (team nicknames), `ml_token_0/1`,
`ml_volume` (Gamma, shares; null for 170 moneylines: 162 in 2026-03, 8 in 2024-11), `ml_volume_trades` (sum of fetched prints),
`ml_closed`, `ml_winner_index` (0/1 = winning outcome from `outcomePrices`, -1 = resolved 50-50, null = unresolved),
`ml_kind_note` (set when Gamma labels the moneyline `sportsMarketType='totals'`: 55 events listing the Thunder first, 2024-25),
`ml_created_at`, `ml_closed_time`, `ml_fees_enabled`, `ml_fee_type`.

**Main spread / total**: `main_spread_market_id`, `main_spread_event_role` (`main` or `v2`), `main_spread_team0`
(team of outcome 0), `main_spread_handicap0` (points added to outcome-0 team: outcome 0 wins iff
score(team0) + handicap0 > score(other)), `main_spread_home_handicap` (the same line from the Polymarket home team's
side, e.g. -4.5 = home favoured by 4.5), `main_spread_volume` (shares), `main_spread_volume_source` (`gamma` or
`trades`); `main_total_market_id`, `main_total_event_role`, `main_total_line` (points; Over wins iff total > line),
`main_total_volume`, `main_total_volume_source`.
"Main" = highest final volume among full-game spread (total) markets with volume > 0. **Look-ahead**: final volume is
known only after the game. `markets.rank_created_in_kind` allows a pre-game choice instead.
`book_spread_home_close`, `book_total_close`, `book_provider`: sportsbook closing line from ESPN (summary pickcenter,
else core odds; live rows excluded), home handicap expressed for the Polymarket home team. `main_spread_minus_book`,
`main_total_minus_book`: Polymarket main line minus that close. The main line is within 1 point of the book close on
85% (spreads) / 64% (totals) of core games but 3+ points away on ~120 / ~265 games: the highest-volume line is
sometimes a far or reverse line (e.g. "Spread: Nets (-13.5)" when the book had BOS -13.5).

**ESPN match**: `espn_event_id`, `match_method` (`et_date`, `slug_date`, `et_date+1`/`-1` - +-1 day only with the
same away/home), `n_candidates`, `espn_sched_utc` (ESPN scheduled start), `espn_date_et`, `espn_season_year`
(ESPN season = end year), `espn_season_type` (1 pre, 2 regular, 3 post, 5 play-in), `espn_season_slug`,
`espn_comp_type` (STD, CC = NBA Cup final, ALLSTAR, RD16/QTR/SEMI/FINAL = playoff rounds), `espn_notes`
(e.g. "East 1st Round - Game 6", postponement notes), `espn_neutral_site`, `espn_status_name`,
`espn_home/away_abbr`, `espn_home/away_name`, `espn_home/away_score` (final, incl. OT; summary header, scoreboard
for the 4 games whose summary is missing), `espn_home/away_linescores` (JSON list of points per period),
`espn_max_period` (5+ = overtime), `espn_n_plays`, `espn_n_pickcenter`, `espn_summary_source` (`site_summary`, or
`core plays + scoreboard header` for the 4 playoff games whose site-API summary returns 404), `espn_score_source`.

**Tip-off and play-by-play QA** (from `espn_pbp`):
| column | meaning |
|---|---|
| espn_tipoff_utc | wallclock of the first period-1 play with a wallclock (normally the jump ball) |
| espn_first_play_type, espn_first_play_clock | that play's type and clock (14 games start with a later play, `flag_first_play_not_jumpball`) |
| espn_last_play_utc, espn_last_play_type | last wallclock / last play in game-clock order |
| espn_n_wallclock_missing | plays without wallclock |
| espn_n_wallclock_backsteps, espn_max_wallclock_backstep_s, espn_max_wallclock_gap_s | wallclock going backwards in sequence order / biggest step back / biggest gap (s) |
| espn_n_score_backsteps_seq | score lower than on the previous play, ESPN sequence order (mostly late-inserted plays) |
| espn_n_score_backsteps_clock | same in game-clock order (`elapsed_s`, `seq`): real rewrites / stale scores around inserted baskets |
| espn_n_score_backsteps_wallclock | same in wallclock order (what a wallclock-aligned user sees) |
| espn_n_score_backsteps_scoring_wallclock | same, scoring plays only, wallclock order |
| espn_final_home/away_pbp | score of the last play in game-clock order; `flag_final_score_pbp_mismatch` if != final |
| espn_wallclock_available | ESPN flag |
| tipoff_minus_espn_sched_min, tipoff_minus_pm_start_min | minutes; normal is 7-15 (median 11) |
| pm_start_minus_espn_sched_min | Polymarket gameStartTime minus ESPN schedule (minutes) |
| flag_wallclock_early / flag_wallclock_late | tip-off < sched - 5 min / > sched + 60 min |
| wallclock_shift_suspect_s | offset suggested from the tip-off alone: +86400 (first play 1 day late), +3600 (2026-03-08 DST start), -3600 (2026-11-01 DST end). Use the per-play column in `espn_pbp` instead: several 2025-10-28..31 games are only partly shifted |
| espn_n_plays_wallclock_shift | plays with a suggested wallclock fix (`espn_pbp.wallclock_shift_suspect_s != 0`) |
| espn_n_plays_wallclock_out_of_window | plays whose (fixed) wallclock is outside [sched - 20 min, sched + 5 h] |
| espn_tipoff_utc_fixed, tipoff_fixed_minus_espn_sched_min | tip-off after the suggested fix |
| flag_wallclock_plays_shifted, flag_wallclock_plays_out_of_window | the two counters above > 0 |
| flag_pm_start_differs | abs(pm_start_minus_espn_sched_min) > 5 |

**Classification and resolution checks**
| column | meaning |
|---|---|
| game_type | `regular`, `play_in`, `playoffs`, `cup_final`, `preseason`, `allstar`, `summer_league` (July 2025, not on ESPN NBA), `postponed` (game not played on this date; see below), `not_played` (if-necessary playoff games never played, resolved 50-50) |
| flag_postponed | ESPN status POSTPONED/CANCELED for the matched event |
| makeup_game_id | for postponed games: the later Polymarket game with the same teams that was played |
| dup_group_size, is_primary_event | number of Polymarket events mapped to the same ESPN game; primary = largest moneyline volume. Duplicates: 8 pairs (UTC-date vs ET-date slugs in 2025-10/11, wrong-date events in 2025-12 and 2026-01) |
| is_core_game | game_type in (regular, play_in, playoffs, cup_final) and primary event |
| pm_away_espn, pm_home_espn, home_away_swapped_vs_espn | slug codes mapped to ESPN abbreviations; swapped = Polymarket home != ESPN home (only the 2025 All-Star semi-final and the 2024 Cup final, neutral sites) |
| ml_outcome0_is_espn_away | moneyline outcome 0 name equals ESPN's away nickname (False only for All-Star team names and the "Twolves" spelling) |
| resolution_checked_against | `same_game` or `makeup_game` |
| ml_names_matched, ml_winner_espn, ml_resolution_ok | ESPN winner as outcome index (by team nickname) and agreement with `ml_winner_index` |
| spread_names_matched, spread_winner_espn, spread_winner_pm, spread_resolution_ok | main spread: outcome index that should win from the ESPN final (-1 push) vs Polymarket |
| total_points_espn, total_winner_espn, total_winner_pm, total_resolution_ok | main total |
| flag_pm_score_differs_espn | Gamma `score` string differs from ESPN final |
| flag_print_gap_in_play | the game was in play during a data hole: >= 5 min with no print in any market of the night while >= 2 core games were in play, and either the CLOB minute prices kept moving (prints missing from the Data API) or there is no minute price either (hole in both sources). 4 windows, 15 core games; `qa/trades_print_gaps.csv` |
| book_spread_home_close, book_total_close, book_provider, main_spread_minus_book, main_total_minus_book | see "Main spread / total" above |

**Coverage counters**: `ml_n_prints`, `ml_prints_sum_size`, `ml_n_price_points_0`, `ml_n_price_points_1`, and the same
for the main spread (`spread_n_prints`, `spread_prints_sum_size`, `spread_n_price_points_0`, `spread_n_price_points_1`)
and main total (`total_n_prints`, `total_prints_sum_size`, `total_n_price_points_0`, `total_n_price_points_1`): rows in
`trades` / `price_history` for that market and token.

---------------------------------------------------------------------------------------------------------------------
## markets.parquet  (every market of the in-scope game events and their sibling events; ~107k rows)
| column | meaning |
|---|---|
| game_id | main game event id |
| event_id, event_slug, event_role | event holding the market; role `main`, `v2` (2025-03/04 sibling with spread/total ladders), `moneyline_dup` (3 playoff events 2025-04-26), `exact_margin` (2026-27) |
| market_id, condition_id, market_slug, question | Gamma ids and text |
| sports_market_type | raw Gamma label |
| kind | normalised type: `moneyline` (main market of the event; fixes the 55 mislabelled), `spreads`, `totals`, `first_half_*`, `q1_*`.., player props (`points`, `rebounds`, ...), `exact_margin`, ... |
| kind_note | why `kind` differs from Gamma's label |
| line_gamma | Gamma `line` (sign NOT consistent across seasons - use handicap0 / total_line) |
| outcome_0/1, n_outcomes, token_0/1 | outcomes and CLOB token ids |
| outcome_price_0/1, winner_index | resolution from `outcomePrices` (0/1 winner, -1 = 50-50, null = unresolved / never opened) |
| volume, volume_clob | Gamma volume (shares). Null on most markets of 2026-03-04..03-25 and on untraded lines |
| volume_trades, n_prints | sum of taker print sizes and count, where prints were pulled |
| volume_effective, volume_source | `volume` if present else `volume_trades`; source `gamma` / `trades` |
| closed, active, uma_resolution_status | Gamma state |
| game_start_time, created_at, accepting_orders_at, closed_time, start_date, end_date | UTC timestamps |
| fees_enabled, fee_type, fee_rate, fee_exponent, fee_taker_only, fee_rebate_rate, fee_schedule_json | fee schedule; taker fee = shares x rate x p x (1-p). Regimes on game markets: none before 2026-03-30; `sports_fees_v2` (rate 0.03, rebate 0.25) 2026-03-30..June 2026; `sports_fees_v3` (rate 0.05, rebate 0.15) for the 2026 preseason and from 2026-11-01; `zero_fees` (rate 0) on the regular-season games of 2026-10-20..31 |
| tick_size, min_order_size, neg_risk | order parameters |
| handicap0, team0, team1 | spreads: points added to outcome 0 (team0); parsed from the question (formats `Spread: Knicks (-2.5)`, `Spread: Pacers (-10.5)` with outcomes Favorite/Underdog, `Favorite(Celtics) vs Underdog(Heat) Line: 8.5`, 2024 Cup final `Will the Thunder beat the Bucks by 5 or more points?` = Thunder -4.5) |
| total_line | totals: line parsed from the question (`O/U 230.5`, `Over 217.5`, `Line: 215.5`, `combine for 215 or more points` = 214.5) |
| role | `moneyline`, `main_spread`, `main_total` or null |
| rank_created_in_kind, rank_volume_in_kind | rank among full-game spreads (totals) of the game by creation time / by final volume |

---------------------------------------------------------------------------------------------------------------------
## trades/season=<season>/month=<YYYY-MM>/part-0.parquet  (taker prints; read the directory with pyarrow/pandas)
Source: Data API v2 `/v2/trades?condition=<conditionId>&taker_only=true`, complete history of the market (walked
to the first print). One row per taker fill as reported by the API. `month` = ET month of the game.
| column | meaning |
|---|---|
| game_id, market_id, market_role | keys; role `moneyline` / `main_spread` / `main_total` |
| time_utc, ts | print time (UTC timestamp / epoch seconds; 1-s resolution) |
| token_index | 0/1 = token of `markets.token_0/1` the taker traded |
| side | `BUY` = taker bought `token_index` at `price`; `SELL` = taker sold it |
| price | execution price (probability) |
| size | shares as reported by the API. For games 2026-03-30..2026-04-28 (first weeks of the `sports_fees_v2` schedule) taker BUY sizes are net of the taker fee (never round numbers; the market's print sum is ~2.6% below Gamma volume), i.e. the shares the taker received; `qa/market_coverage.csv` flags these markets (`net_of_fee_sizes`). Before and after that window sizes are gross and match Gamma volume |
| outcome_index_api | the API's own outcome index (always equal to token_index; kept as a check) |
| tx_hash | Polygon transaction hash (one tx can carry several prints) |
| taker_wallet | taker proxy wallet |
Notes: completeness vs Gamma volume (sum of `size` per market): fee-free markets 4,482 of 4,516 within 1e-4 (18 have 0.01-2.3% fewer shares, 16 more: Gamma undercounts, mostly 2026-03-01..03); `sports_fees_v3` 245/255; `zero_fees` 264/264; games 2026-03-30..04-28 median -2.6% (net-of-fee sizes); later v2 games match. Data holes: 4 windows (137 min) where no print exists while games were in play (`games.flag_print_gap_in_play`, `qa/trades_print_gaps.csv`). 677 prints are stamped after Gamma `closedTime` (`qa/trades_after_close.csv`; three 2026-01-28 moneylines traded for hours after close at mid prices). Prints include pre-game, in-game and post-game (after the result, near 0/1) trading. Several Polymarket
events can exist for one real game (see `games.dup_group_size`).

## price_history/season=<season>/part-0.parquet  (CLOB minute history)
Source: CLOB `/prices-history?market=<token>&startTs&endTs&fidelity=1`, both tokens of the moneyline, main spread
and main total. Window [game_start_utc - 6h, game_start_utc + 4h], widened to include ESPN's scheduled time + 4h
where Polymarket's gameStartTime is off. Points start only when the market was trading. The 17 never-played
'if necessary' playoff moneylines have no points (no game in the window).
| column | meaning |
|---|---|
| game_id, market_id, market_role | keys |
| token_index | 0/1 |
| time_utc, t | point time (UTC; ~1-minute grid, seconds not aligned to :00) |
| p | price (probability) as published by CLOB (a mid/last-trade style series, not an executable quote). 238 points of one market (SAC@GSW 2025-01-05 moneyline) are -0.0035 / 1.0035: clip to [0, 1] |
**Caution**: the minute series lags around in-game events; use `trades` for any event-time entry/exit.

---------------------------------------------------------------------------------------------------------------------
## espn_pbp.parquet  (ESPN play-by-play of every matched game, ESPN sequence order)
| column | meaning |
|---|---|
| espn_event_id | join to `games.espn_event_id` |
| seq, play_id, play_index | ESPN sequenceNumber / play id / 0-based position in sequence order |
| wallclock_utc | ESPN wallclock (UTC), as published. Whole-day / whole-hour errors on some dates: see `wallclock_shift_suspect_s` |
| period | 1-4, 5+ = OT |
| clock_s, clock_display | game clock remaining in the period (seconds) and ESPN text |
| elapsed_s | game seconds elapsed since tip (12-min quarters, 5-min OTs); sort by (elapsed_s, seq) for game chronology |
| home_score, away_score | score after the play as ESPN reports it |
| scoring_play, score_value | scoring flag and points |
| play_type_id, play_type | ESPN play type |
| team_side | `home` / `away` team credited with the play (null for neutral plays) |
| score_backstep | score lower than on the previous play in sequence order |
| wallclock_backstep | wallclock earlier than on the previous play in sequence order |
| wallclock_shift_suspect_s | suggested fix in seconds, NOT applied: +86400 when the wallclock is > 12 h after ESPN's scheduled start (2025-10-28..31, 19 games, whole or partial), +-3600 for every play of a game whose (day-fixed) tip-off is 55-85 min late / 40-75 min early (2026-03-08, 2026-11-01). Fixed time = `wallclock_utc - wallclock_shift_suspect_s` |
| wallclock_out_of_window | fixed wallclock outside [sched - 20 min, sched + 5 h] (a few plays in 4 games) |
Known issues: ESPN inserts late-entered plays out of sequence (their score reflects their own time), so scores
"step back" in sequence order in most games; in game-clock order only real rewrites remain (`work/espn_score_reversals.parquet`, `qa/score_backsteps_clock_order.csv`).

## pickcenter.parquet  (ESPN betting lines, one row per event x provider x source)
| column | meaning |
|---|---|
| espn_event_id | join key |
| source | `summary_pickcenter` (site API `summary.pickcenter`; DraftKings, 2025-26 onward) or `core_odds` (core API `.../competitions/<id>/odds`; ESPN BET for 2024-25, DraftKings later) |
| provider_id, provider_name | sportsbook. Names ending in "- Live Odds" are ESPN's live (in-game) line snapshot, not a pre-game line; use the plain provider rows for open/close |
| details | e.g. "NY -4.5" (favourite and line) |
| spread | HOME team handicap at close/current (negative = home favoured) |
| over_under, over_odds, under_odds | total line and American odds |
| home_ml, away_ml, home_spread_odds, away_spread_odds | American odds |
| home/away_favorite, home/away_favorite_at_open | flags |
| ml_home_open, ml_home_close, ml_away_open, ml_away_close | moneyline American odds at open/close |
| spread_home_open, spread_home_close | home handicap at open/close |
| spread_home_open_odds, spread_home_close_odds, spread_away_open_odds, spread_away_close_odds | American odds of the spread at open/close |
| total_open, total_close | total line at open/close |
| over_open_odds, over_close_odds, under_open_odds, under_close_odds | American odds of over/under at open/close |
| moneyline_winner, spread_winner | core API result flags (when present) |
| raw_json | the provider entry as returned (links/logos removed) - any other open/close/current fields |

## Raw and work files (for rebuilds, not for analysis)
- `raw/gamma_windows/*.json.gz` every Gamma NBA event (all fields) per 3-day endDate window.
- `raw/espn_scoreboard/YYYYMMDD.json.gz`, `raw/espn_summary/<id>.json.gz`, `raw/espn_odds/<id>.json.gz`.
- `work/games_gamma.parquet`, `work/markets_all.parquet` (all discovered game events incl. out-of-scope dates),
  `work/espn_games.parquet` (every ESPN event 2024-10-01..2026-11-04), `work/espn_pbp.parquet`,
  `work/espn_odds.parquet`, `work/espn_score_reversals.parquet`, `work/trades_parts/<market>.parquet` +
  `work/trades_manifest/<market>.json` (incl. the probe of null-volume spread/total markets),
  `work/price_parts/`, `work/price_manifest/`.
- QA lists: `results/nba_data/qa/*.csv`, summary `results/nba_data/qa_summary.json`.
