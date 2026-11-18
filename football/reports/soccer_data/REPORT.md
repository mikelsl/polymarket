# soccer_data: big-5 football dataset (QA report)

Built 2026-11-04 13:32-14:20 CST by the `soccer_data` worker; saved by the main agent (the harness does not let
workers write report files). Read-only: no orders, no wallet, no VPS. Paths are relative to
`polymarket/research/auto-20261104/`. Main-agent spot check 14:30: 6 random 1X2 markets re-read from Gamma, taker
size sums equal Gamma volume (one April 2026 market at 0.97, see 7.10), resolutions agree with scores.

## 1. Scope and result
EPL (`epl`), LaLiga (`lal`), Serie A (`sea`), Bundesliga (`bun`), Ligue 1 (`fl1`) matches on Polymarket:
2025-26 (kickoffs 2025-08-15 to 2026-05-24) and 2026-27 to kickoff 2026-11-03 UTC.

| table (`data/soccer_data/`) | rows | size |
|---|---:|---:|
| `matches.parquet` | 2,204 (2,200 `analysis_ready`) | 0.6 MB |
| `markets.parquet` | 10,171 (6,612 1X2, 1,768 O/U 2.5, 1,791 BTTS) | 4.8 MB |
| `price_history.parquet` | 5,127,223 (10,168 tokens) | 62 MB |
| `trades.parquet` | 6,756,284 (2025-26: 5,511,155; 2026-27: 1,245,129) | 401 MB |
| `events.parquet` | 49,326 (48,071 ESPN key events, 1,255 VAR decisions) | 2.5 MB |
| `odds.parquet` | 2,204 (1,752 + 452) | 0.3 MB |

- Disk 1.3 GB in total, including `trades_parts/` (677 MB, resume only) and `raw/` (183 MB).
- Trades: all 6,612 1X2 markets of both seasons, none empty; 2.62 bn shares, 1.42 bn USDC notional.
- Analysis-ready under the README split: 2025-26 train 1,022, test 726 (from 2026-02-01); 2026-27: 452.

## 2. What Polymarket lists
- Gamma `/events?series_id=` (epl 10188, lal 10193, sea 10203, bun 10194, fl1 10195), 7-day windows: 9,695 events
  (2,231 match events, 1,836 `-more-markets`, 5,628 other siblings).
- Slug `<lg>-<home>-<away>-<YYYY-MM-DD>` in both seasons; three binary 1X2 markets per match with suffixes
  `<homecode>`, `draw`, `<awaycode>`. Slug date is the listing date, not always the played date.
- `sportsMarketType='moneyline'` only from Dec 2025 (earlier null, identified by suffix).
- O/U 2.5 and BTTS sit in the `-more-markets` sibling event from Oct 2025. Exact score, corners, props from Feb 2026,
  not collected.
- 1X2 taker fee rate (`feeSchedule.rate`, fee = shares x rate x p x (1-p)): Aug 2025-Feb 2026 none; Mar 2026 0.0175 on
  33 markets; Apr 2026 0.03 on 489 (none on 54); May 2026 0.03; from Aug 2026 0.05 with `rebateRate` 0.15.
  Detail `results/soccer_data/pm_listing_by_month.csv`.

## 3. Coverage
| season | league | ESPN | PM | O/U | BTTS | prices | trades | odds | agrees |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-26 | bun | 306 | 306 | 223 | 239 | 306 | 306 | 306 | 306 |
| 2025-26 | epl | 380 | 380 | 290 | 297 | 380 | 380 | 380 | 380 |
| 2025-26 | fl1 | 306 | 304 | 216 | 207 | 304 | 304 | 304 | 303 |
| 2025-26 | lal | 380 | 380 | 283 | 291 | 380 | 380 | 380 | 380 |
| 2025-26 | sea | 380 | 379 | 303 | 304 | 379 | 379 | 379 | 379 |
| 2026-27 | bun | 72 | 72 | 72 | 72 | 72 | 72 | 72 | 72 |
| 2026-27 | epl | 90 | 90 | 90 | 90 | 90 | 90 | 90 | 90 |
| 2026-27 | fl1 | 81 | 81 | 81 | 81 | 81 | 81 | 81 | 81 |
| 2026-27 | lal | 109 | 110 | 110 | 110 | 110 | 110 | 109 | 109 |
| 2026-27 | sea | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

2025-26: 1,749 of 1,752 fixtures listed. 2026-27: all 452 plus the postponed `lal-ray-bil-2026-10-10`. 2 secondary
duplicate events kept in `matches`. Monthly detail `qa_coverage_by_month.csv` (O/U and BTTS absent before Oct 2025).

## 4. Sources and method
1. Gamma events -> `pm_matches`, `markets`; 1X2 role from slug suffix; O/U 2.5 = `totals` with line 2.5 in
   `-more-markets`; BTTS = `both_teams_to_score`.
2. CLOB `prices-history` fidelity=1, window [kickoff - 6 h, kickoff + 2.5 h], first outcome token of each market
   (YES for 1X2/BTTS, Over for O/U); median 510 points per token.
3. Data API v2 `/v2/trades`, `taker_only=true`, full history, re-sorted chronologically. Taker size sum equals Gamma
   volume (median ratio 1.0000) except 7.9 and 7.10.
4. ESPN monthly scoreboards (`dates=YYYYMM&limit=1000`) linked on (league, season, home, away): 2,203 of 2,204 events
   link; key events from `summary?event=`; VAR decisions parsed from commentary.
5. football-data.co.uk `2526`/`2627` CSVs (raw in `raw/football_data/`), names via `team_map_fd_espn.csv`; every played
   Polymarket league match has odds; `Time` is UK local (2,197 of 2,204 equal ESPN kickoff); scores equal ESPN.

## 5. Resolution check (`resolution_mismatches.csv`)
2,202 of 2,204 agree, 0 mismatches. Exceptions, both excluded from `analysis_ready`: `lal-ray-bil-2026-10-10`
postponed, unresolved (~1.03M volume); `fl1-nan-tou-2026-05-17` abandoned at 22' at 0-0, Polymarket resolved Draw=Yes
under its cancellation rule (`resolution_check = espn_abandoned`). Cross-check with the existing 404 files: ESPN ids,
scores, kickoffs 404/404; 1X2 resolutions 1,212/1,212; goal wallclocks 403/404 (one moved by 1 s).

## 6. Event timestamp precision
- ESPN wallclocks are second-level UTC for every goal (6,222) and red card (427), not minute-rounded.
- Goal wallclock vs the largest local step in the scoring team's prints (`qa_goal_lag.csv`, 4,584 goals): median +1 s,
  IQR -3..+5 s, 81% within 30 s, 87% within 60 s, p5/p95 -119/+60 s. No hour shift. Actual vs scheduled kickoff
  median +40 s, p95 +201 s.
- Rule: ESPN goal times are good to about 10 s for 75% of goals with a tail of minutes; locate the jump in the prints
  for event-time entries; never use minute history.

## 7. Data-quality problems
1. Stale Polymarket `gameStartTime`: 23 matches differ from ESPN (16 by > 30 min, up to 66 days); `kickoff_ts` uses
   ESPN and 16 matches' price windows were re-fetched.
2. Duplicate events `fl1-olm-psg-2025-09-21/-22`, `lal-val-ovi-2025-09-29/-30`; `is_primary_event` marks ESPN's date.
3. Not listed: Atalanta-Pisa 2025-08-24, PSG-Strasbourg 2025-10-17, Nice-Lyon 2025-10-18.
4. Play-off events excluded (`excluded_playoffs.csv`).
5. Postponed `lal-ray-bil-2026-10-10`. 6. Abandoned `fl1-nan-tou-2026-05-17`.
7. Home/away reversed on Polymarket for `fl1-psg-ren-2026-08-23` (`home_away_swapped`, `odds.pm_orientation_swapped`).
8. Team names change between seasons: 172 names in `team_map_pm_espn.csv`.
9. Gamma `volume` null for 395 1X2 markets (kickoffs 2026-03-04..03-22): use `markets.taker_size_sum`.
10. April 2026 taker BUY sizes are net of a fee taken in shares: for 434 of 543 markets (2026-04-04..04-27) taker
    prints sum to a median 0.975 of Gamma volume; gross BUY shares ~ `size/(1-0.06(1-p))`; do not deduct that fee again.
    May 2026 (same 3% schedule) has no shortfall.
11. In-play trading halts (zero prints, prints still sum to volume): platform-wide 2026-02-07 16:06-16:38 UTC,
    2026-05-01 19:34-20:07, 2026-08-30 17:56-18:28, 2026-09-19 17:00-20:00 in several windows; single-match halts on
    several dates; 24 liquid matches with an in-play silence >= 15 min (`qa_inplay_halts.csv`,
    `matches.inplay_max_gap_s`). Prices can reopen stale (`fl1-tou-hac-2026-09-19` reopened ~0.45 with the home side
    2-0 up, then 0.78).
12. Price history is minute snapshots, not quotes; 3 Oct-2025 BTTS tokens empty; early O/U/BTTS created hours before
    kickoff; points stop at market close.
13. Pinnacle odds stop on 2026-01-15 in the 2025-26 files and are absent in 2026-27: the README test period has no
    Pinnacle benchmark. `Avg*`, `Max*`, `B365*` complete; `BFE*` missing for ~30% of Apr-May 2026.
14. 7 football-data kickoff times differ from ESPN (largest 1 h).
15. ESPN: own goals attributed to the benefiting team; delay events kept; 38 of 1,255 VAR rows lack a wallclock.
16. Bug in project script `polymarket/scripts/soccer-big5-pull.py`: `'red' in type` also matches `penalty---scored`
    (harmless for its saved files); this dataset uses the exact `red-card` type.

## 8. Runtime and API use
Concurrency <= 6, no 429s. Gamma discovery 380 req / 47 s; CLOB prices 10,187 + 76 req / 33 min; trades 10,639 req,
0 retries / 36.2 min; ESPN 2,281 req / ~10 min; total wall time about 50 minutes.

## 9. How to re-run
From this directory, `.venv/bin/python -I scripts/soccer_data/<step>`: `01_discover.py`, `02_build_meta.py`,
`01b_listing_summary.py`, background `04_trades.py --max-hours 4` (`data/soccer_data/trades_concurrency.txt`,
`touch data/soccer_data/trades_STOP`) and `03_prices.py --threads 2`, `05_espn_fixtures.py`,
`05_espn_summaries.py --threads 2`, `06_odds.py`, `07_build_matches_events.py`, `03b_prices_fix.py`,
`08_build_prices_trades.py`, `08b_inplay_stats.py`, `09_qa.py`, `10_data_dictionary.py`.

## 10. Files
Tables `data/soccer_data/{matches,markets,price_history,trades,events,odds}.parquet`; intermediates `pm_matches`,
`pm_espn_link`, `espn_fixtures`, `odds_join`, `qa_price_tokens`, team maps, `trades_parts/`, `raw/`; QA CSV/JSON in
`results/soccer_data/`; `DATA_DICTIONARY.md` generated by `scripts/soccer_data/10_data_dictionary.py`.
