<!-- Saved by the main agent from the nba_pre worker's final message (workers cannot write .md reports). 2026-11-04. -->

I found no profitable pre-game strategy in Polymarket NBA game markets. None of the five hypotheses gives a rule that survives the untouched test period after today's 5% fee. The replay of the current live entry rule also loses money, with a 95% interval below zero for the current exit.

**中文摘要（给 the project owner）：** 赛前没有找到能稳定赚钱、可执行的 NBA 策略。Polymarket 开赛前的价格和 DraftKings 收盘线平均只差约 1 分钱，扣掉 5% 手续费公式和半个价差后没有空间。唯一的线索"90c 以上的大热门"在 2024–26 两季是正的，但在 Polymarket 之前的 2022–24 两季，庄家的同类热门反而赢得比定价少，所以不算稳定。现行实盘入场规则回放 793 笔：持有到结算 −5.6%，现行止盈止损 −10.0%，置信区间在 0 以下。

## Data and method
- **Existing dataset:** all 2,750 core games (2024-10-22..2026-11-02), with 5.55M pre-tip moneyline prints, minute prices, and ESPN open/close odds (ESPN BET for 2024-25 and early 2025-26, DraftKings after).
- **New pulls (read-only, no HTTP 429s):**
  - CLOB minute prices for 17,468 spread/total lines of 2025-26 and 2026-27.
  - Data API prints for every spread line (2.17M prints) and every total line (1.99M).
  - ESPN closing moneylines for 2,629 games of 2022-23 and 2023-24, as an outside check.
- **Prices:** buy at the next 1c tick above the minute mid. I used a trailing 5-minute median because thin 2024-25 books had one-minute spikes such as 0.14 → 0.56 → 0.17. This matches the last real buy print 72% of the time; the print was higher in 15% of cases.
  - For spread/total lines I used the worse of that price and the last real buy print in the previous 15 minutes, and kept only lines listed before tip that actually traded.
- **Costs and statistics:** 10 USDC per trade, fee 0.05·p·(1−p) on every taker leg for all periods (gross also reported), confidence intervals by resampling games. Train is before 2026-02-01; test is 2026-02-01..11-02.
- **Depth:** 10–100 USDC is easy from 2025-26 on (median 42k USDC traded in the last 15 minutes before tip). 2024-25 was thin (median 2.1k; the thinnest tenth 272 USDC).

## Results by hypothesis

**H1 – calibration of moneyline prices: not exploitable.**
- Overall calibration is neutral: slope 1.005 ± 0.048, home bias −0.03 ± 0.04. Train and test pull in opposite directions (slope 0.93 vs 1.26), which looks like period noise.

| Rule (enter at scheduled start) | Train ROI | Test ROI |
|---|---|---|
| Favourite ≥ 50c | −4.2% | −0.3% |
| Underdog < 30c | −1.5% | −37.2% |
| Home team | −3.7% | −12.4% |
| Away team | −1.8% | −7.4% |

- **Lead – favourites priced ≥ 90c:**
  - 123 of 126 won, against an average price of 92.2c.
  - ROI +3.0% on train (65 trades, interval −3.4% to +7.9%) and +6.9% on test (61 of 61 won).
  - Using real print prices instead: +3.4%.
  - Most trades are March–April games against tanking teams.
  - It does not hold up: in 2022-23 and 2023-24, DraftKings favourites priced ≥ 90% won only 88.9% (27 games) and 89.6% (67 games), less than their price.

**H2 – Polymarket vs sportsbook: not exploitable.**
- Polymarket one minute before tip vs the de-vigged book close:
  - Average absolute gap 1.1c; Brier score 0.2065 vs 0.2062 on train, 0.1774 vs 0.1779 on test.
  - In a joint regression neither source adds information beyond the other.
- The book opening line is stale: Polymarket clearly outperforms it.
- The best-side gap at tip has a median of 0.3–0.4c; only 5% of games exceed 2.1c, which is below fee plus half-spread.
- Value rules vs the close:
  - De-vigged with the power method, at least 1c edge: −0.3% train, +4.9% test (131 trades).
  - Proportional de-vig at 2c edge: +27.8% train, −56.8% test.
  - Every rule against the opening line loses.

**H3 – price moves before tip: not exploitable.**
- Price accuracy improves steadily toward tip (Brier 0.2130 a day out → 0.2069 at tip, the same as the book close).
- Drifts are at most 0.4c, far below the ~3.5c round-trip cost:
  - Every buy-then-sell-at-tip rule loses after costs: 0 of 153 positive on test, 6 of 161 on train.
- Momentum in the hour from T−120 to T−60 min looked good on train (+12% to +30%) but did not replicate on test (−15% to +17%, test outcome-regression t = 0.0).

**H4 – spreads and totals (2025-26 onward): not exploitable.**
- Line prices sit within about 1c of a normal-margin model built on the book line (spread σ 13.74, total σ 17.93, both fitted on train game results).
- Value vs book close: spreads −1% to +1% train, −4% to −6% test.
- Over/under on the line closest to 50c at T−60 min:

| Side | Train ROI | Test ROI |
|---|---|---|
| Over | −13.8% | −2.4% |
| Under | +3.2% | −6.5% |

- **Tails** exist only in the test period, and the two markets disagree:

| Expensive side priced 85–99c | Test ROI | Sample |
|---|---|---|
| Spreads | +4.0% | 40 games |
| Totals | −9.0% | 40 games |

- Moneyline vs the moneyline implied by the spread: +2.0% train → −17.9% test.
- **No ladder arbitrage.** With minute mids, 292 pairs of lines looked mispriced at tip; these came from untradeable mids on thin lines. With real print quotes only 2 pairs remain, max 3.5c net.

**H5 – team and situation biases: not exploitable.**
- No situational factor matters: back-to-backs, rest difference, playoffs, popular teams and favourite status all have |t| < 1.3.
- Team effects do not persist: rank correlation −0.02 between train and test.
- Lead to watch, not trade: playoff underdogs were +10.7% (2025) and +23.6% (2026), pooled +16.7% on 181 games. The interval is −5% to +40%, and this was one of 10 situational rules tried.

**H6 – replay of the current live rule.**
- Setup:
  - Rank spread lines by print volume at the scheduled start, top 4 with |line| ≤ 10.
  - Buy the cheapest token with ask ≤ 45c.
  - Exit: Q4 sell if bid ≥ 50c; from Q3 stop if bid ≤ 20c.
- Coverage: 1,421 games, 793 trades. The cheapest token won 40.1% of the time at an average ask of 41.6c.

| Exit variant | ROI | 95% interval |
|---|---|---|
| Hold to resolution | −5.6% | −13.5% to +2.4% |
| Current exit | −10.0% (−817 USDC) | −15.9% to −3.9% |
| Current exit, stop fills at 19c | −6.3% | −12.1% to −0.3% |
| Live-style limit order first, then taker | −8.4% | — |
| Stop only | −11.6% | — |
| Q4 sell only | −6.0% | — |

- The stop does most of the damage: 451 stops lost 80.5%.
- By season: 2025-26 −10.2%, 2026-27 −8.9% with the current exit.
- For the six live days the replay gives +30 USDC with the current exit and −113 USDC holding.
- On 11/01 it picked 2 of the 4 live picks exactly (Pacers −5.5 at 42c; Raptors +1.5); the other two picks differ because the volume ranking differs.

## Multiple testing and look-ahead
- **Out-of-sample check:** in every family, the variant with the best train ROI fails on test. Only 15 of about 650 rule variants had a train interval above zero, fewer than chance would produce. Picks and test results are in `selection.csv`.
- **Look-ahead risks:**
  - The book close is assumed known one minute before tip; the scheduled-start version has up to about 11 minutes of look-ahead.
  - The opening line's posting time is unknown.
  - Final-volume "main" lines were never used to choose lines.
  - In the live-rule replay the book-spread filter is approximated, and stop/sell fills come from print-revealed bids, so they are uncertain (two bounds reported).
  - Four in-play data holes affect 15 games.

The harness blocked my `.md` writes, so please save this message as `results/nba_pre/REPORT.md`. Re-run from `scripts/nba_pre/` (analysis scripts need no network and run in seconds):
```
p1_ml_panel.py → h1_calibration.py → h1b_hist_heavy_fav.py → h2_book.py → h3_dynamics.py → h5_situational.py
c1_consolidate.py → p2_lines_panel.py → h4_lines.py → h6_live_rule.py → s1_select.py
```
The resumable pulls are `f1_fetch.py --mode prices|trades [--kind totals]` and `f2_espn_hist.py`.

Files are in `polymarket/research/auto-20261104/`:
- `scripts/nba_pre/` – all scripts
- `results/nba_pre/` – `h1_*.csv`–`h6_*.csv`, `h1b_hist_heavy_fav.csv`, `selection.csv`, `depth_ml_pregame.csv`, `h6_trades.csv`, fetch logs
- `data/nba_pre/` – `ml_panel.parquet`, `lines_panel.parquet`, `line_prices.parquet`, `spread_trades.parquet`, `total_trades.parquet`, `espn_hist_odds.parquet` (429 MB in total)
