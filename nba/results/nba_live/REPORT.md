<!-- Saved by the main agent from the nba_live worker's final message (workers cannot write .md reports). 2026-11-04. -->

# Workstream `nba_live`: can in-game NBA prices on Polymarket be traded profitably?

Run 2026-11-04, about 15:30–17:20 CST. Read-only: no orders, no VPS access. The only network use was 1,411 Data API requests to fetch prints for pre-game-chosen spread and total lines (2 threads, at most 4 requests/s, no 429s).

## Bottom line

I found no in-game strategy that is both profitable and executable after today's taker fee (0.05·p·(1−p) per share, per leg) at 10–100 USDC per trade.

| Hypothesis | Verdict | Why |
|---|---|---|
| H1 game-state model vs moneyline | **Not exploitable** | The market beats every state model on log loss in liquid seasons. The best train rule went from +1.29 on train to **−1.51 [−2.68, −0.24]** on test. A small late-game effect fades as the information delay grows from 15 to 45 s: it is a latency race. |
| H2 calibration by price × time remaining (incl. 0.90–0.99) | **Not exploitable** | Favourites lose a little on train and win a little on test (ROI −2.6 % / +1.1 %). This flip follows the pre-game calibration of the two periods. Near-certain tokens are not underpriced. In-play longshots are overpriced in 2025-26 and later, but the favourite side nets only about +0.1 USDC per 10 USDC, with a confidence interval that includes 0. |
| H3 momentum / overshoot after big swings | **Not exploitable** | There is mild continuation (+0.1 to +1.7 c over 5 min). Round-trip costs are larger: every follow and fade rule loses 0.13–1.3 USDC per 10 USDC. |
| H4 quarter breaks, halftime, timeouts | **Not exploitable** | Prices hardly move during breaks (mostly under 0.3 c). Trading within a break loses 0.5–1.35 USDC per 10 USDC. |
| H5 in-play totals | **Not exploitable** | Totals do not overreact to early pace. Scoring pace so far adds almost nothing beyond the pre-game total, and the market prices that. |
| H5 in-play spreads | **Inconclusive, and not tradeable at size** | A margin model fitted on train beats thin spread-line prices: all 15 test variants are positive. But the variant picked on train has a test CI that includes 0: +2.51 [−0.47, 6.38]. The effect was in Feb–Mar 2026 and is −1.86 in 2026-27. Profit comes from a few longshot wins. The median entry print is 7 USDC. |
| H6a live bot exit rules (2nd-half stop, Q4 take-profit) | **They cost money in expectation** | The tokens sold at the trigger win about as often as their sale price implies, so the exits only add fees and spread. The spread stop costs **0.28–0.86 USDC per position** compared with holding. |
| H6b/c anchoring to the pre-game price, fading big departures | **Not exploitable** | The first-quarter pattern is noisy. Fading in Q2–Q4 loses. |

Side findings worth knowing:
- **ESPN vs market timing.** The median time between ESPN's play timestamp and the market's reaction (prints shifted back 4 s to fill time) fell from 10 s (2024-25) to 4 s (2025-26) to **0 s (2026-27)**. In the current season the market has moved by the time ESPN stamps a play, so any rule that reacts to ESPN plays is now a pure latency race.
- **NBA leads shrink.** The current margin predicts the result far less than a random-walk model says: fitted σ is 18.8 points per game, against 12.3 implied by the pre-game line. An early lead is expected to shrink by up to half. The moneyline market prices this. A naive random-walk fair value loses badly (test log loss 0.529 vs 0.488 for the market).
- **About 540 rule variants were tried** (about 800 with sensitivity reruns). Scattered positive test results are what chance alone would produce at that count.

## 1. Questions and design

Can a fair value built from game state, or plain price patterns, find in-game mispricings that survive real fills, today's fee, and our latency (bot polls every ~2 s; ESPN feed delay unknown)? The README rules apply:
- Parameters are chosen on train (games before 2026-02-01). Test (2026-02-01..2026-11-02) is reported separately.
- P&L is reported per 10 USDC stake, after fees, with gross shown where relevant.
- 95 % confidence intervals use a bootstrap that resamples whole games (2,000 draws for headline numbers, 300–1,000 for grids).

**Timing conventions** (applied everywhere):
- **Print time.** Every print is moved back to its real fill time: rt = Data API timestamp − 4 s.
- **ESPN time.** ESPN wallclocks get the `wallclock_shift_suspect_s` fix. Each game is then aligned to the market by its estimated lag (section 2.2).
- **Information delay.** ESPN information is used only from aligned time + L, with L = 15 s by default. Sensitivity runs use 30, 45 and 60 s.
- **Fills.** Entry is at the first real print on our side at or after decision time + 2 s, within 60 s:
  - buying home = a taker bought home or sold away;
  - buying away = 1 − the home price of a taker who sold home or bought away.
- **Exits.**
  - A timed exit sells at the first opposite-side print at or after t + h.
  - If the game is over by then, the position is valued at resolution (0 or 1), with no exit fee.
  - If no print exists within 120 s, the exit is the fair mid minus 1 c.
  - I first missed the game-over case: the losing tokens simply had no exit print, so the timed-exit sample silently dropped losers. That is fixed.
- **One trade per game per rule** (the first trigger), unless stated otherwise.

**Score step-backs.** The score used at each moment comes from the play that is latest in game-clock order among the plays published by then. Late-inserted plays therefore do not move the score backwards.

**Exclusions.**
- The 15 games with platform print holes (`flag_print_gap_in_play`) are dropped everywhere.
- Misaligned games are dropped from every test that uses ESPN timing.

## 2. Data

### 2.1 Inputs and splits
- **Games:** 2,750 core games (`data/nba_data`).
  - Train: 2,051 (2024-25: 1,321; 2025-26 to 2026-01-31: 730).
  - Test: 699 (2025-26 from 2026-02-01: 592; 2026-27: 107).
- **Moneyline prints:** 10.73 M, of which about 4.9 M in play. That is 546 per game in 2024-25, 2,949 in 2025-26 and 2,867 in 2026-27.
- **Pre-tip price:** the mid of the last taker buy and sell prints before tip (both within 15 min). It is missing for 4 games.
- **ESPN:** 1.32 M plays with fixed wallclocks.
- **Sportsbook closes:** ESPN close for spread and total.
- **Decision panel:** every 30 s from tip + 60 s to the last play, 759,917 rows. The ESPN state is stored at 15, 30 and 45 s delays.

**Spread and total lines are chosen with pre-game information only.** For each game I take the line closest to the sportsbook close (ties: the earliest-created market).
- Only 33 % of those equal the "main" line, which nba_data picked by final volume (look-ahead).
- Prints for the other 1,356 markets were fetched: 314,568 prints.
- Coverage: 1,419 games each for spreads and totals (2025-26 onward).

### 2.2 ESPN–market alignment (`alignment.py`, `alignment_flags.py`)
- **Method.** For each game I cross-correlate 10-s changes of an ESPN-based win-probability series with the last-print price series. Lags from −180 to +180 s are tried, plus ±1 h and ±1 day.
  - This is done for the whole game and for each half.
  - A mid built from both sides reacts late and biased the lag upward, so I used the last print instead.
- **A game counts as aligned (`align_ok`) when:**
  - its whole-game lag is between −20 and +30 s;
  - the peak correlation is at least 0.15;
  - each half is consistent or only weakly identified.

| Season | Games | Aligned | Median lag (market after ESPN) | IQR |
|---|---|---|---|---|
| 2024-25 | 1,321 | 830 | 10 s | 8–20 s |
| 2025-26 | 1,322 | 1,152 | 4 s | 4–6 s |
| 2026-27 | 107 | 99 | 0 s | 0–0 s |

- **Gross offsets.** About 5 unflagged games peak at +1 h, but with weak correlation.
- **Large positive lags in some halves**, mostly 2024-25 (e.g. `nba-sac-mia-2024-11-04`: the market moves 122 s after ESPN, peak correlation 0.46).
  - They point to delayed print stamps, or ESPN being early.
  - They would leak future information into any rule that uses ESPN, so these games are excluded from ESPN-timed tests.
- **Proof that leakage matters.** 2024-25 rules buying on stale prices showed absurd "profits", for example +9.3 USDC per 10 USDC for the model rule at θ = 0.12 in Q3–Q4.

### 2.3 Costs actually paid (moneyline, in play)
- **Spread paid.** The median cost of buying versus the print mid is 0.5 c in 2025-26 and 2026-27, and 1.0 c in 2024-25.
- **Fee.** 1.25 c/share at p = 0.5, 0.24 c at p = 0.95.
- **Round trip at p ≈ 0.5.** About 3.5 c/share, or about 0.7 USDC per 10 USDC.
- **Buy and hold.** About 1.75 c/share.

## 3. Models (`fit_model.py`, train games only)

- **Pre-game prices are calibrated.**
  - A probit of the result on the inverse-normal of the pre-tip price has slope 0.93 (SE 0.05) on train and 1.23 (SE 0.09) on test. So test-period favourites beat their prices.
  - Book spread and price agree with σ = 12.3 points.
- **Random-walk margin model.** Final margin = current margin + N(μτ, σ²τ), with a continuity correction and an overtime term.
  - The maximum-likelihood σ is **18.8**, not 12.3.
  - By time remaining: 17.2 in the first 12 min, 19.0–19.7 in the middle of the game, 20.7 in the last 2 min.
- **Leads shrink.** I regressed the remaining margin on the current margin and on μτ.
  - The coefficient on the current margin runs from −0.49 (first 6 min) to −0.05 (last 2 min).
  - The coefficient on μτ is 1.0–1.45.
  - So early leads mean-revert strongly and pre-game strength keeps mattering.
- **EMP model:** a logistic per time-remaining bucket on [current margin / √τ, μ·√τ].

| Log loss on panel rows | Market mid | EMP model | Naive random walk (σ 12.34) |
|---|---|---|---|
| 2024-25 (train) | 0.5708 | 0.5705 | 0.6204 |
| 2025-26 to Jan (train) | 0.5598 | 0.5614 | 0.6091 |
| Test | **0.4877** | 0.4945 | 0.5293 |

## 4. H1: state model vs moneyline (`h1_explore.py`, `h1_states.py`, `h1_rules.py`, `h1b_exec.py`)

**Outcome minus mid, binned by model − mid.** Rows are restricted to a stable market (mid within 2 c of its value 20 s earlier) and no known score in the last 30 s.
- 2024-25: monotone from −0.12 at deviation ≤ −0.15 to +0.13 at ≥ +0.15. This is stale prices.
- 2025-26 train: weak.
- Test: no pattern; for deviation 0.08–0.15 the residual is −0.07 [−0.23, +0.09].

**13 state classes.** I checked early big leads, blowouts, close late games, a pre-game favourite trailing, and overtime. None has the same sign across 2024-25, 2025-26 train and test.
- Example, leader up 8+ in the first 6 min: −0.115 [−0.19, −0.03] in 2025-26 train, −0.02 in 2024-25, +0.002 in test.

**Rule grid.** 96 variants: 2 models × θ ∈ {0.03, 0.05, 0.08, 0.12} × 4 windows × 3 exits (hold, 5 min, 10 min).
- 5 and 10 min exits lose 0.3–0.8 USDC per 10 USDC almost everywhere. That is the round-trip cost.
- Hold-to-resolution results are noisy (CIs about ±1.5).

**Executable-price version (H1b).** The signal is model − the next actual fill within 10 s. All 15 test variants are negative (−0.13 to −3.32).

| Train-selected variant (best train t-stat) | Train | Test |
|---|---|---|
| H1: EMP, θ = 0.03, Q1–Q2, hold | +1.29 [0.38, 2.31] (n=1,207) | **−1.51 [−2.68, −0.24]** (n=588) |
| H1b: EMP vs fill, θ = 0.04, Q1–Q2 | +1.32 [0.38, 2.26] | −1.04 [−2.37, 0.29] |

**Latency decay of the one consistent-looking cell** (EMP, θ = 0.05, last 6 min, hold; test / 2025-26 train / 2024-25 train):

| Info delay | Test | 2025-26 train | 2024-25 train |
|---|---|---|---|
| 15 s | +0.60 [−1.14, 2.38] | +1.61 [−0.17, 3.26] | +0.76 |
| 30 s | −0.18 | +0.35 | +0.67 |
| 45 s | −1.40 | +0.08 | +0.09 |

This is a latency race, not a mispricing.

## 5. H2: calibration by price and time remaining (`h2_calibration.py`)

- **Descriptive grid:** 8 price buckets × 8 time buckets (`h2_desc.csv`).
- **Rules:** 72 = (favourite, underdog) × 6 windows (Q1, Q2, Q3, 6–12 min, 2–6 min, 0–2 min) × 6 favourite-price bands. Buy at the ask, hold.

| Pooled (first trigger per game per rule) | Train | Test | 2024-25 | 2025-26 | 2026-27 |
|---|---|---|---|---|---|
| Buy favourite | −0.26 [−0.43, −0.11] | +0.11 [−0.14, 0.34] | −0.26 | −0.10 | +0.10 |
| Buy underdog | −0.40 [−1.52, 0.73] | **−2.90 [−4.20, −1.44]** | +0.13 | −1.93 | −3.39 |

**Near-certain favourites** (any window):
- Band 0.95–0.98: train −0.055, test +0.126 [−0.036, 0.235].
- Band 0.98–0.995: train −0.038, test +0.066 [−0.010, 0.119].
- Last 2 min at 0.95–0.98: train −0.185 [−0.37, −0.02], test −0.12.
- Losses at 95–99 c in the last 2 minutes are mostly real overtime comebacks. A few are stale-mid fills, for example a fill at 0.40 when the mid showed 0.97.

**Best train variant.** Favourite in Q2 at 0.98–0.995 won all 129 train trades: +0.145. On test it made +0.034 [−0.18, 0.15]. The upside is at most about 0.15 USDC per 10 USDC, while one loss costs 10 USDC.

**Why train and test disagree.** The train/test sign flip for favourites tracks the pre-game calibration difference (probit 0.93 vs 1.23). It is outcome noise across about 700 games, not a stable in-play bias.

## 6. H3: runs and momentum (`h3_momentum.py`)

- **Signal:** the print mid moves by at least X ∈ {5, 10, 15} c over W ∈ {60, 180, 300} s, with the mid between 0.10 and 0.90 and a 300-s cooldown.
- **Trade:** enter D ∈ {10, 30} s later, follow or fade the move.
- **Exits:** 1, 3, 5, 10 min, or hold. That is 180 variants on about 1,850 train and 630 test games.

| | Follow | Fade |
|---|---|---|
| Mid drift over 5 min, train | +0.63 to +1.31 c | −0.0 to −0.5 c |
| Mid drift over 5 min, test | +0.13 to +1.72 c | ≈ 0 to −0.5 c |
| P&L, 5-min exit, train / test | −0.45 to −0.66 / −0.13 to −0.64 | −0.92 to −1.15 / −0.81 to −1.31 |
| P&L, hold, train / test | −0.11 to +0.28 / −0.18 to −0.54 | −0.37 to −1.02 / −0.30 to −1.18 |

Best train variant (W = 60 s, X = 15 c, D = 10 s, follow, hold): train +0.28 [−0.19, 0.77], test −0.21 [−0.84, 0.42]. The small continuation never covers half-spread plus fee.

## 7. H4: breaks (`h4_breaks.py`, aligned games)

**Breaks** (break × side rows):
- quarter breaks: 10,759;
- halftime: 6,886;
- timeouts followed by at least 60 s without a play: 50,247;
- other stoppages of at least 90 s: 36,527;
- the break before overtime: 215.

**Trade:** enter at break start + 20 s and exit before play resumes (or hold). Sides tested: leader, trailer, the side that rose over the previous 3 min, the side that fell, and the pre-game favourite.

**Drift during breaks:**
- Halftime: −0.04 to +0.09 c.
- Quarter breaks: within ±0.1 c.
- Timeouts, side that rose: +0.30 c [0.26, 0.37] on train but +0.04 c on test.
- Before overtime, pre-game favourite: +2.84 c [1.08, 5.02] on train (n = 50) and +0.70 c [0.02, 1.45] on test (n = 21).

**P&L:**
- Round trips inside the break lose 0.48–1.35 USDC per 10 USDC in every cell.
- Holding from the break is noise. Best train variant (halftime, trailer): +1.02 [−0.26, 2.39] on train, −1.27 [−2.98, 0.48] on test.

## 8. H5: in-play totals and spreads (`h5_lines.py`, `h5b_exec.py`, `h5c_xmarket.py`, `h5d_deep.py`)

**Models (train, ESPN plus sportsbook close):**
- **Totals.** Remaining points = a + b·(book total·τ) + c·(points so far − expected so far).
  - c is −0.16 in the first 6 min and about 0 after that.
  - So scoring pace so far adds nothing.
- **Spreads.** Remaining margin = a + b·μτ + c·(current margin).
  - c runs from −0.47 to −0.05.
  - The residual SD is 13.5 points early and 3.3 points in the last 2 min.

**Totals.**
- Residual vs pace deviation: slopes of +0.01 to +0.04 per 10 points (mostly within 1 SE). That is not an overreaction; if anything a slight under-reaction.
- Rules lose. Example: θ = 0.05 in Q1–Q2 gives −0.88 on test and −1.51 on train. "Fade the pace" rules are −3.4 to +1.0 with wide CIs.
- Against the executable price (H5b), the residual is flat by model edge. Best train variant: test +0.03 [−1.00, 1.14].

**Spreads, signal = model − actual fill within 10 s (H5b).** The residual is monotone in both periods:
- edge 0.04–0.08: test +5.6 c [1.5, 9.3], train +3.2 c;
- edge 0.08–0.12: train +14.9 c [7.0, 23.1].

All 15 test variants are positive (+0.33 to +19.6). The effect holds when the info delay is 30 or 60 s and when the fill window is 5 or 30 s. With the naive random-walk model it is much weaker (test about 0.2–1.3), so it comes from the mean-reversion structure.

**Cross-market check (H5c).** The spread line priced from the moneyline, with the model's SD, shows no consistent edge. Best train variant: test −0.08 [−1.43, 1.36]. So the spread lines are not clearly inconsistent with the moneyline. Yet the same state model finds nothing against the moneyline (H1b). This undercuts the spread result.

**Train-selected spread rule** (θ = 0.10, regulation with more than 2 min left, hold):
- Train: +5.36 [0.96, 10.46] (n = 108). This is in-sample for the model, which was fitted on train outcomes.
- **Test: +2.51 [−0.47, 6.38]** (n = 212).
  - 2026-02..06: +3.46 [−0.11, 8.22] (n = 174).
  - 2026-27: **−1.86 [−4.82, 1.12]** (n = 38).
- By month: Feb +6.4, Mar +4.6, Apr +0.2, May −1.3, Jun −1.6, Oct −1.8, Nov −2.4.
- By entry price, at θ = 0.06:
  - entries at or below 0.30 make +11.4 on test (n = 39, CI −3.6 to 30.3);
  - the 0.30–0.55 bands are negative on test (−1.84 and −0.71);
  - 0.55–0.70 makes +1.0 [−0.08, 2.09].

**Depth.**
- The median entry print is **7.4 USDC**; 43 % are at least 10 USDC and 10 % at least 100.
- Same-side volume within 40 s: median 20.7 USDC.
- Our fill would compete with the taker who made that print, so the backtest flatters what we would actually get.

**Caveat.** H5b and H5c were designed after the first H5 run, whose tables I saw for test as well.

## 9. H6: other ideas

**(a) The live bot's exit rules on the full sample (`h6_stops.py`).**
- **Population:** every token bought at 0.25–0.50 at the first ask print after tip + 10 min.
- **Stop:** from period 3, a sell print at or below 0.20 while the mid is below 0.50.
- **Take-profit:** from period 4, a sell print at or above 0.50.
- **Execution:** the sale fills at the next sell print at least 2 s later. Values are USDC per 10 USDC position, compared with holding.

| | Moneyline train (n=1,348) | Moneyline test (n=427) | Spread train (n=302) | Spread test (n=617) |
|---|---|---|---|---|
| Stop triggers | 75 % | 76 % | 66 % | 70 % |
| Stop: sale price / later win rate | 0.16 / 0.16 | 0.163 / 0.167 | 0.111 / 0.157 | 0.132 / 0.147 |
| Stop − hold | −0.16 [−0.68, 0.29] | −0.14 [−1.02, 0.71] | **−0.86 [−1.71, −0.18]** | −0.28 [−0.79, 0.16] |
| Take-profit: sale / win rate | 0.664 / 0.664 | 0.659 / 0.712 | 0.747 / 0.741 | 0.691 / 0.700 |
| Take-profit − hold | −0.16 | −0.78 [−1.58, 0.08] | −0.11 | −0.19 |
| Both − hold | −0.25 | −0.41 | **−0.85 [−1.76, −0.05]** | −0.36 |

Caveat: the live bot buys the cheapest of the top-4 spread lines, not the line closest to the book close, so the population is similar but not identical.

**(b) Anchoring (`h6_anchor.py`).** I regressed outcome minus mid on (pre-tip price − mid).
- Q1 slope: +0.39 (SE 0.17) in 2025-26 train, +0.12 (0.16) on test, 0.00 in 2024-25.
- Q2–Q4: about 0.

**(c) Fade large departures from the pre-tip price (`h6_fade_gap.py`).** 40 variants.
- Best train variant (Q1, gap ≥ 25 c, hold): train +0.32 [−2.47, 3.38], test +1.10 [−2.51, 4.88] (n = 70).
- Fading in Q2–Q4 loses (−0.4 to −2.4 per 10 USDC).

## 10. Depth on moneyline entries (`depth.py`)

- **Our fill print:** median 10–24 USDC in play.
- **Taker-buy volume printed within 30 s at no more than our fill + 1 c:**
  - 2025-26 / 2026-27: median 150–650 USDC; at least 100 USDC for 58–87 % of entries.
  - 2024-25: median 55–120 USDC.
- **Near-certain prices (0.97–1.00):** median 560–650 USDC.

10 USDC is always fine and 100 USDC usually fills within 1 c. Depth is not the binding constraint on the moneyline: the absence of an edge is.

## 11. Variants tried and multiple testing

| Family | Primary variants |
|---|---|
| H1 | 96 |
| H1b | 15 |
| H2 | 72 |
| H3 | 180 |
| H4 | 50 |
| H5 (stale mid) | 20 |
| H5b | 30 |
| H5c | 30 |
| H6a | 6 |
| H6c | 40 |
| **Total** | **about 540** |

On top of these:
- about 270 sensitivity reruns (info delay 30/45/60 s, fill window, naive model);
- about 25 descriptive tests (state classes, anchoring regressions).

Picking the best train variant by t-stat in each family and reading its untouched test result (`best_of_train_t_tr.csv`): no family has a test CI above zero.

## 12. Caveats

- **ESPN feed delay is still unmeasured.** The default is 15 s after the market reacted, with 30/45/60 s also reported. In 2026-27 the market already reacts at the ESPN timestamp.
- **Fills are other takers' prints.** For thin spread and total lines that overstates what we could get.
- **2024-25 data is weak.** Sparse prints, stale mids and possibly late print stamps. Train results from that season should be read with caution; this is why the ESPN-timed tests also show 2025-26 train separately.
- **Resolution-based tests have low power.** The SD per 10 USDC trade at p ≈ 0.5 is about 10 USDC. Test-period favourites beat their pre-game prices, which colours every favourite/underdog result.
- **Decisions not cleanly pre-registered:**
  - the anchoring and fade rules (H6b/c) were designed after seeing descriptive tables that included test;
  - H5b and H5c were designed after the first H5 run, whose test results I had seen.
- **Sportsbook closing lines** are taken from ESPN as pre-game information.
- **No book data** before 2026-10-29.

## 13. Recommendations

1. Do not deploy any in-game rule from this study.
2. The live bot's second-half stop and Q4 take-profit only reduce variance, at a cost of about 0.1–0.9 USDC per position. The stop on spreads is the most expensive. If the aim is expected value, holding is better. This is a decision for the project owner.
3. **Optional follow-up, low priority.** Log in-play spread-line books against the margin model at at most 10 USDC to see whether the Feb–Mar 2026 effect returns. The 2026-27 data says it has not, and depth is tiny.

## 14. Files and how to re-run

All paths are under `polymarket/research/auto-20261104/`.

- **Scripts** (`scripts/nba_live/`):
  - shared: `common.py`, `features.py`;
  - base tables and models: `build_base.py`, `alignment.py`, `alignment_flags.py`, `build_panel.py`, `fit_model.py`;
  - H1: `h1_explore.py`, `h1_states.py`, `h1_rules.py`, `h1b_exec.py`, `explore_resid.py`;
  - H2–H4: `h2_calibration.py`, `h3_momentum.py`, `h4_breaks.py`;
  - H5: `fetch_lines.py`, `h5_lines.py`, `h5b_exec.py`, `h5c_xmarket.py`, `h5d_deep.py`;
  - H6: `h6_stops.py`, `h6_anchor.py`, `h6_fade_gap.py`;
  - depth and summaries: `depth.py`, `select_best.py`, `summary.py`.
  - `run_all.sh` runs everything in order. Only `fetch_lines.py` touches the network, and it skips files already present.
- **Results** (`results/nba_live/`):
  - start with `summary.json`, `best_of_train_t_tr.csv` and `best_of_train_mean_tr.csv`;
  - model fit: `model_fit.csv`;
  - H1: `h1_explore_devbins.csv`, `h1_states.csv`, `h1_rules{,_L30,_L45}.csv`, `h1b_*`;
  - H2–H4: `h2_desc.csv`, `h2_rules.csv`, `h3_summary.csv`, `h4_summary.csv`;
  - H5: `h5_fit_*.csv`, `h5_resid_*.csv`, `h5_rules_*.csv`, `h5b_resid_*.csv`, `h5b_rules_*.csv` (spread, total, `_L30`, `_L60`, `_w5`, `_w30`, `_naive`), `h5c_*.csv`;
  - H6: `h6_stops.csv`, `h6_anchor.csv`, `h6_fade_gap.csv`;
  - depth: `depth_by_time.csv`, `depth_by_price.csv`;
  - job logs alongside.
- **Data** (`data/nba_live/`, 277 MB):
  - base tables: `games.parquet`, `prints_{moneyline,spread,total}.parquet`, `states.parquet`;
  - alignment: `alignment.parquet`, `alignment_flags.parquet`;
  - panel and models: `panel.parquet`, `model_params.json`, `model_cp60.parquet`;
  - decision-time lines: `lines_selected.parquet`, `trades_lines/` (1,356 fetched markets);
  - per-hypothesis trade and event files: `h1_trades*`, `h1b_trades_ml`, `h2_trades`, `h3_events`, `h4_breaks`, `h5_panel_*`, `h5_trades_*`, `h5b_trades_*`, `h5c_trades`, `h5d_trades_theta006_all`, `h6_*`.
