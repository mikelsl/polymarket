# Main-agent acceptance: soccer_strat (2026-11-04 16:48 CST)

1. Pre-match efficiency, independent code path (`scripts/verify_soccer_prematch.py`: minute-history price at
   T-30..-5 min, normalised over the three outcomes, vs multiplicatively de-vigged closing odds; Brier summed over the
   three outcomes):
   - Avg close: train n=1,022 Brier PM 0.5735 vs book 0.5740; test n=1,178 0.5810 vs 0.5814; mean |PM-book| 0.76-0.85 c.
   - Betfair close: train 0.5735 vs 0.5735; test n=1,067 0.5794 vs 0.5791; mean |PM-book| 0.35-0.42 c.
   - Pinnacle close (train only, n=895): 0.5712 vs 0.5724.
   Confirms the worker: Polymarket pre-match prices are as accurate as the best closing lines; the typical gap
   (0.4-0.8 c) is below one tick plus the taker fee (about 1.25 c per side at p=0.5), so value betting cannot pay.
2. VAR survivorship: `events.parquet` has 285 `var-decision-no-goal` + 270 `var-goal-overturned` = 555 disallowed
   goals that a "buy after the price jump" rule would also have bought; the worker's anchor-on-confirmation version
   is +5.3% train / +2.0% test to resolution with CIs spanning 0. Accept "not exploitable".
3. Other verdicts (in-play model worse than market from minute 30; late "buy current score" negative/zero on prints;
   O/U and BTTS minute-history edges vanish on prints) are consistent with the micro and xmarket results.
