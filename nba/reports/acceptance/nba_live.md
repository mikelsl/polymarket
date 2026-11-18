# Main-agent acceptance: nba_live (2026-11-04 17:26 CST, corrected run)

1. Live-bot exits (H6a), independent path (`scripts/verify_nba_exits.py`). CORRECTION 17:26: my first run converted
   pandas datetime64[us] with `astype('int64') // 10**9` (a factor 1000 off), so the Q3/Q4 time filters were inactive
   and whole-market prints were used; numbers below are the corrected run (epoch seconds via Timedelta):
   - Second-half taker-SELL print at 0.12-0.20 (first per game-token; "sell minus hold" = price - fee - realised win):
     2024-25 n=1,117 sale 0.165 vs win 0.174, -1.56 c/share [-4.04, +0.65]; 2025-26 n=1,212 0.173 vs 0.178,
     -1.19 c [-3.58, +0.79]; 2026-27 n=102 0.174 vs 0.147, +1.99 c [-5.55, +8.96].
   - Q4 taker-SELL print at 0.50-0.60: 2024-25 n=781 0.548 vs 0.552, -1.61 c [-5.49, +1.86]; 2025-26 n=874 0.549 vs
     0.554, -1.72 c [-4.98, +1.63]; 2026-27 n=70 0.558 vs 0.543, +0.28 c [-10.96, +12.01].
   Conclusion: in-play sale prices at the stop and take-profit triggers are fair; selling costs about 1-2 c per share
   (fee plus spread) versus holding, individually not significant per season. The exits reduce variance, they do not
   add expected value. This contradicts the 29+32-game backtests used on 2026-11-03 to adopt them.
2. Other verdicts (market beats state models on log loss; calibration flips track pre-game calibration; momentum,
   breaks, totals not exploitable; spread-line effect fades in 2026-27 with ~7 USDC median prints) accepted as
   reported; numbers re-runnable from `results/nba_live/`.
