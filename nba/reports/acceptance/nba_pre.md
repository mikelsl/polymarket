# Main-agent acceptance: nba_pre (2026-11-04 17:26 CST)

1. Pre-tip efficiency, independent path (`scripts/verify_nba_prematch.py`: minute-history home price at tip-15..-1 min
   vs proportionally de-vigged ESPN close moneylines): train n=2,045 Brier PM 0.2067 vs book 0.2065; test n=703 0.1772
   vs 0.1777; mean |PM-book| 1.28 / 1.12 c, p90 2.5 / 2.2 c. Confirms the worker's H2 (equal accuracy, gaps below fee
   plus half-spread).
2. Live entry rule replay (`scripts/verify_nba_live_rule.py` re-aggregating `results/nba_pre/h6_trades.csv`, 793
   trades, cheapest token won 40.1% at mean ask 0.416; bootstrap 2,000): hold -5.8% [-14.3, +3.0]; current exits
   -10.3% [-16.8, -3.1]; current exits with stop filled at 19 c -6.5% [-12.9, +0.2]; stop only -11.9% [-19.1, -4.9];
   Q4 sell only -6.2% [-13.9, +1.2]. By season, current exits: 2025-26 -10.5% [-17.3, -3.1], 2026-27 -9.1%
   [-25.5, +9.4]. Gamma outcomes match the file for 6/6 sampled trades.
   Caveat: the replay's stop fills come from the next print-revealed bid (median 0.08; 73% below 0.15), harsher than
   our bot's FOK at the trigger poll, so the realistic figure is between -6.5% and -10.3%. In 2025-26 only 1-3 spread
   lines existed per game, so the replay universe is narrower than today's 33-line ladders.
   Verdict: no evidence the live entry rule has positive expectation; the exits make it worse.
