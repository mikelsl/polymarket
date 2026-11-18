# Football research chronology and evidence bounds

There were **no football live trades**. All P&L percentages discussed below are historical backtests or market simulations, not realized account returns.

## Initial October 30–November 1 study

- An initial cohort held 404 finished five-league matches (EPL, La Liga, Serie A, Bundesliga and Ligue 1); 377 passed the data gate. Prices, ESPN clocks and taker prints were cross-checked.
- A first-goal not-draw trade, late-result hold, Exact Score and draw-timing ideas failed once real executable prints, fees, spread and limited depth replaced lagging 1-minute price history.
- Earlier apparent positives in the minute-history studies were explicitly withdrawn. A late-result 85-minute test using real prints was about +1.4% with a 90% interval spanning zero; Exact Score was too thin for 10-unit size.
- Research was paused November 1. No strategy was deployed live.

## November 4 larger read-only study

- The separate autonomous sample comprised 2,204 match records, 2,200 analysis-ready, with a pre-2026-02-01 train split and an untouched test split. Roughly 6.76 million 1X2 taker prints and 5.13 million minute price observations were analyzed locally; large raw files are retained privately and omitted from this repository.
- About 1,400 football rule variants covered pre-match calibration, book comparisons, game-state models, goals/red cards, totals, both-teams-to-score and cross-market structure. None gave an executable, stable post-fee test-period advantage.
- Apparent profit when entering after a price jump following a goal was VAR survivor bias: cancelled goals were missing from a valid-goals-only sample. Waiting for official confirmation removed the apparent edge.
- Minute-history gains for O/U and BTTS did not survive real-print entry tests. Some intervals were wide and included zero; do not describe a training-period positive cell as a tradable strategy.
- A read-only 30-second VPS book collector covered 48 football match slugs and 17,263 snapshot rows. Only coverage summary is in data/football-book-sample-coverage.csv; raw books remain private pending a redistribution decision.

The 404/377 initial sample and the 2,204/2,200 later sample are **different cohorts**. Method, rule grids, QA exceptions and corrected conclusions are in reports/ and results/.
