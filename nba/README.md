# Polymarket NBA: research and observed live results

Research reports, data dictionaries and checks are in reports/ and results/. Selected research scripts are in scripts/. docs/combined-research-report-2026-11-04.md is a dated cross-sport study; docs/live-trading-methods-and-lessons.md covers the final NBA live day and rule changes.

The wider research set included 2,750 core NBA games and a separate 793-entry historical strategy replay. The replay estimated -5.8% for holding until resolution versus -10.3% with then-current exits under modeled costs; these are simulated figures, not realized results. See reports for uncertainty intervals and assumptions.

Actual live trading ran on seven US game days, with 65 listed games and 51 purchases. data/live-per-day.csv contains only day-level stake and state P&L totals, culminating in -84.96816 pUSD (rounded -84.97). The final day state sum -45.42781 versus independently reconciled chain cash -45.42782 differs by 0.00001. Exact per-game trade amounts, wallet data and raw event streams are not published. data/public-game-index.csv holds 2,750 public-game metadata rows.

No live operation is active as of the final November 5 handoff. The selected scripts alone cannot fully reproduce results without raw source data and external APIs. See ../docs/DATA_POLICY.md.
