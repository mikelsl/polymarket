# Data and privacy boundaries

Only selected derived tables, public-match indices, research scripts and reports are included. Full raw Polymarket prints/books, ESPN event logs, Parquet intermediates (~5.2 GB locally), VPS state/events/order history, credentials and wallet/account records are excluded. Scripts require the original public API sources and raw inputs to reproduce the full pipelines; syntax and published summary consistency were checked, not a complete data rebuild.

NBA live-per-day.csv is computed from seven private read-only VPS state files: 65 listed games, 51 purchases, seven-day total -84.96816 pUSD. The last day state sum -45.42781 differs from the independently checked chain-cash delta -45.42782 by 0.00001 rounding. Full fills, positions, addresses, transaction hashes, order identifiers, exact per-game amounts and private state files are intentionally withheld because exact trade combinations can be linked to on-chain accounts. The underlying audit evidence remains local, not in this repository.

Football has no live trades. The initial 404 finished / 377 gated matches and expanded 2,204 / 2,200 analysis-ready sample are distinct cohorts. A 48-match, 17,263-snapshot read-only weekend book sample is represented only by an aggregate coverage table.

The joint research report is dated November 4, before the final NBA live day. Its prospective live-day recommendations are historical; the final observed live status is in nba/docs/live-trading-methods-and-lessons.md. Research figures rely on stated train/test splits, modeled fees and executable-print checks; some prior minute-price apparent gains were withdrawn. Redistribution of bulk raw third-party feeds was deliberately avoided.
