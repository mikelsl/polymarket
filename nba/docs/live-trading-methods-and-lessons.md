# NBA live research: rule evolution and observed losses

This is a historical record, not advice to run a bot. Dates below are US game dates; values come from seven private VPS state snapshots copied read-only and the parent project handoff. No wallet/order IDs or raw event stream are included.

| US game day | Listed games | Bought | State P&L (pUSD) | Principal observation |
|---|---:|---:|---:|---|
| 2026-10-28 | 12 | 10 | -0.18620 | Initial live spread selection; 8 sold, 2 lost at settlement. |
| 2026-10-29 | 3 | 2 | -7.69660 | Tranche-ladder exit showed sensitivity to available volume. |
| 2026-10-30 | 10 | 8 | +5.75270 | Five-level re-arm ladder; not evidence of positive expectancy. |
| 2026-10-31 | 7 | 5 | +1.89910 | End of the zero-fee promotion. |
| 2026-11-01 | 4 | 4 | -20.18806 | Fees returned; maker entry/exit and spread stop tested. A duplicate-buy bug doubled one exposure; state P&L differs by 0.00001 from cash reconciliation. |
| 2026-11-02 | 15 | 10 | -19.12129 | Stop triggers sampled by bot and 10-second book collector differed by minutes on some games. |
| 2026-11-04 | 14 | 12 | -45.42781 | New rule: bid <=20c from Q3 stops; bid >=50c from Q4 exits. Ten stopped, two took profit. |

State-file total: **-84.96816 pUSD**, rounded **-84.97 pUSD**. The 2026-11-04 state sum (-45.42781) differs from an independently checked chain-cash delta (-45.42782) by 0.00001. No account balance or wallet transfer details are published in this repository.

## What was checked vs what was simulated

- Actual: daily state-file buys, sell summaries, phases and P&L in data/live-per-game.csv and data/live-per-day.csv. Records include skipped games; 65 listed games, 51 purchases.
- Simulated: on November 4, holding the same 12 positions to settlement would have returned -4.78 pUSD, versus actual -45.43; this is an ex-post counterfactual on one day, **not** a proven winning strategy.
- Larger research replay: 793 entries showed hold-to-resolution about -5.8% and then-current stop/take-profit about -10.3% under modeled fees and fills. It is not the seven-day live return.
- Failure modes: after-game volume ranks leak future information; minute price history can be untradeable near scores; 2026-11-01 duplicate maker fill and bot-vs-collector timing disagreement affect inference; strategy and fee rules changed across days.
- The source research report is dated November 4 and contains pre-final-day recommendations; this document and the final-day records supersede it on live status. Live trading stopped November 5 by owner decision.

The private raw state/events and book feeds are retained separately for audit but excluded from this repository. Exact per-game amounts may still be linkable to public on-chain trades; the owner must confirm disclosure before publication.
