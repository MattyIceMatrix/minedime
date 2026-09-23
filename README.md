# MineDime Terminal

A commodity and digital-commodity intelligence terminal with a built-in quant research lab.

**Live site:** https://mattyicematrix.github.io/minedime/

- **13 tabs:** markets, Alpha Lab, tribunal, risk desk with stress tests, digital commodities, positioning,
  crack spreads, forward curves, maritime chokepoints, battery supply chain, mine operations, geopolitical risk,
  and an AI analyst (the analyst runs only when the page is opened inside Claude; on this site it shows a note instead).
- **Alpha Lab:** a genetic-programming engine invents trading formulas. A tribunal then tries to prove each one is luck:
  1. Deflated Sharpe ratio against the best score chance would give after the search's effective number of independent tries.
  2. Independent replication on a validation stretch the search never optimized on, at family-wise 5% (Bonferroni).
  3. Probability of backtest overfitting across the whole field.
  Only survivors reach the risk desk. The last 30% of history stays sealed until the verdict.
- **`quantforge/`:** the same engine in Python. `python3 quantforge/run_missions.py` (numpy, scipy, matplotlib).

## Verified behavior
See [AUDIT.md](AUDIT.md). In short: no look-ahead bias in 600 random formulas across both engines; the JavaScript and
Python statistics agree to 7 decimal places; on simulated markets the tribunal approved real edges and rejected every
zero-edge market. The first gate is conservative: on pure noise it false-alarmed about 0.5–0.7% of the time against a nominal 5%.

Re-run the checks: `python3 tests/test_engine.py` and `node tests/test_site.js`.

## Honest limits
All market, shipping, weather, seismic and geopolitical data on the site is **simulated**. Battery-chain shares and
chokepoint oil flows are rounded public estimates for scale. Simulated edges are cleaner than real ones, and real trading
adds slippage, borrow costs, capacity limits and regime change. This is research software, not investment advice.

## Privacy
The site is one static HTML file: no server, no accounts, no cookies, no analytics, and no outside connections.
Fonts are embedded in the page, so visitors' browsers contact no third party.

## License
Copyright © 2026 MattyIceMatrix. All rights reserved; no open-source license is granted for the code.
Embedded fonts (IBM Plex Mono, Big Shoulders Display) are under the SIL Open Font License 1.1; see [licenses/](licenses/).
