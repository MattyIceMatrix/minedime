# MineDime Terminal

A commodity and digital-commodity intelligence terminal with a built-in quant research lab.

**Live site:** https://mattyicematrix.github.io/minedime/

- 13 tabs: markets, Alpha Lab, tribunal, risk desk with stress tests, digital commodities, positioning,
  crack spreads, forward curves, maritime chokepoints, battery supply chain, mine operations, geopolitical risk,
  and an AI analyst (the analyst works when the page is opened inside Claude).
- **Alpha Lab:** a genetic-programming engine invents trading formulas; a statistical tribunal
  (deflated Sharpe with effective-trial counting, independent validation replication at family-wise 5%,
  and probability of backtest overfitting) tries to prove each one is luck. Only survivors reach the risk desk.
- `quantforge/` is the same engine in Python: `python3 quantforge/run_missions.py` (numpy, scipy, matplotlib).

All market, shipping, weather and seismic data on the site is **simulated**. This is research software,
not investment advice.

The site is a single static HTML file: no backend, no accounts, no API keys, no cookies, no tracking.
