# Second review — 24 September 2026

Two independent reviewers who had not seen the code examined commits `682364a` and `8ef40c6`, one for the Python engine
and one for the website. Every finding below was reproduced before it was fixed, and each fix has a test that fails on
the old code and passes on the new.

| Finding | Severity | Fix | Test |
|---|---|---|---|
| The website's simulator ended every price series at a reference price, so each day's price level depended on all later returns. A formula ranking assets by price earned a vault Sharpe of about 1.0 on zero-edge markets (18 of 20 positive). The scramble-after-cutoff test could not see this. | Critical | Prices compound forward from day 0. Bar noise uses its own random stream, so extending the history changes no earlier day. The display-only chart data keeps its end anchor, and the Alpha Lab never uses it. | `simulator has no future` in both suites |
| Zero-edge markets were not zero-edge for simple returns: higher-volatility assets drifted up by half their variance per day, so a fixed long-volatile, short-calm book made money (t = +3.2 pooled over 12 markets). | Major | Each day's return subtracts half its variance, in both simulators. | `test_zero_edge_market_is_zero_edge` |
| Website rolling statistics used running sums. One spike left a permanent error, and a single-pass variance formula lost precision at high price levels (23 of 48 operator/input/window checks disagreed with an exact calculation). | Major | Windowed moments are kept against a shift that re-syncs exactly. Infinity is treated as missing. | `rolling operators exact` |
| Python: a constant window's std came out as 1e-17 rather than 0, and the backtest scaled that noise to a full-size position. Tied values were ranked by column order. | Major | Variation below 1e-7 of the level counts as zero. Rounding-noise rows get no position. Ties get their average rank (both engines). | `test_operator_edge_cases`, `dollar-neutral, noise-free weights` |
| The third gate, probability of backtest overfitting, was biased toward "not overfit". It used 40 candidates picked by whole-sample fitness, and its trials were chosen by a search that saw the same data. On zero-edge markets it read 10–14%. | Major | Demoted to a diagnostic. It now covers every trial on the validation stretch, which the search never saw. On zero-edge markets it reads 45–73% in the website's calibration and 25–55% in Python runs. That stretch is too short for a hard threshold, so the two remaining gates decide. | Calibration below |
| The website's risk note said "market-neutral", but inverse-volatility scaling after demeaning left up to 49% net exposure. | Major | Weights are demeaned again after scaling, so the book is dollar-neutral (under 1e-13 net), and the note now says so. | `dollar-neutral, noise-free weights` |
| The effective number of independent trials was estimated from 260 sampled trials, about half the full count, so the luck bar was about 8% too lenient. | Minor | Every trial up to 1,500 (the Python cap) is used, with a tridiagonal eigen-solver that matches numpy to 1e-13. | `eigen-solver` |
| The hall of fame stopped at the first negative-Sharpe formula in fitness order, dropping later positive ones (both engines). | Minor | It now skips negative-Sharpe formulas instead of stopping. | `test_hall_of_fame_skips_not_stops` |
| "Monte Carlo VaR" resampled the same days and always equalled historical VaR. | Minor | Replaced by Cornish-Fisher fat-tail VaR, which uses the observed skew and kurtosis. | Browser check |
| Book size of 0, empty or 1e400 silently became $1,000,000. | Minor | The input is validated and the page shows a message. | Browser check |
| Chart range return covered one day fewer than the quotes table; annual roll yield spanned 11 months; Alpha Lab overflowed at phone width. | Minor | All three corrected. | Browser check (WTI 1M: chart −6.4%, table −6.4%) |
| Python: a first-day loss was missing from max drawdown; IC was not recentred after missing data; `ts_rank` treated NaN as a value; `cs_rank` ranked infinity; `run_missions.py` crashed on an empty hall of fame. | Minor | All fixed. | `test_operator_edge_cases` |
| The deflated Sharpe test could pass with a gate that rejected everything. | Test gap | Added a control: a genuinely skilled strategy must pass at least 80% of the time. | Same test |

**Calibration after the fixes (website engine, 12 markets):** real edges were approved in 4 of 6, with a mean vault
Sharpe of 2.39. Zero-edge markets were approved in 0 of 6. Before the fixes the result was 5 of 6, but that run had the
price leak and undercounted trials. Python: 2 approved on the planted-edge mission, none on the trap.

**Verification:**
- ESLint and pyflakes report 0 problems.
- axe-core reports 0 violations on all 13 tabs in both themes, including after a forge run.
- No tab overflows at 390 px.
- The page makes no network requests, and injected scripts are blocked.
- Both CSP hashes match the page's scripts.

**Correction to the first audit:**
- Its accessibility run measured tabs mid-fade. With the fade finished, the shipped build also had 0 contrast failures.
- Its claim that "every test includes a control" was not true for the deflated Sharpe test. It is now.

---

# Audit — 23 September 2026

An audit of the live repository at commit `682364a`, using third-party tools and automated tests that can fail,
rather than review by eye. *(Superseded in part by the second review above: the look-ahead row below was true for the
formulas but missed the website simulator's leak, and the third gate described here has been demoted to a diagnostic.)*

| Area | Method | Before | After |
|---|---|---|---|
| Look-ahead bias | Scramble prices after a cutoff inside the search window; positions on or before it must not change. 300 formulas per engine. | 0 leaks | 0 leaks |
| Math parity | Same inputs through the JavaScript and Python engines, and against SciPy | agree to 7–9 decimals | unchanged |
| Tribunal calibration | 300 simulated searches on pure noise; planted-edge vs zero-edge markets | gate 1 false-alarm 0.5–0.7% (nominal 5%); PBO on noise 0.45–0.52 | unchanged; docs corrected |
| JavaScript lint | ESLint 8, `eslint:recommended` | 3 warnings | 0 |
| Python lint | pyflakes 3 | 1 warning | 0 |
| Accessibility | axe-core 4.13, all 13 tabs, dark and light themes | 1,239 violations (5 root causes) | 0 |
| Security | Map every HTML sink to its data source; check CSP; scan full git history for secrets | no injection path; CSP allowed any inline script | CSP pins 2 script hashes; injection blocked |
| Privacy | Record every network request the page makes | Google Fonts (visitor IPs to Google) | none |
| Performance | Heap over 60 s, frame rate, long tasks | flat 7.1 MB, 60 fps, 0 long tasks; kept running in background tabs | pauses when hidden |

## Fixes applied
- Fonts embedded in the page, removing the only third-party connection (Google Fonts). German courts have found
  remote Google Fonts embedding to breach the GDPR (LG München I, 3 O 17493/20, 2022).
- Content-Security-Policy tightened from `'unsafe-inline'` to the SHA-256 hashes of the page's two scripts.
- Contrast fixed in the light theme (amber, dim text, green, red) and on tab hints and heat-map tiles; added a level-one
  heading, header/nav/aside landmarks, keyboard focus for scrollable regions, and a corrected heading order.
- Price ticks, news and simulated events pause while the tab is hidden.
- Python: a dataclass default instance shared between engines (`cfg=Config()`) replaced with a fresh one per engine;
  unused import removed. Three unused JavaScript variables removed.
- Removed a stale standalone copy of the Alpha Lab page; the terminal supersedes it.
- Added this audit as re-runnable tests: `tests/test_engine.py`, `tests/test_site.js`.

## Not verifiable from outside the repository
GitHub account settings (two-factor authentication, branch protection) can only be checked by the account owner.
