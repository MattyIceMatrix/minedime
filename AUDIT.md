# Audit — 23 September 2026

An audit of the live repository at commit `682364a`, using third-party tools and automated tests that can fail,
rather than review by eye. Every test includes a control that shows it detects the failure it looks for.

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
