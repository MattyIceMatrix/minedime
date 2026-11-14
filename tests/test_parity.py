"""Cross-engine parity: the README claims "the JavaScript and Python statistics agree to
7 decimal places". This test measures that, rather than taking it on trust.

Both engines implement the same tribunal twice, in two languages, with two different normal
CDF / inverse-CDF approximations and two different symmetric eigen-solvers. If one drifts, the
site and `quantforge/` stop being the same experiment.

Run from the repo root:  python3 tests/test_parity.py     (needs node on PATH)
Blast radius: pure computation plus one `node` subprocess. Writes no files.
"""
import json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "quantforge"))
from quantforge.validate import deflated_sharpe, pbo_cscv, effective_trials
from quantforge.backtest import ANN

TOLERANCE = 1e-7   # "agree to 7 decimal places"


def _js(payload):
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node is not on PATH; the parity test needs both engines")
    r = subprocess.run([node, os.path.join(HERE, "_parity_bridge.js")],
                       input=json.dumps(payload), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"the website engine failed to run: {r.stderr.strip()[:400]}")
    return json.loads(r.stdout)


def test_deflated_sharpe_parity():
    """Fat-tailed P&L across a wide span of trial counts, including the extremes where the
    inverse-normal approximations are least comfortable."""
    rng = np.random.default_rng(42)
    cases = []
    for i, n in enumerate([2, 3, 7, 50, 99, 356, 1500, 3000, 50000]):
        pnl = rng.standard_t(5, 900) / np.sqrt(5 / 3) * 0.01 + 0.0004 * i
        cases.append({"pnl": pnl.tolist(), "n": n})
    js = _js({"dsr": cases})["dsr"]
    worst_dsr = worst_bar = 0.0
    for c, j in zip(cases, js):
        p = np.asarray(c["pnl"])
        d, sr0 = deflated_sharpe(p, c["n"])
        worst_dsr = max(worst_dsr, abs(d - j["dsr"]))
        worst_bar = max(worst_bar, abs(sr0 * np.sqrt(ANN) - j["bar"]))
    assert worst_dsr < TOLERANCE, f"deflated Sharpe differs by {worst_dsr:.2e} across the two engines"
    assert worst_bar < TOLERANCE, f"luck bar differs by {worst_bar:.2e} across the two engines"
    # control: the comparison is not trivially passing because both sides are constant
    vals = [j["dsr"] for j in js]
    assert max(vals) - min(vals) > 0.5, f"control failed: all DSR values are alike ({vals})"
    return f"{len(cases)} P&L series, trial counts 2 to 50,000: DSR agrees to {worst_dsr:.1e}, luck bar to {worst_bar:.1e}"


def test_pbo_parity():
    rng = np.random.default_rng(7)
    mats = [rng.standard_normal((600, 12)), rng.standard_normal((400, 9)) * 0.01]
    big = rng.standard_normal((500, 14)); big[:, 0] += 0.3
    mats.append(big)
    js = _js({"pbo": [m.T.tolist() for m in mats]})["pbo"]
    worst = max(abs(pbo_cscv(m) - j) for m, j in zip(mats, js))
    assert worst < TOLERANCE, f"PBO differs by {worst:.2e}"
    assert max(js) - min(js) > 0.3, f"control failed: PBO values are all alike ({js})"
    return f"{len(mats)} candidate matrices (252 splits each): PBO agrees to {worst:.1e}; values {[round(v,3) for v in js]}"


def test_effective_trials_parity():
    """Both engines count eigen-directions, but with different solvers: numpy's LAPACK routine
    against a hand-written Householder + implicit-QL. They must return the same integer."""
    rng = np.random.default_rng(3)
    mats = [rng.standard_normal((800, 40))]
    base = rng.standard_normal(800)
    mats.append(np.column_stack([base + 0.05 * rng.standard_normal(800) for _ in range(30)]))
    mats.append(np.column_stack([rng.standard_normal(800) for _ in range(5)] * 6))
    js = _js({"eff": [m.T.tolist() for m in mats]})["eff"]
    py = [effective_trials(m) for m in mats]
    assert py == js, f"effective trial counts disagree: python {py} vs javascript {js}"
    assert len(set(py)) > 1, f"control failed: every matrix gave the same count ({py})"
    return f"independent / near-clone / repeated-block populations: both engines say {py}"


TESTS = [test_deflated_sharpe_parity, test_pbo_parity, test_effective_trials_parity]

if __name__ == "__main__":
    fails = 0
    for t in TESTS:
        t0 = time.time()
        try:
            print(f"PASS  {t.__name__}: {t()}  [{time.time()-t0:.0f}s]")
        except (AssertionError, RuntimeError) as e:
            fails += 1
            print(f"FAIL  {t.__name__}: {e}")
    sys.exit(1 if fails else 0)
