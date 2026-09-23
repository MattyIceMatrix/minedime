"""QuantForge audit tests. Run from the repo root:  python3 tests/test_engine.py
Blast radius: pure computation in memory. Reads and writes no files. Needs numpy + scipy."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantforge"))
from quantforge.data import synthetic_market
from quantforge import alpha as A
from quantforge.backtest import signal_to_weights
from quantforge.validate import deflated_sharpe, pbo_cscv
from quantforge.engine import AlphaForge, Config

def test_no_look_ahead():
    """Scramble prices after a cutoff day; no formula's positions on or before it may change."""
    D = synthetic_market(T=600, N=20, seed=3); cut = 220
    E = {k: (v.copy() if not k.startswith("_") else v) for k, v in D.items()}
    rng = np.random.default_rng(1)
    for k in ["close", "open", "high", "low", "volume", "vwap"]:
        E[k][cut + 1:] *= rng.uniform(0.5, 1.5, E[k][cut + 1:].shape)
    E["returns"] = np.vstack([np.full((1, 20), np.nan), E["close"][1:] / E["close"][:-1] - 1])
    terms = [k for k in D if not k.startswith("_")]; live = leaks = 0
    for n in range(300):
        node = A.grow(np.random.default_rng(1000 + n), terms, 4)
        wa, wb = signal_to_weights(A.evaluate(node, D, {})), signal_to_weights(A.evaluate(node, E, {}))
        leaks += not np.allclose(wa[:cut + 1], wb[:cut + 1], atol=1e-12, equal_nan=True)
        live += not np.allclose(wa[cut + 2:], wb[cut + 2:], atol=1e-12, equal_nan=True)
    assert live > 100, "control failed: the test could not detect changes at all"
    assert leaks == 0, f"{leaks} formulas used future data"
    return f"300 formulas, 0 look-ahead leaks ({live} reacted after the cutoff, so the test is live)"

def test_deflated_sharpe_on_noise():
    """Best-of-50 zero-skill strategies must rarely pass the first gate (nominal 5%)."""
    rng = np.random.default_rng(7); fp = 0
    for _ in range(300):
        M = rng.standard_t(5, size=(1000, 50)) * 0.01
        fp += deflated_sharpe(M[:, np.argmax(M.mean(0) / M.std(0))], 50)[0] >= 0.95
    assert fp / 300 <= 0.05, f"false alarms {fp/300:.1%} exceed the nominal 5%"
    return f"false-alarm rate on pure noise {fp/300:.1%} (nominal 5%, conservative)"

def test_pbo_on_noise():
    rng = np.random.default_rng(3)
    v = np.mean([pbo_cscv(rng.standard_normal((800, 16))) for _ in range(40)])
    assert 0.3 < v < 0.7, f"PBO on noise {v:.2f}, expected about 0.5"
    return f"PBO on pure noise {v:.2f} (theory about 0.50)"

def test_tribunal_end_to_end():
    """A market with planted edges should ship; a zero-edge market should not."""
    cfg = Config(population=120, generations=8)
    real = AlphaForge(synthetic_market(seed=1, alpha_strength=1.0), cfg); real.evolve(log=lambda *a: None)
    rv, _, _, _ = real.tribunal(real.hall_of_fame())
    trap = AlphaForge(synthetic_market(seed=2, alpha_strength=0.0), cfg); trap.evolve(log=lambda *a: None)
    tv, _, _, _ = trap.tribunal(trap.hall_of_fame())
    assert any(v["passed"] for v in rv), "no alpha approved on the planted-edge market"
    assert not any(v["passed"] for v in tv), "an alpha was approved on the zero-edge market"
    return f"planted edges: {sum(v['passed'] for v in rv)} approved | zero edge: 0 approved"

if __name__ == "__main__":
    fails = 0
    for t in [test_no_look_ahead, test_deflated_sharpe_on_noise, test_pbo_on_noise, test_tribunal_end_to_end]:
        t0 = time.time()
        try: print(f"PASS  {t.__name__}: {t()}  [{time.time()-t0:.0f}s]")
        except AssertionError as e: fails += 1; print(f"FAIL  {t.__name__}: {e}")
    sys.exit(1 if fails else 0)
