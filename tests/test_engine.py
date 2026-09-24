"""QuantForge audit tests. Run from the repo root:  python3 tests/test_engine.py
Blast radius: pure computation in memory. Reads and writes no files. Needs numpy + scipy."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantforge"))
from quantforge.data import synthetic_market
from quantforge import alpha as A
from quantforge import ops
from quantforge.backtest import signal_to_weights, max_drawdown, info_coef
from quantforge.validate import deflated_sharpe, pbo_cscv
from quantforge.engine import AlphaForge, Config, Trial

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
    """Best-of-50 zero-skill strategies must rarely pass the first gate (nominal 5%), and a strategy with
    real skill must usually pass. The second half is the control: a gate that rejects everything fails it."""
    rng = np.random.default_rng(7); fp = hit = 0
    for _ in range(300):
        M = rng.standard_t(5, size=(1000, 50)) * 0.01
        fp += deflated_sharpe(M[:, np.argmax(M.mean(0) / M.std(0))], 50)[0] >= 0.95
        skilled = rng.standard_t(5, size=1000) / np.sqrt(5 / 3) * 0.01 + 0.002   # unit-variance t5: daily Sharpe 0.2, about 3.2 a year
        hit += deflated_sharpe(skilled, 50)[0] >= 0.95
    assert fp / 300 <= 0.05, f"false alarms {fp/300:.1%} exceed the nominal 5%"
    assert hit / 300 >= 0.8, f"control failed: a genuinely skilled strategy passed only {hit/300:.0%} of the time"
    return f"false-alarm rate on pure noise {fp/300:.1%} (nominal 5%); skilled strategy passes {hit/300:.0%}"

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

def test_simulator_has_no_future():
    """Re-simulate with a longer history: every field on every earlier day must be identical. Scrambling data after
    a cutoff (the look-ahead test above) cannot catch a simulator whose past depends on its future; this can."""
    a, b = synthetic_market(T=600, seed=9), synthetic_market(T=700, seed=9)
    moved = [k for k in a if not np.array_equal(a[k][:599], b[k][:599], equal_nan=True)]
    control = synthetic_market(T=600, seed=10)
    assert not np.array_equal(a["close"][:599], control["close"][:599]), "control failed: seeds don't matter"
    assert not moved, f"fields that changed on earlier days when history was extended: {moved}"
    return "extending the history by 100 days changes nothing about the first 599 days"

def test_zero_edge_market_is_zero_edge():
    """On the trap market, a fixed long-high-volatility / short-low-volatility book must earn nothing: pooled over
    12 markets its daily P&L must not be significantly different from zero (|t| < 2). Before the drift correction,
    higher-volatility assets drifted up and this book made money (t about +3)."""
    pnl = []
    for s in range(12):
        D = synthetic_market(seed=300 + s, alpha_strength=0.0); vol = np.nanstd(D["returns"][:500], 0)
        w = signal_to_weights(np.tile(vol, (D["close"].shape[0], 1)))
        pnl.append((w * np.nan_to_num(D["_fwd"]))[500:-1].sum(1))
    p = np.concatenate(pnl); t = p.mean() / p.std() * np.sqrt(len(p))
    assert abs(t) < 2, f"t = {t:+.2f}: the zero-edge market has an edge"
    return f"volatility-tilted book pooled over 12 zero-edge markets: t = {t:+.2f} (|t| < 2 required)"

def test_operator_edge_cases():
    """Ties, constant windows, missing data and first-day losses."""
    x = np.array([[1.0, 1.0, 1.0, 2.0]]); r = ops.cs_rank(x)
    assert np.allclose(r[0, :3], r[0, 0]) and r[0, 3] == 0.5, f"tied values ranked by column order: {r}"
    flat = np.full((30, 4), 0.123456789); assert np.all(ops.ts_std(flat, 20)[19:] == 0), "constant window std not 0"
    assert np.all(signal_to_weights(np.full((5, 6), 0.3) + 1e-17 * np.arange(6)) == 0), "rounding noise got a position"
    tr = ops.ts_rank(np.array([[1.], [2], [3], [np.nan], [5]]), 3)
    assert np.isnan(tr[3, 0]) and np.isnan(tr[4, 0]), f"missing data ranked as a real value: {tr.ravel()}"
    assert np.isnan(ops.cs_rank(np.array([[np.inf, 1.0, 2.0, 3.0]]))[0, 0]), "infinity ranked as a real value"
    assert max_drawdown(np.array([-0.1, 0.0, 0.0])) == 0.1, "first-day loss ignored"
    sig = np.tile(np.arange(10.0), (30, 1)); fwd = sig.copy(); sig[:, 3] = np.nan
    assert abs(info_coef(sig, fwd) - 1.0) < 1e-12, "IC of a perfect signal with a missing asset is not 1"
    # control: the checks do detect a wrong answer
    assert ops.cs_rank(np.array([[1.0, 2.0, 3.0, 4.0]]))[0, 0] == -0.5
    return "ties average, constant std is 0, noise gets no position, missing stays missing, drawdown and IC exact"

def test_hall_of_fame_skips_not_stops():
    """A positive-Sharpe formula ranked below a negative one by fitness must still be considered."""
    f = AlphaForge(synthetic_market(T=300, N=10, seed=1), Config(population=10, generations=0))
    p = np.random.default_rng(0).normal(0, 0.01, 50)
    f.trials = {"a": Trial("a", ("close", (), None), False, -0.01, -0.04, p), "b": Trial("b", ("open", (), None), False, 0.40, -0.20, -p)}
    hof = f.hall_of_fame()
    assert [t.expr for t in hof] == ["b"], f"hall of fame {[t.expr for t in hof]}"
    return "positive-Sharpe formula kept after a higher-fitness negative one"

if __name__ == "__main__":
    fails = 0
    for t in [test_no_look_ahead, test_simulator_has_no_future, test_zero_edge_market_is_zero_edge, test_operator_edge_cases,
              test_hall_of_fame_skips_not_stops, test_deflated_sharpe_on_noise, test_pbo_on_noise, test_tribunal_end_to_end]:
        t0 = time.time()
        try: print(f"PASS  {t.__name__}: {t()}  [{time.time()-t0:.0f}s]")
        except AssertionError as e: fails += 1; print(f"FAIL  {t.__name__}: {e}")
    sys.exit(1 if fails else 0)
