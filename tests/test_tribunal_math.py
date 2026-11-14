"""QuantForge tribunal maths: does the statistics actually do what the README claims?

The README makes three specific, checkable methodological claims about the Alpha Lab:

  1. Deflated Sharpe ratio "against the best score chance would give after the search's
     EFFECTIVE number of independent tries" (not the raw formula count).
  2. Replication on the validation stretch "at family-wise 5% (Bonferroni)" - one error
     budget shared across the candidates tested, not 5% each.
  3. Probability of backtest overfitting reported "across every formula on the VALIDATION
     stretch", which the search never optimised on.

Each test below pins one of those claims, prefers a case whose answer is knowable on paper,
and carries a control so a gate that simply rejects (or accepts) everything fails it.

Run from the repo root:  python3 tests/test_tribunal_math.py
Blast radius: pure computation in memory. Reads and writes no files. Needs numpy + scipy.
"""
import os, sys, time
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantforge"))
from quantforge.data import synthetic_market

from quantforge import engine as E
from quantforge.engine import AlphaForge, Config
from quantforge.validate import (expected_max_sharpe, deflated_sharpe, pbo_cscv,
                                 effective_trials, autocorr_inflation)

GAMMA = 0.5772156649015329   # Euler-Mascheroni, to full double precision


def _iid_pnl_with_sharpe(sr, T=20000, seed=0):
    """A P&L series with (near) zero skew, kurtosis 3, no autocorrelation and an exact
    daily Sharpe of `sr`. Built from normal quantiles, then shuffled so eta == 1."""
    z = norm.ppf((np.arange(1, T + 1) - 0.5) / T)
    z = (z - z.mean()) / z.std()
    np.random.default_rng(seed).shuffle(z)
    return 0.01 * (z + sr)


# --- claim 1: the luck bar -----------------------------------------------------------
def test_luck_bar_matches_the_published_formula():
    """E[max Sharpe of n zero-skill trials] = sqrt(V) * ((1-g) Z^-1(1-1/n) + g Z^-1(1-1/(n e)))
    (Bailey & Lopez de Prado 2014). Recompute it from scratch and compare."""
    worst = 0.0
    for n in [2, 5, 10, 50, 100, 356, 1000, 2852, 10000]:
        ref = (1 - GAMMA) * norm.ppf(1 - 1 / n) + GAMMA * norm.ppf(1 - 1 / (n * np.e))
        worst = max(worst, abs(expected_max_sharpe(n, 1.0) - ref))
    assert worst < 1e-9, f"luck bar differs from the published formula by {worst:.2e}"
    # scales as sqrt of the Sharpe variance, and rises with the number of tries
    bars = [expected_max_sharpe(n, 1.0 / 1000) for n in [10, 100, 1000, 10000]]
    assert all(b < c for b, c in zip(bars, bars[1:])), f"luck bar not increasing in n: {bars}"
    assert abs(expected_max_sharpe(50, 4.0) - 2 * expected_max_sharpe(50, 1.0)) < 1e-12
    # control: the test can tell the real formula from the usual sqrt(2 ln n) shortcut
    assert abs(expected_max_sharpe(1000, 1.0) - np.sqrt(2 * np.log(1000))) > 0.2
    return f"matches Bailey & Lopez de Prado (2014) to {worst:.1e}; monotone in n; scales as sqrt(V)"


def test_deflated_sharpe_is_exactly_one_half_at_the_luck_bar():
    """A strategy whose Sharpe sits exactly on the luck bar has a 50/50 case. Knowable on paper:
    the DSR argument is (sr - sr0) * sqrt(T_eff - 1) / denom, so sr == sr0 gives Phi(0) = 0.5."""
    n_eff, T = 356, 20000
    bar = expected_max_sharpe(n_eff, 1.0 / T)
    at = deflated_sharpe(_iid_pnl_with_sharpe(bar, T), n_eff)[0]
    assert abs(at - 0.5) < 1e-6, f"DSR at the luck bar is {at:.8f}, expected 0.5"
    # control: it moves the right way, and only the right way
    above = deflated_sharpe(_iid_pnl_with_sharpe(bar * 1.5, T), n_eff)[0]
    below = deflated_sharpe(_iid_pnl_with_sharpe(bar * 0.5, T), n_eff)[0]
    assert above > 0.9 and below < 0.1, f"DSR not responsive: {below:.3f} / {at:.3f} / {above:.3f}"
    return f"sr == luck bar gives DSR {at:.8f} (exactly 0.5); 1.5x the bar gives {above:.3f}, half the bar {below:.3f}"


def test_deflated_sharpe_tightens_as_the_search_widens():
    """More tries must raise the bar and lower the verdict for the SAME P&L. A gate that
    ignored the trial count (the classic mistake) would return a flat line here."""
    pnl = _iid_pnl_with_sharpe(0.06, 4000, seed=3)
    out = [(n, *deflated_sharpe(pnl, n)) for n in [2, 10, 100, 1000, 10000]]
    assert all(a[1] > b[1] for a, b in zip(out, out[1:])), f"DSR not falling with n: {out}"
    assert all(a[2] < b[2] for a, b in zip(out, out[1:])), f"luck bar not rising with n: {out}"
    assert out[0][1] - out[-1][1] > 0.3, f"trial count barely matters: {out[0][1]:.3f} -> {out[-1][1]:.3f}"
    return "DSR " + " > ".join(f"{d:.3f}" for _, d, _ in out) + " for n = 2, 10, 100, 1k, 10k"


def test_autocorrelated_pnl_is_penalised():
    """Sticky P&L carries less information than its length suggests (Lo 2002). An AR(1) with
    phi = 0.6 has long-run variance ratio (1+phi)/(1-phi) = 4; the Bartlett window at q = 10
    recovers most of it. The same Sharpe on sticky P&L must score lower."""
    rng = np.random.default_rng(11)
    T, phi = 60000, 0.6
    e = rng.standard_normal(T); ar = np.zeros(T)
    for t in range(1, T):
        ar[t] = phi * ar[t - 1] + e[t]
    eta = autocorr_inflation(ar)
    assert abs(autocorr_inflation(rng.standard_normal(T)) - 1.0) < 0.05, "iid P&L should not be inflated"
    assert 3.0 < eta < 4.0, f"AR(1) phi=0.6 inflation {eta:.2f}, expected just under (1+phi)/(1-phi)=4"
    # Same daily Sharpe, same length, same trial count - the only difference is the stickiness.
    # Keep the Sharpe modest so neither verdict saturates and the comparison is visible.
    sticky = 0.01 * (ar / ar.std() + 0.012)
    sr = sticky.mean() / sticky.std()
    iid = _iid_pnl_with_sharpe(sr, T, seed=4)
    d_sticky, bar_sticky = deflated_sharpe(sticky, 500)
    d_iid, bar_iid = deflated_sharpe(iid, 500)
    assert d_iid - d_sticky > 0.1, f"sticky P&L barely penalised: {d_sticky:.4f} vs iid {d_iid:.4f}"
    assert bar_sticky > bar_iid, "autocorrelation should raise the luck bar"
    return (f"iid eta 1.00, AR(1) phi=0.6 eta {eta:.2f} (theory 4.0, Bartlett q=10 truncates); "
            f"daily Sharpe {sr:.4f} scores DSR {d_sticky:.3f} sticky vs {d_iid:.3f} iid")


def test_effective_trials_counts_directions_not_formulas():
    """Near-clone formulas are one bet, not many. Structural cases with a knowable answer."""
    rng = np.random.default_rng(0); T = 1200
    clones = np.tile(rng.standard_normal(T)[:, None], (1, 300))
    assert effective_trials(clones) == 1, "300 copies of one formula should count as 1 bet"
    for k, per in [(1, 50), (4, 25), (12, 20)]:
        cols = []
        for _ in range(k):
            b = rng.standard_normal(T)
            cols += [b + 0.02 * rng.standard_normal(T) for _ in range(per)]
        got = effective_trials(np.column_stack(cols))
        assert got == k, f"{k} families of {per} near-clones counted as {got}, expected {k}"
    # a population of independent columns is NOT collapsed
    assert effective_trials(rng.standard_normal((T, 40))) >= 35, "independent columns collapsed"
    return "300 clones -> 1 bet; 1/4/12 near-clone families -> 1/4/12 bets; 40 independent columns stay ~40"


# --- a small real search, shared by the three claim tests below ----------------------
def _small_forge():
    cfg = Config(population=60, generations=4)
    f = AlphaForge(synthetic_market(T=1400, N=30, seed=1, alpha_strength=1.0), cfg)
    f.evolve(log=lambda *a: None)
    return f


_FORGE = None


def _forge():
    global _FORGE
    if _FORGE is None:
        _FORGE = _small_forge()
    return _FORGE


def test_tribunal_deflates_by_effective_trials_not_the_raw_count():
    """README claim 1. The tribunal must hand the DSR the EFFECTIVE number of independent
    tries (x2, because every formula is tested long and short), not len(trials)."""
    f = _forge()
    trials = list(f.trials.values())
    P = np.column_stack([t.pnl_is for t in trials])
    expected = 2 * effective_trials(P)
    _, _, n_eff, _ = f.tribunal(f.hall_of_fame())
    assert n_eff == expected, f"tribunal used n_eff={n_eff}, effective count says {expected}"
    assert n_eff != 2 * len(trials) and n_eff != len(trials), \
        f"n_eff={n_eff} looks like the raw trial count ({len(trials)}), not the effective one"
    assert n_eff < len(trials), f"n_eff {n_eff} should be well below the {len(trials)} formulas tried"
    return (f"{len(trials)} formulas tried, {len(trials)*2} directions, deflated by "
            f"n_eff={n_eff} independent bets (= 2 x {expected//2})")


def test_replication_gate_is_family_wise_bonferroni():
    """README claim 2. The validation test spends ONE 5% error budget across the candidates
    that reached it: z_crit = Phi^-1(1 - 0.05/k), k = number of DSR survivors. Per-formula 5%
    (k = 1, z = 1.645) would be wrong, and the control below shows the test can tell them apart."""
    f = _forge()
    verdicts, _, _, z_crit = f.tribunal(f.hall_of_fame())
    k = max(sum(v["dsr_pass"] for v in verdicts), 1)
    assert abs(z_crit - norm.ppf(1 - 0.05 / k)) < 1e-12, \
        f"z_crit {z_crit:.4f} is not Bonferroni at k={k} (expected {norm.ppf(1-0.05/k):.4f})"
    assert all(v["passed"] == (v["dsr_pass"] and v["val_z"] >= z_crit) for v in verdicts), \
        "a verdict does not follow the two gates it claims to apply"
    # Force every hall-of-fame candidate through gate 1 and the family must grow with it.
    # Per-formula 5% would leave z_crit at 1.645 no matter how many candidates were tested.
    real_dsr = E.deflated_sharpe
    try:
        E.deflated_sharpe = lambda pnl, n: (1.0, real_dsr(pnl, n)[1])
        hof = f.hall_of_fame()
        v_all, _, _, z_all = f.tribunal(hof)
        n = len(hof)
        assert sum(v["dsr_pass"] for v in v_all) == n, "the stub did not open gate 1"
        assert abs(z_all - norm.ppf(1 - 0.05 / n)) < 1e-12, \
            f"with {n} candidates through gate 1, z_crit is {z_all:.4f}, not {norm.ppf(1-0.05/n):.4f}"
        assert z_all > norm.ppf(0.95) + 0.1, \
            f"z_crit {z_all:.4f} did not rise above the uncorrected 1.645 for a family of {n}"
    finally:
        E.deflated_sharpe = real_dsr
    # The property itself, checked by simulation: k null candidates on independent validation
    # data must trigger a family-wise false approval at most 5% of the time.
    rng = np.random.default_rng(1)
    reps, fam = 4000, 6
    z = rng.standard_normal((reps, fam))                     # k null z-statistics per family
    fw_bonf = (z.max(1) >= norm.ppf(1 - 0.05 / fam)).mean()
    fw_each = (z.max(1) >= norm.ppf(0.95)).mean()            # control: uncorrected
    assert fw_bonf <= 0.05, f"Bonferroni family-wise rate {fw_bonf:.3f} exceeds 5%"
    assert fw_each > 0.2, f"control failed: uncorrected rate {fw_each:.3f} should be near {1-0.95**fam:.2f}"
    return (f"z_crit {z_crit:.3f} == Phi^-1(1 - 0.05/{k}); forcing all {n} candidates through gate 1 "
            f"moves it to {z_all:.3f} == Phi^-1(1 - 0.05/{n}); simulated family-wise rate over {fam} "
            f"candidates {fw_bonf:.1%} corrected vs {fw_each:.1%} uncorrected")


def test_pbo_is_measured_on_the_validation_stretch():
    """README claim 3. Catch the matrix on its way into the PBO and prove it is the validation
    stretch - the days between vstart and split - and not the search stretch or the vault."""
    f = _forge()
    seen = {}
    real = E.pbo_cscv

    def spy(matrix, *a, **kw):
        seen["shape"] = matrix.shape
        seen["matrix"] = matrix.copy()
        return real(matrix, *a, **kw)

    E.pbo_cscv = spy
    try:
        _, pbo, _, _ = f.tribunal(f.hall_of_fame())
    finally:
        E.pbo_cscv = real
    want_rows = f.split - f.vstart
    assert seen["shape"][0] == want_rows, \
        f"PBO ran on {seen['shape'][0]} days; the validation stretch is {want_rows}"
    assert seen["shape"][1] == len(f.trials), \
        f"PBO covered {seen['shape'][1]} formulas, not all {len(f.trials)}"
    search_days = f.vstart - f.warm
    vault_days = len(f.fwd) - f.split
    assert seen["shape"][0] not in (search_days, vault_days), "PBO stretch is the search window or the vault"
    # and the columns really are validation-stretch P&L, recomputed here from scratch
    t = next(iter(f.trials.values()))
    ref, _ = f._pnl(f._weights(t), slice(f.vstart, f.split))
    col = seen["matrix"][:, list(f.trials.values()).index(t)]
    assert np.allclose(col, ref, atol=1e-12), "PBO column is not the validation-stretch P&L"
    assert 0.0 <= pbo <= 1.0, f"PBO out of range: {pbo}"
    return (f"PBO ran on {seen['shape'][0]} validation days x {seen['shape'][1]} formulas "
            f"(search stretch is {search_days} days, vault {vault_days}); columns match a fresh recompute")


def test_pbo_endpoints_are_exact():
    """Two constructions whose PBO is knowable on paper.

    Zero: one strategy beats every rival in every block, so the in-sample winner is also the
    out-of-sample best in all 252 splits - never in the bottom half.

    One: strategy m earns its whole P&L inside block m alone. The in-sample winner is always a
    strategy whose one good block sits in the in-sample half, so out of sample it has nothing
    left while the five strategies whose blocks landed out of sample do well. The winner is in
    the bottom half every single split.
    """
    rng = np.random.default_rng(2); T, M = 800, 12
    noise = rng.standard_normal((T, M)) * 0.01
    champ = noise.copy(); champ[:, 0] += 0.05
    assert pbo_cscv(champ) == 0.0, f"a dominant strategy gave PBO {pbo_cscv(champ):.3f}, expected exactly 0"
    blocks = np.array_split(np.arange(T), 10)
    trap = rng.standard_normal((T, 10)) * 0.01
    for m in range(10):
        trap[blocks[m], m] += 0.08
    assert pbo_cscv(trap) == 1.0, f"a one-block-wonder population gave PBO {pbo_cscv(trap):.3f}, expected exactly 1"
    mid = pbo_cscv(rng.standard_normal((T, M)))
    assert 0.3 < mid < 0.7, f"PBO on pure noise {mid:.3f}, expected about 0.5"
    return "dominant strategy PBO exactly 0.00, one-block-wonders PBO exactly 1.00, pure noise " + f"{mid:.2f}"


def test_the_vault_stays_sealed():
    """README: "The last 30% of history stays sealed until the verdict." Scramble every price
    after the split and the search, the hall of fame and both gates must come out bit-identical.
    Control: the vault report itself must change, or the scramble did nothing."""
    cfg = Config(population=40, generations=2)
    D = synthetic_market(T=1200, N=20, seed=5, alpha_strength=1.0)
    S = {k: (v.copy() if not k.startswith("_") else v.copy()) for k, v in D.items()}
    f0 = AlphaForge(D, cfg)
    rng = np.random.default_rng(99)
    cut = f0.split
    for k in ["close", "open", "high", "low", "volume", "vwap", "returns", "_fwd"]:
        S[k][cut:] *= rng.uniform(0.5, 1.5, S[k][cut:].shape)
    fa, fb = AlphaForge(D, cfg), AlphaForge(S, cfg)
    fa.evolve(log=lambda *a: None); fb.evolve(log=lambda *a: None)
    assert sorted(fa.trials) == sorted(fb.trials), "the search itself saw the vault"
    va, _, na, za = fa.tribunal(fa.hall_of_fame())
    vb, _, nb, zb = fb.tribunal(fb.hall_of_fame())
    assert (na, za) == (nb, zb), f"trial count or critical value moved: {(na, za)} vs {(nb, zb)}"
    assert [v["trial"].expr for v in va] == [v["trial"].expr for v in vb], "hall of fame changed"
    for a, b in zip(va, vb):
        assert a["dsr"] == b["dsr"] and a["val_z"] == b["val_z"] and a["passed"] == b["passed"], \
            f"verdict for {a['trial'].expr} changed when the vault was scrambled"
    oa, _ = fa.oos_report(fa.hall_of_fame()[:1])
    ob, _ = fb.oos_report(fb.hall_of_fame()[:1])
    assert abs(oa[0]["oos_sharpe"] - ob[0]["oos_sharpe"]) > 1e-9, \
        "control failed: scrambling the vault did not change the vault report either"
    return (f"{len(fa.trials)} formulas, {len(va)} verdicts, n_eff {na} and z_crit {za:.3f} all identical "
            f"after scrambling the sealed 30%; vault Sharpe moved {oa[0]['oos_sharpe']:.2f} -> {ob[0]['oos_sharpe']:.2f}")


def test_pure_luck_does_not_survive_both_gates():
    """The negative case the whole tribunal exists for. Search M zero-skill strategies, crown the
    in-sample best, then run it through gate 1 (deflated Sharpe at the true trial count) and gate 2
    (Bonferroni replication on fresh data). Luck must almost never get through. Control: a strategy
    with genuine skill must get through most of the time, so a gate that rejects everything fails."""
    rng = np.random.default_rng(13)
    reps, M, T_is, T_val = 400, 60, 1000, 500
    approved = skilled_approved = 0
    z_crit = norm.ppf(1 - 0.05 / 1)     # one candidate reaches replication, so k = 1
    for _ in range(reps):
        IS = rng.standard_t(5, size=(T_is, M)) / np.sqrt(5 / 3) * 0.01
        best = int(np.argmax(IS.mean(0) / IS.std(0)))
        if deflated_sharpe(IS[:, best], M)[0] >= 0.95:
            v = rng.standard_t(5, size=T_val) / np.sqrt(5 / 3) * 0.01   # no edge on fresh data
            if v.mean() / v.std() * np.sqrt(T_val / autocorr_inflation(v)) >= z_crit:
                approved += 1
        real = rng.standard_t(5, size=T_is) / np.sqrt(5 / 3) * 0.01 + 0.002   # daily SR 0.2
        if deflated_sharpe(real, M)[0] >= 0.95:
            v = rng.standard_t(5, size=T_val) / np.sqrt(5 / 3) * 0.01 + 0.002
            if v.mean() / v.std() * np.sqrt(T_val / autocorr_inflation(v)) >= z_crit:
                skilled_approved += 1
    assert approved / reps <= 0.05, f"pure luck approved {approved/reps:.1%} of the time"
    assert skilled_approved / reps >= 0.8, \
        f"control failed: real skill approved only {skilled_approved/reps:.0%} of the time"
    return (f"best of {M} zero-skill strategies approved {approved/reps:.2%} of {reps} searches; "
            f"genuine skill approved {skilled_approved/reps:.0%}")


TESTS = [test_luck_bar_matches_the_published_formula,
         test_deflated_sharpe_is_exactly_one_half_at_the_luck_bar,
         test_deflated_sharpe_tightens_as_the_search_widens,
         test_autocorrelated_pnl_is_penalised,
         test_effective_trials_counts_directions_not_formulas,
         test_pbo_endpoints_are_exact,
         test_the_vault_stays_sealed,
         test_pure_luck_does_not_survive_both_gates,
         test_tribunal_deflates_by_effective_trials_not_the_raw_count,
         test_replication_gate_is_family_wise_bonferroni,
         test_pbo_is_measured_on_the_validation_stretch]

if __name__ == "__main__":
    fails = 0
    for t in TESTS:
        t0 = time.time()
        try:
            print(f"PASS  {t.__name__}: {t()}  [{time.time()-t0:.0f}s]")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {t.__name__}: {e}")
    sys.exit(1 if fails else 0)
