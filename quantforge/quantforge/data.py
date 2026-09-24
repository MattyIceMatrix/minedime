"""Market data: a synthetic 'flight simulator' market with optionally planted alpha,
plus a loader for real wide-format CSVs (date column + one column per ticker).
Blast radius: load_wide_csv READS the CSV paths you pass it. Nothing is written."""
import os
import numpy as np


def _cz(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def synthetic_market(T=2520, N=50, seed=0, alpha_strength=1.0):
    """Fat tails, GARCH market volatility, persistent idiosyncratic vol regimes.
    alpha_strength=1 plants two weak real edges: 5-day reversal and 60-day momentum.
    alpha_strength=0 is a market with NO edge at all: the perfect overfitting trap.
    Day t depends only on draws up to day t, so extending T never changes earlier days."""
    rng = np.random.default_rng(seed)
    a_rev, a_mom = 0.035 * alpha_strength, 0.02 * alpha_strength
    beta = rng.uniform(0.6, 1.4, N)
    base_vol = rng.uniform(0.012, 0.028, N)
    ret = np.zeros((T, N)); hv = np.zeros(N); m_var = 1e-4; m_prev = 0.0
    lvol_state = np.zeros(N)
    logvol = np.zeros((T, N)); base_lv = rng.uniform(12, 16, N)
    for t in range(T):
        m_var = 2e-6 + 0.08 * m_prev ** 2 + 0.9 * m_var
        m = np.sqrt(m_var) * rng.standard_t(5) / np.sqrt(5 / 3)
        hv = 0.98 * hv + 0.2 * rng.standard_normal(N) * np.sqrt(1 - 0.98 ** 2) * 5
        sig = base_vol * np.exp(0.25 * np.tanh(hv))
        z = rng.standard_t(5, N) / np.sqrt(5 / 3)
        edge = np.zeros(N)
        if t >= 60:
            edge = a_rev * -_cz(ret[t - 5:t].sum(0)) + a_mom * _cz(ret[t - 60:t - 5].sum(0))
        # minus half the day's variance: every asset's expected SIMPLE return is zero (plus any planted edge).
        # Without it, higher-volatility assets drift up and a zero-edge market has an edge for long/short books.
        ret[t] = beta * m + sig * (z + edge) - 0.5 * (beta ** 2 * m_var + sig ** 2)
        lvol_state = 0.9 * lvol_state + 0.3 * rng.standard_normal(N)
        logvol[t] = base_lv + lvol_state + 0.6 * np.abs(z)
        m_prev = m
    close = 50 * np.exp(np.cumsum(ret, 0))
    prev = np.vstack([close[:1], close[:-1]])
    bar = np.random.default_rng([seed, 1]).standard_normal((T, 2, N))  # own stream, filled day by day
    opn = prev * np.exp(0.3 * ret + 0.002 * bar[:, 0])
    rng_hl = np.abs(bar[:, 1]) * base_vol * 0.5
    high = np.maximum(opn, close) * np.exp(rng_hl)
    low = np.minimum(opn, close) * np.exp(-rng_hl)
    vwap = (high + low + close) / 3
    return _finish(dict(close=close, open=opn, high=high, low=low,
                        volume=np.exp(logvol), vwap=vwap))


def _finish(d):
    c = d["close"]
    d["returns"] = np.vstack([np.full((1, c.shape[1]), np.nan), c[1:] / c[:-1] - 1])
    fwd = np.full_like(c, np.nan)
    fwd[:-1] = c[1:] / c[:-1] - 1
    d["_fwd"] = fwd  # target, never visible to formulas
    return d


def load_wide_csv(close_csv, volume_csv=None):
    """Real data path. Each CSV: first column date, then one column per ticker, same order."""
    for p in [close_csv, volume_csv]:
        if p is not None and not os.path.isfile(p):
            raise FileNotFoundError(f"QuantForge: data file not found: {p}")
    read = lambda p: np.genfromtxt(p, delimiter=",", skip_header=1)[:, 1:]
    d = {"close": read(close_csv)}
    if volume_csv:
        d["volume"] = read(volume_csv)
        if d["volume"].shape != d["close"].shape:
            raise ValueError("QuantForge: close and volume CSVs have different shapes")
    return _finish(d)
