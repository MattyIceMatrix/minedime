"""Vectorised cross-sectional long/short backtest with transaction costs.
Blast radius: pure computation, touches no files, network, or processes."""
import numpy as np

ANN = 252


def signal_to_weights(sig):
    """Dollar-neutral, gross exposure 1: long top, short bottom, proportional to signal."""
    s = sig - np.nanmean(sig, 1, keepdims=True)
    s = s / np.nansum(np.abs(s), 1, keepdims=True)
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


def run(weights, fwd_ret, cost_bps=5.0):
    """weights[t] set at close t, earn fwd_ret[t] (close t -> close t+1). Returns daily net PnL."""
    gross = np.nansum(weights * np.nan_to_num(fwd_ret), 1)
    turnover = np.abs(np.diff(weights, axis=0, prepend=0.0)).sum(1)
    return gross - turnover * cost_bps / 1e4, turnover


def sharpe(pnl):
    sd = pnl.std()
    return 0.0 if sd == 0 or not np.isfinite(sd) else pnl.mean() / sd * np.sqrt(ANN)


def max_drawdown(pnl):
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def info_coef(sig, fwd_ret):
    """Mean daily cross-sectional rank correlation between signal and next-day return."""
    from .ops import cs_rank
    a, b = cs_rank(sig), cs_rank(fwd_ret)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = np.where(ok, a, 0), np.where(ok, b, 0)
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    ic = num / den
    return float(np.nanmean(ic[np.isfinite(ic)]))
