"""Vectorised cross-sectional long/short backtest with transaction costs.
Blast radius: pure computation, touches no files, network, or processes."""
import numpy as np

ANN = 252


def signal_to_weights(sig):
    """Dollar-neutral, gross exposure 1: long top, short bottom, proportional to signal.
    Non-finite values are missing. A day whose cross-sectional spread is rounding noise relative to the
    signal's level (or has fewer than 3 valid assets) carries no position instead of being blown up to full size."""
    x = np.where(np.isfinite(sig), sig, np.nan)
    n = np.isfinite(x).sum(1, keepdims=True)
    s = x - np.nanmean(x, 1, keepdims=True)
    spread = np.nansum(np.abs(s), 1, keepdims=True); level = np.nansum(np.abs(x), 1, keepdims=True)
    live = (n >= 3) & (spread > 1e-9 * level)
    s = np.where(live, s / np.where(live, spread, 1.0), 0.0)
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


def run(weights, fwd_ret, cost_bps=5.0):
    """weights[t] set at close t, earn fwd_ret[t] (close t -> close t+1). Returns daily net PnL."""
    gross = np.nansum(weights * np.nan_to_num(fwd_ret), 1)
    turnover = np.abs(np.diff(weights, axis=0, prepend=0.0)).sum(1)
    return gross - turnover * cost_bps / 1e4, turnover


def sharpe(pnl):
    if len(pnl) == 0: return 0.0
    sd = pnl.std()
    return 0.0 if sd == 0 or not np.isfinite(sd) else pnl.mean() / sd * np.sqrt(ANN)


def max_drawdown(pnl):
    eq = np.concatenate([[0.0], np.cumsum(pnl)])  # start from zero, so a first-day loss counts
    return float((np.maximum.accumulate(eq) - eq).max())


def info_coef(sig, fwd_ret):
    """Mean daily cross-sectional rank correlation between signal and next-day return."""
    from .ops import cs_rank
    ok = np.isfinite(sig) & np.isfinite(fwd_ret)          # rank both over the SAME valid assets each day
    a, b = cs_rank(np.where(ok, sig, np.nan)), cs_rank(np.where(ok, fwd_ret, np.nan))
    a, b = np.where(ok, a, 0), np.where(ok, b, 0)           # ranks are centred on 0 over the valid set
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    ic = num / den
    return float(np.nanmean(ic[np.isfinite(ic)]))
