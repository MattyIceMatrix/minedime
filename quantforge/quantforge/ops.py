"""Operator library on (T days x N assets) panels. Every op looks only backward in time.
Blast radius: pure computation, touches no files, network, or processes."""
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


def _clean(x):
    """NaN and +/-inf both mean 'missing'."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, np.nan)


def _win(x, d):
    return _swv(_clean(x), d, axis=0)  # (T-d+1, N, d); any missing value in a window -> NaN result


def _pad(out, d):
    return np.vstack([np.full((d - 1, out.shape[1]), np.nan), out])


def ts_mean(x, d):   return _pad(_win(x, d).mean(-1), d)


def ts_std(x, d):
    """Population std over [t-d+1, t]. Variation below 1e-7 of the level counts as none, so a constant window gives
    exactly 0 rather than 1e-17 of rounding noise that a later division would blow up into a full-size signal."""
    w = _win(x, d); m = w.mean(-1); s = w.std(-1)
    return _pad(np.where(s <= 1e-7 * np.abs(m), 0.0, s), d)


def ts_max(x, d):    return _pad(_win(x, d).max(-1), d)
def ts_min(x, d):    return _pad(_win(x, d).min(-1), d)
def ts_sum(x, d):    return _pad(_win(x, d).sum(-1), d)


def ts_delay(x, d):
    out = np.full_like(x, np.nan)
    out[d:] = x[:-d]
    return out


def ts_delta(x, d):  x = _clean(x); return x - ts_delay(x, d)


def ts_rank(x, d):
    w = _win(x, d)
    r = (w < w[..., -1:]).sum(-1) / (d - 1)
    return _pad(np.where(np.isnan(w).any(-1), np.nan, r), d)  # missing data stays missing


def ts_zscore(x, d):
    s = ts_std(x, d)
    return np.where(s > 0, (_clean(x) - ts_mean(x, d)) / np.where(s > 0, s, 1.0), np.nan)


def ts_corr(x, y, d):
    wx, wy = _win(x, d), _win(y, d)
    bad = np.isnan(wx).any(-1) | np.isnan(wy).any(-1)
    mx, my = wx.mean(-1, keepdims=True), wy.mean(-1, keepdims=True)
    dx, dy = wx - mx, wy - my
    vx, vy = (dx ** 2).mean(-1), (dy ** 2).mean(-1)
    flat = (np.sqrt(vx) <= 1e-7 * np.abs(mx[..., 0])) | (np.sqrt(vy) <= 1e-7 * np.abs(my[..., 0]))
    c = np.clip((dx * dy).mean(-1) / np.sqrt(np.where(flat, 1.0, vx * vy)), -1, 1)
    return _pad(np.where(bad | flat, np.nan, c), d)


def cs_rank(x):
    """Cross-sectional rank scaled to [-0.5, 0.5]. Tied values share their average rank, so equal inputs never
    turn into a ranking by column order."""
    from scipy.stats import rankdata
    x = _clean(x)
    r = rankdata(x, method="average", axis=1, nan_policy="omit")
    n = np.isfinite(x).sum(1, keepdims=True)
    r = (r - 1) / np.maximum(n - 1, 1) - 0.5
    return np.where(np.isfinite(x), r, np.nan)


def cs_zscore(x):
    x = _clean(x); m = np.nanmean(x, 1, keepdims=True); s = np.nanstd(x, 1, keepdims=True)
    ok = (s > 1e-7 * np.abs(m)) & (s > 0) & (np.isfinite(x).sum(1, keepdims=True) >= 2)
    return np.where(ok, (x - m) / np.where(ok, s, 1.0), np.nan)


def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def div(a, b): return np.where(np.abs(b) > 1e-12, a / b, np.nan)
def neg(a):    return -a
def sabs(a):   return np.abs(a)
def slog(a):   return np.sign(a) * np.log1p(np.abs(a))
def sign(a):   return np.sign(a)

# name -> (function, number of child expressions, takes a lookback window)
OPS = {
    "ts_mean": (ts_mean, 1, True), "ts_std": (ts_std, 1, True),
    "ts_max": (ts_max, 1, True), "ts_min": (ts_min, 1, True),
    "ts_sum": (ts_sum, 1, True), "ts_delta": (ts_delta, 1, True),
    "ts_rank": (ts_rank, 1, True), "ts_zscore": (ts_zscore, 1, True),
    "ts_corr": (ts_corr, 2, True),
    "cs_rank": (cs_rank, 1, False), "cs_zscore": (cs_zscore, 1, False),
    "add": (add, 2, False), "sub": (sub, 2, False),
    "mul": (mul, 2, False), "div": (div, 2, False),
    "neg": (neg, 1, False), "abs": (sabs, 1, False),
    "slog": (slog, 1, False), "sign": (sign, 1, False),
}
WINDOWS = [3, 5, 10, 20, 40, 60]
