"""Operator library on (T days x N assets) panels. Every op looks only backward in time.
Blast radius: pure computation, touches no files, network, or processes."""
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


def _win(x, d):
    return _swv(x, d, axis=0)  # (T-d+1, N, d)


def _pad(out, d):
    return np.vstack([np.full((d - 1, out.shape[1]), np.nan), out])


def ts_mean(x, d):   return _pad(_win(x, d).mean(-1), d)
def ts_std(x, d):    return _pad(_win(x, d).std(-1), d)
def ts_max(x, d):    return _pad(_win(x, d).max(-1), d)
def ts_min(x, d):    return _pad(_win(x, d).min(-1), d)
def ts_sum(x, d):    return _pad(_win(x, d).sum(-1), d)


def ts_delay(x, d):
    out = np.full_like(x, np.nan)
    out[d:] = x[:-d]
    return out


def ts_delta(x, d):  return x - ts_delay(x, d)


def ts_rank(x, d):
    w = _win(x, d)
    return _pad((w < w[..., -1:]).sum(-1) / (d - 1), d)


def ts_zscore(x, d):
    return (x - ts_mean(x, d)) / ts_std(x, d)


def ts_corr(x, y, d):
    wx, wy = _win(x, d), _win(y, d)
    dx = wx - wx.mean(-1, keepdims=True)
    dy = wy - wy.mean(-1, keepdims=True)
    c = (dx * dy).sum(-1) / np.sqrt((dx ** 2).sum(-1) * (dy ** 2).sum(-1))
    return _pad(c, d)


def cs_rank(x):
    nan = np.isnan(x)
    r = np.argsort(np.argsort(np.where(nan, np.inf, x), axis=1), axis=1).astype(float)
    n = (~nan).sum(1, keepdims=True)
    r = r / np.maximum(n - 1, 1) - 0.5
    r[nan] = np.nan
    return r


def cs_zscore(x):
    return (x - np.nanmean(x, 1, keepdims=True)) / np.nanstd(x, 1, keepdims=True)


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
