"""The tribunal: statistics that try to prove a backtest is a fluke.
Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) and Probability of Backtest
Overfitting via Combinatorially Symmetric Cross-Validation (Bailey et al., 2017).
Blast radius: pure computation, touches no files, network, or processes."""
from itertools import combinations
import numpy as np
from scipy.stats import norm, skew, kurtosis

EULER = 0.5772156649


def expected_max_sharpe(n_trials, var_sr):
    """Sharpe you'd expect from the luckiest of n_trials strategies that have zero true skill."""
    n = max(n_trials, 2)
    return np.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / n)
                              + EULER * norm.ppf(1 - 1 / (n * np.e)))


def autocorr_inflation(pnl, q=10):
    """Newey-West factor: how much autocorrelated P&L overstates Sharpe precision (Lo, 2002)."""
    x = pnl - pnl.mean(); v = (x * x).sum()
    if v == 0: return 1.0
    eta = 1 + sum(2 * (1 - k / (q + 1)) * (x[k:] * x[:-k]).sum() / v for k in range(1, q + 1))
    return float(max(eta, 1.0))


def deflated_sharpe(pnl, n_trials):
    """Probability that the true Sharpe exceeds the best-of-n-trials luck benchmark.
    Null: zero skill, per-period Sharpe variance ~ eta/T. Sharpe here is daily."""
    eta = autocorr_inflation(pnl); T_eff = len(pnl) / eta
    sr = pnl.mean() / pnl.std()
    sr0 = expected_max_sharpe(n_trials, 1.0 / T_eff)
    g3, g4 = skew(pnl), kurtosis(pnl, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0) * np.sqrt(T_eff - 1) / denom)), float(sr0)


def pbo_cscv(pnl_matrix, n_blocks=10):
    """pnl_matrix: (T, M) daily PnL of M candidate strategies.
    Returns the fraction of splits where the in-sample winner lands in the bottom
    half out-of-sample. ~0.5 means the selection process is picking noise."""
    T, M = pnl_matrix.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    logits = []
    for is_idx in combinations(range(n_blocks), n_blocks // 2):
        oos_idx = [b for b in range(n_blocks) if b not in is_idx]
        IS = pnl_matrix[np.concatenate([blocks[b] for b in is_idx])]
        OS = pnl_matrix[np.concatenate([blocks[b] for b in oos_idx])]
        sr_is = IS.mean(0) / (IS.std(0) + 1e-12)
        sr_os = OS.mean(0) / (OS.std(0) + 1e-12)
        best = int(np.argmax(sr_is))
        rank = (sr_os < sr_os[best]).sum() + 1  # 1..M
        w = rank / (M + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return float((logits <= 0).mean())


def effective_trials(pnl_matrix, var_explained=0.95, max_cols=1500, seed=0):
    """How many genuinely independent bets did the search make? Near-clone formulas
    share one bet. Counts correlation-matrix eigen-directions covering 95% of variance
    (deliberately stricter than the participation ratio)."""
    P = pnl_matrix
    if P.shape[1] > max_cols:
        P = P[:, np.random.default_rng(seed).choice(P.shape[1], max_cols, replace=False)]
    C = np.nan_to_num(np.corrcoef(P.T))
    lam = np.sort(np.clip(np.linalg.eigvalsh(C), 0, None))[::-1]
    return int(np.searchsorted(np.cumsum(lam) / lam.sum(), var_explained) + 1)
