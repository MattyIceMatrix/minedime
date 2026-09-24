"""AlphaForge engine: genetic-programming search (the attacker) + validation gate (the tribunal).
Timeline: search | validation (replication test only) | sealed vault (final report only).
Blast radius: pure computation, touches no files, network, or processes."""
from dataclasses import dataclass, field
import time
import numpy as np
from . import alpha as A
from .backtest import signal_to_weights, sharpe, max_drawdown, info_coef, ANN
from scipy.stats import norm
from .validate import deflated_sharpe, pbo_cscv, effective_trials, autocorr_inflation


@dataclass
class Config:
    population: int = 300
    generations: int = 25
    max_depth: int = 4
    tournament: int = 4
    elite: int = 10
    cost_bps: float = 5.0
    oos_frac: float = 0.30        # sealed vault
    val_frac: float = 0.20        # replication stretch, never optimized on
    family_alpha: float = 0.05    # family-wise error for the replication test
    complexity_penalty: float = 0.03
    max_corr: float = 0.5         # diversity rule for the hall of fame
    hall_size: int = 8
    dsr_threshold: float = 0.95
    min_coverage: float = 0.8
    seed: int = 7


@dataclass
class Trial:
    expr: str
    node: tuple
    flip: bool
    is_sharpe: float
    fitness: float
    pnl_is: np.ndarray = field(repr=False)
    turnover: float = 0.0
    pnl_val: np.ndarray = field(default=None, repr=False)  # validation stretch; only the PBO diagnostic reads it


class AlphaForge:
    def __init__(self, data, cfg=None):
        cfg = cfg if cfg is not None else Config()  # fresh Config per engine; never a shared default
        self.d, self.cfg = data, cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.terms = [k for k in data if not k.startswith("_")]
        T = data["close"].shape[0]
        self.split = int(T * (1 - cfg.oos_frac))
        self.vstart = int(T * (1 - cfg.oos_frac - cfg.val_frac))
        self.warm = 120  # burn-in so long windows are populated
        self.cache, self.trials = {}, {}
        self.fwd = data["_fwd"]

    # --- scoring --------------------------------------------------------------
    def _pnl(self, w, sl, flip=False):
        fwd = np.nan_to_num(self.fwd[sl])
        gross = (w[sl] * fwd).sum(1)
        to = np.abs(np.diff(w, axis=0, prepend=0.0)).sum(1)[sl]
        cost = to * self.cfg.cost_bps / 1e4
        return (-gross if flip else gross) - cost, to.mean()

    def _score(self, node):
        key = A.to_str(node)
        if key in self.trials:
            return self.trials[key]
        try:
            sig = A.evaluate(node, self.d, self.cache)
        except Exception:
            return None
        IS = slice(self.warm, self.vstart)  # search sees only this
        s = sig[IS]
        if np.isfinite(s).mean() < self.cfg.min_coverage:
            return None
        sd = np.nanstd(np.where(np.isfinite(s), s, np.nan))
        if not sd > max(1e-12, 1e-9 * np.nanmean(np.abs(np.where(np.isfinite(s), s, np.nan)))):
            return None  # constant, or only rounding noise
        w = signal_to_weights(sig)
        # Every formula is tested in both directions; keep the better one.
        # (Both directions count as tested, which the tribunal accounts for below.)
        pnl, to = self._pnl(w, IS, False)
        pnl_f, _ = self._pnl(w, IS, True)
        flip = pnl_f.mean() > pnl.mean()
        if flip:
            pnl = pnl_f
        sr = sharpe(pnl)
        fit = sr - self.cfg.complexity_penalty * A.size(node)
        val, _ = self._pnl(w, slice(self.vstart, self.split), flip)
        t = Trial(key, node, flip, sr, fit, pnl, to, val)
        self.trials[key] = t
        return t

    # --- search ---------------------------------------------------------------
    def _tournament(self, pop):
        idx = self.rng.integers(len(pop), size=self.cfg.tournament)
        return max((pop[i] for i in idx), key=lambda t: t.fitness)

    def evolve(self, log=print):
        c = self.cfg
        pop = []
        while len(pop) < c.population:
            t = self._score(A.grow(self.rng, self.terms, c.max_depth))
            if t: pop.append(t)
        for g in range(c.generations):
            t0 = time.time()
            self.cache.clear()
            pop.sort(key=lambda t: -t.fitness)
            nxt = pop[:c.elite]
            tries = 0
            while len(nxt) < c.population and tries < c.population * 5:
                tries += 1
                if self.rng.random() < 0.5:
                    child = A.crossover(self.rng, self._tournament(pop).node,
                                        self._tournament(pop).node, c.max_depth)
                else:
                    child = A.mutate(self.rng, self._tournament(pop).node,
                                     self.terms, c.max_depth)
                t = self._score(child)
                if t: nxt.append(t)
            pop = nxt
            best = max(pop, key=lambda t: t.fitness)
            log(f"  gen {g+1:>2}/{c.generations}  formulas tested: {len(self.trials):>5}  "
                f"best in-sample Sharpe {best.is_sharpe:5.2f}  ({time.time()-t0:4.1f}s)")
        return pop

    def hall_of_fame(self):
        ranked = sorted(self.trials.values(), key=lambda t: -t.fitness)
        hof = []
        for t in ranked:
            if not t.is_sharpe > 0: continue  # skip; a later, lower-fitness formula may still be positive
            if all(abs(np.corrcoef(t.pnl_is, h.pnl_is)[0, 1]) < self.cfg.max_corr for h in hof):
                hof.append(t)
            if len(hof) >= self.cfg.hall_size: break
        return hof

    # --- tribunal -------------------------------------------------------------
    def _weights(self, t):
        sig = A.evaluate(t.node, self.d, self.cache)
        return signal_to_weights(-sig if t.flip else sig)

    def tribunal(self, hof):
        """Two gates. 1) Deflated Sharpe on the search stretch: beat the best Sharpe pure
        luck gives after this many independent tries, with 95% confidence (autocorrelation-
        aware). 2) Replication: DSR survivors must be significant on the validation stretch,
        which the search never optimized on, at family-wise 5% (Bonferroni).
        Plus a diagnostic, not a gate: PBO over every trial on the validation stretch. On the search
        stretch it would be biased toward 'not overfit', because the search chose its trials on that
        same data; on the shorter validation stretch it is too noisy for a hard threshold."""
        trials = list(self.trials.values())
        P = np.column_stack([t.pnl_is for t in trials])
        n_eff = 2 * effective_trials(P)  # x2: each formula was tried long and short
        verdicts = []
        for t in hof:
            dsr, sr0 = deflated_sharpe(t.pnl_is, n_eff)
            verdicts.append(dict(trial=t, dsr=dsr, luck_bar_annual=sr0 * np.sqrt(ANN),
                                 dsr_pass=dsr >= self.cfg.dsr_threshold))
        k = max(sum(v["dsr_pass"] for v in verdicts), 1)
        z_crit = float(norm.ppf(1 - self.cfg.family_alpha / k))
        V = slice(self.vstart, self.split)
        for v in verdicts:
            p, _ = self._pnl(self._weights(v["trial"]), V)
            v["val_sharpe"] = sharpe(p)
            v["val_z"] = float(p.mean() / p.std() * np.sqrt(len(p) / autocorr_inflation(p)))
            v["passed"] = v["dsr_pass"] and v["val_z"] >= z_crit
        pbo = pbo_cscv(np.column_stack([t.pnl_val for t in trials])) if len(trials) > 1 else float("nan")
        return verdicts, pbo, n_eff, z_crit

    # --- deployment -----------------------------------------------------------
    def oos_report(self, trials):
        """Open the sealed vault: evaluate chosen alphas and their ensemble on unseen data."""
        OOS = slice(self.split, len(self.fwd) - 1)
        out, ws = [], []
        for t in trials:
            sig = A.evaluate(t.node, self.d, self.cache)
            w = self._weights(t)
            pnl, to = self._pnl(w, OOS)
            ic = info_coef(-sig[OOS] if t.flip else sig[OOS], self.fwd[OOS])
            out.append(dict(expr=t.expr, flip=t.flip, is_sharpe=t.is_sharpe,
                            oos_sharpe=sharpe(pnl), oos_ic=ic, turnover=to, pnl=pnl))
            ws.append(w / (t.pnl_is.std() + 1e-12))
        ens = None
        if ws:
            W = sum(ws)
            W = W / np.maximum(np.abs(W).sum(1, keepdims=True), 1e-12)
            ens_is, _ = self._pnl(W, slice(self.warm, self.split))
            ens_oos, to = self._pnl(W, OOS)
            ens = dict(is_sharpe=sharpe(ens_is), oos_sharpe=sharpe(ens_oos),
                       oos_ann_return=ens_oos.mean() * ANN, oos_max_dd=max_drawdown(ens_oos),
                       turnover=to, pnl_is=ens_is, pnl_oos=ens_oos)
        return out, ens
