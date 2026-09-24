"""QuantForge live-fire test.
Blast radius: CREATES a timestamped folder under ./runs/ next to this file and writes
report.md + equity.png there. Reads nothing else, no network, no installs, no processes.
Needs: numpy, scipy, matplotlib (already present in a standard scientific Python)."""
import os, sys, time
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("QuantForge: matplotlib is required for charts (install it yourself; this script never installs).")

from quantforge.data import synthetic_market
from quantforge.engine import AlphaForge, Config
from quantforge.backtest import signal_to_weights, sharpe
from quantforge import ops


def mission(name, data, cfg, lines):
    print(f"\n=== {name} ===")
    forge = AlphaForge(data, cfg)
    forge.evolve()
    hof = forge.hall_of_fame()
    verdicts, pbo, n, z_crit = forge.tribunal(hof)
    survivors = [v["trial"] for v in verdicts if v["passed"]]
    raw, ens_raw = forge.oos_report(hof)            # what a naive quant would ship
    kept, ens_kept = forge.oos_report(survivors)    # what the tribunal lets through
    lines += [f"## {name}", "",
              f"Formulas tried: {len(forge.trials)} (each long and short). Effective independent "
              f"bets: {n}. Luck bar (best Sharpe pure chance would hand you): "
              + (f"{verdicts[0]['luck_bar_annual']:.2f}." if verdicts else "n/a (no formula made money in the search)."), "",
              f"For information, probability of backtest overfitting across all formulas on the validation "
              f"stretch: **{pbo:.0%}** (about 50% means their ranking is luck; this is a diagnostic, not a gate).", "",
              f"To pass: Deflated Sharpe >= 0.95 on the search stretch AND validation z >= {z_crit:.2f} "
              "on a stretch the search never optimized on.", "",
              "| Alpha | Search Sharpe | Deflated Sharpe | Validation Sharpe (z) | Verdict | Vault (OOS) Sharpe |",
              "|---|---|---|---|---|---|"]
    for v, r in zip(verdicts, raw):
        e = ("-" if r["flip"] else "") + r["expr"]
        lines.append(f"| `{e}` | {r['is_sharpe']:.2f} | {v['dsr']:.2f} | {v['val_sharpe']:.2f} ({v['val_z']:.1f}) | "
                     f"{'PASS' if v['passed'] else 'REJECT'} | {r['oos_sharpe']:.2f} |")
    lines.append("")
    if ens_raw:
        lines.append(f"Naive ensemble (ship everything): IS Sharpe {ens_raw['is_sharpe']:.2f} "
                     f"-> vault Sharpe **{ens_raw['oos_sharpe']:.2f}**")
    if ens_kept:
        lines.append(f"Tribunal-approved ensemble: IS Sharpe {ens_kept['is_sharpe']:.2f} -> vault "
                     f"Sharpe **{ens_kept['oos_sharpe']:.2f}**, annual return "
                     f"{ens_kept['oos_ann_return']:.1%}, max drawdown {ens_kept['oos_max_dd']:.1%}")
    else:
        lines.append("Tribunal-approved ensemble: **nothing survived. Nothing ships.**")
    lines.append("")
    print(f"  PBO {pbo:.0%} | survivors {len(survivors)}/{len(hof)}")
    return forge, ens_raw, ens_kept


def main():
    out = os.path.join(HERE, "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=False)
    cfg = Config()
    lines = ["# QuantForge live-fire report", "",
             "Synthetic market: 50 assets, 10 years daily, fat tails, volatility clustering, "
             f"{cfg.cost_bps:.0f} bps cost per unit traded. Timeline: search, then a {cfg.val_frac:.0%} validation "
             f"stretch for replication, then the last {cfg.oos_frac:.0%} sealed in a vault nobody sees until the end.", ""]
    t0 = time.time()
    real = synthetic_market(seed=1, alpha_strength=1.0)
    trap = synthetic_market(seed=2, alpha_strength=0.0)
    fA, rawA, keptA = mission("Mission 1: market with two real, planted edges", real, cfg, lines)
    fB, rawB, keptB = mission("Mission 2: the trap (market with zero edge)", trap, cfg, lines)

    # Ceiling: the true planted signal, known only because we built this market.
    r = real["returns"]
    oracle = 0.035 * -ops.cs_zscore(ops.ts_sum(r, 5)) + 0.02 * ops.cs_zscore(ops.ts_delay(ops.ts_sum(r, 55), 5))
    w = signal_to_weights(oracle)
    fwd = np.nan_to_num(real["_fwd"])
    to = np.abs(np.diff(w, axis=0, prepend=0.0)).sum(1)
    pnl = (w * fwd).sum(1) - to * cfg.cost_bps / 1e4
    s = fA.split
    lines += ["## The ceiling", "",
              f"The true planted formula (the answer key) earns vault Sharpe "
              f"**{sharpe(pnl[s:-1]):.2f}** after costs. That is the best any engine could do here.", "",
              f"Runtime: {time.time()-t0:.0f}s"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, (title, raw, kept) in zip(ax, [("Mission 1: real edges", rawA, keptA),
                                           ("Mission 2: zero edge (trap)", rawB, keptB)]):
        if not raw:  # nothing made money in the search, so there is no naive book to plot
            a.set_title(title + " (nothing to plot)"); continue
        n_is = len(raw["pnl_is"])
        x_is, x_os = np.arange(n_is), n_is + np.arange(len(raw["pnl_oos"]))
        a.plot(x_is, np.cumsum(raw["pnl_is"]), color="tab:red", label="naive: ship everything")
        a.plot(x_os, raw["pnl_is"].sum() + np.cumsum(raw["pnl_oos"]), color="tab:red", ls="--")
        if kept:
            a.plot(x_is, np.cumsum(kept["pnl_is"]), color="tab:green", label="tribunal-approved")
            a.plot(x_os, kept["pnl_is"].sum() + np.cumsum(kept["pnl_oos"]), color="tab:green", ls="--")
        a.axvline(n_is, color="k", lw=1); a.text(n_is, a.get_ylim()[1]*0.95, " vault opens", va="top")
        a.set_title(title); a.set_xlabel("trading days"); a.set_ylabel("cumulative return")
        a.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "equity.png"), dpi=110)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
