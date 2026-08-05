"""Iterated (boosting-style) residualisation baseline — spectra (issue #20 / E4).

Reviewer E4: "repeat the residualisation loop twice/thrice — presumably less and
less information about y can be retrieved." This tests exactly that: residualise
the embedding against redshift `z` repeatedly (1x, 2x, ... with a linear or MLP
regressor, each pass subtracting the fitted E[X|z]) and track how much redshift
remains recoverable (linear + MLP probes) and how much physics survives, vs
iteration. Flower's `cond` embedding is the reference asymptote.

Expectation (and the point): mean-residualisation is ~idempotent — after one pass
the residual already has zero conditional mean, so further passes remove little,
and the curve flattens *above* Flower, because the surviving y-information is in
higher-order structure that mean-subtraction cannot reach.

Run from this directory (needs `DATA_ROOT`):

    python iterated_residual.py --spender spender_I [--n-iter 5]
"""

import argparse
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ivae_sweep import RANDOM_STATE, SPENDER_MAP, TARGETS, _r2, load_data
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _regressor(kind):
    if kind == "linear":
        return LinearRegression()
    return MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE
    )


def _metrics(x_tr, x_te, z_tr, z_te, tg_tr, tg_te, kind, it):
    phys = [_r2(x_tr, x_te, tg_tr[t], tg_te[t], "mlp") for t in TARGETS]
    return {
        "residualiser": kind,
        "iteration": it,
        "z_r2_linreg": _r2(x_tr, x_te, z_tr, z_te, "linreg"),
        "z_r2_mlp": _r2(x_tr, x_te, z_tr, z_te, "mlp"),
        "mean_phys_r2_mlp": float(np.mean(phys)),
    }


def run_iterated(x_tr, x_te, cond_tr, cond_te, z_tr, z_te, tg_tr, tg_te, kind, n_iter):
    xt, xe = x_tr.copy(), x_te.copy()
    rows = [_metrics(xt, xe, z_tr, z_te, tg_tr, tg_te, kind, 0)]  # iter 0 = raw
    for it in range(1, n_iter + 1):
        reg = _regressor(kind)
        reg.fit(cond_tr, xt)
        xt = xt - reg.predict(cond_tr)
        xe = xe - reg.predict(cond_te)
        rows.append(_metrics(xt, xe, z_tr, z_te, tg_tr, tg_te, kind, it))
        print(
            f"  {kind} iter {it}: z_mlp={rows[-1]['z_r2_mlp']:.3f} "
            f"phys={rows[-1]['mean_phys_r2_mlp']:.3f}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spender", default="spender_I", choices=list(SPENDER_MAP))
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-filter", type=int, default=300000)
    parser.add_argument("--outdir", default="ivae_sweep_results")
    parser.add_argument("--results-csv", default=None)
    args = parser.parse_args()
    results_csv = args.results_csv or os.path.join(args.outdir, "results.csv")

    (o_tr, _, _, z_tr, tg_tr), (o_te, _, _, z_te, tg_te) = load_data(
        args.spender, args.n_train, args.n_filter
    )
    sc = StandardScaler().fit(o_tr)
    x_tr = sc.transform(o_tr).astype(np.float32)
    x_te = sc.transform(o_te).astype(np.float32)
    cond_tr, cond_te = z_tr.reshape(-1, 1), z_te.reshape(-1, 1)

    rows = []
    for kind in ("linear", "mlp"):
        print(f"Iterating {kind} residualisation...")
        rows += run_iterated(
            x_tr, x_te, cond_tr, cond_te, z_tr, z_te, tg_tr, tg_te, kind, args.n_iter
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "iterated_residual.csv"), index=False)

    # Flower reference from the sweep results.
    flow_z = flow_phys = None
    if os.path.exists(results_csv):
        rc = pd.read_csv(results_csv)
        fc = rc[rc.source == "Flower-cond"]
        if not fc.empty:
            fc = fc.iloc[0]
            flow_z = fc.z_r2_mlp
            flow_phys = float(np.mean([fc[f"{t}_r2_mlp"] for t in TARGETS]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for kind, color in {"linear": "tab:blue", "mlp": "tab:orange"}.items():
        d = df[df.residualiser == kind]
        axes[0].plot(
            d.iteration, d.z_r2_mlp, "-o", color=color, label=f"{kind} residualiser"
        )
        axes[1].plot(
            d.iteration,
            d.mean_phys_r2_mlp,
            "-o",
            color=color,
            label=f"{kind} residualiser",
        )
    if flow_z is not None:
        axes[0].axhline(flow_z, ls="--", color="crimson", label="Flower cond")
        axes[1].axhline(flow_phys, ls="--", color="crimson", label="Flower cond")
    axes[0].set_title("recoverable redshift vs iteration")
    axes[0].set_xlabel("residualisation iterations")
    axes[0].set_ylabel("z R² — MLP  (lower = better removal)")
    axes[1].set_title("physics preserved vs iteration")
    axes[1].set_xlabel("residualisation iterations")
    axes[1].set_ylabel("mean physics R² — MLP")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{args.spender}: iterated (boosting) residualisation of redshift — "
        "mean-removal is ~idempotent, asymptotes above Flower"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "iterated_residual.png"), dpi=140)

    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved iterated_residual.csv, iterated_residual.png in {args.outdir}/")


if __name__ == "__main__":
    main()
