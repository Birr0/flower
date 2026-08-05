"""Probe-free correlation metrics on spectra embeddings (issue #20 / E2).

Every removal/preservation number we currently quote on spectra is a *probe*
score: fit a regressor on the representation, read off R^2. That is aggregate and
capacity-dependent — an MLP R^2 of 0.05 conflates "no redshift left" with "no
redshift this MLP found", and it moves with ``hidden_layer_sizes``. This script
adds the complementary family from ``flower.evaluation.dependence``:
per-dimension, closed-form, hyperparameter-free correlations, reported for the
*same* representations as ``ivae_sweep.py`` so the two families sit side by side.

Three things it is meant to settle:

1. **Worst-case leak.** ``z_spearman_max`` is the largest |Spearman| between
   redshift and any single coordinate. A probe cannot make that statement; a
   representation with mean probe R^2 ~ 0 can still have one coordinate tracking
   z at 0.6.
2. **Linear vs monotone.** Redshift enters the embedding through a curved but
   order-preserving map, so ``z_pearson_max`` understates the leak. The gap
   between the Pearson and Spearman columns is the size of that understatement.
3. **The preservation confound.** logM*, logSFR and A_v are themselves correlated
   with redshift, so a raw correlation between a residual embedding and logM*
   partly re-measures the z that was supposed to have been removed. The
   ``*_partial_max`` columns control for z (rank-partial: a *linear* control
   removal barely dents a nonlinear confound — see
   ``tests/test_evaluation_dependence.py``), so raw-minus-partial is the share of
   apparent "preserved physics" that is really redshift.

Correlations are scale-invariant, so standardising the embedding does not move
them, and they need no train/test split — they are computed on the test split
directly. Probe columns are reproduced from ``ivae_sweep.evaluate`` so a reader
can compare families row by row; pass ``--no-probes`` for the fast table.

Run from this directory (needs `DATA_ROOT`):

    python correlation_metrics.py --spender spender_I 2>&1 \
        | tee ivae_sweep_results/correlation_metrics.log
"""

import argparse
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ivae_sweep import (
    CONDITION,
    RANDOM_STATE,
    SPENDER_MAP,
    TARGETS,
    encode,
    evaluate,
    load_data,
    model_residual,
    train_ivae,
)
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from flower.evaluation.dependence import dependence_report
from flower.evaluation.ica import (
    conditional_prior_residual,
    drop_top_k_dependent,
    regression_residual,
)


def correlation_metrics(x, z, targets):
    """Probe-free dependence of one representation on z and on the targets.

    ``x`` is the test-split representation, ``z`` the condition, ``targets`` the
    ``{name: values}`` physical factors. Everything is computed on this one split
    — no fitting, so no leakage to worry about.
    """
    out = {"n_dims": x.shape[1]}
    for metric in ("pearson", "spearman"):
        rep = dependence_report(x, z, metric)
        out[f"{CONDITION}_{metric}_max"] = rep["max"]
        out[f"{CONDITION}_{metric}_mean"] = rep["mean"]
        out[f"{CONDITION}_{metric}_n_above"] = rep["n_above"]
        out[f"{CONDITION}_{metric}_null"] = rep["null_level"]

    for t, values in targets.items():
        # Raw: how much of this factor any single coordinate tracks.
        out[f"{t}_raw_max"] = dependence_report(x, values, "spearman")["max"]
        # Partial: the same, with redshift removed from both sides.
        out[f"{t}_partial_max"] = dependence_report(
            x, values, "partial_spearman", control=z
        )["max"]
    return out


def build_representations(args, x_tr, x_te, z_tr, z_te, flower, d):
    """Reproduce the ``ivae_sweep`` representation set, in the same order.

    Yields ``(source, method, k, train, test)``. Train arrays are only needed for
    the optional probe columns; the correlation metrics use the test split alone.
    """
    yield "Raw", "none", 0, x_tr, x_te

    print("Fitting FastICA...")
    ica = FastICA(
        n_components=d, random_state=args.seed, max_iter=1000, whiten="unit-variance"
    )
    f_tr, f_te = ica.fit_transform(x_tr), ica.transform(x_te)
    fB_tr, coef = regression_residual(f_tr, z_tr)
    fB_te, _ = regression_residual(f_te, z_te, coef)
    yield "FastICA", "residB", 0, fB_tr, fB_te

    print(f"Training iVAE ({args.epochs} epochs)...")
    u_tr = z_tr.reshape(-1, 1).astype(np.float32)
    u_te = z_te.reshape(-1, 1).astype(np.float32)
    ivae = train_ivae(x_tr, u_tr, epochs=args.epochs, seed=args.seed)
    s_tr, pm_tr = encode(ivae, x_tr, u_tr)
    s_te, pm_te = encode(ivae, x_te, u_te)
    yield (
        "iVAE",
        "residB",
        0,
        conditional_prior_residual(s_tr, pm_tr),
        conditional_prior_residual(s_te, pm_te),
    )

    k_grid = sorted(set(np.linspace(1, d - 1, args.n_k).round().astype(int).tolist()))
    for name, (src_tr, src_te) in {
        "FastICA": (f_tr, f_te),
        "iVAE": (s_tr, s_te),
    }.items():
        for k in k_grid:
            _, dropped = drop_top_k_dependent(
                src_tr, z_tr, k=k, dependence="continuous"
            )
            keep = np.setdiff1d(np.arange(d), dropped)
            yield name, "residA", int(k), src_tr[:, keep], src_te[:, keep]

    resid_models = {
        "Resid-linear": LinearRegression(),
        "Resid-mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=args.seed
        ),
        "Resid-rf": RandomForestRegressor(n_estimators=100, random_state=args.seed),
    }
    for name, mdl in resid_models.items():
        print(f"Building {name}...")
        r_tr, r_te = model_residual(x_tr, z_tr, x_te, z_te, mdl)
        yield name, "direct", 0, r_tr, r_te

    # Flower embeddings are evaluated directly (no training). Scaled the same way
    # as ivae_sweep so the probe columns are comparable; the correlation columns
    # are scale-invariant either way.
    for label, (train, test) in flower.items():
        sc = StandardScaler().fit(train)
        yield (
            label,
            "embedding",
            0,
            sc.transform(train).astype(np.float32),
            sc.transform(test).astype(np.float32),
        )


def make_plot(df, spender, outpath):
    """Left: do the probe and correlation families agree on removal?
    Right: how much apparent preservation is really redshift?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    if "z_r2_mlp" in df:
        for method, marker in {
            "none": "*",
            "residB": "o",
            "residA": ".",
            "direct": "s",
            "embedding": "D",
        }.items():
            sub = df[df.method == method]
            if sub.empty:
                continue
            axes[0].scatter(
                sub.z_r2_mlp,
                sub[f"{CONDITION}_spearman_max"],
                marker=marker,
                s=60,
                alpha=0.8,
                label=method,
            )
        for _, r in df[df.method.isin(["embedding", "direct", "residB"])].iterrows():
            axes[0].annotate(
                r.source,
                (r.z_r2_mlp, r[f"{CONDITION}_spearman_max"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        axes[0].set_xlabel("z R² — MLP probe  (lower = better removal)")
        axes[0].set_ylabel("max |Spearman| with z  (lower = better removal)")
        axes[0].set_title("probe vs probe-free: same verdict?")
        axes[0].legend(fontsize=8)
    else:
        axes[0].set_visible(False)

    for t in TARGETS:
        axes[1].scatter(df[f"{t}_raw_max"], df[f"{t}_partial_max"], s=40, label=t)
    lim = [
        0,
        max(
            df[[f"{t}_raw_max" for t in TARGETS]].to_numpy().max(),
            df[[f"{t}_partial_max" for t in TARGETS]].to_numpy().max(),
        )
        * 1.05,
    ]
    axes[1].plot(lim, lim, "k--", lw=1, label="no confound")
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].set_xlabel("max |Spearman| with factor (raw)")
    axes[1].set_ylabel("max |Spearman| with factor, z partialled out")
    axes[1].set_title("distance below the line = redshift leaking through targets")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{spender}: probe-free dependence metrics on spectra embeddings")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spender", default="spender_I", choices=list(SPENDER_MAP))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-filter", type=int, default=300000)
    parser.add_argument("--n-k", type=int, default=8)
    parser.add_argument("--outdir", default="ivae_sweep_results")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--no-probes",
        action="store_true",
        help="skip the R² probe columns (much faster; correlations only)",
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    (o_tr, c_tr, u_tr_e, z_tr, tg_tr), (o_te, c_te, u_te_e, z_te, tg_te) = load_data(
        args.spender, args.n_train, args.n_filter
    )
    flower = {"Flower-cond": (c_tr, c_te), "Flower-uncond": (u_tr_e, u_te_e)}
    print(f"train {o_tr.shape}, test {o_te.shape}")

    sc = StandardScaler().fit(o_tr)
    x_tr = sc.transform(o_tr).astype(np.float32)
    x_te = sc.transform(o_te).astype(np.float32)
    d = x_tr.shape[1]

    # How entangled the targets are with z to begin with — the baseline the
    # partial columns are measured against.
    print("\nTarget-vs-redshift entanglement (|Spearman|):")
    for t in TARGETS:
        rho = dependence_report(z_te.reshape(-1, 1), tg_te[t], "spearman")["max"]
        print(f"  {t:8s} {rho:.3f}")

    rows = []
    for source, method, k, tr, te in build_representations(
        args, x_tr, x_te, z_tr, z_te, flower, d
    ):
        row = {"source": source, "method": method, "k": k}
        row.update(correlation_metrics(te, z_te, tg_te))
        if not args.no_probes:
            row.update(evaluate(tr, te, z_tr, z_te, tg_tr, tg_te))
        rows.append(row)
        print(
            f"  {source:14s} {method:9s} k={k:<3d} "
            f"z_sp_max={row[f'{CONDITION}_spearman_max']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "correlation_metrics.csv"), index=False)

    cols = ["source", "method", "k", "n_dims"]
    cols += [f"{CONDITION}_{m}_max" for m in ("pearson", "spearman")]
    cols += [f"{CONDITION}_spearman_n_above"]
    if "z_r2_mlp" in df:
        cols += ["z_r2_mlp"]
    cols += [f"{t}_partial_max" for t in TARGETS]
    table = df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    chance = df[f"{CONDITION}_spearman_null"].iloc[0]
    header = (
        f"Probe-free dependence metrics — {args.spender} (condition = redshift z)\n"
        "z_*_max: LOWER = better removal | *_partial_max: HIGHER = better "
        "preservation of non-redshift structure\n"
        f"chance level for |rho| at n={len(z_te)}: {chance:.4f}\n"
    )
    with open(os.path.join(args.outdir, "correlation_metrics.txt"), "w") as fh:
        fh.write(header + "\n" + table + "\n")
    print("\n" + header + "\n" + table)

    make_plot(df, args.spender, os.path.join(args.outdir, "correlation_metrics.png"))
    print(f"\nSaved correlation_metrics.{{csv,txt,png}} in {args.outdir}/")


if __name__ == "__main__":
    main()
