"""Probe-free correlation metrics on the 2D-Gaussians toy (issue #20 / E2).

The toy counterpart of the spectra and MNIST ``correlation_metrics.py`` scripts,
and the only one of the three where the **ground-truth sources are known**: the
generator returns ``diff``, the 2-D condition-independent "seed", alongside the
mode. That makes this the calibration case — we can check the probe-free metrics
against MCC, which measures identifiability directly, and see whether they tell
the same story on data where the answer is not in doubt.

Metrics:

- **mode** (categorical condition, 4 clusters) -> ``eta``, unsquared so it shares
  the ``|Pearson r|`` scale. Should fall to chance for a good residual.
- **radial distance** (continuous, condition-independent by construction) ->
  ``|Spearman|``, raw and with the mode partialled out. The two should agree; the
  mode is in the cluster *mean*, and the distance is measured from that mean.
- **seed recovery** -> ``multiple_correlation`` of each coordinate against the
  2-D ``diff``, reported as a max, plus ``compute_mcc`` for the standard
  identifiability number. MCC matches whole representations one-to-one; the
  multiple-R max asks the narrower question of whether *some* coordinate tracks
  the seed.

Because ``d = 2`` here, residual A leaves a single coordinate — the max and mean
columns coincide for those rows.

Run from this directory (``examples/2d_gaussians``):

    python correlation_metrics.py [--epochs 30] 2>&1 \
        | tee correlation_metrics_results/correlation_metrics.log
"""

import argparse
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from data import generate_quad_gmm
from ivae_benchmark import (
    N_MODES,
    RANDOM_STATE,
    _one_hot,
    _to_numpy,
    encode_features,
    fastica_residuals,
    train_ivae,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

from flower.evaluation.dependence import dependence_report
from flower.evaluation.ica import (
    compute_mcc,
    conditional_prior_residual,
    drop_top_k_dependent,
)


def _mode_control(modes):
    """One-hot modes as a numeric control, first column dropped (see the MNIST
    script) — ``partial_correlation`` removes its control linearly."""
    return _one_hot(np.asarray(modes).astype(int), N_MODES)[:, 1:]


def correlation_metrics(x, modes, dist, diff):
    """Probe-free dependence on the mode, the radial distance and the seed."""
    rep = dependence_report(x, modes, "eta")
    control = _mode_control(modes)
    return {
        "n_dims": x.shape[1],
        "mode_eta_max": rep["max"],
        "mode_eta_mean": rep["mean"],
        "mode_eta_null": rep["null_level"],
        "dist_raw_max": dependence_report(x, dist, "spearman")["max"],
        "dist_partial_max": dependence_report(
            x, dist, "partial_spearman", control=control
        )["max"],
        "seed_multiple_max": dependence_report(x, diff, "multiple")["max"],
        "seed_mcc": compute_mcc(diff, x),
    }


def probe_metrics(x_tr, x_te, y_tr, y_te, d_tr, d_te, seed):
    """Point-estimate probes matching the other correlation_metrics scripts.

    ``ivae_benchmark.py`` reports the same quantities with 1000-sample bootstrap
    CIs; point estimates keep this table to one row per representation.
    """
    logreg = LogisticRegression(random_state=seed, max_iter=1000).fit(x_tr, y_tr)
    mlp_c = MLPClassifier(
        hidden_layer_sizes=(64, 64), max_iter=1000, random_state=seed
    ).fit(x_tr, y_tr)
    linreg = LinearRegression().fit(x_tr, d_tr)
    mlp_r = MLPRegressor(
        hidden_layer_sizes=(64, 64), max_iter=1000, random_state=seed
    ).fit(x_tr, d_tr)
    return {
        "mode_acc_logreg": accuracy_score(y_te, logreg.predict(x_te)),
        "mode_acc_mlp": accuracy_score(y_te, mlp_c.predict(x_te)),
        "dist_r2_linreg": r2_score(d_te, linreg.predict(x_te)),
        "dist_r2_mlp": r2_score(d_te, mlp_r.predict(x_te)),
    }


def build_representations(args, x_tr, x_te, y_tr, y_te):
    """Reproduce the ``ivae_benchmark`` representation set."""
    yield "Raw", "none", x_tr, x_te

    (fA_tr, fA_te), (fB_tr, fB_te), _ = fastica_residuals(
        x_tr, x_te, y_tr, y_te, seed=args.seed
    )
    yield "FastICA", "residA", fA_tr, fA_te
    yield "FastICA", "residB", fB_tr, fB_te

    print(f"Training iVAE ({args.epochs} epochs)...")
    u_tr, u_te = _one_hot(y_tr, N_MODES), _one_hot(y_te, N_MODES)
    ivae = train_ivae(x_tr, u_tr, epochs=args.epochs, seed=args.seed)
    s_tr, pmu_tr = encode_features(ivae, x_tr, u_tr)
    s_te, pmu_te = encode_features(ivae, x_te, u_te)

    _, dropped = drop_top_k_dependent(s_tr, y_tr, k=1)
    keep = np.setdiff1d(np.arange(s_tr.shape[1]), dropped)
    yield "iVAE", "residA", s_tr[:, keep], s_te[:, keep]
    yield (
        "iVAE",
        "residB",
        conditional_prior_residual(s_tr, pmu_tr),
        conditional_prior_residual(s_te, pmu_te),
    )


def make_plot(df, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    labels = df.source + "\n" + df.method

    if "mode_acc_mlp" in df:
        axes[0].scatter(df.mode_acc_mlp, df.mode_eta_max, s=70, c="tab:blue")
        for _, r in df.iterrows():
            axes[0].annotate(
                f"{r.source}-{r.method}",
                (r.mode_acc_mlp, r.mode_eta_max),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
        axes[0].axvline(0.25, ls=":", color="grey", lw=1, label="chance accuracy")
        axes[0].set_xlabel("mode accuracy — MLP probe  (lower = better removal)")
        axes[0].set_ylabel("max eta with mode  (lower = better removal)")
        axes[0].set_title("probe vs probe-free: same verdict?")
        axes[0].legend(fontsize=8)

    idx = np.arange(len(df))
    axes[1].bar(idx - 0.2, df.seed_mcc, width=0.4, label="MCC (whole representation)")
    axes[1].bar(
        idx + 0.2, df.seed_multiple_max, width=0.4, label="max multiple R (per-dim)"
    )
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("seed recovery")
    axes[1].set_title("ground-truth seed: MCC vs the probe-free per-dim measure")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("2D-Gaussians toy: probe-free dependence metrics (condition = mode)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--outdir", type=str, default="correlation_metrics_results")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--no-probes", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # `generate_quad_gmm` takes no seed and `data.py` never seeds torch, so the
    # toy is redrawn on every run. Seed here so this table is reproducible —
    # without it the iVAE rows move materially between runs (mode_eta_max 0.109
    # vs 0.038, MCC 0.754 vs 0.813 across two draws). `--seed` also drives
    # FastICA, the iVAE and the probes.
    torch.manual_seed(args.seed)
    x_tr, y_tr, d_tr, _ = _to_numpy(*generate_quad_gmm(args.n_train))
    x_te, y_te, d_te, diff_te = _to_numpy(*generate_quad_gmm(args.n_test))
    y_tr, y_te = y_tr.astype(int), y_te.astype(int)
    print(f"train {x_tr.shape}, test {x_te.shape}")

    rows = []
    for source, method, tr, te in build_representations(args, x_tr, x_te, y_tr, y_te):
        row = {"source": source, "method": method}
        row.update(correlation_metrics(te, y_te, d_te, diff_te))
        if not args.no_probes:
            row.update(probe_metrics(tr, te, y_tr, y_te, d_tr, d_te, args.seed))
        rows.append(row)
        print(f"  {source:8s} {method:7s} mode_eta_max={row['mode_eta_max']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "correlation_metrics.csv"), index=False)

    cols = ["source", "method", "n_dims", "mode_eta_max"]
    if "mode_acc_mlp" in df:
        cols += ["mode_acc_mlp"]
    cols += ["dist_raw_max", "dist_partial_max", "seed_multiple_max", "seed_mcc"]
    table = df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    header = (
        "Probe-free dependence metrics — 2D-Gaussians toy (condition = mode)\n"
        "mode_eta_max: LOWER = better removal (chance acc 0.25) | "
        "dist/seed: HIGHER = better preservation\n"
        f"chance level for eta at n={len(y_te)}: {df.mode_eta_null.iloc[0]:.4f}\n"
    )
    with open(os.path.join(args.outdir, "correlation_metrics.txt"), "w") as fh:
        fh.write(header + "\n" + table + "\n")
    print("\n" + header + "\n" + table)

    make_plot(df, os.path.join(args.outdir, "correlation_metrics.png"))
    print(f"\nSaved correlation_metrics.{{csv,txt,png}} in {args.outdir}/")


if __name__ == "__main__":
    main()
