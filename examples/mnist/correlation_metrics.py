"""Probe-free correlation metrics on RGB-MNIST embeddings (issue #20 / E2).

The MNIST counterpart of ``examples/spectra/correlation_metrics.py``, but the
condition — the **digit** — is categorical, so the removal metric is the
correlation ratio ``eta`` rather than a correlation coefficient. Reported for the
same representations as ``ivae_sweep.py`` so the probe and probe-free families sit
side by side.

Metric per variable:

- **digit** (categorical condition) -> ``eta``, reported unsquared so it shares
  the ``|Pearson r|`` scale. Caveat that belongs in any write-up using this
  column: ``eta`` sees group *means* only. A coordinate whose variance depends on
  the digit but whose per-digit means coincide scores ~0, and the MLP probe would
  still find it. This is the one place the probe is strictly stronger.
- **colour b** (continuous, independent of digit by construction) -> ``|Spearman|``,
  raw and with the digit partialled out. The two should agree here; a gap would
  mean the colour channel is not as digit-independent as assumed.
- **rotation** (optional) -> see below.

Rotation is **circular, with period 180 deg**. ``compute_rotation.py`` estimates a
principal-axis orientation via ``0.5 * atan2(2*mu11, mu20 - mu02)``, and a
principal axis has no direction, so -89.9 deg and +89.9 deg are 0.2 deg apart, not
179.8 deg. Correlating against raw degrees therefore charges a representation for
the wrap. This script reports both:

- ``rot_circular_max`` — ``multiple_correlation`` against ``[sin 2t, cos 2t]``,
  which maps period-180 onto a full turn and is the correct measure;
- ``rot_naive_max`` — ``|Spearman|`` on raw degrees, the wrapped version.

The gap between them is the size of the artefact, and it applies equally to the
existing ``rot_r2_linreg``/``rot_r2_mlp`` probe columns, which regress on raw
degrees.

Run from this directory (needs `DATA_ROOT`):

    python correlation_metrics.py --rotation-dir . 2>&1 \
        | tee ivae_sweep_results/correlation_metrics.log
"""

import argparse
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from ivae_sweep import (
    DEFAULT_EMBED_SUBPATH,
    N_DIGITS,
    RANDOM_STATE,
    _one_hot,
    encode,
    evaluate,
    train_ivae,
)
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

from flower.evaluation.dependence import dependence_report
from flower.evaluation.ica import (
    conditional_mean_residual,
    conditional_prior_residual,
    drop_top_k_dependent,
)
from flower.evaluation.metrics import prepare_data


def _digit_control(digits):
    """One-hot digits as a numeric control, first column dropped.

    ``partial_correlation`` removes the control linearly, so a categorical
    control has to be encoded by the caller; the drop keeps the design from being
    collinear with the intercept.
    """
    return _one_hot(np.asarray(digits).astype(int), N_DIGITS)[:, 1:]


def correlation_metrics(x, digits, b, rot=None):
    """Probe-free dependence of one representation on the digit and the factors."""
    rep = dependence_report(x, digits, "eta")
    out = {
        "n_dims": x.shape[1],
        "digit_eta_max": rep["max"],
        "digit_eta_mean": rep["mean"],
        "digit_eta_n_above": rep["n_above"],
        "digit_eta_null": rep["null_level"],
    }

    control = _digit_control(digits)
    out["b_raw_max"] = dependence_report(x, b, "spearman")["max"]
    out["b_partial_max"] = dependence_report(x, b, "partial_spearman", control=control)[
        "max"
    ]

    if rot is not None:
        # Period-180 orientation -> double the angle to get a full turn.
        theta = np.deg2rad(np.asarray(rot, dtype=float)) * 2.0
        components = np.column_stack([np.sin(theta), np.cos(theta)])
        out["rot_circular_max"] = dependence_report(x, components, "multiple")["max"]
        out["rot_naive_max"] = dependence_report(x, rot, "spearman")["max"]
    return out


def build_representations(args, x_tr, x_te, dig_tr, dig_te, u_tr, u_te, d, flower):
    """Reproduce the ``ivae_sweep`` representation set, in the same order.

    ``flower`` adds the pretrained Flower embeddings (``cond``/``uncond``), which
    ``ivae_sweep.py`` does not carry — on MNIST those are evaluated separately by
    ``flower_cond_eval.py``. They need no training; they are scaled the same way
    as the raw embedding so the probe columns are comparable (the correlation
    columns are scale-invariant either way).
    """
    yield "Raw", "none", 0, x_tr, x_te

    for label, (train, test) in flower.items():
        sc = StandardScaler().fit(train)
        yield (
            label,
            "embedding",
            0,
            sc.transform(train).astype(np.float32),
            sc.transform(test).astype(np.float32),
        )

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    sources = {}

    if "FastICA" in selected:
        print("Fitting FastICA...")
        ica = FastICA(
            n_components=d,
            random_state=args.seed,
            max_iter=1000,
            whiten="unit-variance",
        )
        sources["FastICA"] = (ica.fit_transform(x_tr), ica.transform(x_te), None, None)

    for name, cond_enc in (("iVAE-cond", True), ("iVAE-fair", False)):
        if name not in selected:
            continue
        print(f"Training {name} ({args.epochs} epochs)...")
        ivae, _ = train_ivae(
            x_tr, u_tr, condition_encoder=cond_enc, epochs=args.epochs, seed=args.seed
        )
        s_tr, pm_tr = encode(ivae, x_tr, u_tr)
        s_te, pm_te = encode(ivae, x_te, u_te)
        sources[name] = (s_tr, s_te, pm_tr, pm_te)

    k_grid = sorted(set(np.linspace(1, d - 1, args.n_k).round().astype(int).tolist()))
    for name, (s_tr, s_te, pm_tr, pm_te) in sources.items():
        if name == "FastICA":
            rB_tr, means = conditional_mean_residual(s_tr, dig_tr)
            rB_te, _ = conditional_mean_residual(s_te, dig_te, means)
        else:
            rB_tr = conditional_prior_residual(s_tr, pm_tr)
            rB_te = conditional_prior_residual(s_te, pm_te)
        yield name, "residB", 0, rB_tr, rB_te

        for k in k_grid:
            _, dropped = drop_top_k_dependent(s_tr, dig_tr, k=k)
            keep = np.setdiff1d(np.arange(d), dropped)
            yield name, "residA", int(k), s_tr[:, keep], s_te[:, keep]


def make_plot(df, outpath, has_rot):
    n_panels = 2 + int(has_rot)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))

    if "digit_acc_mlp" in df:
        markers = {"none": "*", "residB": "o", "residA": ".", "embedding": "D"}
        for method, marker in markers.items():
            sub = df[df.method == method]
            if sub.empty:
                continue
            axes[0].scatter(
                sub.digit_acc_mlp, sub.digit_eta_max, marker=marker, s=60, label=method
            )
        axes[0].axvline(0.1, ls=":", color="grey", lw=1, label="chance acc")
        axes[0].set_xlabel("digit accuracy — MLP probe  (lower = better removal)")
        axes[0].set_ylabel("max eta with digit  (lower = better removal)")
        axes[0].set_title("probe vs probe-free: same verdict?")
        axes[0].legend(fontsize=8)
    else:
        axes[0].set_visible(False)

    axes[1].scatter(df.b_raw_max, df.b_partial_max, s=40, c="tab:green")
    lim = [0, max(df.b_raw_max.max(), df.b_partial_max.max()) * 1.05]
    axes[1].plot(lim, lim, "k--", lw=1)
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].set_xlabel("max |Spearman| with colour b (raw)")
    axes[1].set_ylabel("digit partialled out")
    axes[1].set_title("colour b: digit-independent as assumed?")

    if has_rot:
        axes[2].scatter(df.rot_naive_max, df.rot_circular_max, s=40, c="tab:purple")
        lim = [0, max(df.rot_naive_max.max(), df.rot_circular_max.max()) * 1.05]
        axes[2].plot(lim, lim, "k--", lw=1)
        axes[2].set_xlim(lim)
        axes[2].set_ylim(lim)
        axes[2].set_xlabel("max |Spearman| vs raw degrees (wrapped)")
        axes[2].set_ylabel("multiple R vs [sin 2t, cos 2t] (correct)")
        axes[2].set_title("rotation: cost of ignoring the 180° wrap")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("RGB-MNIST: probe-free dependence metrics (condition = digit)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embed-subpath", type=str, default=DEFAULT_EMBED_SUBPATH)
    parser.add_argument("--outdir", type=str, default="ivae_sweep_results")
    parser.add_argument("--n-k", type=int, default=11)
    parser.add_argument("--models", type=str, default="FastICA,iVAE-cond,iVAE-fair")
    parser.add_argument("--rotation-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--no-probes",
        action="store_true",
        help="skip the probe columns (much faster; correlations only)",
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    load_dotenv()
    embed_path = f"{os.getenv('DATA_ROOT')}/{args.embed_subpath}"
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"{embed_path}/train/*.parquet",
            "test": f"{embed_path}/test/*.parquet",
        },
    )
    x_tr, dig_tr, x_te, dig_te = prepare_data(ds, "orig", "digit")
    _, b_tr, _, b_te = prepare_data(ds, "orig", "b")
    dig_tr, dig_te = dig_tr.astype(int), dig_te.astype(int)

    # Flower's pretrained embeddings live in the same parquet as `orig`.
    flower = {
        f"Flower-{col}": (
            np.array(ds["train"][col], dtype=float),
            np.array(ds["test"][col], dtype=float),
        )
        for col in ("cond", "uncond")
    }

    rot_tr = rot_te = None
    if args.rotation_dir:
        rot_tr = pd.read_csv(
            os.path.join(args.rotation_dir, "train_rotation_aligned.csv")
        )["Rotation_Deg"].to_numpy()
        rot_te = pd.read_csv(
            os.path.join(args.rotation_dir, "test_rotation_aligned.csv")
        )["Rotation_Deg"].to_numpy()
        if len(rot_tr) != len(dig_tr) or len(rot_te) != len(dig_te):
            msg = "rotation CSV length does not match embeddings (row misalignment)"
            raise ValueError(msg)

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)
    u_tr, u_te = _one_hot(dig_tr, N_DIGITS), _one_hot(dig_te, N_DIGITS)
    d = x_tr.shape[1]
    print(f"embeddings: train {x_tr.shape}, test {x_te.shape}")

    rows = []
    for source, method, k, tr, te in build_representations(
        args, x_tr, x_te, dig_tr, dig_te, u_tr, u_te, d, flower
    ):
        row = {"source": source, "method": method, "k": k}
        row.update(correlation_metrics(te, dig_te, b_te, rot_te))
        if not args.no_probes:
            row.update(evaluate(tr, te, dig_tr, dig_te, b_tr, b_te, rot_tr, rot_te))
        rows.append(row)
        print(
            f"  {source:10s} {method:7s} k={k:<3d} "
            f"digit_eta_max={row['digit_eta_max']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "correlation_metrics.csv"), index=False)

    cols = ["source", "method", "k", "n_dims", "digit_eta_max", "digit_eta_n_above"]
    if "digit_acc_mlp" in df:
        cols += ["digit_acc_mlp"]
    cols += ["b_raw_max", "b_partial_max"]
    if rot_te is not None:
        cols += ["rot_naive_max", "rot_circular_max"]
    table = df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    header = (
        "Probe-free dependence metrics — RGB-MNIST (condition = digit)\n"
        "digit_eta_max: LOWER = better removal | b/rot: HIGHER = better "
        "preservation\n"
        f"chance level for eta at n={len(dig_te)}: "
        f"{df.digit_eta_null.iloc[0]:.4f}\n"
    )
    with open(os.path.join(args.outdir, "correlation_metrics.txt"), "w") as fh:
        fh.write(header + "\n" + table + "\n")
    print("\n" + header + "\n" + table)

    make_plot(
        df,
        os.path.join(args.outdir, "correlation_metrics.png"),
        has_rot=rot_te is not None,
    )
    print(f"\nSaved correlation_metrics.{{csv,txt,png}} in {args.outdir}/")


if __name__ == "__main__":
    main()
