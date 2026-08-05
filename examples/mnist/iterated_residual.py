"""Iterated (boosting-style) residualisation baseline — RGB-MNIST (issue #20 / E4).

Reviewer E4 on cMNIST: "repeat the residualisation loop twice/thrice — presumably
less and less information about y can be retrieved." Residualise the VAE embedding
against the digit repeatedly (each pass fits a regressor `one-hot(digit) → X` and
subtracts the fitted per-digit mean), with a linear and an MLP regressor, 1x…5x.
Track recoverable digit (logreg + MLP) and preservation of colour `b` and rotation
per iteration. Flower's `cond` embedding is the reference.

Same expectation as the spectra version: mean-residualisation is ~idempotent, so
iterating does not remove more of the digit — the curve flattens after one pass.

Run from this directory (needs `DATA_ROOT`; rotation CSVs from compute_rotation.py):

    python iterated_residual.py [--n-iter 5]
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
    _b_r2,
    _digit_acc,
    _one_hot,
)
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from flower.evaluation.metrics import prepare_data


def _regressor(kind):
    if kind == "linear":
        return LinearRegression()
    return MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE
    )


def _metrics(x_tr, x_te, dig_tr, dig_te, b_tr, b_te, rot_tr, rot_te, kind, it):
    b = _b_r2(x_tr, x_te, b_tr, b_te, "mlp")
    rot = _b_r2(x_tr, x_te, rot_tr, rot_te, "mlp")
    return {
        "residualiser": kind,
        "iteration": it,
        "digit_acc_logreg": _digit_acc(x_tr, x_te, dig_tr, dig_te, "logreg"),
        "digit_acc_mlp": _digit_acc(x_tr, x_te, dig_tr, dig_te, "mlp"),
        "b_r2_mlp": b,
        "rot_r2_mlp": rot,
        "mean_preserv_mlp": (b + rot) / 2,
    }


def run_iterated(x_tr, x_te, cond_tr, cond_te, dig, b, rot, kind, n_iter):
    xt, xe = x_tr.copy(), x_te.copy()
    rows = [_metrics(xt, xe, *dig, *b, *rot, kind, 0)]
    for it in range(1, n_iter + 1):
        reg = _regressor(kind)
        reg.fit(cond_tr, xt)
        xt = xt - reg.predict(cond_tr)
        xe = xe - reg.predict(cond_te)
        rows.append(_metrics(xt, xe, *dig, *b, *rot, kind, it))
        print(
            f"  {kind} iter {it}: digit_mlp={rows[-1]['digit_acc_mlp']:.3f} "
            f"preserv={rows[-1]['mean_preserv_mlp']:.3f}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-subpath", default=DEFAULT_EMBED_SUBPATH)
    parser.add_argument("--rotation-dir", default=".")
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--outdir", default="ivae_sweep_results")
    parser.add_argument("--flower-csv", default="flower_cond_results/results.csv")
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
    rot_tr = pd.read_csv(os.path.join(args.rotation_dir, "train_rotation_aligned.csv"))[
        "Rotation_Deg"
    ].to_numpy()
    rot_te = pd.read_csv(os.path.join(args.rotation_dir, "test_rotation_aligned.csv"))[
        "Rotation_Deg"
    ].to_numpy()

    sc = StandardScaler().fit(x_tr)
    x_tr = sc.transform(x_tr).astype(np.float32)
    x_te = sc.transform(x_te).astype(np.float32)
    cond_tr = _one_hot(dig_tr, N_DIGITS)
    cond_te = _one_hot(dig_te, N_DIGITS)

    rows = []
    for kind in ("linear", "mlp"):
        print(f"Iterating {kind} residualisation...")
        rows += run_iterated(
            x_tr,
            x_te,
            cond_tr,
            cond_te,
            (dig_tr, dig_te),
            (b_tr, b_te),
            (rot_tr, rot_te),
            kind,
            args.n_iter,
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "iterated_residual.csv"), index=False)

    flow_dig = flow_preserv = None
    if os.path.exists(args.flower_csv):
        fc = pd.read_csv(args.flower_csv)
        fc = fc[fc.embedding == "Flower cond"]
        if not fc.empty:
            fc = fc.iloc[0]
            flow_dig = fc.digit_acc_mlp
            flow_preserv = (fc.b_r2_mlp + fc.rot_r2_mlp) / 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for kind, color in {"linear": "tab:blue", "mlp": "tab:orange"}.items():
        d = df[df.residualiser == kind]
        axes[0].plot(
            d.iteration,
            d.digit_acc_mlp,
            "-o",
            color=color,
            label=f"{kind} residualiser",
        )
        axes[1].plot(
            d.iteration,
            d.mean_preserv_mlp,
            "-o",
            color=color,
            label=f"{kind} residualiser",
        )
    if flow_dig is not None:
        axes[0].axhline(flow_dig, ls="--", color="crimson", label="Flower cond")
        axes[1].axhline(flow_preserv, ls="--", color="crimson", label="Flower cond")
    axes[0].axhline(0.10, ls=":", color="grey", label="digit chance")
    axes[0].set_title("recoverable digit vs iteration")
    axes[0].set_xlabel("residualisation iterations")
    axes[0].set_ylabel("digit accuracy — MLP  (lower = better removal)")
    axes[1].set_title("colour b + rotation preserved vs iteration")
    axes[1].set_xlabel("residualisation iterations")
    axes[1].set_ylabel("mean preservation R² — MLP")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "RGB-MNIST: iterated (boosting) residualisation of the digit — "
        "mean-removal is ~idempotent, digit-recoverability flat above Flower"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "iterated_residual.png"), dpi=140)

    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved iterated_residual.csv, iterated_residual.png in {args.outdir}/")


if __name__ == "__main__":
    main()
