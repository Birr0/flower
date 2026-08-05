"""Distance correlation between each representation and the test variables.

Companion to ``correlation_metrics.py``, which scores every latent coordinate
separately. That is basis-dependent: rotating a representation moves the same
information onto different coordinates and changes every ``*_max`` column
without changing what the representation knows. FastICA and an iVAE do not agree
on axes, so those columns cannot carry a comparison *across* methods.

This script computes one number per representation instead: ``dcor`` between the
**whole** matrix and each variable. It depends on the representation only
through its pairwise distances, so rotations, permutations and sign flips leave
it alone — and dCor is zero only under independence, so unlike |Pearson| or eta
it also sees nonlinear structure.

Variables:

- **digit** (the condition, want removed) — one-hot encoded, so the distance
  between two labels is "same or different" rather than ``|3 - 5| = 2``.
- **colour b** (want preserved) — as is.
- **rotation** (want preserved, optional) — as ``[sin 2t, cos 2t]``, because the
  principal-axis orientation has period 180 deg.

dCor is positively biased at small ``n``, so every column is reported against
``*_null``: the same statistic with the variable shuffled. Read a score against
its null, never against zero.

Cost is O(n^2), hence ``--n-subsample``.

Run from this directory (needs `DATA_ROOT`):

    python dcor_metrics.py --rotation-dir . 2>&1 \
        | tee ivae_sweep_results/dcor_metrics.log
"""

import argparse
import os

import dcor
import numpy as np
import pandas as pd
from correlation_metrics import build_representations
from datasets import load_dataset
from dotenv import load_dotenv
from ivae_sweep import DEFAULT_EMBED_SUBPATH, N_DIGITS, RANDOM_STATE, _one_hot
from sklearn.preprocessing import StandardScaler

from flower.evaluation.metrics import prepare_data


def dcor_row(x, variables, idx, rng):
    """dCor of the whole of ``x`` against each variable, plus a shuffled null."""
    x_sub = x[idx]
    out = {"n_dims": x.shape[1]}
    for name, values in variables.items():
        v_sub = np.asarray(values, dtype=float).reshape(len(x), -1)[idx]
        out[f"{name}_dcor"] = float(dcor.distance_correlation(x_sub, v_sub))
        out[f"{name}_null"] = float(
            dcor.distance_correlation(x_sub, v_sub[rng.permutation(len(idx))])
        )
    return out


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
        "--n-subsample",
        type=int,
        default=3000,
        help="test rows used per dCor (O(n^2) in time and memory)",
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
    _, _b_tr, _, b_te = prepare_data(ds, "orig", "b")
    dig_tr, dig_te = dig_tr.astype(int), dig_te.astype(int)

    flower = {
        f"Flower-{col}": (
            np.array(ds["train"][col], dtype=float),
            np.array(ds["test"][col], dtype=float),
        )
        for col in ("cond", "uncond")
    }

    variables = {"digit": _one_hot(dig_te, N_DIGITS), "b": b_te}
    if args.rotation_dir:
        rot_te = pd.read_csv(
            os.path.join(args.rotation_dir, "test_rotation_aligned.csv")
        )["Rotation_Deg"].to_numpy()
        if len(rot_te) != len(dig_te):
            msg = "rotation CSV length does not match embeddings (row misalignment)"
            raise ValueError(msg)
        theta = np.deg2rad(rot_te.astype(float)) * 2.0
        variables["rot"] = np.column_stack([np.sin(theta), np.cos(theta)])

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)
    u_tr, u_te = _one_hot(dig_tr, N_DIGITS), _one_hot(dig_te, N_DIGITS)
    d = x_tr.shape[1]
    print(f"embeddings: train {x_tr.shape}, test {x_te.shape}")

    # One subsample, shared by every row, so differences between rows are not
    # differences between draws.
    rng = np.random.default_rng(args.seed)
    take = min(args.n_subsample, len(x_te))
    idx = rng.choice(len(x_te), take, replace=False) if take < len(x_te) else None
    idx = np.arange(len(x_te)) if idx is None else idx
    print(f"dCor on {take} test rows")

    rows = []
    for source, method, k, tr, te in build_representations(
        args, x_tr, x_te, dig_tr, dig_te, u_tr, u_te, d, flower
    ):
        del tr
        row = {"source": source, "method": method, "k": k}
        row.update(dcor_row(te, variables, idx, np.random.default_rng(args.seed)))
        rows.append(row)
        print(f"  {source:14s} {method:9s} k={k:<3d} digit={row['digit_dcor']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "dcor_metrics.csv"), index=False)

    cols = ["source", "method", "k", "n_dims"]
    for name in variables:
        cols += [f"{name}_dcor", f"{name}_null"]
    table = df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    header = (
        "Distance correlation, whole representation vs variable — RGB-MNIST\n"
        "digit: LOWER = better removal | b/rot: HIGHER = better preservation\n"
        f"n_subsample={take}; read each score against its *_null column\n"
    )
    with open(os.path.join(args.outdir, "dcor_metrics.txt"), "w") as fh:
        fh.write(header + "\n" + table + "\n")
    print("\n" + header + "\n" + table)
    print(f"\nSaved dcor_metrics.{{csv,txt}} in {args.outdir}/")


if __name__ == "__main__":
    main()
