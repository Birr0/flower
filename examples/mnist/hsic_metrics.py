"""HSIC independence test between each representation and the test variables.

Companion to ``dcor_metrics.py``, which found distance correlation underpowered
on RGB-MNIST: `conditional_mean_residual` equalises the per-digit means by
construction, and dCor reads that mean-matched residual as near-independent
(0.130 against a 0.109 null) even though an MLP recovers the digit at 0.858.

HSIC with a Gaussian kernel does have power against covariance-only dependence,
so it is the better candidate for a model-free independence test. This script
reports the normalized HSIC (equivalently CKA) together with a permutation
p-value, which is the calibrated statement a probe accuracy cannot make on its
own: *reject* independence, with an error rate.

Two comparability choices, both needed for a cross-method table:

- **Fixed bandwidth.** Each representation is standardized per dimension and the
  kernel width is set to ``sqrt(d)``, the scale implied by ``E||a-b||^2 = 2d``
  for standardized data. A median heuristic would fit a *different* kernel to
  every row, so effect-size differences could be bandwidth artefacts rather than
  dependence.
- **One shared subsample** across rows, so row-to-row differences are not
  draw-to-draw differences.

Read the result one-sided. Rejecting independence is evidence of dependence;
failing to reject is not evidence of independence — the dCor result is the
cautionary case. The normalized statistic orders rows by *detected* dependence
under this kernel, which is not the same as ordering them by how recoverable the
condition is: see the note for where it disagrees with the MLP probe.

Run from this directory (needs `DATA_ROOT`):

    python hsic_metrics.py --rotation-dir . 2>&1 \
        | tee ivae_sweep_results/hsic_metrics.log
"""

import argparse
import os

import numpy as np
import pandas as pd
from correlation_metrics import build_representations
from datasets import load_dataset
from dotenv import load_dotenv
from ivae_sweep import DEFAULT_EMBED_SUBPATH, N_DIGITS, RANDOM_STATE, _one_hot
from sklearn.preprocessing import StandardScaler

from flower.evaluation.metrics import prepare_data


def _centered_rbf(a):
    """Double-centered Gaussian kernel at the fixed ``sqrt(d)`` bandwidth."""
    a = np.asarray(a, dtype=float)
    a = StandardScaler().fit_transform(a)
    sq = ((a[:, None, :] - a[None, :, :]) ** 2).sum(-1)
    k = np.exp(-sq / (2.0 * a.shape[1]))
    # Double-centering commutes with permutation, so this can be done once and
    # the permuted statistic read off by indexing — which is what makes 2000
    # permutations affordable.
    return k - k.mean(0) - k.mean(1)[:, None] + k.mean()


def hsic_test(kx, ky, n_perm, rng):
    """Normalized HSIC (CKA) plus a permutation p-value and null mean."""
    x_norm = np.sqrt((kx * kx).sum())

    def stat(k):
        den = x_norm * np.sqrt((k * k).sum())
        return float((kx * k).sum() / den) if den > 0 else 0.0

    obs = stat(ky)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(ky))
        null[i] = stat(ky[np.ix_(p, p)])
    return {
        "hsic": obs,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embed-subpath", type=str, default=DEFAULT_EMBED_SUBPATH)
    parser.add_argument("--outdir", type=str, default="ivae_sweep_results")
    parser.add_argument("--n-k", type=int, default=11)
    parser.add_argument("--models", type=str, default="FastICA,iVAE-cond,iVAE-fair")
    parser.add_argument("--rotation-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument(
        "--n-subsample",
        type=int,
        default=2000,
        help="test rows per kernel (O(n^2) memory, O(n^2) per permutation)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="none,embedding,residB",
        help="restrict to these `method` values (residA sweeps are expensive)",
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    keep_methods = {m.strip() for m in args.methods.split(",") if m.strip()}

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

    rng = np.random.default_rng(args.seed)
    take = min(args.n_subsample, len(x_te))
    idx = (
        rng.choice(len(x_te), take, replace=False)
        if take < len(x_te)
        else np.arange(len(x_te))
    )
    print(f"HSIC on {take} test rows, {args.n_perm} permutations")

    var_kernels = {
        name: _centered_rbf(np.asarray(v, dtype=float).reshape(len(x_te), -1)[idx])
        for name, v in variables.items()
    }

    rows = []
    for source, method, k, tr, te in build_representations(
        args, x_tr, x_te, dig_tr, dig_te, u_tr, u_te, d, flower
    ):
        del tr
        if method not in keep_methods:
            continue
        kx = _centered_rbf(te[idx])
        row = {"source": source, "method": method, "k": k, "n_dims": te.shape[1]}
        for name, ky in var_kernels.items():
            res = hsic_test(kx, ky, args.n_perm, np.random.default_rng(args.seed))
            row[f"{name}_hsic"] = res["hsic"]
            row[f"{name}_null"] = res["null_mean"]
            row[f"{name}_p95"] = res["null_p95"]
            row[f"{name}_p"] = res["p_value"]
        rows.append(row)
        print(
            f"  {source:14s} {method:9s} k={k:<3d} "
            f"digit={row['digit_hsic']:.4f} p={row['digit_p']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "hsic_metrics.csv"), index=False)

    cols = ["source", "method", "k", "n_dims"]
    for name in variables:
        cols += [f"{name}_hsic", f"{name}_null", f"{name}_p"]
    table = df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    header = (
        "HSIC (normalized, fixed sqrt(d) bandwidth) — RGB-MNIST\n"
        "digit: LOWER = less detected dependence | b/rot: HIGHER = more preserved\n"
        f"n_subsample={take}, n_perm={args.n_perm}; "
        f"smallest attainable p = {1 / (1 + args.n_perm):.5f}\n"
        "Read one-sided: rejecting independence is evidence of dependence;\n"
        "failing to reject is NOT evidence of independence.\n"
    )
    with open(os.path.join(args.outdir, "hsic_metrics.txt"), "w") as fh:
        fh.write(header + "\n" + table + "\n")
    print("\n" + header + "\n" + table)
    print(f"\nSaved hsic_metrics.{{csv,txt}} in {args.outdir}/")


if __name__ == "__main__":
    main()
