"""Per-component dependence on redshift *and* on each downstream physical parameter.

Answers the follow-up from NeurIPS Reviewer #1, who objected that our removal/
preservation comparison is not controlled: Flower retains all 10 dimensions while the
FastICA and iVAE rows retain 2 and 1, and "the component-selection protocol is unclear".

``redshift_motivation.py`` already measures per-component dependence, but it collapses
the physical targets with ``np.max`` before plotting, so the figure cannot answer "which
parameter lives on which component". This dumps the uncollapsed table: for every
component of every representation, its dependence on ``z`` and on each of logM*, logSFR
and A_v, as data rather than as bars.

Two measures per pair, because neither dominates (see ``flower.evaluation.dependence``):

- ``|rho|`` -- absolute Pearson correlation. This is the *actual selection rule*
  used by ``drop_top_k_dependent(dependence="continuous")``, so the ``rho_z``
  ranking below is literally the order in which the sweep deletes components.
- ``eta`` -- correlation ratio against a quantile-binned target, on the ``|rho|``
  scale (``squared=False``). Unlike Pearson it detects *nonlinear* dependence.
  Where ``eta`` greatly exceeds ``|rho|`` for ``z``, the linear selection rule is
  failing to see redshift a nonlinear probe can still recover -- which matters
  most for iVAE, whose identifiability holds only up to a pointwise transform.

Representations are all 10-dim and are compared at equal dimensionality: the raw Spender
embedding, FastICA sources, iVAE sources, and Flower's conditional seed.

**The iVAE arm is confounded and the CSV says so.** ``ivae_sweep.train_ivae`` hardcodes
``condition_encoder=True``, so its encoder takes ``z`` as an input and can re-encode it
into every source. Read its rows as an upper bound on iVAE's z-dependence, not as a
property of nonlinear ICA. FastICA has no such conditioning and is the clean baseline.

Usage:
    python component_table.py --spender spender_I --outdir component_table_results
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ivae_sweep import RANDOM_STATE, SPENDER_MAP, TARGETS, encode, load_data, train_ivae
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

from flower.evaluation.dependence import abs_pearson, correlation_ratio

N_BINS = 20  # quantile bins used to make a continuous target discrete for eta


def _bin(y: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Quantile-bin a continuous variable into labels for ``correlation_ratio``."""
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(y, edges)


def _scores(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-component (|Pearson|, eta) of features ``x`` against continuous ``y``."""
    return abs_pearson(x, y), correlation_ratio(x, _bin(y), squared=False)


def component_table(
    name: str, x: np.ndarray, z: np.ndarray, targets: dict
) -> pd.DataFrame:
    """One row per component: dependence on z and on every physical target."""
    rho_z, eta_z = _scores(x, z)
    rows = []
    # rank 0 = deleted first by the sweep's |rho|-with-z rule
    order = np.argsort(rho_z)[::-1]
    for rank, j in enumerate(order):
        row = {
            "representation": name,
            "component": int(j),
            "drop_rank": rank,
            "rho_z": rho_z[j],
            "eta_z": eta_z[j],
        }
        for t in TARGETS:
            rho_t, eta_t = _scores(x, targets[t])
            row[f"rho_{t}"] = rho_t[j]
            row[f"eta_{t}"] = eta_t[j]
        rows.append(row)
    return pd.DataFrame(rows)


def null_level(x: np.ndarray, z: np.ndarray, seed: int) -> dict:
    """Dependence against shuffled z -- the floor below which a score is noise."""
    rng = np.random.default_rng(seed)
    rho, eta = _scores(x, rng.permutation(z))
    return {"rho_null": float(rho.max()), "eta_null": float(eta.max())}


def _panel(ax, name: str, frame: pd.DataFrame) -> None:
    sub = frame.sort_values("drop_rank")
    xs = np.arange(len(sub))
    ax.bar(xs, sub["rho_z"], color="tab:red", alpha=0.75, label="|rho| with z")
    for t, marker in zip(TARGETS, ["-o", "-s", "-^"], strict=False):
        ax.plot(xs, sub[f"rho_{t}"], marker, ms=4, lw=1.2, label=f"|rho| with {t}")
    ax.set_title(name)
    ax.set_xlabel("component (sorted by |rho| with z = deletion order)")
    ax.set_ylabel("|correlation|")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spender", default="spender_I", choices=list(SPENDER_MAP))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-filter", type=int, default=300000)
    parser.add_argument("--outdir", default="component_table_results")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    (o_tr, c_tr, _, z_tr, tg_tr), _ = load_data(
        args.spender, args.n_train, args.n_filter
    )
    x_tr = StandardScaler().fit_transform(o_tr).astype(np.float32)
    cond = StandardScaler().fit_transform(c_tr).astype(np.float32)
    d = x_tr.shape[1]
    print(f"train {x_tr.shape}; z range [{z_tr.min():.3f}, {z_tr.max():.3f}]")

    print("Fitting FastICA...")
    ica = FastICA(
        n_components=d, random_state=args.seed, max_iter=1000, whiten="unit-variance"
    )
    f_tr = ica.fit_transform(x_tr)

    print(f"Training iVAE ({args.epochs} epochs, condition_encoder=True)...")
    u_tr = z_tr.reshape(-1, 1).astype(np.float32)
    ivae = train_ivae(x_tr, u_tr, epochs=args.epochs, seed=args.seed)
    s_tr, _ = encode(ivae, x_tr, u_tr)

    reps = {
        "Raw (Spender)": x_tr,
        "FastICA": f_tr,
        "iVAE (cond-encoder)": s_tr,
        "Flower (cond seed)": cond,
    }
    df = pd.concat(
        [component_table(n, x, z_tr, tg_tr) for n, x in reps.items()], ignore_index=True
    )
    out_csv = os.path.join(args.outdir, "component_table.csv")
    df.to_csv(out_csv, index=False)

    nulls = {n: null_level(x, z_tr, args.seed) for n, x in reps.items()}
    with open(os.path.join(args.outdir, "params.json"), "w") as fh:
        json.dump({"args": vars(args), "n_bins": N_BINS, "nulls": nulls}, fh, indent=2)

    fig, axes = plt.subplots(1, len(reps), figsize=(6 * len(reps), 5))
    for ax, (n, _) in zip(axes, reps.items(), strict=False):
        _panel(ax, n, df[df["representation"] == n])
    fig.suptitle(
        f"{args.spender}: per-component dependence on z and on each physical parameter",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "component_table.png"), dpi=140)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    for n in reps:
        sub = df[df["representation"] == n].drop(columns=["representation"])
        print(f"\n=== {n} ===  (null: {nulls[n]})")
        print(sub.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
