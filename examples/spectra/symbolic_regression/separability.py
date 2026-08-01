"""Are the cond+z equations additively separable in latent and redshift? (TODO item 3c)

Intro claim 1 says the representation is factorised: redshift lives in its own
coordinate and ``cond`` carries what is left. The equation-level version of that claim
is additive separability. For a fitted ``f(x, z)``, define

    delta(x, z) = f(x, z) - f(0, z)

If ``f`` is additively separable -- ``f = g(x) + h(z)`` -- then delta = g(x) - g(0),
which does not depend on z at all. So probing z -> delta measures how far the fitted
equation is from separable, in the paper's own idiom.

**This is not implied by item 3(a).** A probe cannot recover z from ``cond``, but
statistical independence of the coordinates does not give additive separability of a
fitted function: a multiplicative ``f = g(x)h(z)`` has delta = (g(x) - g(0))h(z), which
still depends on z. Separability is a genuinely additional property.

**What this script cannot tell you on its own.** It produces a fraction -- "N% of front
equations are separable" -- with no scale to read it against. That fraction could
reflect the representation, or simply what this search on this target produces. Settling
that needs the same measurement on an ``orig+z`` arm, which does **not currently exist**
(``utils.fit_sym_fn`` has branches only for ``cond+z`` and ``z``; ``orig`` carries no
redshift feature). Run this first to see whether a control would even be informative --
a result at 100% or near 0% is read very differently from one in the middle.

**Two measures of z-dependence in delta**, because a linear one alone would miss
curvature:

- ``r2_linear`` -- ordinary least squares, z -> delta.
- ``r2_binned`` -- nonparametric: bin by z quantile, take the bin means, and report the
  variance they explain. Catches any monotone or non-monotone shape a linear fit misses.

``r2_binned`` is biased upward at finite n, so a permutation null (z shuffled against
delta) gives the chance level to compare against. Both are reported per equation.

Rows where delta is constant are flagged, not scored: those are parents that never used
a latent, so they are trivially "separable" and say nothing about factorisation.

Usage:
    python separability.py
    python separability.py --feature lgm_tot_p50
    python separability.py --feature vollim_z0.150_lgm_tot_p50 --embeddings-cut 0.150
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_orig_z import swap_cond_for_orig
from val_rescore import build_matrices, build_merged, evaluate_equation

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a897f"
SURFACE = "#fcfcfb"
MAIN = "#2a78d6"
NULL = "#8a897f"

N_BINS = 40
N_PERM = 20
FLAT_TOL = 1e-9


def binned_r2(z: np.ndarray, delta: np.ndarray, n_bins: int = N_BINS) -> float:
    """Variance of delta explained by its conditional mean given z."""
    edges = np.quantile(z, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return float("nan")
    idx = np.clip(np.digitize(z, edges[1:-1]), 0, len(edges) - 2)
    fitted = np.empty_like(delta)
    for b in np.unique(idx):
        m = idx == b
        fitted[m] = delta[m].mean()
    ss_tot = np.sum((delta - delta.mean()) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - np.sum((delta - fitted) ** 2) / ss_tot


def linear_r2(z: np.ndarray, delta: np.ndarray) -> float:
    if np.std(z) <= 0 or np.std(delta) <= 0:
        return float("nan")
    return float(np.corrcoef(z, delta)[0, 1] ** 2)


def permutation_null(
    z: np.ndarray, delta: np.ndarray, rng: np.random.Generator, n: int = N_PERM
) -> float:
    """Mean binned R^2 under z shuffled against delta -- the finite-n floor."""
    vals = [binned_r2(z, rng.permutation(delta)) for _ in range(n)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def load_fronts(feature: str, embed_type: str) -> pd.DataFrame:
    pattern = f"job_results_{feature}_{embed_type}_seed*/pareto_fronts.csv"
    paths = sorted(glob.glob(pattern))
    if not paths:
        message = f"no fronts matched {pattern!r} -- run from symbolic_regression/"
        raise SystemExit(message)
    df = pd.concat([pd.read_csv(p) for p in paths]).reset_index(drop=True)
    return df.drop_duplicates(subset=["Seed", "Length"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = os.environ.get("DATA_ROOT", "/home/birr0/local_data")
    parser.add_argument("--feature", default="LGM_FIB_P50")
    parser.add_argument(
        "--target",
        default=None,
        help="catalogue column if it differs from --feature (vollim runs prefix it)",
    )
    parser.add_argument("--embed-type", default="cond+z")
    parser.add_argument(
        "--embeddings",
        default=f"{data_root}/sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0",
    )
    parser.add_argument(
        "--embeddings-cut",
        default=None,
        help="use the volume-limited embeddings at this cut instead",
    )
    parser.add_argument("--galspec", default=f"{data_root}/galSpecExtra-dr8.fits")
    parser.add_argument("--specgals-home", default=f"{data_root}/sdss")
    parser.add_argument(
        "--swap-cond-for-orig",
        action="store_true",
        help="score the orig+z control arm from run_orig_z.py (same column swap)",
    )
    parser.add_argument("--outdir", default="separability_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target = args.target
    if target is None:
        # the volume-limited fronts prefix the job dir with vollim_z{cut}_
        target = (
            args.feature.split("_", 2)[-1]
            if args.feature.startswith("vollim_")
            else args.feature
        )
    root = args.embeddings
    if args.embeddings_cut:
        root = (
            f"{data_root}/vol_limited_embeddings_7655991_0_allsplits"
            f"/z={args.embeddings_cut}"
        )

    os.makedirs(args.outdir, exist_ok=True)
    arm = args.embed_type.replace("+", "_")
    stem = f"{args.outdir}/separability_{args.feature}_{arm}"

    print(f"fronts: {args.feature} | {args.embed_type}   target column: {target}")
    merged = build_merged(root, args.galspec, args.specgals_home)
    if args.swap_cond_for_orig:
        print("  scoring the orig+z control: cond column <- orig")
        merged = swap_cond_for_orig(merged)
    mats = build_matrices(merged, target, args.embed_type)
    X = mats["X_test"]
    z = X[:, -1].copy()
    X0 = X.copy()
    X0[:, :-1] = 0.0  # zero every latent, keep raw redshift in the last column
    print(f"test rows {len(X)}, z range [{z.min():.4f}, {z.max():.4f}]")

    rng = np.random.default_rng(args.seed)
    fronts = load_fronts(args.feature, args.embed_type)
    rows = []
    for row in fronts.itertuples():
        record = {"Seed": row.Seed, "Length": row.Length, "Test_R2": row.Test_R2}
        try:
            full = evaluate_equation(row.Equation, X)
            zeroed = evaluate_equation(row.Equation, X0)
            delta = full - zeroed
            if not np.all(np.isfinite(delta)):
                record["status"] = "non-finite"
            elif np.std(delta) <= FLAT_TOL:
                record["status"] = "delta constant (parent uses no latent)"
            else:
                record["status"] = "scored"
                record["r2_linear"] = linear_r2(z, delta)
                record["r2_binned"] = binned_r2(z, delta)
                record["r2_binned_null"] = permutation_null(z, delta, rng)
                record["delta_std"] = float(np.std(delta))
        except Exception as exc:  # a bad equation must not stop the sweep
            record["status"] = f"error: {type(exc).__name__}"
        rows.append(record)

    out = pd.DataFrame(rows)
    out.to_csv(f"{stem}.csv", index=False)
    print(f"\nwrote {stem}.csv")

    print(out["status"].value_counts().to_string())
    scored = out[out["status"] == "scored"]
    if scored.empty:
        print("nothing scored")
        return

    null = scored["r2_binned_null"].median()
    print(
        f"\n{len(scored)} equations scored. Permutation null (binned R^2): {null:.4f}"
    )
    print("z-dependence remaining in delta:")
    for col in ("r2_linear", "r2_binned"):
        q = scored[col].quantile([0.25, 0.5, 0.75])
        print(
            f"  {col:10s} median {q[0.5]:.4f}  IQR [{q[0.25]:.4f}, {q[0.75]:.4f}]"
            f"  max {scored[col].max():.4f}"
        )
    for thresh in (0.01, 0.05, 0.10):
        n = int((scored["r2_binned"] < thresh).sum())
        print(
            f"  binned R^2 < {thresh:.2f}: {n}/{len(scored)}"
            f" ({100 * n / len(scored):.0f}%)"
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.scatter(
        scored["Length"],
        scored["r2_binned"],
        s=34,
        color=MAIN,
        edgecolor=SURFACE,
        linewidth=1.0,
        zorder=3,
        label="z-dependence left in $\\Delta$",
    )
    ax.axhline(null, color=NULL, linestyle="--", linewidth=1.4, zorder=2)
    ax.text(
        0.99,
        null,
        f" permutation null {null:.3f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax.set_xlabel("expression length", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel(
        "binned $R^2$ of  z $\\rightarrow \\Delta$", fontsize=9.5, color=INK_SECONDARY
    )
    ax.set_title(
        f"Additive separability of the {args.embed_type} equations",
        fontsize=11,
        color=INK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.text(
        0.0,
        -0.06,
        "$\\Delta(x,z) = f(x,z) - f(0,z)$. Separable equations leave no z-dependence in"
        " $\\Delta$; points at the null are separable.\nA fraction here has no scale"
        " without the same measurement on a non-factorised arm — see TODO item 3(c).",
        fontsize=7.5,
        color=INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
