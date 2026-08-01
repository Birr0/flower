"""What do the cond+z equations reduce to when every latent dimension is set to zero?

The cond+z arm fits log stellar mass from the redshift-removed embedding plus redshift
handed back as an explicit input (``X11``). Setting ``X1..X10 = 0`` collapses each
fitted expression to a pure function of redshift, which is the cleanest way to see what
the symbolic regressor thinks the mass-redshift relation is, separately from what the
latent adds on top of it.

**The recorded R^2 does not survive the reduction.** ``pareto_fronts.csv`` stores the
score of the *full* equation, latents active. Zeroing them leaves a strictly weaker
model whose score is not recorded anywhere, so ``parent_Test_R2`` here is an upper bound
and usually a wild one -- the best full equation scores 0.920 while the same sweep's
z-only arm tops out at 0.657, and that arm searched specifically for the best function
of z alone. The score transfers only for the handful of front entries whose parent never
used a latent (``reduced_R2_known``). Computing the reductions' true R^2 needs the test
split, which these scripts never load.

``parent_Test_R2`` orders and shades the curves here; it is never a quotable result.
``val_rescore.py`` re-scores the same fronts on the untouched val split, and flags 15
entries -- all in the ``z`` and ``cond+z`` arms -- whose stored strings cannot be
evaluated at all, because pyoperon printed a degenerate constant pair in fixed point.
Five are L=5 entries whose reductions appear in this figure, so check ``Refit_Kind`` in
``val_rescore_results/`` before quoting any individual reduced equation.

Note what zero means here. ``fit_sym_fn`` standardises the embedding columns (fit on
train) but appends redshift raw, so ``X_i = 0`` is the *sample mean* of latent dimension
i, not an absence of information. The reduced curve is therefore a conditional slice
through the fit at the mean latent -- it is not the marginal M*(z) relation of the
sample, and should not be compared against a literature M*-z fit as though it were.

Reductions are symbolic (sympy), so the printed form is exact given the fitted
constants; only the final constant rounding is cosmetic.

Usage:
    python latent_zero_limit.py
    python latent_zero_limit.py --seed 42 --precision 4
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
import sympy as sp
from matplotlib.colors import LinearSegmentedColormap, Normalize

RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", RAMP)

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a897f"
SURFACE = "#fcfcfb"

Z_INDEX = 11
Z = sp.Symbol("z", positive=True)


def to_sympy(equation: str, n_vars: int = 11) -> sp.Expr:
    """Parse a printed pyoperon expression into a sympy expression in X1..Xn, z."""
    text = equation.replace("^", "**")
    local = {f"X{i}": sp.Symbol(f"X{i}") for i in range(1, n_vars + 1)}
    local["X" + str(Z_INDEX)] = Z
    local.update({"exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Abs": sp.Abs})
    return sp.sympify(text, locals=local)


def reduce_to_redshift(expr: sp.Expr, precision: int) -> sp.Expr:
    """Zero every latent dimension and simplify what is left."""
    zeros = {sp.Symbol(f"X{i}"): 0 for i in range(1, Z_INDEX)}
    reduced = sp.simplify(expr.subs(zeros))
    return sp.N(reduced, precision)


def load_fronts(feature: str, embed_type: str) -> pd.DataFrame:
    pattern = f"job_results_{feature}_{embed_type}_seed*/pareto_fronts.csv"
    paths = sorted(glob.glob(pattern))
    if not paths:
        message = f"no fronts matched {pattern!r} -- run from symbolic_regression/"
        raise SystemExit(message)
    df = pd.concat([pd.read_csv(p) for p in paths]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Seed", "Length"])
    return df[np.isfinite(df["Test_R2"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature", default="LGM_FIB_P50")
    parser.add_argument("--embed-type", default="cond+z")
    parser.add_argument("--seed", type=int, default=None, help="default: every seed")
    parser.add_argument("--precision", type=int, default=6, help="constant digits kept")
    parser.add_argument(
        "--zmin", type=float, default=0.02, help="low-z end of the plot"
    )
    parser.add_argument(
        "--zmax", type=float, default=0.3, help="z_x cut used at fit time"
    )
    parser.add_argument("--outdir", default="latent_zero_limit_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    arm = args.embed_type.replace("+", "_")
    stem = f"{args.outdir}/latent_zero_limit_{arm}"

    df = load_fronts(args.feature, args.embed_type)
    if args.seed is not None:
        df = df[df["Seed"] == args.seed]
    print(f"{len(df)} front entries across {df['Seed'].nunique()} seeds")

    # Below ~0.02 the reductions are extrapolating past the sample and diverge; the
    # grid starts inside the data so that divergence does not set the y-scale.
    z_grid = np.linspace(args.zmin, args.zmax, 200)
    rows = []
    for row in df.itertuples():
        try:
            expr = to_sympy(row.Equation)
            reduced = reduce_to_redshift(expr, args.precision)
            # Zeroing the latents can put a zero in a denominator or a negative base
            # under a fractional power; such an expression has no real reduction.
            if reduced.has(sp.zoo, sp.oo, sp.nan, sp.I):
                message = "singular once the latents are zeroed"
                raise ValueError(message)
            fn = sp.lambdify(Z, reduced, "numpy")
            values = np.asarray(fn(z_grid), dtype=float) * np.ones_like(z_grid)
            if not np.all(np.isfinite(values)):
                message = "non-finite over the z grid"
                raise ValueError(message)
        except (
            TypeError,
            ValueError,
            KeyError,
            ZeroDivisionError,
            sp.SympifyError,
        ) as exc:
            print(f"[warn] seed {row.Seed} length {row.Length}: {exc}")
            continue

        uses_z = Z in reduced.free_symbols
        # The recorded R^2 belongs to the *full* equation. It carries over to the
        # reduction only when the parent never used a latent in the first place --
        # otherwise the reduction is a strictly weaker model with an unknown score.
        latents = {sp.Symbol(f"X{i}") for i in range(1, Z_INDEX)}
        parent_uses_latents = bool(to_sympy(row.Equation).free_symbols & latents)
        rows.append(
            {
                "Seed": row.Seed,
                "Length": row.Length,
                "parent_Test_R2": row.Test_R2,
                "parent_uses_latents": parent_uses_latents,
                "reduced_R2_known": not parent_uses_latents,
                "depends_on_z": uses_z,
                "reduced": str(reduced),
                "logM_at_z0.05": float(fn(0.05)),
                "logM_at_z0.10": float(fn(0.10)),
                "logM_at_z0.20": float(fn(0.20)),
                "logM_at_z0.30": float(fn(0.30)),
                "curve": values,
                "Equation": row.Equation,
            }
        )

    out = pd.DataFrame(rows).sort_values(["Seed", "Length"]).reset_index(drop=True)
    csv_path = f"{stem}.csv"
    out.drop(columns=["curve"]).to_csv(csv_path, index=False)

    known = out[out["reduced_R2_known"]]
    print()
    print(
        "NOTE: parent_R2 is the score of the FULL equation, with all latents active."
        "\nIt is the reduction's own score only for the "
        f"{len(known)}/{len(out)} entries whose parent never used a latent"
        " (marked [R2 valid]).\nFor the rest the reduction is a strictly weaker,"
        " z-only model whose R^2 is not recorded anywhere;\nthe z-only arm of the same"
        " sweep is the reference for what a function of z alone can reach."
    )

    print()
    for seed, group in out.groupby("Seed"):
        print(f"=== seed {seed} ===")
        for r in group.itertuples():
            flags = "" if r.depends_on_z else "   [z drops out]"
            flags += "   [R2 valid]" if r.reduced_R2_known else ""
            print(f"  L={r.Length:2d}  parent_R2={r.parent_Test_R2:.4f}{flags}")
            print(f"      log M* = {r.reduced}")
        print()

    # --- figure: every reduced curve, shaded by the parent equation's test R^2 -------
    fig, ax = plt.subplots(figsize=(9.0, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    norm = Normalize(vmin=out["parent_Test_R2"].min(), vmax=out["parent_Test_R2"].max())

    for r in out.itertuples():
        ax.plot(
            z_grid,
            r.curve,
            color=SEQ(norm(r.parent_Test_R2)),
            linewidth=2.0,
            alpha=0.85,
            solid_capstyle="round",
            zorder=3,
        )

    best = out.loc[out["parent_Test_R2"].idxmax()]
    ax.plot(z_grid, best["curve"], color="#0d366b", linewidth=3.0, zorder=4)
    ax.text(
        0.025,
        0.97,
        f"thick line: reduction of the best full equation (seed {best['Seed']},"
        f" L={best['Length']})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=INK,
    )

    ax.set_xlabel("redshift $z$", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel(
        "$\\log_{10} M_*$ at mean latent  (all $X_i = 0$)",
        fontsize=10,
        color=INK_SECONDARY,
    )
    ax.set_title(
        f"{args.feature}: the {args.embed_type} equations with every latent"
        " dimension zeroed"
        f"\n{len(out)} Pareto-front expressions reduced to functions of $z$ alone",
        fontsize=12,
        color=INK,
        loc="left",
        pad=12,
        linespacing=1.5,
    )
    ax.grid(color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, length=3, labelsize=9)

    smap = plt.cm.ScalarMappable(cmap=SEQ, norm=norm)
    cbar = fig.colorbar(smap, ax=ax, pad=0.015, fraction=0.04)
    cbar.set_label(
        "test $R^2$ of the FULL parent equation\n(not of the reduced curve)",
        fontsize=9,
        color=INK_SECONDARY,
    )
    cbar.ax.tick_params(labelsize=8, colors=INK_MUTED)
    cbar.outline.set_visible(False)

    fig.text(
        0.008,
        0.005,
        "Latents are standardised, so X_i = 0 is the sample mean of each dimension:"
        " this is a conditional slice at the mean latent, not the marginal M*-z"
        " relation.\nShading is the FULL equation's R^2 and overstates the reduced"
        " curve, which is a z-only model: the z-only arm of the same sweep tops out"
        " at R^2 = 0.657.\nCurves are symbolic reductions of the fitted expressions;"
        " no refitting.",
        fontsize=8,
        color=INK_MUTED,
        va="bottom",
        linespacing=1.5,
    )

    png_path = f"{stem}.png"
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(png_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
