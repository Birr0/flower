"""Does the X=0 curve land on the real M*(z) relation? cond+z vs orig+z (F12).

``mass_limit_overlay.py`` draws the ``cond+z`` X=0 reductions against the observed
density, the empirical conditional mean and the completeness limit, and finds the
learned curve sits on the conditional mean. This adds the **``orig+z`` control** to
the same axes, which turns that from a description into a test.

**Why the comparison is a factorisation test.** Zeroing the latents evaluates the
fitted ``f`` at the *mean* latent. That equals the marginal E[log M* | z] only if the
latent is independent of z -- which is exactly what conditioning is for. So if
``cond+z``'s curve lands on the empirical conditional mean and ``orig+z``'s does not,
the conditioned coordinates are behaving as independent of redshift and the original
ones are not. This is the equation-level factorisation test in the paper's own idiom,
and it succeeds where the additive-separability route failed (see the F9 retraction).

**Internal agreement is not the test.** Both arms produce tightly self-consistent
bundles of curves -- structurally unrelated forms agreeing to a few hundredths of a
dex -- and the two bundles are mutually incompatible, up to 0.27 dex apart.
Consistency measures whether the fitting reproduces itself, not whether the answer is
right. Only the comparison against the observed conditional mean separates them.

Recomputes the observed summaries here rather than reading
``mass_limit_overlay_results/``, so the empirical curve and the reductions are
guaranteed to come from the same rows.

Usage:
    python plot_x0_arms_overlay.py
    python plot_x0_arms_overlay.py --zmax 0.15
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from astropy.cosmology import FlatLambdaCDM
from figstyle import INK, INK_MUTED, SURFACE, target_label, tidy, use_science
from latent_zero_limit import Z, load_fronts, reduce_to_redshift, to_sympy
from mass_limit_overlay import kcorrection_model, limit_curve
from val_rescore import build_merged

COND = "#2a78d6"
ORIG = "#eb6834"
TRUTH = "#0b0b0b"
LIMIT = "#6b6b6b"

DEFAULT_COND_FRONTS = "LGM_FIB_P50"
DEFAULT_ORIG_FRONTS = "origz_LGM_FIB_P50"


def reductions(feature: str, z_grid: np.ndarray) -> np.ndarray:
    """Every usable X=0 reduction of an arm's fronts, evaluated on the grid."""
    curves = []
    for row in load_fronts(feature, "cond+z").itertuples():
        try:
            reduced = reduce_to_redshift(to_sympy(row.Equation), 6)
            if (
                reduced.has(sp.zoo, sp.oo, sp.nan, sp.I)
                or Z not in reduced.free_symbols
            ):
                continue
            fn = sp.lambdify(Z, reduced, "numpy")
            vals = np.asarray(fn(z_grid), dtype=float) * np.ones_like(z_grid)
            if np.all(np.isfinite(vals)):
                curves.append(vals)
        except (TypeError, ValueError, KeyError, ZeroDivisionError, sp.SympifyError):
            continue
    return np.asarray(curves)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = os.environ.get("DATA_ROOT", "/home/birr0/local_data")
    parser.add_argument("--feature", default="LGM_FIB_P50")
    parser.add_argument(
        "--embeddings",
        default=f"{data_root}/sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0",
    )
    parser.add_argument("--galspec", default=f"{data_root}/galSpecExtra-dr8.fits")
    parser.add_argument("--specgals-home", default=f"{data_root}/sdss")
    parser.add_argument(
        "--vol-root", default=f"{data_root}/vol_limited_embeddings_7655991_0"
    )
    parser.add_argument("--nbins", type=int, default=30)
    parser.add_argument("--envelope-pct", type=float, default=5.0)
    parser.add_argument("--zmin", type=float, default=0.02)
    parser.add_argument("--zmax", type=float, default=0.3)
    parser.add_argument(
        "--cond-fronts",
        default=DEFAULT_COND_FRONTS,
        help="job_results_{this}_cond+z_seed*/ for the conditioned arm",
    )
    parser.add_argument(
        "--orig-fronts",
        default=DEFAULT_ORIG_FRONTS,
        help="job_results_{this}_cond+z_seed*/ for the orig+z control",
    )
    parser.add_argument(
        "--embeddings-cut",
        default=None,
        help="use the volume-limited embeddings at this cut for the observed curve",
    )
    parser.add_argument("--tag", default="", help="suffix for the output filenames")
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="scale the y-axis to the curves rather than the completeness limit, which"
        " otherwise compresses them when the relation is shallow",
    )
    parser.add_argument("--outdir", default="plot_x0_arms_overlay_results")
    args = parser.parse_args()

    arms = [
        ("cond+z", args.cond_fronts, COND),
        ("orig+z", args.orig_fronts, ORIG),
    ]
    root = args.embeddings
    if args.embeddings_cut:
        root = (
            f"{data_root}/vol_limited_embeddings_7655991_0_allsplits"
            f"/z={args.embeddings_cut}"
        )

    style = use_science()
    print(f"style: {style}")

    os.makedirs(args.outdir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    stem = f"{args.outdir}/plot_x0_arms_overlay{suffix}"

    merged = build_merged(root, args.galspec, args.specgals_home)
    test = merged["test"]
    test = test[np.isfinite(test[args.feature]) & (test[args.feature] != -9999.0)]
    z_obs = test["z_x"].to_numpy()
    m_obs = test[args.feature].to_numpy()
    keep = (z_obs >= args.zmin) & (z_obs <= args.zmax)
    z_obs, m_obs = z_obs[keep], m_obs[keep]
    print(f"{len(z_obs)} galaxies with valid {args.feature}")

    edges = np.linspace(args.zmin, args.zmax, args.nbins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(z_obs, edges) - 1
    running_mean, envelope = [], []
    for b in range(args.nbins):
        sel = idx == b
        if sel.sum() < 20:
            running_mean.append(np.nan)
            envelope.append(np.nan)
            continue
        running_mean.append(float(np.mean(m_obs[sel])))
        envelope.append(float(np.percentile(m_obs[sel], args.envelope_pct)))
    running_mean = np.asarray(running_mean)
    envelope = np.asarray(envelope)

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    shape = limit_curve(centres, kcorrection_model(args.vol_root), cosmo)
    good = np.isfinite(envelope)
    limit = shape + float(np.mean(envelope[good] - shape[good]))

    z_grid = np.linspace(args.zmin, args.zmax, 200)
    emp_on_grid = np.interp(z_grid, centres[good], running_mean[good])

    rows = []
    bundles = {}
    for name, feature, _ in arms:
        curves = reductions(feature, z_grid)
        bundles[name] = curves
        median = np.median(curves, axis=0)
        dev = median - emp_on_grid
        print(
            f"{name:7s}: {len(curves):2d} usable reductions, "
            f"mean |dev| from the conditional mean {np.mean(np.abs(dev)):.3f} dex, "
            f"RMS {np.sqrt(np.mean(dev**2)):.3f} dex"
        )
        for z, med, e in zip(z_grid, median, emp_on_grid, strict=True):
            rows.append(
                {
                    "arm": name,
                    "z": z,
                    "median_curve": med,
                    "empirical_mean": e,
                    "deviation": med - e,
                }
            )
    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)
    print(f"wrote {stem}.csv")

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(6.8, 5.6),
        sharex=True,
        height_ratios=[3.0, 1.0],
        gridspec_kw={"hspace": 0.08},
    )
    ax.hist2d(z_obs, m_obs, bins=[120, 120], cmap="Greys", cmin=1, alpha=0.55, zorder=1)

    for name, _, colour in arms:
        for i, c in enumerate(bundles[name]):
            ax.plot(
                z_grid,
                c,
                color=colour,
                linewidth=0.9,
                alpha=0.22,
                zorder=3,
                label=f"{name} reductions" if i == 0 else None,
            )
        ax.plot(
            z_grid,
            np.median(bundles[name], axis=0),
            color=colour,
            linewidth=2.8,
            zorder=5,
            solid_capstyle="round",
        )

    ax.plot(
        centres[good],
        running_mean[good],
        color=TRUTH,
        linewidth=2.2,
        linestyle=(0, (1, 1.6)),
        zorder=6,
        label="observed conditional mean",
    )
    ax.plot(
        centres,
        limit,
        color=LIMIT,
        linewidth=1.6,
        linestyle="--",
        zorder=4,
        label="completeness limit ($r<17.77$)",
    )

    ax.set_xlim(args.zmin, args.zmax)
    medians = [np.median(bundles[n], axis=0) for n, _, _ in arms]
    if args.zoom:
        # The completeness limit runs far below a shallow relation and would flatten it.
        lo = min(np.nanmin(m) for m in [*medians, emp_on_grid])
        hi = max(np.nanmax(m) for m in [*medians, emp_on_grid])
        pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        tops = [np.nanmax(m) for m in medians]
        ax.set_ylim(
            np.nanmin(limit) - 0.4, max(*tops, np.nanpercentile(m_obs, 99.5)) + 0.25
        )

    # --- residual panel: what the mean |deviation| numbers actually look like ------
    rax.axhline(0.0, color=TRUTH, linewidth=1.6, linestyle=(0, (1, 1.6)), zorder=3)
    for (_name, _, colour), med in zip(arms, medians, strict=True):
        rax.plot(z_grid, med - emp_on_grid, color=colour, linewidth=2.4, zorder=4)
    span = max(np.nanmax(np.abs(m - emp_on_grid)) for m in medians)
    rax.set_ylim(-1.15 * span, 1.15 * span)
    tidy(rax)
    rax.set_ylabel("curve $-$ observed\n(dex)", fontsize=8)
    rax.set_xlabel("redshift $z$", fontsize=9)
    tidy(ax, grid_axis=None)
    ax.set_ylabel(f"$\\log_{{10}}$ M$_*$  ({target_label(args.feature)})", fontsize=9)
    ax.set_title(
        "Zeroing the latents recovers the real M$_*$(z) relation -- only when z"
        " was conditioned out",
        fontsize=10,
        color=INK,
        loc="left",
        pad=8,
    )
    leg = ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    for line in leg.get_lines():
        line.set_alpha(1.0)
        line.set_linewidth(2.2)

    cond_dev = np.mean(np.abs(np.median(bundles["cond+z"], axis=0) - emp_on_grid))
    orig_dev = np.mean(np.abs(np.median(bundles["orig+z"], axis=0) - emp_on_grid))
    ax.text(
        0.015,
        0.965,
        f"mean |deviation| from the observed mean:\n"
        f"  cond+z   {cond_dev:.3f} dex\n"
        f"  orig+z   {orig_dev:.3f} dex",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        color=INK_MUTED,
        family="monospace",
    )
    fig.text(
        0.0,
        -0.045,
        "The X=0 slice evaluates each fitted equation at the mean latent, which"
        " equals the marginal E[log M*|z] only if the latent is independent of z."
        "\nBoth bundles are internally tight; only cond+z lands on the observed"
        " relation, so internal agreement cannot separate them."
        "\nDensity is the observed test split.",
        fontsize=6.5,
        color=INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
