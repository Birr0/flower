"""Figure 4: every representation, when the selection function is flattened.

The paper's central result, and the one figure that carries it. Each arm is drawn as
a slope between its ceiling on the flux-limited sample and its ceiling on the
volume-limited one, so two things are visible at once:

1. **Redshift stops predicting stellar mass entirely** -- the ``z`` arm falls from
   0.589 to ~0.000. On the sample as observed, redshift alone explains more of the
   variance in stellar mass than four of the five representations do. Remove the
   flux limit and it explains none of it.
2. **The four latent representations converge.** Flux-limited they span 0.604 in
   test R^2; volume-limited they span 0.082. Most of what distinguished them was how
   much selection signal each carried, not how well each encodes the physics. ``z``
   is excluded from that span deliberately -- it does not converge with them, it
   drops out of their range altogether.

``cond`` is the only arm that *improves*, which is the same fact from the other
side: what conditioning removed was selection-carried rather than spectral.

Ceilings are computed exactly as in Figure 1 -- best accuracy achievable at
complexity <= L within a seed, then median over seeds -- so the two figures cannot
disagree.

Usage:
    python plot_selection_collapse.py
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figstyle import (
    ARM_COLOUR,
    ARM_ORDER,
    INK,
    INK_MUTED,
    SURFACE,
    target_label,
    tidy,
    use_science,
)


def ceiling(prefix: str, arm: str, min_seeds: int = 2) -> float:
    """Best achievable accuracy, matching Figure 1's construction."""
    paths = sorted(glob.glob(f"job_results_{prefix}_{arm}_seed*/pareto_fronts.csv"))
    if not paths:
        return float("nan")
    df = pd.concat([pd.read_csv(p) for p in paths])
    df = df[np.isfinite(df["Test_R2"])]
    best = []
    for seed, group in df.groupby("Seed"):
        ordered = group.sort_values("Length").copy()
        ordered["Test_R2"] = ordered["Test_R2"].cummax()
        ordered["Seed"] = seed
        best.append(ordered)
    df = pd.concat(best)
    g = df.groupby("Length")["Test_R2"]
    out = pd.DataFrame({"median": g.median(), "n_seeds": g.count()})
    out = out[out["n_seeds"] >= min_seeds]
    return float(out["median"].max()) if len(out) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flux-prefix", default="lgm_tot_p50")
    parser.add_argument("--vol-prefix", default="vollim_z0.150_lgm_tot_p50")
    parser.add_argument("--cut", default="0.150")
    parser.add_argument("--outdir", default="plot_selection_collapse_results")
    args = parser.parse_args()

    style = use_science()
    print(f"style: {style}")

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{args.outdir}/plot_selection_collapse"

    rows = []
    for arm in ARM_ORDER:
        flux = ceiling(args.flux_prefix, arm)
        vol = ceiling(args.vol_prefix, arm)
        rows.append(
            {
                "arm": arm,
                "flux_limited": flux,
                "volume_limited": vol,
                "change": vol - flux,
            }
        )
        print(f"  {arm:7s}: {flux:.3f} -> {vol:.3f}   ({vol - flux:+.3f})")
    df = pd.DataFrame(rows)
    df.to_csv(f"{stem}.csv", index=False)
    print(f"wrote {stem}.csv")

    # Span over the *latent* arms only. z is not converging with them: volume-limited
    # it falls below every one, so including it would measure its collapse rather than
    # their convergence.
    latent = df[df["arm"] != "z"]
    span_flux = latent["flux_limited"].max() - latent["flux_limited"].min()
    span_vol = latent["volume_limited"].max() - latent["volume_limited"].min()
    print(f"latent-arm span: {span_flux:.3f} -> {span_vol:.3f}")

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    x = [0.0, 1.0]

    # Label positions, decluttered: arms that land close together on the right would
    # otherwise print on top of each other (0.547 and 0.536 differ by 0.011).
    def declutter(values: dict[str, float], min_gap: float) -> dict[str, float]:
        placed: dict[str, float] = {}
        for arm, v in sorted(values.items(), key=lambda kv: kv[1]):
            y = v
            for other in placed.values():
                if abs(y - other) < min_gap:
                    y = other + min_gap
            placed[arm] = y
        return placed

    span_axis = df[["flux_limited", "volume_limited"]].to_numpy()
    gap = 0.030 * (np.nanmax(span_axis) - np.nanmin(span_axis))
    left_y = declutter(dict(zip(df["arm"], df["flux_limited"], strict=True)), gap)
    right_y = declutter(dict(zip(df["arm"], df["volume_limited"], strict=True)), gap)

    for _, r in df.iterrows():
        colour = ARM_COLOUR[r["arm"]]
        ax.plot(
            x,
            [r["flux_limited"], r["volume_limited"]],
            "-o",
            color=colour,
            linewidth=2.4,
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.4,
            zorder=4,
        )
        ax.annotate(
            f"{r['arm']}  {r['flux_limited']:.3f}",
            xy=(0.0, left_y[r["arm"]]),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7.5,
            color=colour,
        )
        ax.annotate(
            f"{r['volume_limited']:.3f}",
            xy=(1.0, right_y[r["arm"]]),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color=colour,
        )

    # The convergence, as a bracket on the right.
    lo, hi = latent["volume_limited"].min(), latent["volume_limited"].max()
    ax.annotate(
        "",
        xy=(1.30, lo),
        xytext=(1.30, hi),
        arrowprops={"arrowstyle": "<->", "color": INK_MUTED, "linewidth": 1.3},
    )
    ax.text(
        1.34,
        0.5 * (lo + hi),
        f"the four latent arms\nconverge: span {span_vol:.3f}\n(was {span_flux:.3f})",
        fontsize=7,
        color=INK_MUTED,
        va="center",
    )

    ax.set_xlim(-0.42, 1.62)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "flux-limited\n(the sample as observed)",
            f"volume-limited\n$z \\leq$ {args.cut}",
        ],
        fontsize=8,
    )
    tidy(ax, grid_axis="y")
    ax.tick_params(bottom=False)
    ax.set_ylabel("best test $R^2$ for stellar mass", fontsize=9)
    ax.set_title(
        "Flatten the selection function and the mass-redshift relation disappears",
        fontsize=10,
        color=INK,
        loc="left",
        pad=8,
    )
    fig.text(
        0.0,
        -0.045,
        f"Target: {target_label(args.flux_prefix)}, 3 seeds; ceilings as in Figure 1."
        " The z arm's"
        " collapse is corroborated by the reference models (MLP 0.587 to -0.020;"
        "\nlinear 0.542 to -0.000, no better than predicting the mean); it falls below"
        " every latent arm rather than converging with them.\ncond is the only arm that"
        " improves: what conditioning removed was selection-carried, not spectral.",
        fontsize=6.5,
        color=INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
