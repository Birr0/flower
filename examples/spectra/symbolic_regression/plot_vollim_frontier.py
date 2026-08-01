"""Does cond+z's parsimony advantage over orig survive volume-limiting? (TODO item 4)

Figure 1 of the paper claims ``cond+z`` dominates ``orig`` on the accuracy-complexity
frontier at every expression length >= 8. This re-plots that frontier on the
flux-limited sample beside the volume-limited one, and the ordering **inverts**:
``cond+z`` leads at 14 of 17 shared lengths >= 8 on the full sample and **0 of 16** once
the flux limit is removed.

The mechanism is visible in the companion arms rather than here: ``cond+z``'s edge comes
from redshift handed back as an explicit feature, and redshift is informative on a
flux-limited sample largely because it proxies selection. The z-only arm scores 0.589
flux-limited and ~0.000 volume-limited, so the edge has nothing left to stand on and
``cond`` -- having had z-correlated structure removed -- lands slightly behind ``orig``.

Recomputes nothing: reads the stored ``pareto_fronts.csv`` files only.

**Two data-handling choices, both load-bearing.**

1. *Median over seeds, not mean.* Volume-limited ``orig`` seed 42 at L=22 has Train_R2
   0.573 but Test_R2 **-27.48** -- a single blown-up front member, and the only negative
   Test_R2 among 65 ``orig`` entries. L=22 is also the one length where only that seed
   contributed, so a mean-over-seeds frontier reports the broken value verbatim and
   manufactures a spurious ``cond+z`` win there.
2. *Lengths with >= 2 seeds only.* Single-seed points at the long end are noise, and
   including them is what turns "0 of 16" into a misleading "1 of 22".

Both panels share a y-axis: the levels differ (volume-limiting compresses the target
spread, so R^2 falls everywhere) and so does the ordering, and the reader needs to see
that those are two separate facts.

Usage:
    python plot_vollim_frontier.py
    python plot_vollim_frontier.py --cut 0.100 --feature lgm_tot_p50
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

# House palette, shared with vol_limited_probe.py / latent_zero_limit.py.
# Validated as a 2-slot categorical pair on surface #fcfcfb: worst adjacent CVD
# separation dE 24.7 (protan), 33.6 normal vision, both well clear of the floors.
ARM_COLOR = {"orig": "#2a78d6", "cond+z": "#eb6834"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a897f"
SURFACE = "#fcfcfb"
LEAD_TINT = "#f6d9cb"  # where cond+z leads


def frontier(pattern: str, min_seeds: int = 2) -> pd.DataFrame:
    """Median test R^2 per expression length, over seeds."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        message = f"no fronts matched {pattern!r} -- run from symbolic_regression/"
        raise SystemExit(message)
    df = pd.concat([pd.read_csv(p) for p in paths])
    df = df[np.isfinite(df["Test_R2"])]
    g = df.groupby("Length")["Test_R2"]
    out = pd.DataFrame({"median": g.median(), "n_seeds": g.count()})
    return out[out["n_seeds"] >= min_seeds]


def panel(ax, orig: pd.DataFrame, cond: pd.DataFrame, title: str, subtitle: str) -> int:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, color=INK_MUTED, alpha=0.16, linewidth=0.6)
    ax.set_axisbelow(True)

    common = sorted(set(orig.index) & set(cond.index))
    eligible = [x for x in common if x >= 8]
    lead = [x for x in eligible if cond.loc[x, "median"] > orig.loc[x, "median"]]

    # Shade where cond+z leads -- secondary encoding for the point of the figure.
    if lead:
        o = orig.loc[common, "median"].to_numpy()
        c = cond.loc[common, "median"].to_numpy()
        ax.fill_between(
            common,
            o,
            c,
            where=(c > o),
            color=LEAD_TINT,
            alpha=0.55,
            linewidth=0,
            interpolate=True,
            zorder=1,
        )

    for arm, frame in (("orig", orig), ("cond+z", cond)):
        ax.plot(
            frame.index,
            frame["median"],
            "-o",
            color=ARM_COLOR[arm],
            linewidth=2,
            markersize=4.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=arm,
            zorder=3,
        )
        last = frame.index.max()
        ax.annotate(
            arm,
            xy=(last, frame.loc[last, "median"]),
            xytext=(5, 0),
            textcoords="offset points",
            color=ARM_COLOR[arm],
            fontsize=9,
            va="center",
            fontweight="medium",
        )

    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=12)
    ax.text(
        0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax.set_xlabel("expression length", fontsize=9.5, color=INK_SECONDARY)
    return len(lead), len(eligible)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature", default="lgm_tot_p50")
    parser.add_argument("--cut", default="0.150")
    parser.add_argument("--outdir", default="plot_vollim_frontier_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{args.outdir}/plot_vollim_frontier"
    f, cut = args.feature, args.cut

    frames = {
        "full": {
            arm: frontier(f"job_results_{f}_{arm}_seed*/pareto_fronts.csv")
            for arm in ARM_COLOR
        },
        "vollim": {
            arm: frontier(
                f"job_results_vollim_z{cut}_{f}_{arm}_seed*/pareto_fronts.csv"
            )
            for arm in ARM_COLOR
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), facecolor=SURFACE, sharey=True)
    lead_full = panel(
        axes[0],
        frames["full"]["orig"],
        frames["full"]["cond+z"],
        "Flux-limited (full sample)",
        "the sample Figure 1 reports",
    )
    lead_vol = panel(
        axes[1],
        frames["vollim"]["orig"],
        frames["vollim"]["cond+z"],
        f"Volume-limited, z $\\leq$ {cut}",
        "selection function flattened",
    )
    axes[0].set_ylabel(
        "test $R^2$ (median over 3 seeds)", fontsize=9.5, color=INK_SECONDARY
    )
    axes[0].legend(
        frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left"
    )

    for ax, (n, tot) in zip(axes, (lead_full, lead_vol), strict=True):
        ax.text(
            0.5,
            0.04,
            f"cond+z ahead at {n}/{tot} lengths $\\geq$ 8",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color=INK if n else INK_SECONDARY,
        )

    fig.suptitle(
        "The parsimony advantage is selection-driven,"
        " and inverts without the flux limit",
        fontsize=12.5,
        color=INK,
        x=0.008,
        ha="left",
        y=1.06,
    )
    fig.text(
        0.008,
        -0.19,
        f"Target {f}, seeds 42/43/44. Shaded where cond+z leads. Median over seeds"
        " at lengths where $\\geq$ 2 seeds contributed: volume-limited orig seed 42"
        "\nat L=22 has Test $R^2$ $-27.48$ against Train $R^2$ 0.573, the only"
        " negative among 65 entries, and is the sole seed at that length — a"
        "\nmean-based frontier would report it verbatim. Shared y-axis: $R^2$ falls"
        " everywhere under volume-limiting because the target spread compresses"
        "\n(y_std 0.585 → 0.240 dex); the change in *ordering* is the separate and"
        " load-bearing fact.",
        fontsize=7.5,
        color=INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {stem}.png")

    rows = []
    for sample, arms in frames.items():
        for arm, frame in arms.items():
            for length, r in frame.iterrows():
                rows.append(
                    {
                        "sample": sample,
                        "arm": arm,
                        "Length": length,
                        "median_Test_R2": r["median"],
                        "n_seeds": int(r["n_seeds"]),
                    }
                )
    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)
    print(f"wrote {stem}.csv")
    print(f"full sample:    cond+z ahead at {lead_full[0]}/{lead_full[1]} lengths >=8")
    print(f"volume-limited: cond+z ahead at {lead_vol[0]}/{lead_vol[1]} lengths >=8")


if __name__ == "__main__":
    main()
