"""Figure 1: the accuracy-complexity frontier for all five arms.

Replaces the notebook-produced ``r2_pareto_front_test.png``, and regenerates it on the
**aperture-corrected** ``lgm_tot_p50`` target, which is the paper's primary: a fibre
mass has aperture selection built in, so a selection-function claim resting on it
invites the reply that we have rediscovered the aperture.

Reads stored fronts only; recomputes nothing.

**Three data-handling choices, all load-bearing.**

1. *The frontier is the best accuracy achievable at complexity <= L*, taken within a
   seed before aggregating. This is what a Pareto front means -- a length-20 solution
   is still available on a length-27 budget -- and it is also what makes the curve
   robust. Without it the figure inherits the numerical pathologies documented in the
   methodology appendix: the ``z`` arm seed 43 has Test_R2 **-6.4e18** at length 24
   and **0.119** at length 27, both with Train_R2 0.593, because raw unscaled redshift
   lets the optimiser find constant pairs that overflow on test rows outside the
   fitting subsample.
2. *Medians over seeds, not means.*
3. *Only lengths reached by at least two seeds.* Note the median does not protect
   against a pathological entry when only two seeds contribute -- it returns their
   mean -- which is why (1) is doing the real work.

**What this figure does and does not claim.** It characterises how accuracy trades
against expression length for each representation. ``cond+z`` leading ``orig`` is a
real result on the sample as observed and is stated as one. That the advantage does
not survive volume-limiting, and that the information-matched ``orig+z`` control is
more accurate still, belong with the interpretation and are deferred to the discussion
-- see ``OUTLINE.md`` section 5.1.

Usage:
    python plot_frontier_all_arms.py
    python plot_frontier_all_arms.py --fronts-prefix vollim_z0.150_lgm_tot_p50 \
        --tag vollim
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


def frontier(prefix: str, arm: str, min_seeds: int = 2) -> pd.DataFrame:
    """Median test R^2 per expression length, over seeds."""
    paths = sorted(glob.glob(f"job_results_{prefix}_{arm}_seed*/pareto_fronts.csv"))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths])
    df = df[np.isfinite(df["Test_R2"])]

    # Best achievable at complexity <= L, within each seed, before aggregating.
    best = []
    for seed, group in df.groupby("Seed"):
        ordered = group.sort_values("Length").copy()
        ordered["Test_R2"] = ordered["Test_R2"].cummax()
        ordered["Seed"] = seed
        best.append(ordered)
    df = pd.concat(best)

    g = df.groupby("Length")["Test_R2"]
    out = pd.DataFrame({"median": g.median(), "n_seeds": g.count()})
    return out[out["n_seeds"] >= min_seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fronts-prefix", default="lgm_tot_p50")
    parser.add_argument("--arms", nargs="+", default=ARM_ORDER)
    parser.add_argument("--tag", default="")
    parser.add_argument("--outdir", default="plot_frontier_all_arms_results")
    args = parser.parse_args()

    style = use_science()
    print(f"style: {style}")

    os.makedirs(args.outdir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    stem = f"{args.outdir}/plot_frontier_all_arms{suffix}"

    curves, rows = {}, []
    for arm in args.arms:
        f = frontier(args.fronts_prefix, arm)
        if f.empty:
            print(f"  [warn] no fronts for {arm}")
            continue
        curves[arm] = f
        print(f"  {arm:7s}: {len(f)} lengths, ceiling {f['median'].max():.3f}")
        for length, r in f.iterrows():
            rows.append(
                {
                    "arm": arm,
                    "Length": length,
                    "median_Test_R2": r["median"],
                    "n_seeds": int(r["n_seeds"]),
                }
            )
    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)
    print(f"wrote {stem}.csv")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    tidy(ax)

    xmax = max(f.index.max() for f in curves.values())
    for arm in args.arms:
        if arm not in curves:
            continue
        f = curves[arm]
        colour = ARM_COLOUR[arm]
        ax.plot(
            f.index,
            f["median"],
            "-o",
            color=colour,
            linewidth=2.0,
            markersize=4.2,
            markeredgecolor=SURFACE,
            markeredgewidth=1.1,
            label=arm,
            zorder=4,
        )
        last = f.index.max()
        ax.annotate(
            arm,
            xy=(last, f.loc[last, "median"]),
            xytext=(6, 0),
            textcoords="offset points",
            color=colour,
            fontsize=8,
            va="center",
        )

    ax.set_xlim(right=xmax + 4.5)
    ax.set_xlabel("expression length", fontsize=9)
    ax.set_ylabel("test $R^2$ (median over 3 seeds)", fontsize=9)
    ax.set_title(
        "Accuracy against expression complexity, by representation",
        fontsize=10,
        color=INK,
        loc="left",
        pad=8,
    )
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    fig.text(
        0.0,
        -0.06,
        f"Target: {target_label(args.fronts_prefix)}, seeds 42/43/44. Medians over"
        " seeds at lengths reached by at least two seeds; each curve is the best"
        "\naccuracy achievable at that complexity or below. Redshift alone (z) reaches"
        " 0.589 -- the first sign that much of what these representations\nencode is"
        " the survey's selection function rather than the physics.",
        fontsize=6.5,
        color=INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
