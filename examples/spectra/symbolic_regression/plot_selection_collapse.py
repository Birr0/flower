"""Figure 3: every representation, across progressively wider volume-limited cuts.

The paper's central result. Each arm's ceiling is drawn on the flux-limited sample and
then on each volume-limited cut in order of increasing redshift band, so the trend is
visible rather than a single before/after pair:

1. **Redshift stops predicting stellar mass.** On the sample as observed, redshift alone
   explains more of the variance in stellar mass than three of the five representations
   do. Remove the flux limit and it explains none of it, at every cut.
2. **The four latent representations converge.** Most of what distinguished them was how
   much selection signal each carried, not how well each encodes the physics. ``z`` is
   excluded from that span deliberately -- it does not converge with them, it falls out
   of their range altogether.
3. ``cond`` is the only arm that *improves*, which is the same fact from the other side:
   what conditioning removed was selection-carried rather than spectral.

**The flux-limited sample is not a cut**, so it is drawn separated from the sequence
rather than as its first point -- it reaches z ~ 0.3 and is selected on apparent
magnitude, whereas each volume-limited point is a complete sample to its own limit.

Ceilings are computed exactly as in Figure 1 -- best accuracy achievable at complexity
<= L within a seed, then mean over seeds -- so the two figures cannot disagree.

Cuts with no fronts on disk are skipped with a warning, so this runs before a full sweep
has finished.

Usage:
    python plot_selection_collapse.py
    python plot_selection_collapse.py --cuts 0.100 0.150
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
from val_rescore import build_matrices, build_merged


def target_sigma(root: str, feature: str, galspec: str, specgals: str) -> float:
    """Standard deviation of the target on a sample's test split.

    This is the yardstick the normalised metric divides by, and it moves a lot between
    samples -- 0.585 dex flux-limited against 0.240 at z <= 0.150 -- because a wider
    volume-limited cut carries a brighter magnitude limit and so admits a narrower range
    of stellar masses.
    """
    merged = build_merged(root, galspec, specgals)
    return float(np.std(build_matrices(merged, feature, "z")["y_test"]))


def best(prefix: str, arm: str, min_seeds: int = 2) -> tuple[float, float]:
    """Best achievable (R^2, RMSE) within the complexity budget, matching Figure 1.

    Both are the running best within a seed -- cummax on R^2, cummin on MSE -- then
    averaged over seeds, so a front entry that blew up on test cannot set the value.
    """
    paths = sorted(glob.glob(f"job_results_{prefix}_{arm}_seed*/pareto_fronts.csv"))
    if not paths:
        return float("nan"), float("nan")
    df = pd.concat([pd.read_csv(p) for p in paths])
    df = df[np.isfinite(df["Test_R2"]) & np.isfinite(df["Test_MSE"])]
    best_rows = []
    for seed, group in df.groupby("Seed"):
        ordered = group.sort_values("Length").copy()
        ordered["Test_R2"] = ordered["Test_R2"].cummax()
        ordered["Test_MSE"] = ordered["Test_MSE"].cummin()
        ordered["Seed"] = seed
        best_rows.append(ordered)
    df = pd.concat(best_rows)
    g = df.groupby("Length")
    out = pd.DataFrame(
        {
            "r2": g["Test_R2"].mean(),
            "mse": g["Test_MSE"].mean(),
            "n_seeds": g["Test_R2"].count(),
        }
    )
    out = out[out["n_seeds"] >= min_seeds]
    if not len(out):
        return float("nan"), float("nan")
    return float(out["r2"].max()), float(np.sqrt(out["mse"].min()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = os.environ.get("DATA_ROOT", "/home/birr0/local_data")
    parser.add_argument("--flux-prefix", default="lgm_tot_p50")
    parser.add_argument(
        "--cuts", nargs="+", default=["0.050", "0.075", "0.100", "0.125", "0.150"]
    )
    parser.add_argument("--target", default="lgm_tot_p50")
    parser.add_argument(
        "--embeddings",
        default=f"{data_root}/sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0",
    )
    parser.add_argument(
        "--vol-root", default=f"{data_root}/vol_limited_embeddings_7655991_0_allsplits"
    )
    parser.add_argument("--galspec", default=f"{data_root}/galSpecExtra-dr8.fits")
    parser.add_argument("--specgals-home", default=f"{data_root}/sdss")
    parser.add_argument("--outdir", default="plot_selection_collapse_results")
    args = parser.parse_args()

    style = use_science()
    print(f"style: {style}")

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{args.outdir}/plot_selection_collapse"

    rows = []
    present = []
    # A cut is only included once every arm has at least two seeds. Without this a
    # sweep still in progress contributes half-built points that move on the next run.
    for cut in args.cuts:
        prefix = f"vollim_z{cut}_{args.target}"
        counts = {
            arm: len(glob.glob(f"job_results_{prefix}_{arm}_seed*/pareto_fronts.csv"))
            for arm in ARM_ORDER
        }
        short = {a: n for a, n in counts.items() if n < 2}
        if short:
            have = sum(counts.values())
            print(f"  [skip] cut {cut}: incomplete ({have}/15 cells; short: {short})")
            continue
        present.append(cut)

    print("target sigma per sample (dex):")
    sigma = {
        "flux_limited": target_sigma(
            args.embeddings, args.target, args.galspec, args.specgals_home
        )
    }
    print(f"  flux-limited  {sigma['flux_limited']:.3f}")
    for cut in present:
        sigma[cut] = target_sigma(
            f"{args.vol_root}/z={cut}", args.target, args.galspec, args.specgals_home
        )
        print(f"  z<={cut}      {sigma[cut]:.3f}")

    for arm in ARM_ORDER:
        r2, rmse = best(args.flux_prefix, arm)
        row = {
            "arm": arm,
            "flux_limited_r2": r2,
            "flux_limited_rmse": rmse,
            "flux_limited_norm": rmse / sigma["flux_limited"],
        }
        for cut in present:
            r2c, rmsec = best(f"vollim_z{cut}_{args.target}", arm)
            row[f"{cut}_r2"] = r2c
            row[f"{cut}_rmse"] = rmsec
            row[f"{cut}_norm"] = rmsec / sigma[cut]
        rows.append(row)
        trail = "  ".join(f"{c}:{row[f'{c}_norm']:.3f}" for c in present)
        print(f"  {arm:7s}: RMSE/sigma flux {row['flux_limited_norm']:.3f} | {trail}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{stem}.csv", index=False)
    print(f"wrote {stem}.csv")

    latent = df[df["arm"] != "z"]
    spans = {c: latent[f"{c}_rmse"].max() - latent[f"{c}_rmse"].min() for c in present}
    span_flux = latent["flux_limited_norm"].max() - latent["flux_limited_norm"].min()
    print(
        f"latent-arm RMSE/sigma span: flux {span_flux:.3f} | "
        + "  ".join(f"{c}:{v:.3f}" for c, v in spans.items())
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    # Flux-limited sits apart: it is not a cut, and reaches z ~ 0.3.
    x_flux = 0.0
    x_cuts = [1.0 + i for i in range(len(present))]

    for _, r in df.iterrows():
        colour = ARM_COLOUR[r["arm"]]
        ys = [r[f"{c}_norm"] for c in present]
        ax.plot(
            [x_flux],
            [r["flux_limited_norm"]],
            "o",
            color=colour,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.1,
            zorder=5,
        )
        ax.plot(
            x_cuts,
            ys,
            "-o",
            color=colour,
            linewidth=2.0,
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=4,
        )
        # Dotted tie from the flux-limited point into the sequence, so the eye connects
        # them without implying the flux-limited sample is part of the ordering.
        ax.plot(
            [x_flux, x_cuts[0]],
            [r["flux_limited_norm"], ys[0]],
            ":",
            color=colour,
            linewidth=1.1,
            alpha=0.55,
            zorder=3,
        )
        ax.annotate(
            r["arm"],
            xy=(x_cuts[-1], ys[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color=colour,
        )

    # 1.0 is "predicting the sample mean": error equal to the target's own spread.
    ax.axhline(1.0, color=INK, linewidth=1.1, linestyle=(0, (1, 1.6)), zorder=2)
    ax.text(
        x_cuts[-1] + 0.80,
        1.0,
        "predicting\nthe mean",
        fontsize=6.5,
        color=INK_MUTED,
        va="center",
        ha="right",
    )
    ax.axvline(0.5, color=INK_MUTED, linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)
    ax.set_xticks([x_flux, *x_cuts])
    ax.set_xticklabels(
        ["flux-\nlimited", *[f"$\\leq${c}" for c in present]], fontsize=8
    )
    ax.set_xlim(-0.55, x_cuts[-1] + 0.85)
    tidy(ax, grid_axis="y")
    ax.tick_params(bottom=False)
    ax.set_ylabel("best test RMSE / target $\\sigma$", fontsize=9)
    ax.set_xlabel("volume-limited cut", fontsize=9)
    ax.set_title(
        "Flatten the selection function and redshift stops predicting stellar mass",
        fontsize=10,
        color=INK,
        loc="left",
        pad=8,
    )
    fig.text(
        0.0,
        -0.045,
        f"Target: {target_label(args.flux_prefix)}, 3 seeds; best within the complexity"
        " budget, as in Figure 1. Error is divided by each sample's own target"
        "\nspread, because a wider volume-limited cut carries a brighter magnitude"
        " limit and so a narrower range of stellar masses (sigma falls 0.585"
        "\nto 0.240 dex); raw RMSE is not comparable across samples. A value of 1"
        " means the model does no better than predicting the sample mean."
        "\nThe flux-limited sample is not a cut -- it reaches z ~ 0.3 -- so it is drawn"
        " apart, tied to the sequence by a dotted line.",
        fontsize=6.5,
        color=INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
