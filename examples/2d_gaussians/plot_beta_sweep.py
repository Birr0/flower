"""Plot the β sweep: removal, preservation, the KL bound, and transport cost.

Combines the coarse sweep (`beta_sweep_results/`) with the finer transition pass
(`beta_sweep_transition_results/`) into one figure, so the shape of the
removal-vs-transport-cost trade-off is visible in one look.

Four panels rather than twin axes — the measures live on different scales and
must not share one y-axis. β=0 has no place on a log axis, so it is drawn at the
far left with an explicit axis break.

Run from this directory:

    python plot_beta_sweep.py
"""

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULT_DIRS = ["beta_sweep_results", "beta_sweep_transition_results"]
OUTPATH = "beta_sweep_trend.png"

# dataviz reference palette, categorical slots 1-2 (validated all-pairs, light
# mode: CVD ΔE 9.2, normal-vision 24.0, both clear of 3:1 contrast).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
BAND = "#e9e8e3"

CHANCE = 0.25
ZERO_X = 3e-4  # pseudo-position for β=0 on the log axis
TRANSITION = (0.01, 0.15)


def load():
    frames = []
    for d in RESULT_DIRS:
        try:
            frames.append(pd.read_csv(f"{d}/results.csv"))
        except FileNotFoundError:
            print(f"skipping missing {d}/results.csv")
    df = pd.concat(frames, ignore_index=True)
    agg = df.groupby("beta").agg(["mean", "std"]).sort_index()
    # A β with a single seed has std NaN; treat as zero-width band.
    return agg.fillna(0.0)


def x_of(betas):
    x = np.asarray(betas, dtype=float)
    return np.where(x == 0, ZERO_X, x)


def series(ax, x, agg, col, color, label, marker="o"):
    m, s = agg[(col, "mean")].to_numpy(), agg[(col, "std")].to_numpy()
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, m, color=color, linewidth=2, marker=marker, markersize=5, label=label)


def style(ax, title, ylabel):
    ax.set_xscale("log")
    ax.axvspan(*TRANSITION, color=BAND, zorder=0)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_xticks([ZERO_X, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    ax.set_xticklabels(["0", "0.001", "0.01", "0.1", "1", "10", "100", "1000"])
    # axis break between β=0 and the log range
    ax.annotate(
        "⁄⁄",  # noqa: RUF001 — fraction slash is the axis-break glyph
        xy=(6e-4, 0),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
        annotation_clip=False,
    )


def main():
    agg = load()
    x = x_of(agg.index)
    n_seeds = "3 seeds" if agg[("seed", "std")].max() > 0 else "1 seed"

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")

    # (a) removal
    ax = axes[0, 0]
    series(ax, x, agg, "cls_acc_mlp", BLUE, "MLP probe")
    series(ax, x, agg, "cls_acc_logreg", ORANGE, "linear probe", marker="s")
    ax.axhline(CHANCE, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(
        "chance (0.25)",
        xy=(1.5e-3, CHANCE),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
    )
    style(ax, "Condition leakage in the seed — lower is better removal", "accuracy")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")

    # (b) preservation
    ax = axes[0, 1]
    series(ax, x, agg, "dist_r2_mlp", BLUE, "distance R²")
    style(ax, "Preserved residual factor — higher is better", "R² (distance)")

    # (c) the bound. KL underflows to exactly 0 for large β, which a log axis
    # cannot draw — clip to a floor and say so rather than dropping the points.
    ax = axes[1, 0]
    kl_floor = 1e-6
    kl = agg[("kl", "mean")].to_numpy().copy()
    clipped = kl < kl_floor
    kl[clipped] = kl_floor
    ax.plot(x, kl, color=BLUE, linewidth=2, marker="o", markersize=5)
    if clipped.any():
        ax.plot(
            x[clipped],
            kl[clipped],
            linestyle="none",
            marker="v",
            markersize=6,
            color=BLUE,
        )
        ax.annotate(
            f"▼ = KL below {kl_floor:g} (numerically 0)",
            xy=(0.98, 0.06),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
            color=MUTED,
        )
    ax.set_yscale("log")
    ax.set_ylim(kl_floor / 3, 60)
    style(ax, "The KL bound itself — β's direct target", "KL (log scale)")

    # (d) transport cost
    ax = axes[1, 1]
    series(ax, x, agg, "cfm_loss", BLUE, "CFM loss")
    series(ax, x, agg, "straightness", ORANGE, "S (coupling-confounded)", marker="s")
    style(ax, "Transport cost — what the KL is paid for", "loss / S")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="center right")

    for ax in axes.flat:
        ax.set_xlabel("β (KL weight)", fontsize=9, color=MUTED)

    fig.text(
        0.006,
        0.985,
        "β is a switch for removal, a knob for removal-vs-transport-cost",
        fontsize=13,
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.006,
        0.951,
        f"2D-Gaussian toy · condition = GMM component · {n_seeds} per β · "
        "shaded band = transition region (β 0.01–0.15)",  # noqa: RUF001 — en dash in a numeric range
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.subplots_adjust(
        top=0.90, bottom=0.075, left=0.065, right=0.985, hspace=0.42, wspace=0.20
    )
    fig.savefig(OUTPATH, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {OUTPATH}  (β values: {list(agg.index)})")


if __name__ == "__main__":
    main()
