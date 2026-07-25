"""Combined removal-vs-preservation figure: blunt ICA vs surgical Flower.

Overlays the FastICA source-dropping frontiers (colour b and rotation, MLP
probes, swept over k) with Flower's cond / uncond embeddings and the Raw
embedding, all on the same axes: digit removal (x) vs factor preservation (y).

Reads the two saved result CSVs; run from this directory:

    python plot_combined_frontier.py
"""

import argparse
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fastica-csv", default="ivae_sweep_results_fastica_rot/results.csv"
    )
    parser.add_argument("--flower-csv", default="flower_cond_results/results.csv")
    parser.add_argument("--out", default="combined_frontier.png")
    args = parser.parse_args()

    fica = pd.read_csv(args.fastica_csv)
    flow = pd.read_csv(args.flower_csv)
    a = fica[fica.method == "residA"].sort_values("k")
    rb = fica[fica.method == "residB"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.5, 6))

    # FastICA source-dropping frontiers (blue): b solid, rotation dashed.
    ax.plot(
        a.digit_acc_mlp,
        a.b_r2_mlp,
        "-o",
        color="tab:blue",
        ms=4,
        label="FastICA drop — b",
    )
    ax.plot(
        a.digit_acc_mlp,
        a.rot_r2_mlp,
        "--s",
        color="tab:blue",
        ms=4,
        alpha=0.7,
        label="FastICA drop — rotation",
    )
    ax.scatter(
        rb.digit_acc_mlp,
        rb.b_r2_mlp,
        marker="*",
        s=180,
        color="tab:blue",
        edgecolor="k",
        zorder=5,
        label="FastICA residB — b",
    )
    ax.scatter(
        rb.digit_acc_mlp,
        rb.rot_r2_mlp,
        marker="*",
        s=180,
        color="tab:cyan",
        edgecolor="k",
        zorder=5,
        label="FastICA residB — rotation",
    )

    # Flower / Raw embeddings: b = circle, rotation = square; dotted link per method.
    styles = {
        "Flower cond": "tab:green",
        "uncond": "tab:orange",
        "Raw (orig)": "grey",
    }
    for name, color in styles.items():
        r = flow[flow.embedding == name]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.plot(
            [r.digit_acc_mlp, r.digit_acc_mlp],
            [r.b_r2_mlp, r.rot_r2_mlp],
            ":",
            color=color,
            lw=1,
            zorder=4,
        )
        ax.scatter(
            r.digit_acc_mlp,
            r.b_r2_mlp,
            marker="o",
            s=150,
            color=color,
            edgecolor="k",
            zorder=6,
            label=f"{name} — b",
        )
        ax.scatter(
            r.digit_acc_mlp,
            r.rot_r2_mlp,
            marker="s",
            s=150,
            color=color,
            edgecolor="k",
            zorder=6,
            label=f"{name} — rotation",
        )

    ax.axvline(0.10, ls=":", color="red", lw=1.2, label="digit chance (0.10)")
    ax.axhline(0.0, ls="-", color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("digit accuracy — MLP probe   (lower = better removal)")
    ax.set_ylabel("preservation R² — MLP   (higher = better)")
    ax.set_title(
        "Surgical vs blunt removal on RGB-MNIST (class-only)\n"
        "b (circles/solid) is separable; rotation (squares/dashed) is entangled"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Saved {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
