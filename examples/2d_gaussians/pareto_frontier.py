"""Removal-vs-preservation Pareto frontier on the 2D toy (issue #27 / E8).

Reviewer 1 (W4): "The paper should better quantify the tradeoff between condition
removal and preservation of residual information."

Everything else we have plots Flower as a *single point* against baselines swept
over their own knob (`examples/mnist/plot_combined_frontier.py`). That understates
the answer: Flower has knobs too, and this toy is the only setting where both are
already swept end-to-end, so it is the only place we can draw *Flower's own*
frontier rather than one operating point.

Two Flower curves, from results already on disk (no new compute):

- **beta** (`beta_sweep_results/`, 13 values x 3 seeds, all at guidance omega=1) —
  the training-time knob: how hard the KL constrains the base.
- **omega** (`omega_probe_results/`, 11 values x 3 seeds, all at beta=1) — the
  inference-time knob: how strongly the inversion is guided.

They intersect at (beta=1, omega=1), which is the consistency check printed in the
summary: the two sweeps are separate scripts with slightly different probe
settings, so agreement there is what licenses drawing them on shared axes.

Baselines (`correlation_metrics_results/`) go on the same axes as points.

Axes: x = mode accuracy under an **MLP** probe (removal, LOWER better, chance
0.25); y = distance R^2 under an **MLP** probe (preservation, HIGHER better). The
nonlinear probe is the honest choice on both axes -- a linear probe reads the toy's
radial distance at R^2 ~ 0 no matter what (see `omega_probe_note.md`), and reads
the condition as removed when it is not.

**Read the toy result with care** -- see `pareto_frontier_note.md`. The toy's
condition is *exactly* a per-class mean offset, so subtracting a conditional mean
is the exactly-correct operation here and FastICA residB attains both axes at once.
That is a property of the toy, not a general result; on cMNIST and spectra, where
the condition is nonlinearly encoded, the same construction collapses under an MLP
probe. Do not present this panel without that pairing.

Run from this directory (``examples/2d_gaussians``):

    python pareto_frontier.py 2>&1 \
        | tee pareto_frontier_results/pareto_frontier.log
"""

import argparse
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CHANCE = 0.25


def load_beta(paths):
    """Flower's training-time knob. beta_sweep.py probes at cfg_scale=1.

    Takes several CSVs because the sweep was run in two passes: the wide
    log-spaced one, and `beta_sweep_transition_results/` which fills in
    beta = 0.02-0.3. That gap is not cosmetic -- the wide sweep jumps 0.01 ->
    0.05 -> 0.1, which makes a continuous transition look like a step. Same
    script, same probes, same seeds, so the rows concatenate directly.
    """
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    g = df.groupby("beta")[["cls_acc_mlp", "dist_r2_mlp"]].agg(["mean", "std"])
    return pd.DataFrame(
        {
            "knob": "beta",
            "value": g.index,
            "removal": g[("cls_acc_mlp", "mean")].values,
            "removal_sd": g[("cls_acc_mlp", "std")].values,
            "preservation": g[("dist_r2_mlp", "mean")].values,
            "preservation_sd": g[("dist_r2_mlp", "std")].values,
        }
    )


def load_omega(path):
    """Flower's inference-time knob, from the E6 sweep (beta=1 checkpoint)."""
    df = pd.read_csv(path)
    df = df[df.kind == "seed"]
    g = df.groupby("omega")[["mode_acc_mlp", "dist_r2_mlp"]].agg(["mean", "std"])
    return pd.DataFrame(
        {
            "knob": "omega",
            "value": g.index,
            "removal": g[("mode_acc_mlp", "mean")].values,
            "removal_sd": g[("mode_acc_mlp", "std")].values,
            "preservation": g[("dist_r2_mlp", "mean")].values,
            "preservation_sd": g[("dist_r2_mlp", "std")].values,
        }
    )


def load_baselines(path):
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "knob": "baseline",
            "value": df.source + "-" + df.method,
            "removal": df.mode_acc_mlp,
            "removal_sd": float("nan"),
            "preservation": df.dist_r2_mlp,
            "preservation_sd": float("nan"),
        }
    )


def pareto_front(df):
    """Non-dominated rows: nothing else removes more AND preserves more.

    Removal is minimised, preservation maximised, so row i is dominated if some
    row j has removal_j <= removal_i and preservation_j >= preservation_i with at
    least one strict.
    """
    keep = []
    for i, a in df.iterrows():
        dominated = (
            (df.removal <= a.removal)
            & (df.preservation >= a.preservation)
            & ((df.removal < a.removal) | (df.preservation > a.preservation))
        ).any()
        if not dominated:
            keep.append(i)
    return df.loc[keep].sort_values("removal")


def _draw(ax, beta, omega, base, annotate):
    for df, colour, label in [
        (beta, "tab:blue", r"Flower, $\beta$ sweep ($\omega$=1)"),
        (omega, "tab:green", r"Flower, $\omega$ sweep ($\beta$=1)"),
    ]:
        d = df.sort_values("removal")
        ax.plot(d.removal, d.preservation, "-o", color=colour, ms=4, label=label)
        ax.errorbar(
            d.removal,
            d.preservation,
            xerr=d.removal_sd,
            yerr=d.preservation_sd,
            fmt="none",
            ecolor=colour,
            alpha=0.3,
        )

    markers = {
        "Raw-none": "s",
        "FastICA-residA": "^",
        "FastICA-residB": "v",
        "iVAE-residA": "<",
        "iVAE-residB": ">",
    }
    for _, r in base.iterrows():
        ax.scatter(
            r.removal,
            r.preservation,
            marker=markers.get(r.value, "o"),
            s=110,
            c="tab:red",
            zorder=5,
        )
        if annotate:
            ax.annotate(
                r.value,
                (r.removal, r.preservation),
                fontsize=8,
                xytext=(7, -11),
                textcoords="offset points",
            )

    # Endpoint labels: the knob values a reader needs to locate the operating point.
    if annotate:
        for df, colour in [(beta, "tab:blue"), (omega, "tab:green")]:
            for _, r in df.iloc[[0, -1]].iterrows():
                ax.annotate(
                    f"{r.knob}={r.value:g}",
                    (r.removal, r.preservation),
                    fontsize=7,
                    color=colour,
                    xytext=(4, 6),
                    textcoords="offset points",
                )

    ax.axvline(CHANCE, ls=":", color="grey", lw=1.2)
    ax.set_xlabel("mode accuracy, MLP probe  —  removal (lower = better)")
    ax.set_ylabel(r"distance $R^2$, MLP probe  —  preservation (higher = better)")
    ax.grid(alpha=0.3)


def make_plot(beta, omega, base, outpath):
    """Two panels: the full trade-off, and a zoom on the top-left corner where
    every good operating point piles up and labels collide."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    _draw(axes[0], beta, omega, base, annotate=True)
    axes[0].annotate(
        "chance (perfect removal)",
        (CHANCE, axes[0].get_ylim()[0]),
        fontsize=8,
        color="grey",
        rotation=90,
        xytext=(4, 8),
        textcoords="offset points",
    )
    axes[0].set_title("full trade-off — top-left is better")
    axes[0].legend(fontsize=9, loc="lower center")

    _draw(axes[1], beta, omega, base, annotate=True)
    axes[1].set_xlim(0.24, 0.36)
    axes[1].set_ylim(0.90, 1.01)
    axes[1].set_title(
        "zoom: the usable corner\nFastICA residB is exactly optimal HERE — see the note"
    )

    fig.suptitle(
        "Removal vs preservation, 2D-Gaussian toy (issue #27 / E8) — "
        "Flower traced over both its knobs"
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beta-csv",
        nargs="+",
        default=[
            "beta_sweep_results/results.csv",
            "beta_sweep_transition_results/results.csv",
        ],
    )
    parser.add_argument("--omega-csv", default="omega_probe_results/omega_probe.csv")
    parser.add_argument(
        "--baseline-csv",
        default="correlation_metrics_results/correlation_metrics.csv",
    )
    parser.add_argument("--outdir", default="pareto_frontier_results")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    beta, omega = load_beta(args.beta_csv), load_omega(args.omega_csv)
    base = load_baselines(args.baseline_csv)
    allpts = pd.concat([beta, omega, base], ignore_index=True)
    allpts.to_csv(f"{args.outdir}/pareto_frontier.csv", index=False)

    # The two sweeps are separate scripts (different probe max_iter, different
    # train fractions). They share one setting -- beta=1, omega=1 -- so if that
    # point disagrees, the curves are not on comparable axes and the figure is
    # invalid. Print it rather than assume it.
    b1 = beta[beta.value == 1.0].iloc[0]
    o1 = omega[omega.value == 1.0].iloc[0]
    check = (
        "Consistency check at the shared setting (beta=1, omega=1):\n"
        f"  beta sweep : removal {b1.removal:.3f}  preservation {b1.preservation:.3f}\n"
        f"  omega sweep: removal {o1.removal:.3f}  preservation {o1.preservation:.3f}\n"
        f"  delta      : removal {abs(b1.removal - o1.removal):.3f}  "
        f"preservation {abs(b1.preservation - o1.preservation):.3f}\n"
    )

    front = pareto_front(allpts)
    summary = (
        "Removal vs preservation — 2D-Gaussian toy (issue #27 / E8)\n"
        f"removal = mode accuracy, MLP probe (chance {CHANCE:.2f}, lower better)\n"
        "preservation = distance R^2, MLP probe (higher better)\n\n"
        + check
        + "\nPareto-optimal points (nothing removes more AND preserves more):\n"
        + front[["knob", "value", "removal", "preservation"]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
        + "\n\nAll points:\n"
        + allpts[["knob", "value", "removal", "preservation"]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
        + "\n"
    )
    with open(f"{args.outdir}/pareto_frontier.txt", "w") as fh:
        fh.write(summary)
    print(summary)

    make_plot(beta, omega, base, f"{args.outdir}/pareto_frontier.png")
    print(f"Saved pareto_frontier.{{csv,txt,png}} in {args.outdir}/")


if __name__ == "__main__":
    main()
