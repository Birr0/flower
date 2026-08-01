"""Which variables does symbolic regression add, in what order, and what does each buy?

The pyoperon Pareto fronts under
``job_results_{FEATURE}_{EMBED}_seed{SEED}/pareto_fronts.csv``
are indexed by expression *length*, which conflates two different things: adding a new
input variable, and re-fitting the same variables in a fancier functional form. Much of
each front is the latter -- in the ``cond`` arm even-length entries gain ~0.0000 test
R^2
over the odd-length entry below them, being the same expression wrapped in exp().

This script collapses each front to its distinct *variable sets* and reports what each
added variable is worth. For the ``cond+z`` arm the chain starts at redshift alone, so
it
reads directly as "what does each latent dimension add on top of z?" -- the question
that
matters once z has been removed from the embedding by the conditional flow.

Two forms, since they answer different questions:

``--form waterfall`` (default)
    Cumulative R^2 and per-variable gain along the nested spine: at each step the
    highest-R^2 variable set that strictly extends the previous one. Use this to show
    what each dimension buys and where the fit saturates.

``--form tree``
    The branching view over all seeds. Use this only to show *search instability* --
    where the seeds disagree about which variable comes next. Note it is a tree drawn
    over what is really a subset lattice: sets reachable from two predecessors are
    attached to the higher-R^2 one, so re-merging paths are not drawn.

**Split usage.** The stored fronts carry only test scores, so ``Test_R2`` is what orders
the entries here. ``val_rescore.py`` re-scores those fronts on the untouched val split;
rebuilding the spine on ``Val_R2`` instead selects the *identical* variable set at every
depth in all four arms, and the measured optimism from test-selection is +0.0001 R^2
overall. So these figures are unchanged by the fix -- but quote the val-selected numbers
from ``val_rescore_results/`` in text, not the raw ``Test_R2`` used for ordering here.

Usage:
    python variable_additions.py
    python variable_additions.py --form tree
    python variable_additions.py --embed-type orig
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from itertools import combinations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Sequential blue ramp, light -> dark (steps 100..700). Magnitude gets one hue.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", RAMP)

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a897f"
SURFACE = "#fcfcfb"
# Gains have polarity (the last variable is worth less than nothing), so the one
# negative bar takes the opposite pole rather than another step of the blue ramp.
NEGATIVE = "#eb6834"

# pyoperon prints variables 1-indexed; in the cond+z arm the appended redshift column
# is the last feature. Anything else is a standardised embedding dimension.
Z_INDEX = 11


def variable_set(equation: str) -> tuple[int, ...]:
    """Variables referenced by a printed pyoperon expression, as a sorted tuple."""
    return tuple(sorted({int(m) for m in re.findall(r"X(\d+)", equation)}))


def label_for(index: int, embed_type: str) -> str:
    """Human label for a variable index."""
    if embed_type == "cond+z" and index == Z_INDEX:
        return "z"
    return f"X{index}"


def load_fronts(feature: str, embed_type: str) -> pd.DataFrame:
    """Concatenate every seed's Pareto front for one (feature, embedding) arm."""
    pattern = f"job_results_{feature}_{embed_type}_seed*/pareto_fronts.csv"
    paths = sorted(glob.glob(pattern))
    if not paths:
        message = f"no fronts matched {pattern!r} -- run from symbolic_regression/"
        raise SystemExit(message)

    df = pd.concat([pd.read_csv(p) for p in paths]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Seed", "Length"])

    # train_logm.py records -inf / inf for trees with non-finite predictions.
    n_bad = int((~np.isfinite(df["Test_R2"])).sum())
    if n_bad:
        print(f"[warn] dropping {n_bad} front entries with non-finite Test_R2")
        df = df[np.isfinite(df["Test_R2"])]

    df["vars"] = df["Equation"].map(variable_set)
    return df


def collapse_to_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct variable set: its best equation, R^2, length and seeds."""
    rows = []
    for varset, group in df.groupby("vars"):
        best = group.loc[group["Test_R2"].idxmax()]
        rows.append(
            {
                "vars": varset,
                "depth": len(varset),
                "Test_R2": best["Test_R2"],
                "Test_R2_min": group["Test_R2"].min(),
                "min_length": int(group["Length"].min()),
                "n_seeds": group["Seed"].nunique(),
                "seeds": tuple(sorted(group["Seed"].unique())),
                "n_entries": len(group),
                "Equation": best["Equation"],
            }
        )
    return pd.DataFrame(rows).sort_values(["depth", "Test_R2"]).reset_index(drop=True)


def link_parents(nodes: pd.DataFrame) -> dict[tuple[int, ...], tuple[int, ...] | None]:
    """Attach each variable set to the observed subset it most plausibly extends.

    A set can extend more than one observed predecessor (e.g. {3,6,z} extends both
    {3,z} and {6,z}); we take the highest-R^2 one, so a weaker sibling shows up as
    the dead end it was. Sets whose immediate subsets were never visited fall back to
    the largest observed proper subset, and finally to the root.
    """
    observed = {row.vars: row.Test_R2 for row in nodes.itertuples()}
    parents: dict[tuple[int, ...], tuple[int, ...] | None] = {}

    for varset in observed:
        if len(varset) == 1:
            parents[varset] = None
            continue

        candidates = [
            subset
            for size in range(len(varset) - 1, 0, -1)
            for subset in combinations(varset, size)
            if subset in observed
        ]
        if not candidates:
            parents[varset] = None
            continue

        best_size = len(candidates[0])
        top = [c for c in candidates if len(c) == best_size]
        parents[varset] = max(top, key=lambda c: observed[c])

    return parents


def assign_layout(
    nodes: pd.DataFrame, parents: dict[tuple[int, ...], tuple[int, ...] | None]
) -> dict[tuple[int, ...], float]:
    """Depth-first vertical placement so sibling subtrees never overlap."""
    children: dict[tuple[int, ...] | None, list[tuple[int, ...]]] = {}
    for varset, parent in parents.items():
        children.setdefault(parent, []).append(varset)
    for group in children.values():
        group.sort(key=lambda v: -nodes.set_index("vars").loc[[v], "Test_R2"].iloc[0])

    y_positions: dict[tuple[int, ...], float] = {}
    cursor = [0.0]

    def place(varset: tuple[int, ...]) -> float:
        kids = children.get(varset, [])
        if not kids:
            y = cursor[0]
            cursor[0] += 1.0
            y_positions[varset] = y
            return y
        kid_ys = [place(k) for k in kids]
        y = float(np.mean(kid_ys))
        y_positions[varset] = y
        return y

    for root in children.get(None, []):
        place(root)

    return y_positions


def extract_spine(nodes: pd.DataFrame) -> list[pd.Series]:
    """The best strictly-nested chain of variable sets, one per depth.

    At each depth we take the highest-R^2 set that actually extends the one chosen
    below it, so the returned chain is a genuine sequence of variable additions --
    unlike the full front, where consecutive lengths sometimes swap a variable out.
    """
    chain: list[pd.Series] = []
    for depth in sorted(nodes["depth"].unique()):
        at_depth = nodes[nodes["depth"] == depth].sort_values(
            "Test_R2", ascending=False
        )
        if not chain:
            chain.append(at_depth.iloc[0])
            continue
        previous = set(chain[-1]["vars"])
        supersets = [row for row in at_depth.itertuples() if previous < set(row.vars)]
        if not supersets:
            print(
                f"[warn] depth {depth} extends nothing at depth {depth - 1};"
                " spine stops here"
            )
            break
        chain.append(at_depth.loc[supersets[0].Index])
    return chain


def draw_waterfall(
    nodes: pd.DataFrame,
    feature: str,
    embed_type: str,
    outpath: str,
) -> pd.DataFrame:
    """Cumulative R^2 and per-variable gain along the nested spine."""
    chain = extract_spine(nodes)
    all_seeds = sorted({s for seeds in nodes["seeds"] for s in seeds})

    steps = []
    for i, row in enumerate(chain):
        previous = set(chain[i - 1]["vars"]) if i else set()
        added = sorted(set(row["vars"]) - previous)
        steps.append(
            {
                "order": i,
                "added": ",".join(label_for(v, embed_type) for v in added),
                "vars_label": ",".join(label_for(v, embed_type) for v in row["vars"]),
                "depth": row["depth"],
                "Test_R2": row["Test_R2"],
                "base": chain[i - 1]["Test_R2"] if i else 0.0,
                "min_length": row["min_length"],
                "n_seeds": row["n_seeds"],
                "seeds": row["seeds"],
            }
        )
    steps_df = pd.DataFrame(steps)
    steps_df["delta_R2"] = steps_df["Test_R2"] - steps_df["base"]

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(max(9.0, 1.25 * len(steps_df) + 3.0), 7.6),
        height_ratios=[2.15, 1.0],
        sharex=True,
        facecolor=SURFACE,
    )
    x = np.arange(len(steps_df))
    norm = Normalize(
        vmin=steps_df["Test_R2"].min() - 0.06, vmax=steps_df["Test_R2"].max()
    )

    # -- top: cumulative R^2 as a waterfall ---------------------------------------
    ax_top.set_facecolor(SURFACE)
    for i, row in steps_df.iterrows():
        bottom = row["base"] if i else 0.0
        height = row["Test_R2"] - bottom
        colour = SEQ(norm(row["Test_R2"]))
        ax_top.bar(
            x[i],
            height,
            bottom=bottom,
            width=0.62,
            color=colour if height >= 0 else NEGATIVE,
            edgecolor=SURFACE,
            linewidth=2.0,  # 2px surface gap between adjacent fills
            zorder=3,
        )
        ax_top.text(
            x[i],
            row["Test_R2"] + 0.018,
            f"{row['Test_R2']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=INK,
            zorder=4,
        )
        if i:
            ax_top.plot(
                [x[i - 1] + 0.31, x[i] - 0.31],
                [row["base"], row["base"]],
                color=INK_MUTED,
                linewidth=1.0,
                linestyle=(0, (3, 3)),
                zorder=2,
            )

    ax_top.set_ylim(0, 1.0)
    ax_top.set_ylabel("cumulative test $R^2$", fontsize=10, color=INK_SECONDARY)
    ax_top.axhline(1.0, color=INK_MUTED, linewidth=0.8, linestyle=":", zorder=1)
    ax_top.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
    ax_top.set_axisbelow(True)

    # -- bottom: the increment each variable buys, on its own scale ---------------
    # The baseline (z alone) is not a "gain" -- plotting it here would compress every
    # increment it exists to make readable, so it is annotated instead of drawn.
    ax_bot.set_facecolor(SURFACE)
    gains = steps_df.iloc[1:]
    for i, row in gains.iterrows():
        delta = row["delta_R2"]
        colour = SEQ(norm(row["Test_R2"])) if delta >= 0 else NEGATIVE
        ax_bot.bar(
            x[i],
            delta,
            width=0.62,
            color=colour,
            edgecolor=SURFACE,
            linewidth=2.0,
            zorder=3,
        )
        span = gains["delta_R2"].max() - min(0.0, gains["delta_R2"].min())
        offset = 0.02 * span * (1 if delta >= 0 else -1)
        ax_bot.text(
            x[i],
            delta + offset,
            f"{delta:+.3f}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=8.5,
            color=INK if delta >= 0 else NEGATIVE,
            zorder=4,
        )
    ax_bot.text(
        x[0],
        gains["delta_R2"].max() * 0.5,
        f"baseline\n{steps_df['Test_R2'].iloc[0]:.3f}",
        ha="center",
        va="center",
        fontsize=8.5,
        color=INK_MUTED,
        style="italic",
        linespacing=1.4,
        zorder=4,
    )
    ax_bot.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=2)
    ax_bot.set_ylabel(
        "gain from this\nvariable ($\\Delta R^2$)", fontsize=10, color=INK_SECONDARY
    )
    ax_bot.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
    ax_bot.set_axisbelow(True)
    span = gains["delta_R2"].max() - min(0.0, gains["delta_R2"].min())
    ax_bot.set_ylim(
        min(0.0, gains["delta_R2"].min()) - 0.18 * span,
        gains["delta_R2"].max() + 0.16 * span,
    )

    ticks = []
    for _, row in steps_df.iterrows():
        star = "*" if row["n_seeds"] < len(all_seeds) else ""
        ticks.append(f"{row['added']}{star}\nL={row['min_length']}")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(ticks, fontsize=9, color=INK_SECONDARY, linespacing=1.5)
    ax_bot.set_xlabel(
        "variable added (cumulative, left to right)   |   "
        "L = shortest expression using that set",
        fontsize=9.5,
        color=INK_SECONDARY,
        labelpad=8,
    )
    ax_bot.set_xlim(-0.7, len(steps_df) - 0.3)

    for axis in (ax_top, ax_bot):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.spines["left"].set_color(INK_MUTED)
        axis.spines["bottom"].set_color(INK_MUTED)
        axis.tick_params(colors=INK_MUTED, length=3, labelsize=8.5)

    root_label = steps_df["vars_label"].iloc[0]
    ax_top.set_title(
        f"{feature}: what each latent dimension adds on top of {root_label}"
        f"\n{embed_type} arm, pyoperon Pareto fronts pooled over"
        f" {len(all_seeds)} seeds",
        fontsize=12,
        color=INK,
        loc="left",
        pad=12,
        linespacing=1.5,
    )
    partial = steps_df[steps_df["n_seeds"] < len(all_seeds)]
    footnote = (
        "Nested spine through the front: at each step the highest-R^2 variable set"
        " that extends the previous one.\nR^2 is test-set and was also used to select,"
        " so read it as an upper"
        " bound -- reselect on the unused val split before quoting."
    )
    if len(partial):
        found = ", ".join(
            f"{r['added']} (seed {','.join(str(s) for s in r['seeds'])})"
            for _, r in partial.iterrows()
        )
        footnote += f"\n* Not found by every seed: {found}."
    fig.text(
        0.008,
        0.005,
        footnote,
        fontsize=8,
        color=INK_MUTED,
        va="bottom",
        linespacing=1.5,
    )

    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(outpath, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return steps_df


def draw_tree(
    nodes: pd.DataFrame,
    parents: dict[tuple[int, ...], tuple[int, ...] | None],
    y_positions: dict[tuple[int, ...], float],
    feature: str,
    embed_type: str,
    outpath: str,
) -> None:
    by_vars = nodes.set_index("vars")
    all_seeds = sorted({s for seeds in nodes["seeds"] for s in seeds})

    # Spread siblings vertically so the variable-set captions under each node clear
    # both their neighbours and the axis.
    spacing = 1.9
    y_positions = {v: y * spacing for v, y in y_positions.items()}

    norm = Normalize(vmin=nodes["Test_R2"].min(), vmax=nodes["Test_R2"].max())
    depths = sorted(nodes["depth"].unique())
    width = max(11.0, 1.85 * len(depths) + 2.0)
    height = max(4.5, 1.25 * (max(y_positions.values()) + 1.6))
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Edges first, so nodes sit on top of them.
    for varset, parent in parents.items():
        if parent is None:
            continue
        x0, y0 = len(parent), y_positions[parent]
        x1, y1 = len(varset), y_positions[varset]
        gain = (
            by_vars.loc[[varset], "Test_R2"].iloc[0]
            - by_vars.loc[[parent], "Test_R2"].iloc[0]
        )

        ax.annotate(
            "",
            xy=(x1 - 0.30, y1),
            xytext=(x0 + 0.30, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": INK_MUTED,
                "linewidth": 1.4,
                "shrinkA": 0,
                "shrinkB": 0,
                "connectionstyle": "arc3,rad=0.0",
            },
        )

        added = set(varset) - set(parent)
        added_label = "+" + ",".join(label_for(i, embed_type) for i in sorted(added))
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2 + 0.17,
            f"{added_label}\n{gain:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK_SECONDARY,
            linespacing=1.25,
        )

    for row in nodes.itertuples():
        x, y = row.depth, y_positions[row.vars]
        colour = SEQ(norm(row.Test_R2))
        # Ink flips to white once the fill is dark enough to swallow black text.
        text_colour = "#ffffff" if norm(row.Test_R2) > 0.55 else INK

        partial = row.n_seeds < len(all_seeds)
        ax.scatter(
            [x],
            [y],
            s=1750,
            c=[colour],
            marker="o",
            edgecolors=INK if not partial else "#c0392b",
            linewidths=1.1 if not partial else 1.8,
            zorder=3,
        )
        ax.text(
            x,
            y,
            f"{row.Test_R2:.3f}",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=text_colour,
            zorder=4,
        )
        # Wrap wide variable sets so deep nodes don't overrun their neighbours.
        labels = [label_for(i, embed_type) for i in row.vars]
        wrapped = "\n".join(
            ",".join(labels[i : i + 4]) + ("," if i + 4 < len(labels) else "")
            for i in range(0, len(labels), 4)
        )
        ax.text(
            x,
            y - 0.42,
            wrapped,
            ha="center",
            va="top",
            fontsize=7,
            color=INK_SECONDARY,
            linespacing=1.3,
            zorder=4,
        )
        if partial:
            ax.text(
                x,
                y + 0.44,
                "seed " + ",".join(str(s) for s in row.seeds),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#c0392b",
                zorder=4,
            )

    depths = sorted(nodes["depth"].unique())
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths], fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("number of input variables", fontsize=10, color=INK_SECONDARY)
    ax.set_yticks([])
    ax.set_xlim(min(depths) - 0.6, max(depths) + 0.6)
    ax.set_ylim(-0.9, max(y_positions.values()) + 0.9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(axis="x", colors=INK_MUTED, length=3)

    smap = plt.cm.ScalarMappable(cmap=SEQ, norm=norm)
    cbar = fig.colorbar(smap, ax=ax, pad=0.015, fraction=0.03)
    cbar.set_label("test $R^2$", fontsize=9, color=INK_SECONDARY)
    cbar.ax.tick_params(labelsize=8, colors=INK_MUTED)
    cbar.outline.set_visible(False)

    root = nodes.loc[nodes["depth"].idxmin(), "vars"]
    root_label = ", ".join(label_for(i, embed_type) for i in root)
    ax.set_title(
        f"{feature} -- variable additions along the {embed_type} Pareto front"
        f"\nrooted at {root_label}; node = best test $R^2$ for that variable set,"
        f" edge = variable added and $\\Delta R^2$",
        fontsize=11.5,
        color=INK,
        loc="left",
        pad=14,
        linespacing=1.5,
    )
    ax.text(
        0.0,
        -0.085,
        "Red outline: variable set found by only some seeds. Nodes pool all seeds and"
        " lengths;"
        " R^2 is test-set and test-selected, so treat it as an upper bound.",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_MUTED,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature", default="LGM_FIB_P50", help="target column")
    parser.add_argument(
        "--embed-type",
        default="cond+z",
        help="embedding arm: cond+z, cond, orig, uncond, z",
    )
    parser.add_argument(
        "--form",
        default="waterfall",
        choices=("waterfall", "tree"),
        help="waterfall: gain per added variable along the nested spine (default). "
        "tree: the branching view, for showing where seeds disagree",
    )
    parser.add_argument("--outdir", default="variable_additions_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    arm = args.embed_type.replace("+", "_")
    stem = f"{args.outdir}/variable_additions_{args.form}_{arm}"

    df = load_fronts(args.feature, args.embed_type)
    print(f"{len(df)} front entries across {df['Seed'].nunique()} seeds")

    nodes = collapse_to_nodes(df)
    print(f"collapsed to {len(nodes)} distinct variable sets")

    parents = link_parents(nodes)
    y_positions = assign_layout(nodes, parents)

    nodes = nodes.copy()
    nodes["parent"] = nodes["vars"].map(parents.get)
    nodes["added"] = [
        ",".join(
            label_for(i, args.embed_type)
            for i in sorted(set(row.vars) - set(parents.get(row.vars) or ()))
        )
        for row in nodes.itertuples()
    ]
    nodes["delta_R2"] = [
        row.Test_R2
        - nodes.set_index("vars").loc[[parents[row.vars]], "Test_R2"].iloc[0]
        if parents.get(row.vars) is not None
        else np.nan
        for row in nodes.itertuples()
    ]
    nodes["vars_label"] = nodes["vars"].map(
        lambda v: ",".join(label_for(i, args.embed_type) for i in v)
    )

    png_path = f"{stem}.png"
    csv_path = f"{stem}.csv"

    if args.form == "waterfall":
        steps = draw_waterfall(nodes, args.feature, args.embed_type, png_path)
        steps.assign(seeds=steps["seeds"].map(lambda s: " ".join(map(str, s)))).to_csv(
            csv_path, index=False
        )
        print()
        print(
            steps[
                [
                    "order",
                    "added",
                    "vars_label",
                    "Test_R2",
                    "delta_R2",
                    "min_length",
                    "n_seeds",
                ]
            ].to_string(index=False)
        )
    else:
        nodes.drop(columns=["vars", "parent", "seeds"]).assign(
            vars=nodes["vars"].map(lambda v: " ".join(map(str, v))),
            parent=nodes["parent"].map(lambda v: " ".join(map(str, v)) if v else ""),
            seeds=nodes["seeds"].map(lambda s: " ".join(map(str, s))),
        ).to_csv(csv_path, index=False)
        print()
        print(
            nodes[
                [
                    "depth",
                    "vars_label",
                    "added",
                    "Test_R2",
                    "delta_R2",
                    "min_length",
                    "n_seeds",
                ]
            ].to_string(index=False)
        )
        draw_tree(nodes, parents, y_positions, args.feature, args.embed_type, png_path)

    print(f"\nwrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
