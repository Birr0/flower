"""Motivation: redshift is distributed across the ICA/iVAE basis (issue #20 / E2).

Expounds *why* a conditional flow (Flower) beats fixed-basis ICA/iVAE removal for
a real nuisance factor (redshift). Two panels:

**Panel 1 — redshift is distributed.** For the FastICA and iVAE source bases,
plot each component's dependence on redshift `z` (|Pearson correlation|), sorted,
with the components' dependence on the physical targets overlaid. Redshift is
smeared across *many* components, and the *same* components carry the physics — so
no small set of axes isolates `z`. Quantified by the **participation ratio**
(effective number of z-carrying axes).

**Panel 2 — the cost of axis-deletion.** From the residual-A sweep
(`ivae_sweep_results/results.csv`): as components are dropped in `z`-dependence
order, redshift removal (z R² ↓) only reaches ~0 after most axes are gone, by
which point the physics has collapsed. Flower's `cond` embedding is a single
point *off* that frontier — it removes `z` as a whole-representation conditional
transform, not by deleting axes (and, being invertible, can also re-inject `z`).

Run from this directory (needs `DATA_ROOT`; Panel 2 needs a prior sweep run):

    python redshift_motivation.py --spender spender_I
"""

import argparse
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ivae_sweep import RANDOM_STATE, SPENDER_MAP, TARGETS, encode, load_data, train_ivae
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler


def abs_corr(x, y):
    """Per-column |Pearson correlation| of features ``x`` (n, d) with 1-D ``y``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float) - np.mean(y)
    xc = x - x.mean(axis=0)
    denom = np.sqrt((xc**2).sum(axis=0) * (y**2).sum())
    return np.abs(
        np.divide(
            (xc * y[:, None]).sum(axis=0),
            denom,
            out=np.zeros(x.shape[1]),
            where=denom > 0,
        )
    )


def participation_ratio(w):
    """Effective number of contributing entries: (Σw)² / Σw².  1 = concentrated in
    one, len(w) = spread uniformly."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (np.square(w).sum() + 1e-12))


def per_component_dependence(sources, z, targets):
    z_dep = abs_corr(sources, z)
    phys_dep = {t: abs_corr(sources, targets[t]) for t in TARGETS}
    return z_dep, phys_dep


def _panel_distribution(ax, name, z_dep, phys_dep):
    order = np.argsort(z_dep)[::-1]
    x = np.arange(len(z_dep))
    ax.bar(x, z_dep[order], color="tab:red", alpha=0.75, label="|corr| with z")
    # overlay the strongest physics dependence per component (same ordering)
    phys_max = np.max(np.stack([phys_dep[t] for t in TARGETS]), axis=0)[order]
    ax.plot(x, phys_max, "-o", color="tab:blue", ms=4, label="max |corr| with physics")
    n_eff = participation_ratio(z_dep**2)
    ax.set_title(f"{name}: z spread over ~{n_eff:.1f} of {len(z_dep)} components")
    ax.set_xlabel("component (sorted by z-dependence)")
    ax.set_ylabel("|correlation|")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    return n_eff


def _panel_cost(ax, results_csv, targets):
    df = pd.read_csv(results_csv)
    d = int(df["n_dims"].max())
    for name, color in {"FastICA": "tab:blue", "iVAE": "tab:orange"}.items():
        a = df[(df.source == name) & (df.method == "residA")].copy()
        if a.empty:
            continue
        a["dropped"] = d - a["n_dims"]
        a = a.sort_values("dropped")
        phys = a[[f"{t}_r2_mlp" for t in targets]].mean(axis=1)
        ax.plot(
            a.dropped,
            a.z_r2_mlp,
            "--s",
            color=color,
            ms=4,
            label=f"{name}: z R² (removal)",
        )
        ax.plot(
            a.dropped, phys, "-o", color=color, ms=4, label=f"{name}: mean physics R²"
        )
    fc = df[df.source == "Flower-cond"]
    if not fc.empty:
        fc = fc.iloc[0]
        phys_f = np.mean([fc[f"{t}_r2_mlp"] for t in targets])
        ax.scatter(
            0,
            fc.z_r2_mlp,
            marker="*",
            s=260,
            color="crimson",
            edgecolor="k",
            zorder=6,
            label="Flower: z R² (removed)",
        )
        ax.scatter(
            0,
            phys_f,
            marker="*",
            s=260,
            color="green",
            edgecolor="k",
            zorder=6,
            label="Flower: mean physics R²",
        )
    ax.set_title("cost of axis-deletion: removing z destroys the physics")
    ax.set_xlabel("number of components dropped")
    ax.set_ylabel("R² — MLP")
    ax.legend(fontsize=7, loc="center right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spender", default="spender_I", choices=list(SPENDER_MAP))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-filter", type=int, default=300000)
    parser.add_argument("--outdir", default="ivae_sweep_results")
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    results_csv = args.results_csv or os.path.join(args.outdir, "results.csv")

    (o_tr, _, _, z_tr, tg_tr), _ = load_data(args.spender, args.n_train, args.n_filter)
    x_tr = StandardScaler().fit_transform(o_tr).astype(np.float32)
    d = x_tr.shape[1]
    print(f"train {x_tr.shape}; z range [{z_tr.min():.3f}, {z_tr.max():.3f}]")

    print("Fitting FastICA...")
    ica = FastICA(
        n_components=d, random_state=args.seed, max_iter=1000, whiten="unit-variance"
    )
    f_tr = ica.fit_transform(x_tr)
    fica_zdep, fica_pdep = per_component_dependence(f_tr, z_tr, tg_tr)

    print(f"Training iVAE ({args.epochs} epochs)...")
    u_tr = z_tr.reshape(-1, 1).astype(np.float32)
    ivae = train_ivae(x_tr, u_tr, epochs=args.epochs, seed=args.seed)
    s_tr, _ = encode(ivae, x_tr, u_tr)
    ivae_zdep, ivae_pdep = per_component_dependence(s_tr, z_tr, tg_tr)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    n_fica = _panel_distribution(axes[0], "FastICA", fica_zdep, fica_pdep)
    n_ivae = _panel_distribution(axes[1], "iVAE", ivae_zdep, ivae_pdep)
    _panel_cost(axes[2], results_csv, TARGETS)
    fig.suptitle(
        f"{args.spender}: redshift is distributed across the ICA/iVAE basis — "
        "axis-deletion cannot remove it without destroying the physics; "
        "Flower removes it compositely",
        fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(args.outdir, "redshift_motivation.png")
    fig.savefig(out, dpi=140)
    print(
        f"\nParticipation ratio (effective # z-carrying axes of {d}): "
        f"FastICA {n_fica:.2f}, iVAE {n_ivae:.2f}\nSaved {out}"
    )


if __name__ == "__main__":
    main()
