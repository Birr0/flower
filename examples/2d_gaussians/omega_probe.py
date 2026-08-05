"""Fig 3b explainer: is the residual injected by guidance, or linearised? (#25 / E6)

Reviewer 3 reads Fig 3b as counterintuitive: "R^2 increases as omega moves from 1
to 0, which seems to imply that the flow-matching process injects more
information about the distance variable."

Two things are going on, and this script separates them.

1. **The omega axis reads the other way.** In the source figure
   (``distance_trajectory.ipynb``, ``omegas=linspace(0,1,50)``, viridis 0->1) the
   yellow (omega=1) curves sit at the *top* at t=0 and the dark purple (omega=0)
   curves at the bottom, so R^2 *decreases* as omega goes 1->0. The reviewer
   inverted the direction.

2. **Even read correctly, "more R^2" is not "more information".** Fig 3b probes
   the residual with a **linear** regressor only. A linear probe conflates *how
   much* of the residual is present with *how accessibly* it is encoded. The
   residual here is an intrinsic property of the data (the displacement of each
   point from its own cluster mean, fixed at generation time); the flow cannot
   create it.

So we re-probe the t=0 seed across omega with a linear *and* a nonlinear probe.
Prediction, if guidance linearises rather than injects: **MLP R^2 approximately
flat and high across all omega**, while **linear R^2 rises with omega**. The gap
between the two curves is the part of Fig 3b that is about accessibility rather
than content.

**Outcome: the anti-injection claim holds, the flatness prediction does not.**
The nonlinear probe never beats the raw-data ceiling (so nothing is injected),
but it is not flat -- it dips at intermediate omega, where the blended field is
the transport of neither model. See ``omega_probe_note.md``; ``--capacity-omegas``
is the ladder that rules out probe capacity as the cause.

Targets, all evaluated on the same inverted seed:

- ``dx0``/``dx1`` — the two components of ``diff``, the ground-truth
  condition-independent seed. **This is Fig 3b's actual target** (its legend
  reads delta-x_0 / delta-x_1), not the scalar distance the reviewer names.
- ``dist`` — the scalar radial distance ``||diff||``, reported because that is
  the variable the review refers to and the one R3 A6 asks us to define.
- ``mode`` — the condition itself (4 clusters, chance 0.25), as the trade-off
  context: the same omega that makes the residual linearly readable is the omega
  that removes the class.

Uses the shipped checkpoint (``checkpoints/cond_fm.pth``) — the same weights
behind Fig 3b — so only the data draw and the probe fits vary across ``--seeds``.
``--train`` instead fits a fresh model per seed via ``beta_sweep.train_one``.

Run from this directory (``examples/2d_gaussians``):

    python omega_probe.py 2>&1 | tee omega_probe_results/omega_probe.log
"""

import argparse
import json
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from beta_sweep import ITERATIONS, N_CLASSES, train_one
from data import generate_quad_gmm
from flow_matching.solver import ODESolver
from model import MLP, ConditionedVelocityModelWrapper, Prior
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch import Tensor

CHANCE = 1.0 / N_CLASSES
# Endpoints included: omega=1 is the shipped conditional inversion, omega=0 the
# purely unconditional one. 11 points is enough to show a monotone trend without
# refitting four probes 50 times.
OMEGAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [0, 1, 2]
N_EVAL = 10000
# Same 3:1 probe split as beta_sweep.py, expressed as a fraction so --n is free.
PROBE_TRAIN_FRAC = 0.75
CKPT = "checkpoints/cond_fm.pth"


def load_model(ckpt_path, device):
    """The Fig 3b weights. Constructed fresh rather than importing model.vf, so
    the module-level singletons in model.py stay untouched."""
    vf, prior = MLP(dim=2, h=64).to(device), Prior().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    vf.load_state_dict(ckpt["vf_state_dict"])
    prior.load_state_dict(ckpt["prior_state_dict"])
    return vf, prior


@torch.no_grad()
def invert(vf, x_1, y, omega, device, n_steps=100):
    """Data -> seed at t=0 through the CFG field at guidance weight omega.

    Same construction as ``beta_sweep.make_seed``, but with cfg_scale swept:
    ``u = (1-omega)*u_null + omega*u_cond`` (``ConditionedVelocityModelWrapper``),
    so omega=1 is the conditional field and omega=0 the unconditional one.
    """
    solver = ODESolver(
        velocity_model=ConditionedVelocityModelWrapper(vf, cfg_scale=float(omega))
    )
    x_0 = solver.sample(
        x_init=x_1,
        y=y,
        time_grid=torch.linspace(1, 0, 2).to(device),
        step_size=1.0 / n_steps,
        method="midpoint",
    )
    return x_0.cpu().numpy()


def probe(x, targets, modes, seed):
    """Linear vs nonlinear probe of every target on the same representation.

    The linear row is what Fig 3b plots; the MLP row is what the figure is
    missing. Features are standardised so the MLP is not handicapped by scale --
    this cannot change the linear R^2, which is affine-invariant.
    """
    n = int(PROBE_TRAIN_FRAC * len(x))
    sc = StandardScaler().fit(x[:n])
    xt, xe = sc.transform(x[:n]), sc.transform(x[n:])

    out = {}
    for name, t in targets.items():
        out[f"{name}_r2_lin"] = r2_score(
            t[n:], LinearRegression().fit(xt, t[:n]).predict(xe)
        )
        out[f"{name}_r2_mlp"] = r2_score(
            t[n:],
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=seed)
            .fit(xt, t[:n])
            .predict(xe),
        )
    out["mode_acc_logreg"] = accuracy_score(
        modes[n:],
        LogisticRegression(max_iter=1000, random_state=seed)
        .fit(xt, modes[:n])
        .predict(xe),
    )
    out["mode_acc_mlp"] = accuracy_score(
        modes[n:],
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=seed)
        .fit(xt, modes[:n])
        .predict(xe),
    )
    return out


def capacity_ladder(vf, x_1, y, targets, omega, device, seed, n_steps):
    """Is the mid-omega dip in the MLP curve lost information, or a probe limit?

    The inversion is a deterministic ODE map, so it is a diffeomorphism and
    cannot destroy information -- but a *blended* field (0<omega<1) is not the
    transport of either the conditional or the unconditional model, so it can
    scramble the residual into a form a fixed-capacity probe cannot read. If the
    dip is a probe artifact, R^2 climbs with probe capacity; if the residual were
    genuinely gone, no probe would recover it.
    """
    x_0 = invert(vf, x_1, y, omega, device, n_steps)
    n = int(PROBE_TRAIN_FRAC * len(x_0))
    sc = StandardScaler().fit(x_0[:n])
    xt, xe = sc.transform(x_0[:n]), sc.transform(x_0[n:])

    probes = {
        "linear": LinearRegression(),
        "mlp_64_32": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=seed
        ),
        "mlp_256_128": MLPRegressor(
            hidden_layer_sizes=(256, 128), max_iter=2000, random_state=seed
        ),
        "knn_10": KNeighborsRegressor(n_neighbors=10),
    }
    rows = []
    for pname, est in probes.items():
        row = {"seed": seed, "omega": omega, "probe": pname}
        for tname, t in targets.items():
            row[f"{tname}_r2"] = round(
                r2_score(t[n:], est.fit(xt, t[:n]).predict(xe)), 4
            )
        rows.append(row)
        print(
            f"  capacity omega={omega:<4g} {pname:<12s} "
            + "  ".join(f"{k}={row[f'{k}_r2']:.3f}" for k in targets),
            flush=True,
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omegas", type=float, nargs="+", default=OMEGAS)
    parser.add_argument(
        "--capacity-omegas",
        type=float,
        nargs="+",
        default=[],
        help="re-probe these omegas with a ladder of probe capacities",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--n", type=int, default=N_EVAL)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument(
        "--train",
        action="store_true",
        help="train a fresh model per seed instead of loading --ckpt",
    )
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--outdir", default="omega_probe_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/params.json", "w") as fh:
        json.dump(
            {
                "args": vars(args),
                "probe_train_frac": PROBE_TRAIN_FRAC,
                "chance_accuracy": CHANCE,
                "cfg": "u = (1-omega)*u_null + omega*u_cond",
                "targets": {
                    "dx0/dx1": "components of `diff` — Fig 3b's target",
                    "dist": "||diff||, the scalar the review calls 'distance'",
                    "mode": "the condition (4 clusters)",
                },
                "probes": {
                    "linear": ["LinearRegression", "LogisticRegression"],
                    "nonlinear": ["MLPRegressor(64,32)", "MLPClassifier(64,32)"],
                },
                "device": str(device),
            },
            fh,
            indent=2,
        )

    rows, cap_rows = [], []
    for seed in args.seeds:
        # generate_quad_gmm takes no seed and data.py never seeds torch (TODO.md),
        # so the draw must be pinned here for this table to reproduce.
        torch.manual_seed(seed)
        if args.train:
            vf, _, _, _ = train_one(1.0, seed, device, args.iterations)
        else:
            vf, _ = load_model(args.ckpt, device)

        x_1, y, dist, diff = generate_quad_gmm(args.n)
        modes = y.numpy().astype(int)
        targets = {
            "dx0": diff[:, 0].numpy(),
            "dx1": diff[:, 1].numpy(),
            "dist": dist.numpy(),
        }
        x_1, y = Tensor(x_1).to(device), Tensor(y).to(device)

        # The raw data is the ceiling: whatever the seed retains, it cannot
        # exceed what was in x_1 to begin with. This is the row that decides
        # injection vs preservation.
        row = {"seed": seed, "kind": "raw", "omega": np.nan}
        row.update(probe(x_1.cpu().numpy(), targets, modes, seed))
        rows.append(row)
        print(
            f"seed={seed} raw            "
            f"dx0 lin={row['dx0_r2_lin']:.3f} mlp={row['dx0_r2_mlp']:.3f}  "
            f"mode_mlp={row['mode_acc_mlp']:.3f}",
            flush=True,
        )

        for omega in args.omegas:
            x_0 = invert(vf, x_1, y, omega, device, args.n_steps)
            row = {"seed": seed, "kind": "seed", "omega": omega}
            row.update(probe(x_0, targets, modes, seed))
            rows.append(row)
            print(
                f"seed={seed} omega={omega:<5g}     "
                f"dx0 lin={row['dx0_r2_lin']:.3f} mlp={row['dx0_r2_mlp']:.3f}  "
                f"dist lin={row['dist_r2_lin']:.3f} mlp={row['dist_r2_mlp']:.3f}  "
                f"mode_mlp={row['mode_acc_mlp']:.3f}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(f"{args.outdir}/omega_probe.csv", index=False)

        for omega in args.capacity_omegas:
            cap_rows.extend(
                capacity_ladder(vf, x_1, y, targets, omega, device, seed, args.n_steps)
            )
            pd.DataFrame(cap_rows).to_csv(
                f"{args.outdir}/omega_probe_capacity.csv", index=False
            )

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.outdir}/omega_probe.csv", index=False)
    write_summary(df, args)
    make_plot(df, f"{args.outdir}/omega_probe.png")
    print(f"\nSaved omega_probe.{{csv,txt,png}} in {args.outdir}/")


def write_summary(df, args):
    seed_df = df[df.kind == "seed"]
    agg = seed_df.groupby("omega").mean(numeric_only=True).drop(columns="seed").round(4)
    raw = df[df.kind == "raw"].mean(numeric_only=True).round(4)

    cols = [
        "dx0_r2_lin",
        "dx0_r2_mlp",
        "dx1_r2_lin",
        "dx1_r2_mlp",
        "dist_r2_lin",
        "dist_r2_mlp",
        "mode_acc_logreg",
        "mode_acc_mlp",
    ]
    summary = (
        "Fig 3b explainer — linear vs nonlinear probe of the residual across omega\n"
        "2D-Gaussian toy, seed taken at t=0 by CFG inversion of the shipped model\n"
        f"{len(args.seeds)} seeds x {args.n} samples; chance accuracy "
        f"{CHANCE:.2f}\n\n"
        "The test for injection is the raw-data row below: the seed cannot\n"
        "contain more residual than x_1 did. If the nonlinear probe on the seed\n"
        "stays at or under that ceiling while the linear probe climbs, guidance\n"
        "made the residual READABLE rather than adding any.\n\n"
        f"raw data x_1 (ceiling): dx0 lin={raw.dx0_r2_lin:.3f} "
        f"mlp={raw.dx0_r2_mlp:.3f} | dist lin={raw.dist_r2_lin:.3f} "
        f"mlp={raw.dist_r2_mlp:.3f} | mode_mlp={raw.mode_acc_mlp:.3f}\n\n"
        + agg[cols].to_string()
        + "\n"
    )
    with open(f"{args.outdir}/omega_probe.txt", "w") as fh:
        fh.write(summary)
    print("\n" + summary)


def make_plot(df, outpath):
    seed_df = df[df.kind == "seed"]
    g = seed_df.groupby("omega")
    mean, std = g.mean(numeric_only=True), g.std(numeric_only=True)
    om = mean.index.values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    def band(ax, col, label, color, ls="-"):
        ax.plot(om, mean[col], ls, color=color, marker="o", ms=4, label=label)
        ax.fill_between(
            om, mean[col] - std[col], mean[col] + std[col], color=color, alpha=0.15
        )

    band(axes[0], "dx0_r2_lin", r"$\delta x_0$ — linear probe", "tab:red")
    band(axes[0], "dx0_r2_mlp", r"$\delta x_0$ — MLP probe", "tab:blue")
    band(axes[0], "dx1_r2_lin", r"$\delta x_1$ — linear probe", "tab:orange", ls="--")
    band(axes[0], "dx1_r2_mlp", r"$\delta x_1$ — MLP probe", "tab:cyan", ls="--")
    axes[0].set_ylabel(r"$R^2$")
    axes[0].set_title(
        "Fig 3b's target: the 2-D seed\n"
        r"(at $\omega$=0 nonlinear $\gg$ linear; they converge by $\omega$=1)"
    )

    band(axes[1], "dist_r2_lin", "linear probe", "tab:red")
    band(axes[1], "dist_r2_mlp", "MLP probe", "tab:blue")
    axes[1].set_ylabel(r"$R^2$")
    axes[1].set_title(r"scalar distance $d=\|\delta x\|$" + "\n(the review's variable)")

    band(axes[2], "mode_acc_logreg", "logistic probe", "tab:red")
    band(axes[2], "mode_acc_mlp", "MLP probe", "tab:blue")
    axes[2].axhline(CHANCE, ls=":", color="grey", lw=1, label="chance")
    axes[2].set_ylabel("accuracy")
    axes[2].set_title("the condition itself\n(what guidance actually removes)")

    for ax in axes:
        ax.set_xlabel(r"guidance weight $\omega$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Guidance linearises the residual, it does not inject it "
        "(2D-Gaussians, mean ± s.d. over seeds)"
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)


if __name__ == "__main__":
    main()
