"""Is β a knob or a switch? A wide-range β sweep on the 2D-Gaussian toy.

Fig 7 (cMNIST) flattens after β=1, which suggests the KL is a *necessary switch
that saturates* rather than a graded control — a distinction that matters because
the rebuttal must not promise a "controllable" residual-MI ceiling it cannot
deliver (R2 §A1c, R3 §A4b). The toy is the right instrument: training is seconds,
so β can be swept over orders of magnitude with multiple seeds, which neither
cMNIST nor spectra can afford.

Per (β, seed) this trains the toy conditional flow with a learnable conditional
base — the same loss as ``train.py``, ``flow_loss + β·KL`` with linear warmup —
then measures, on held-out samples:

- **KL at convergence** — has the base collapsed to the prior? (β's direct target)
- **CFM loss** — what the KL costs in fit
- **residual class information in the seed** — the condition is the GMM component
  (4 classes, chance 0.25), probed with logistic regression and an MLP. This is
  the quantity that decides knob-vs-switch.
- **preserved residual factor** — distance from the component mean (the toy's
  y-independent structure, Fig 3b), probed by R²
- **straightness integral** ``S = E[int_0^1 ||(x_1 - x_0) - v_t(x_t)||^2 dt]`` — the E1
  primary metric (`rebuttal_issue_plan.md`), never measured until now

The seed is obtained the way the shipped pipeline obtains it: integrate the
*conditional* field from t=1 to t=0 (CFG), not by sampling the base.

Run from this directory:

    python beta_sweep.py --outdir beta_sweep_results
"""

import argparse
import json
import os

import pandas as pd
import torch
from data import generate_quad_gmm
from flow_matching.solver import ODESolver
from model import MLP, ConditionedVelocityModelWrapper, Prior
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch import Tensor

# Matches train.py so the β=1 point of the sweep reproduces the shipped model.
LR = 0.001
BATCH_SIZE = 1024
ITERATIONS = 20001
WARMUP_ITERS = 10000
Y_NULL_VAL = -1.0
N_CLASSES = 4
CHANCE = 1.0 / N_CLASSES
# Wide, log-spaced, plus 0. If β were a knob, residual class info would fall
# smoothly across this range; if a switch, it drops once and then flattens.
BETAS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
SEEDS = [0, 1, 2]
N_EVAL = 20000
N_PROBE_TRAIN = 15000


def train_one(beta_target, seed, device, iterations=ITERATIONS):
    torch.manual_seed(seed)
    vf = MLP(dim=2, h=64).to(device)
    prior = Prior().to(device)
    opt = torch.optim.Adam(list(vf.parameters()) + list(prior.parameters()), lr=LR)

    for i in range(iterations):
        opt.zero_grad()
        x_1, y, _, _ = generate_quad_gmm(BATCH_SIZE)
        x_1 = Tensor(x_1).to(device)
        y = Tensor(y).to(device)
        y_null = torch.ones_like(y) * Y_NULL_VAL

        mu_model, log_var = prior(y.unsqueeze(1).float())
        eps = torch.randn_like(x_1)
        x_0_cond = mu_model + torch.exp(0.5 * log_var) * eps
        x_0_uncond = torch.randn_like(x_1)

        t = torch.rand(x_1.shape[0]).to(device).unsqueeze(-1)
        t = torch.cat([t, t], dim=0)
        x_0 = torch.cat([x_0_cond, x_0_uncond], dim=0)
        x_1d = torch.cat([x_1, x_1], dim=0)
        y_d = torch.cat([y, y_null], dim=0)

        x_t = t * x_1d + (1 - t) * x_0
        v_t = vf(x_t=x_t, y=y_d, t=t)
        v_tgt = x_1d - x_0

        kl = (
            0.5 * torch.sum(torch.exp(log_var) + mu_model**2 - 1 - log_var, dim=-1)
        ).mean()
        flow_loss = torch.pow(v_t - v_tgt, 2).mean()
        # Warmup identical to train.py; with beta_target=0 this is a no-op.
        beta = min(beta_target, beta_target * (i / WARMUP_ITERS))
        (flow_loss + beta * kl).backward()
        opt.step()

    return vf, prior, float(flow_loss.item()), float(kl.item())


@torch.no_grad()
def straightness(vf, prior, device, n=8192, n_t=32):
    """S = E[∫ ||(x1-x0) - v_t(x_t)||² dt], Monte-Carlo over t."""
    x_1, y, _, _ = generate_quad_gmm(n)
    x_1, y = Tensor(x_1).to(device), Tensor(y).to(device)
    mu_model, log_var = prior(y.unsqueeze(1).float())
    x_0 = mu_model + torch.exp(0.5 * log_var) * torch.randn_like(x_1)
    target = x_1 - x_0
    total = 0.0
    for t_val in torch.linspace(0.0, 1.0, n_t):
        t = torch.full((n, 1), float(t_val), device=device)
        x_t = t * x_1 + (1 - t) * x_0
        total += torch.pow(target - vf(x_t=x_t, y=y, t=t), 2).sum(-1).mean().item()
    return total / n_t


@torch.no_grad()
def make_seed(vf, device, n=N_EVAL, n_steps=100):
    """Invert data → seed through the conditional field (t: 1 → 0), as shipped."""
    x_1, y, dist, _ = generate_quad_gmm(n)
    x_1, y = Tensor(x_1).to(device), Tensor(y).to(device)
    # ConditionedVelocityModelWrapper already exposes the ODESolver signature
    # (x, t, **extras) and calls the raw MLP as vf(x_t=..., t=..., y=...), so it
    # must wrap the velocity field directly — not a WrappedModel.
    wrapped = ConditionedVelocityModelWrapper(vf, cfg_scale=1.0)
    solver = ODESolver(velocity_model=wrapped)
    x_0 = solver.sample(
        x_init=x_1,
        y=y,
        time_grid=torch.linspace(1, 0, 2).to(device),
        step_size=1.0 / n_steps,
        method="midpoint",
    )
    return x_0.cpu().numpy(), y.cpu().numpy().astype(int), dist.numpy()


def probe(x, y_cls, dist):
    n = N_PROBE_TRAIN
    sc = StandardScaler().fit(x[:n])
    xt, xe = sc.transform(x[:n]), sc.transform(x[n:])
    out = {}
    out["cls_acc_logreg"] = accuracy_score(
        y_cls[n:], LogisticRegression(max_iter=1000).fit(xt, y_cls[:n]).predict(xe)
    )
    out["cls_acc_mlp"] = accuracy_score(
        y_cls[n:],
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=0)
        .fit(xt, y_cls[:n])
        .predict(xe),
    )
    out["dist_r2_linreg"] = r2_score(
        dist[n:], LinearRegression().fit(xt, dist[:n]).predict(xe)
    )
    out["dist_r2_mlp"] = r2_score(
        dist[n:],
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=0)
        .fit(xt, dist[:n])
        .predict(xe),
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", type=float, nargs="+", default=BETAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--outdir", default="beta_sweep_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/params.json", "w") as fh:
        json.dump(
            {
                "args": vars(args),
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "warmup_iters": WARMUP_ITERS,
                "n_eval": N_EVAL,
                "n_probe_train": N_PROBE_TRAIN,
                "chance_accuracy": CHANCE,
                "probes": {
                    "classifier": ["LogisticRegression", "MLP(64,32)"],
                    "regressor": ["LinearRegression", "MLP(64,32)"],
                },
                "device": str(device),
            },
            fh,
            indent=2,
        )

    rows = []
    for beta in args.betas:
        for seed in args.seeds:
            vf, prior, cfm, kl = train_one(beta, seed, device, args.iterations)
            s = straightness(vf, prior, device)
            x0, y_cls, dist = make_seed(vf, device)
            row = {
                "beta": beta,
                "seed": seed,
                "cfm_loss": round(cfm, 5),
                "kl": round(kl, 6),
                "straightness": round(s, 5),
                **{k: round(v, 4) for k, v in probe(x0, y_cls, dist).items()},
            }
            rows.append(row)
            print(
                f"beta={beta:<7g} seed={seed}  kl={row['kl']:.5f}  "
                f"cfm={row['cfm_loss']:.4f}  "
                f"S={row['straightness']:.4f}  cls_mlp={row['cls_acc_mlp']:.3f}  "
                f"dist_r2_mlp={row['dist_r2_mlp']:.3f}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(f"{args.outdir}/results.csv", index=False)

    df = pd.DataFrame(rows)
    agg = df.groupby("beta").agg(["mean", "std"]).round(4)
    summary = (
        "2D-Gaussian β sweep — knob or switch?\n"
        f"condition = GMM component (4 classes, chance = {CHANCE:.2f}); "
        "LOWER cls_acc = better removal\n"
        "dist_r2 = preserved residual factor (HIGHER = better); "
        "S = straightness integral (LOWER = straighter)\n"
        f"{len(args.seeds)} seeds per β, {args.iterations} iterations each\n\n"
        + agg[
            [
                c
                for c in agg.columns
                if c[0]
                in {"kl", "cfm_loss", "straightness", "cls_acc_mlp", "dist_r2_mlp"}
            ]
        ].to_string()
        + "\n"
    )
    with open(f"{args.outdir}/summary.txt", "w") as fh:
        fh.write(summary)
    print("\n" + summary + f"\nSaved to {args.outdir}/")


if __name__ == "__main__":
    main()
