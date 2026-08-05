"""What does the conditional prior buy? Final-loss ablation on the 2D-Gaussian toy.

`train.py` learns a conditional base `p(x_0|y) = N(mu(y), sigma(y))` alongside the
velocity field and regularises it toward `N(0, I)` with `beta * KL`. This script asks
what happens when that base is removed entirely — plain conditional flow matching from
a fixed standard-Gaussian base — and how the *final* flow-matching loss compares.

Three arms, identical optimiser, iterations, batch size, warmup and CFG batch layout;
the only difference is where `x_0` comes from:

- ``learned_prior_b1``  shipped setup: learnable `Prior`, beta=1 KL with linear warmup
- ``learned_prior_b0``  same module, KL weight 0 (base free to drift anywhere)
- ``no_prior``          no `Prior` module at all: x_0 ~ N(0, I), no KL term

Losses are evaluated on held-out samples, not the last training minibatch, and split
by CFG half — the base only ever affects the conditional half, so the training-time
number (which averages both) understates the difference by roughly 2x:

- ``cfm_cond``       conditional half: x_0 from that arm's base, y = true label
- ``cfm_uncond``     unconditional half: x_0 ~ N(0, I), y = Y_NULL_VAL  (identical
                     across arms by construction, so it doubles as a sanity check)
- ``cfm_total``      mean of the two = the flow term of the training objective
- ``cfm_cond_fixed`` conditional pathway scored from a *common* N(0, I) base

CAVEAT worth keeping in mind when reading the table: `cfm_cond` is *not* a like-for-like
model-quality score across arms, because each arm is scored against its own base. A
learned conditional base shortens the transport, so it can post a lower loss for an
easier problem, not a better fit — the same coupling confound that makes the
straightness integral unusable for cross-arm claims. `cfm_cond_fixed` is the fix: every
arm transports from the same N(0, I) to the same targets, so only the velocity field
varies. Read it knowing it handicaps the learned-prior arms (see `heldout_cfm`).

Run from this directory:

    python no_prior_ablation.py --outdir no_prior_ablation_results
"""

import argparse
import json
import os

import pandas as pd
import torch
from beta_sweep import CHANCE, make_seed, probe
from data import generate_quad_gmm
from model import MLP, Prior
from torch import Tensor

# Matches train.py / beta_sweep.py so arms are comparable to the shipped model.
LR = 0.001
BATCH_SIZE = 1024
ITERATIONS = 20001
WARMUP_ITERS = 10000
Y_NULL_VAL = -1.0
SEEDS = [0, 1, 2]
N_EVAL = 20000
N_T = 32
EVAL_SEED = 1234

VARIANTS = {
    # name: (use_prior, beta)
    "learned_prior_b1": (True, 1.0),
    "learned_prior_b0": (True, 0.0),
    "no_prior": (False, 0.0),
}


def train_one(use_prior, beta_target, seed, device, iterations=ITERATIONS):
    """Identical to `beta_sweep.train_one` except that `use_prior=False` drops the
    `Prior` module (and the KL) entirely, sampling x_0 ~ N(0, I) for both halves."""
    torch.manual_seed(seed)
    vf = MLP(dim=2, h=64).to(device)
    prior = Prior().to(device) if use_prior else None
    params = list(vf.parameters()) + (list(prior.parameters()) if use_prior else [])
    opt = torch.optim.Adam(params, lr=LR)

    for i in range(iterations):
        opt.zero_grad()
        x_1, y, _, _ = generate_quad_gmm(BATCH_SIZE)
        x_1 = Tensor(x_1).to(device)
        y = Tensor(y).to(device)
        y_null = torch.ones_like(y) * Y_NULL_VAL

        if use_prior:
            mu_model, log_var = prior(y.unsqueeze(1).float())
            eps = torch.randn_like(x_1)
            x_0_cond = mu_model + torch.exp(0.5 * log_var) * eps
            kl = (
                0.5 * torch.sum(torch.exp(log_var) + mu_model**2 - 1 - log_var, dim=-1)
            ).mean()
        else:
            x_0_cond = torch.randn_like(x_1)
            kl = torch.zeros((), device=device)

        x_0_uncond = torch.randn_like(x_1)

        t = torch.rand(x_1.shape[0]).to(device).unsqueeze(-1)
        t = torch.cat([t, t], dim=0)
        x_0 = torch.cat([x_0_cond, x_0_uncond], dim=0)
        x_1d = torch.cat([x_1, x_1], dim=0)
        y_d = torch.cat([y, y_null], dim=0)

        x_t = t * x_1d + (1 - t) * x_0
        v_t = vf(x_t=x_t, y=y_d, t=t)
        v_tgt = x_1d - x_0

        flow_loss = torch.pow(v_t - v_tgt, 2).mean()
        beta = min(beta_target, beta_target * (i / WARMUP_ITERS))
        (flow_loss + beta * kl).backward()
        opt.step()

    return vf, prior, float(flow_loss.item()), float(kl.item())


@torch.no_grad()
def straightness(vf, prior, device, n=8192, n_t=N_T):
    """`beta_sweep.straightness`, with the base swapped for N(0, I) when there is no
    prior — the only line that can differ, since S integrates the same residual."""
    x_1, y, _, _ = generate_quad_gmm(n)
    x_1, y = Tensor(x_1).to(device), Tensor(y).to(device)
    if prior is not None:
        mu_model, log_var = prior(y.unsqueeze(1).float())
        x_0 = mu_model + torch.exp(0.5 * log_var) * torch.randn_like(x_1)
    else:
        x_0 = torch.randn_like(x_1)
    target = x_1 - x_0
    total = 0.0
    for t_val in torch.linspace(0.0, 1.0, n_t):
        t = torch.full((n, 1), float(t_val), device=device)
        x_t = t * x_1 + (1 - t) * x_0
        total += torch.pow(target - vf(x_t=x_t, y=y, t=t), 2).sum(-1).mean().item()
    return total / n_t


@torch.no_grad()
def heldout_cfm(vf, prior, device, n=N_EVAL, n_t=N_T):
    """Held-out CFM loss, three scorings. The eval batch is drawn under a fixed seed
    so every arm and seed is scored on the same x_1, y.

    - cfm_cond        x_0 from this arm's own base, true y. Biased TOWARD a learned
                      base: a base near the class mode is graded on a shorter
                      transport, so a lower value need not mean a better field.
    - cfm_cond_fixed  x_0 ~ N(0, I) for every arm, true y. Identical transport problem
                      across arms, so only the velocity field varies — the
                      unconfounded conditional comparison. Biased toward `no_prior`,
                      for which this *is* the training distribution while the
                      learned-prior arms are scored off-distribution. A learned-prior
                      win here is therefore strong evidence; a loss is weak.
    - cfm_uncond      x_0 ~ N(0, I), y = Y_NULL_VAL. Trained identically in all arms,
                      so it is a control rather than a discriminator.

    The `z_fixed` draw deliberately comes *after* the other two, leaving their RNG
    stream untouched so cfm_cond/cfm_uncond reproduce the previously recorded values.
    """
    g = torch.Generator().manual_seed(EVAL_SEED)
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(g.get_state())
    x_1, y, _, _ = generate_quad_gmm(n)
    torch.random.set_rng_state(state)

    x_1, y = Tensor(x_1).to(device), Tensor(y).to(device)
    y_null = torch.ones_like(y) * Y_NULL_VAL

    if prior is not None:
        mu_model, log_var = prior(y.unsqueeze(1).float())
        x_0_cond = mu_model + torch.exp(0.5 * log_var) * torch.randn_like(x_1)
        kl = (
            0.5 * torch.sum(torch.exp(log_var) + mu_model**2 - 1 - log_var, dim=-1)
        ).mean()
    else:
        x_0_cond = torch.randn_like(x_1)
        kl = torch.zeros((), device=device)

    z_uncond = torch.randn_like(x_1)
    z_fixed = torch.randn_like(x_1)  # after z_uncond — see docstring

    out = {"kl_heldout": float(kl.item())}
    for name, x_0, y_use in (
        ("cfm_cond", x_0_cond, y),
        ("cfm_uncond", z_uncond, y_null),
        ("cfm_cond_fixed", z_fixed, y),
    ):
        target = x_1 - x_0
        total = 0.0
        for t_val in torch.linspace(0.0, 1.0, n_t):
            t = torch.full((n, 1), float(t_val), device=device)
            x_t = t * x_1 + (1 - t) * x_0
            # `.mean()` over batch and both coords, matching train_one's reduction.
            total += torch.pow(target - vf(x_t=x_t, y=y_use, t=t), 2).mean().item()
        out[name] = total / n_t
    out["cfm_total"] = 0.5 * (out["cfm_cond"] + out["cfm_uncond"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument(
        "--learned-betas",
        type=float,
        nargs="+",
        default=[],
        help="extra learned-prior arms at these β, named learned_prior_b{β}",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--outdir", default="no_prior_ablation_results")
    args = parser.parse_args()

    variants = dict(VARIANTS)
    for b in args.learned_betas:
        name = f"learned_prior_b{b:g}"
        variants[name] = (True, b)
        if name not in args.variants:
            args.variants.append(name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/params.json", "w") as fh:
        json.dump(
            {
                "args": vars(args),
                "variants": {k: variants[k] for k in args.variants},
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "warmup_iters": WARMUP_ITERS,
                "n_eval": N_EVAL,
                "n_t": N_T,
                "eval_seed": EVAL_SEED,
                "device": str(device),
            },
            fh,
            indent=2,
        )

    rows = []
    for variant in args.variants:
        use_prior, beta = variants[variant]
        for seed in args.seeds:
            vf, prior, cfm_train, kl_train = train_one(
                use_prior, beta, seed, device, args.iterations
            )
            # Probes are run exactly as `beta_sweep.py` runs them — same imported
            # `make_seed` (invert the conditional field t: 1→0, never the base) and
            # same imported `probe` — so these rows drop straight into the β table.
            x0, y_cls, dist = make_seed(vf, device)
            row = {
                "variant": variant,
                "use_prior": use_prior,
                "beta": beta,
                "seed": seed,
                # `cfm_loss` / `straightness` keep beta_sweep's column names so the
                # two results.csv files can be concatenated directly.
                "cfm_loss": round(cfm_train, 5),
                "kl": round(kl_train, 6),
                "straightness": round(straightness(vf, prior, device), 5),
                **{k: round(v, 4) for k, v in probe(x0, y_cls, dist).items()},
                **{k: round(v, 5) for k, v in heldout_cfm(vf, prior, device).items()},
            }
            rows.append(row)
            print(
                f"{variant:<17} seed={seed}  cfm={row['cfm_loss']:.4f}  "
                f"S={row['straightness']:.4f}  "
                f"cls_logreg={row['cls_acc_logreg']:.3f}  "
                f"cls_mlp={row['cls_acc_mlp']:.3f}  "
                f"dist_r2_mlp={row['dist_r2_mlp']:.3f}  "
                f"cond={row['cfm_cond']:.4f}  "
                f"cond_fixed={row['cfm_cond_fixed']:.4f}  "
                f"uncond={row['cfm_uncond']:.4f}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(f"{args.outdir}/results.csv", index=False)

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("variant", sort=False)[
            [
                "cls_acc_logreg",
                "cls_acc_mlp",
                "dist_r2_linreg",
                "dist_r2_mlp",
                "cfm_loss",
                "straightness",
                "cfm_cond",
                "cfm_cond_fixed",
                "cfm_uncond",
                "cfm_total",
            ]
        ]
        .agg(["mean", "std"])
        .round(5)
    )
    summary = (
        "2D-Gaussian: conditional prior ablation — probes + final flow-matching loss\n"
        f"{len(args.seeds)} seeds per arm, {args.iterations} iterations each, "
        f"{N_EVAL} held-out samples (fixed eval seed {EVAL_SEED}), {N_T} t-points\n"
        f"condition = GMM component (4 classes, chance = {CHANCE:.2f}); "
        "LOWER cls_acc = better removal\n"
        "dist_r2 = preserved residual factor (HIGHER = better); "
        "S = straightness integral (LOWER = straighter)\n"
        "Probes/S/cfm_loss use beta_sweep.py's definitions verbatim, so these rows "
        "concatenate with its results.csv.\n"
        "cfm_cond scores each arm against ITS OWN base, so it is not a like-for-like\n"
        "model-quality score: a learned base shortens transport and thus lowers the\n"
        "loss for an easier problem. cfm_uncond should match across arms.\n\n"
        + agg.to_string()
        + "\n"
    )
    with open(f"{args.outdir}/summary.txt", "w") as fh:
        fh.write(summary)
    print("\n" + summary + f"\nSaved to {args.outdir}/")


if __name__ == "__main__":
    main()
