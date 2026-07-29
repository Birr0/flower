"""Comprehensive ICA-baseline sweep on RGB-MNIST embeddings (issue #20 / E2).

Covers all the removal configurations for the linear (FastICA) and nonlinear
(iVAE) ICA baselines, so we can see the full removal-vs-preservation picture and
the cross-probe (linear vs MLP) behaviour:

Source models
-------------
- **FastICA** — linear ICA floor (unsupervised).
- **iVAE-cond** — reference iVAE, encoder conditioned on ``u = one-hot(digit)``.
- **iVAE-fair** — iVAE with ``condition_encoder=False`` (encoder sees only ``x``),
  so the sources cannot bake the digit back in through the encoder.

Residuals
---------
- **residual B** — subtract the digit mean (empirical conditional-mean for
  FastICA; learned conditional-prior ``lambda_mu(u)`` for the iVAEs).
- **residual A(k)** — drop the ``k`` most digit-dependent sources, swept over
  ``k = 1 .. d-1``.

Every residual is probed for digit removal with a **linear** (logreg) and a
**nonlinear** (MLP) classifier, and for colour-``b`` preservation with linear and
MLP regressors (point estimates — the earlier ``ivae_benchmark.py`` gives
bootstrap CIs for the headline configs).

All outputs are saved under ``--outdir``: ``results.csv`` (every row),
``summary.txt`` (the printed table), and ``tradeoff.png`` (digit-MLP-accuracy vs
colour-``b`` MLP-R², the removal/preservation frontier).

Run from this directory (needs ``DATA_ROOT``):

    python ivae_sweep.py [--epochs 30] [--outdir ivae_sweep_results]
"""

import argparse
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from flower.evaluation.ica import (
    conditional_mean_residual,
    conditional_prior_residual,
    drop_top_k_dependent,
)
from flower.evaluation.metrics import prepare_data
from flower.models.ivae import IVAE, LightningIVAE

RANDOM_STATE = 42
N_DIGITS = 10
DEFAULT_EMBED_SUBPATH = "rgbmnist/rgbmnist_Flow_cond_prior/embeddings/7518770_0"


def _one_hot(labels, n):
    return np.eye(n, dtype=np.float32)[labels]


def train_ivae(
    x, u, *, condition_encoder, epochs, lr=1e-2, batch_size=256, seed=RANDOM_STATE
):
    torch.manual_seed(seed)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    u_t = torch.as_tensor(u, dtype=torch.float32)
    ivae = IVAE(
        data_dim=x.shape[1],
        aux_dim=u.shape[1],
        latent_dim=x.shape[1],
        hidden_dim=128,
        n_layers=3,
        activation="xtanh",
        learn_prior_mean=True,
        condition_encoder=condition_encoder,
    )
    lit = LightningIVAE(ivae, lr=lr, batch_size=batch_size, beta=1.0)
    opt = torch.optim.Adam(ivae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, epochs // 3), gamma=0.3
    )
    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss, _, _ = lit._losses(ivae(x_t[idx], u_t[idx]), x_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        out_tr = ivae(x_t, u_t)
    return ivae, out_tr


def encode(ivae, x, u):
    with torch.no_grad():
        out = ivae(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(u, dtype=torch.float32),
        )
    return out["mu"].numpy(), out["prior_mu"].numpy()


def _digit_acc(x_tr, x_te, y_tr, y_te, kind):
    if kind == "logreg":
        clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE
        )
    clf.fit(x_tr, y_tr)
    return accuracy_score(y_te, clf.predict(x_te))


def _b_r2(x_tr, x_te, y_tr, y_te, kind):
    if kind == "linreg":
        reg = LinearRegression()
    else:
        reg = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE
        )
    reg.fit(x_tr, y_tr)
    return r2_score(y_te, reg.predict(x_te))


def evaluate(x_tr, x_te, dig_tr, dig_te, b_tr, b_te, rot_tr=None, rot_te=None):
    """All probes; returns a dict of point estimates.

    Digit = condition (want removed). Colour ``b`` and rotation = independent
    factors (want preserved); rotation is optional (needs the cached CSVs).

    Every representation is standardised (scaler fit on train) immediately before
    probing. The raw embedding and the FastICA sources are already unit-scale (the
    latter by ``whiten="unit-variance"``), but iVAE sources are raw encoder ``mu``
    values whose per-dimension scales differ by more than an order of magnitude.
    Probing those unscaled makes ``MLPClassifier``/``MLPRegressor`` fail to converge
    erratically depending on which columns survive the drop sweep, and additionally
    distorts ``LogisticRegression``, whose L2 penalty is not scale-invariant. On the
    spectra analogue this produced accuracy curves that *improved* as coordinates
    were deleted — impossible for nested feature sets — in the direction that
    flatters the baseline. See ``examples/spectra/ivae_sweep_paper_eval.py``.
    """
    sc = StandardScaler().fit(x_tr)
    x_tr, x_te = sc.transform(x_tr), sc.transform(x_te)
    out = {
        "digit_acc_logreg": _digit_acc(x_tr, x_te, dig_tr, dig_te, "logreg"),
        "digit_acc_mlp": _digit_acc(x_tr, x_te, dig_tr, dig_te, "mlp"),
        "b_r2_linreg": _b_r2(x_tr, x_te, b_tr, b_te, "linreg"),
        "b_r2_mlp": _b_r2(x_tr, x_te, b_tr, b_te, "mlp"),
        "n_dims": x_tr.shape[1],
    }
    if rot_tr is not None:
        out["rot_r2_linreg"] = _b_r2(x_tr, x_te, rot_tr, rot_te, "linreg")
        out["rot_r2_mlp"] = _b_r2(x_tr, x_te, rot_tr, rot_te, "mlp")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embed-subpath", type=str, default=DEFAULT_EMBED_SUBPATH)
    parser.add_argument("--outdir", type=str, default="ivae_sweep_results")
    parser.add_argument("--n-k", type=int, default=11, help="number of k values swept")
    parser.add_argument(
        "--models",
        type=str,
        default="FastICA,iVAE-cond,iVAE-fair",
        help="comma-separated source models to run",
    )
    parser.add_argument(
        "--rotation-dir",
        type=str,
        default=None,
        help="dir with {train,test}_rotation_aligned.csv to add rotation as a "
        "second preservation target (see compute_rotation.py)",
    )
    parser.add_argument(
        "--skip-residb",
        action="store_true",
        help="skip the residual-B (mean / conditional-prior subtraction) rows and "
        "sweep only residual A. Mean-residualisation is near-idempotent and is "
        "covered by iterated_residual.py, so the drop sweep is the informative arm.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    load_dotenv()
    embed_path = f"{os.getenv('DATA_ROOT')}/{args.embed_subpath}"
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"{embed_path}/train/*.parquet",
            "test": f"{embed_path}/test/*.parquet",
        },
    )
    x_tr, dig_tr, x_te, dig_te = prepare_data(ds, "orig", "digit")
    _, b_tr, _, b_te = prepare_data(ds, "orig", "b")
    dig_tr, dig_te = dig_tr.astype(int), dig_te.astype(int)

    # Optional second preservation target: rotation (row-aligned CSVs).
    rot_tr = rot_te = None
    if args.rotation_dir:
        rot_tr = pd.read_csv(
            os.path.join(args.rotation_dir, "train_rotation_aligned.csv")
        )["Rotation_Deg"].to_numpy()
        rot_te = pd.read_csv(
            os.path.join(args.rotation_dir, "test_rotation_aligned.csv")
        )["Rotation_Deg"].to_numpy()
        if len(rot_tr) != len(dig_tr) or len(rot_te) != len(dig_te):
            msg = "rotation CSV length does not match embeddings (row misalignment)"
            raise ValueError(msg)
        print(f"rotation targets loaded (train {rot_tr.shape}, test {rot_te.shape})")

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)
    u_tr, u_te = _one_hot(dig_tr, N_DIGITS), _one_hot(dig_te, N_DIGITS)
    d = x_tr.shape[1]
    print(f"embeddings: train {x_tr.shape}, test {x_te.shape}")

    # --- source models (only those selected via --models) ---
    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    sources = {}

    if "FastICA" in selected:
        print("Fitting FastICA...")
        ica = FastICA(
            n_components=d,
            random_state=args.seed,
            max_iter=1000,
            whiten="unit-variance",
        )
        fica_tr, fica_te = ica.fit_transform(x_tr), ica.transform(x_te)
        sources["FastICA"] = (fica_tr, fica_te, None, None)

    if "iVAE-cond" in selected:
        print(f"Training iVAE-cond ({args.epochs} epochs)...")
        ivae_c, _ = train_ivae(
            x_tr, u_tr, condition_encoder=True, epochs=args.epochs, seed=args.seed
        )
        sc_tr, pmc_tr = encode(ivae_c, x_tr, u_tr)
        sc_te, pmc_te = encode(ivae_c, x_te, u_te)
        sources["iVAE-cond"] = (sc_tr, sc_te, pmc_tr, pmc_te)

    if "iVAE-fair" in selected:
        print(f"Training iVAE-fair ({args.epochs} epochs)...")
        ivae_f, _ = train_ivae(
            x_tr, u_tr, condition_encoder=False, epochs=args.epochs, seed=args.seed
        )
        sf_tr, pmf_tr = encode(ivae_f, x_tr, u_tr)
        sf_te, pmf_te = encode(ivae_f, x_te, u_te)
        sources["iVAE-fair"] = (sf_tr, sf_te, pmf_tr, pmf_te)

    k_grid = sorted(set(np.linspace(1, d - 1, args.n_k).round().astype(int).tolist()))
    rows = []

    def ev(xt, xe):
        return evaluate(xt, xe, dig_tr, dig_te, b_tr, b_te, rot_tr, rot_te)

    # Raw baseline.
    print("Evaluating: Raw embedding")
    rows.append({"source": "Raw", "method": "none", "k": 0, **ev(x_tr, x_te)})

    for name, (s_tr, s_te, pm_tr, pm_te) in sources.items():
        # Residual B (subtract the digit mean).
        if not args.skip_residb:
            if name == "FastICA":
                rB_tr, means = conditional_mean_residual(s_tr, dig_tr)
                rB_te, _ = conditional_mean_residual(s_te, dig_te, means)
            else:
                rB_tr = conditional_prior_residual(s_tr, pm_tr)
                rB_te = conditional_prior_residual(s_te, pm_te)
            print(f"Evaluating: {name} residual B")
            rows.append(
                {"source": name, "method": "residB", "k": 0, **ev(rB_tr, rB_te)}
            )

        # Residual A(k): drop the k most digit-dependent sources (rank on train).
        for k in k_grid:
            _, dropped = drop_top_k_dependent(s_tr, dig_tr, k=k)
            keep = np.setdiff1d(np.arange(d), dropped)
            print(f"Evaluating: {name} residual A(k={k})")
            rows.append(
                {
                    "source": name,
                    "method": "residA",
                    "k": int(k),
                    **ev(s_tr[:, keep], s_te[:, keep]),
                }
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "results.csv")
    df.to_csv(csv_path, index=False)

    # Printed summary (also saved to summary.txt).
    cols = [
        "source",
        "method",
        "k",
        "n_dims",
        "digit_acc_logreg",
        "digit_acc_mlp",
        "b_r2_linreg",
        "b_r2_mlp",
    ]
    if rot_tr is not None:
        cols += ["rot_r2_linreg", "rot_r2_mlp"]
    table = df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    rot_line = (
        " | rotation R2: higher = better preservation" if rot_tr is not None else ""
    )
    header = (
        "ICA-baseline sweep on RGB-MNIST (class-only; condition = digit)\n"
        "digit acc: lower = better removal (chance 0.10) | "
        "b R2: higher = better preservation" + rot_line + "\n"
    )
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(header + "\n" + table + "\n")
    print("\n" + header + "\n" + table)

    # Tradeoff plot: removal (digit MLP acc, x) vs preservation (MLP R2, y).
    # Colour b is solid; rotation (if available) is dashed — both should stay
    # high as the digit is removed, since both are independent of the digit.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name in sources:
        a = df[(df.source == name) & (df.method == "residA")].sort_values("k")
        (line,) = ax.plot(
            a.digit_acc_mlp, a.b_r2_mlp, "-o", label=f"{name} resid A — b"
        )
        b = df[(df.source == name) & (df.method == "residB")].iloc[0]
        ax.scatter(
            b.digit_acc_mlp,
            b.b_r2_mlp,
            marker="*",
            s=220,
            color=line.get_color(),
            edgecolor="k",
            zorder=5,
            label=f"{name} resid B — b",
        )
        if rot_tr is not None:
            ax.plot(
                a.digit_acc_mlp,
                a.rot_r2_mlp,
                "--s",
                color=line.get_color(),
                alpha=0.7,
                label=f"{name} resid A — rotation",
            )
    raw = df[df.source == "Raw"].iloc[0]
    ax.scatter(
        raw.digit_acc_mlp,
        raw.b_r2_mlp,
        marker="s",
        s=90,
        color="grey",
        zorder=5,
        label="Raw — b",
    )
    ax.axvline(0.10, ls=":", color="red", lw=1, label="digit chance (0.10)")
    ax.set_xlabel("digit accuracy — MLP probe  (lower = better removal)")
    ax.set_ylabel("preservation R² — MLP  (higher = better)")
    ax.set_title("RGB-MNIST: digit removal vs factor preservation (class-only)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "tradeoff.png"), dpi=140)

    print(f"\nSaved: {csv_path}, summary.txt, tradeoff.png  (in {args.outdir}/)")


if __name__ == "__main__":
    main()
