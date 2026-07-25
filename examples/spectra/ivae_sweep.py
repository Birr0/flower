"""ICA-baseline sweep on spectra (spender) embeddings (issue #20 / E2).

The spectra analogue of ``examples/mnist/ivae_sweep.py``, but the condition —
**redshift z** — is *continuous*, so everything is regression and the residual
mechanics use the continuous variants:

- residual B: FastICA -> ``regression_residual`` (subtract E[sources | z]);
  iVAE -> ``conditional_prior_residual`` (subtract the learned lambda_mu(z)).
- residual A: drop the k sources most |correlated| with z
  (``drop_top_k_dependent(..., dependence="continuous")``).

Metrics (MLP + linear probes, point estimates):
- **z** regression  -> condition; LOWER R^2 = better removal.
- **logM*, logSFR, A_v** regression -> physical residual factors; HIGHER = better
  preservation. These are entangled with redshift to differing degrees, so they
  are the spectra counterpart of "rotation" (the meaningful preservation test).

Flower's ``cond`` / ``uncond`` embeddings are evaluated directly (no training) for
the surgical-vs-blunt comparison.

Embeddings load from ``$DATA_ROOT/sdss_II/...`` (redshift is the ``Z`` column, so
the condition needs no catalog); physical targets come from the HF catalog
``Birr001/spectra_catalog`` (index-aligned to the embeddings, as in
``flower_benchmark.py``). Run from this directory:

    python ivae_sweep.py --spender spender_I [--n-train 40000] [--epochs 30]
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
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from flower.evaluation.ica import (
    conditional_prior_residual,
    drop_top_k_dependent,
    regression_residual,
)
from flower.models.ivae import IVAE, LightningIVAE

RANDOM_STATE = 42
CONDITION = "z"
TARGETS = ["logM*", "logSFR", "A_v"]  # physical factors to preserve
SPENDER_MAP = {
    "spender_I": "sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7526202_0",
    "spender_II": "sdss_II/spender_I_flow_v2/spender_II_flow_v2/embeddings/7527549_0",
}


def _r2(x_tr, x_te, y_tr, y_te, kind):
    if kind == "linreg":
        reg = LinearRegression()
    else:
        reg = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE
        )
    reg.fit(x_tr, y_tr)
    return r2_score(y_te, reg.predict(x_te))


def evaluate(x_tr, x_te, z_tr, z_te, targ_tr, targ_te):
    """z removal (linear + MLP) and physical-target preservation (MLP)."""
    out = {
        "z_r2_linreg": _r2(x_tr, x_te, z_tr, z_te, "linreg"),
        "z_r2_mlp": _r2(x_tr, x_te, z_tr, z_te, "mlp"),
        "n_dims": x_tr.shape[1],
    }
    for name in TARGETS:
        out[f"{name}_r2_mlp"] = _r2(x_tr, x_te, targ_tr[name], targ_te[name], "mlp")
    return out


def _dump_params(args):
    """Write the resolved run configuration to <outdir>/params.json for
    reproducibility. Fixed model/probe hyperparameters are documented in
    examples/ICA_HYPERPARAMETERS.md."""
    params = {
        "args": vars(args),
        "condition": CONDITION,
        "targets": TARGETS,
        "embed_path": SPENDER_MAP[args.spender],
        "ivae": {
            "hidden_dim": 128,
            "n_layers": 3,
            "activation": "xtanh",
            "slope": 0.1,
            "decoder_var": 0.01,
            "learn_prior_mean": True,
            "condition_encoder": True,
            "lr": 1e-2,
            "batch_size": 256,
            "beta": 1.0,
            "scheduler": "StepLR(step_size=epochs//3, gamma=0.3)",
        },
        "fastica": {"n_components": "d", "max_iter": 1000, "whiten": "unit-variance"},
        "probes": {
            "regressor_mlp": {"hidden_layer_sizes": [64, 32], "max_iter": 300},
            "regressor_linear": "LinearRegression",
        },
        "direct_residualisers": {
            "linear": "LinearRegression",
            "mlp": {"hidden_layer_sizes": [64, 32], "max_iter": 300},
            "rf": {"n_estimators": 100},
        },
    }
    with open(os.path.join(args.outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)


def model_residual(x_tr, z_tr, x_te, z_te, model):
    """Direct residualisation of the embedding against a continuous condition:
    fit ``model`` to predict the embedding from ``z`` on train, subtract its
    prediction on both splits. Linear = same as ``regression_residual``; an MLP/RF
    model removes *nonlinear* z-dependence (the standard strong baseline)."""
    z2_tr, z2_te = z_tr.reshape(-1, 1), z_te.reshape(-1, 1)
    model.fit(z2_tr, x_tr)
    return x_tr - model.predict(z2_tr), x_te - model.predict(z2_te)


def train_ivae(x, u, *, epochs, lr=1e-2, batch_size=256, seed=RANDOM_STATE):
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
    return ivae


def encode(ivae, x, u):
    with torch.no_grad():
        out = ivae(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(u, dtype=torch.float32),
        )
    return out["mu"].numpy(), out["prior_mu"].numpy()


def _clean(v):
    return pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=float)


def load_data(spender, n_train, n_filter):
    load_dotenv()
    embed_path = f"{os.getenv('DATA_ROOT')}/{SPENDER_MAP[spender]}"
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"{embed_path}/train/*.parquet",
            "test": f"{embed_path}/test/*.parquet",
        },
    )
    cat = load_dataset("Birr001/spectra_catalog")

    def split(embed_split, cat_split, cap):
        emb = ds[embed_split]
        orig = np.array(emb["orig"], dtype=float)[:cap]
        cond = np.array(emb["cond"], dtype=float)[:cap]
        uncond = np.array(emb["uncond"], dtype=float)[:cap]
        z = _clean(emb["Z"])[:cap]
        targ = {t: _clean(cat[cat_split][t])[:cap] for t in TARGETS}
        mask = np.isfinite(z) & (z != -99.0) & np.isfinite(orig).all(axis=1)
        for t in TARGETS:
            mask &= np.isfinite(targ[t]) & (targ[t] != -99.0)
        idx = np.where(mask)[0]
        targ_m = {t: targ[t][idx] for t in TARGETS}
        return orig[idx], cond[idx], uncond[idx], z[idx], targ_m

    tr = split("train", "train", n_filter)
    te = split("test", "test", None)
    # subsample train
    if n_train and len(tr[0]) > n_train:
        rng = np.random.default_rng(RANDOM_STATE)
        sel = rng.choice(len(tr[0]), n_train, replace=False)
        tr = (
            tr[0][sel],
            tr[1][sel],
            tr[2][sel],
            tr[3][sel],
            {t: tr[4][t][sel] for t in TARGETS},
        )
    return tr, te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spender", default="spender_I", choices=list(SPENDER_MAP))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-filter", type=int, default=300000)
    parser.add_argument("--n-k", type=int, default=8)
    parser.add_argument("--outdir", default="ivae_sweep_results")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    _dump_params(args)
    (o_tr, c_tr, u_tr_e, z_tr, tg_tr), (o_te, c_te, u_te_e, z_te, tg_te) = load_data(
        args.spender, args.n_train, args.n_filter
    )
    print(
        f"train {o_tr.shape}, test {o_te.shape}; "
        f"z range [{z_tr.min():.3f}, {z_tr.max():.3f}]"
    )

    # Scale embeddings (fit on train) for each embedding type.
    def scaled(train, test):
        sc = StandardScaler().fit(train)
        return (
            sc.transform(train).astype(np.float32),
            sc.transform(test).astype(np.float32),
        )

    x_tr, x_te = scaled(o_tr, o_te)
    u_tr = z_tr.reshape(-1, 1).astype(np.float32)  # continuous condition
    u_te = z_te.reshape(-1, 1).astype(np.float32)
    d = x_tr.shape[1]

    def ev(xt, xe):
        return evaluate(xt, xe, z_tr, z_te, tg_tr, tg_te)

    rows = []
    rows.append({"source": "Raw", "method": "none", "k": 0, **ev(x_tr, x_te)})

    # FastICA (linear floor).
    print("Fitting FastICA...")
    ica = FastICA(
        n_components=d, random_state=args.seed, max_iter=1000, whiten="unit-variance"
    )
    f_tr, f_te = ica.fit_transform(x_tr), ica.transform(x_te)
    fB_tr, coef = regression_residual(f_tr, z_tr)
    fB_te, _ = regression_residual(f_te, z_te, coef)
    rows.append({"source": "FastICA", "method": "residB", "k": 0, **ev(fB_tr, fB_te)})

    # iVAE (nonlinear).
    print(f"Training iVAE ({args.epochs} epochs)...")
    ivae = train_ivae(x_tr, u_tr, epochs=args.epochs, seed=args.seed)
    s_tr, pm_tr = encode(ivae, x_tr, u_tr)
    s_te, pm_te = encode(ivae, x_te, u_te)
    iB_tr = conditional_prior_residual(s_tr, pm_tr)
    iB_te = conditional_prior_residual(s_te, pm_te)
    rows.append({"source": "iVAE", "method": "residB", "k": 0, **ev(iB_tr, iB_te)})

    # Residual A(k) sweeps (continuous dependence on z).
    k_grid = sorted(set(np.linspace(1, d - 1, args.n_k).round().astype(int).tolist()))
    src_sets = {"FastICA": (f_tr, f_te), "iVAE": (s_tr, s_te)}
    for name, (src_tr, src_te) in src_sets.items():
        for k in k_grid:
            _, dropped = drop_top_k_dependent(
                src_tr, z_tr, k=k, dependence="continuous"
            )
            keep = np.setdiff1d(np.arange(d), dropped)
            rows.append(
                {
                    "source": name,
                    "method": "residA",
                    "k": int(k),
                    **ev(src_tr[:, keep], src_te[:, keep]),
                }
            )

    # Direct residualisation of the embedding against z (the standard baselines):
    # linear (== regression residual) and nonlinear (MLP, RF) mean-removal.
    resid_models = {
        "Resid-linear": LinearRegression(),
        "Resid-mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=args.seed
        ),
        "Resid-rf": RandomForestRegressor(n_estimators=100, random_state=args.seed),
    }
    for name, mdl in resid_models.items():
        print(f"Evaluating {name}...")
        r_tr, r_te = model_residual(x_tr, z_tr, x_te, z_te, mdl)
        rows.append({"source": name, "method": "direct", "k": 0, **ev(r_tr, r_te)})

    # Flower cond / uncond (evaluate embeddings directly).
    for label, (train, test) in {
        "Flower-cond": (c_tr, c_te),
        "Flower-uncond": (u_tr_e, u_te_e),
    }.items():
        xt, xe = scaled(train, test)
        rows.append({"source": label, "method": "embedding", "k": 0, **ev(xt, xe)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "results.csv"), index=False)

    cols = ["source", "method", "k", "n_dims", "z_r2_linreg", "z_r2_mlp"] + [
        f"{t}_r2_mlp" for t in TARGETS
    ]
    table = df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    header = (
        f"ICA-baseline sweep on {args.spender} spectra (condition = redshift z)\n"
        "z R2: LOWER = better removal | logM*/logSFR/A_v R2: HIGHER = better "
        "preservation\n"
    )
    with open(os.path.join(args.outdir, "summary.txt"), "w") as fh:
        fh.write(header + "\n" + table + "\n")
    print("\n" + header + "\n" + table)

    make_tradeoff_plot(
        df, TARGETS, args.spender, os.path.join(args.outdir, "tradeoff.png")
    )
    print(f"\nSaved results.csv, summary.txt, tradeoff.png in {args.outdir}/")


def make_tradeoff_plot(df, targets, spender, outpath):
    """One panel per physical factor: redshift removal (x) vs that factor's
    preservation (y). Line = source-dropping sweep; markers = the direct
    embeddings / residual-B points."""
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), sharex=True)
    axes = np.atleast_1d(axes)
    sweep_colors = {"FastICA": "tab:blue", "iVAE": "tab:orange"}
    points = {  # label -> (color, marker, size)
        "Flower-cond": ("crimson", "*", 320),
        "Flower-uncond": ("purple", "P", 120),
        "Raw": ("grey", "s", 110),
        "Resid-linear": ("tab:green", "v", 90),
        "Resid-mlp": ("tab:red", "D", 90),
        "Resid-rf": ("saddlebrown", "h", 90),
    }
    for ax, t in zip(axes, targets, strict=False):
        col = f"{t}_r2_mlp"
        for name, color in sweep_colors.items():
            a = df[(df.source == name) & (df.method == "residA")].sort_values("k")
            ax.plot(
                a.z_r2_mlp, a[col], "-o", color=color, ms=4, label=f"{name} drop (k↑)"
            )
            rb = df[(df.source == name) & (df.method == "residB")]
            if not rb.empty:
                ax.scatter(
                    rb.iloc[0].z_r2_mlp,
                    rb.iloc[0][col],
                    marker="X",
                    s=110,
                    color=color,
                    edgecolor="k",
                    zorder=5,
                    label=f"{name} residB",
                )
        for label, (color, mk, size) in points.items():
            r = df[df.source == label]
            if not r.empty:
                ax.scatter(
                    r.iloc[0].z_r2_mlp,
                    r.iloc[0][col],
                    marker=mk,
                    s=size,
                    color=color,
                    edgecolor="k",
                    zorder=6,
                    label=label,
                )
        ax.set_title(t)
        ax.set_xlabel("redshift z R²  (← better removal)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("factor R² — MLP  (higher = better preservation)")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(
        f"{spender}: redshift removal vs physical-factor preservation "
        "(surgical Flower vs blunt ICA source-dropping)"
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)


if __name__ == "__main__":
    main()
