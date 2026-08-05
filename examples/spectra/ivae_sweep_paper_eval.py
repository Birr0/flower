"""ICA baselines vs Flower under the *paper's* evaluation protocol (R2 Table 1).

``ivae_sweep.py`` and the spectra benchmark the paper reports
(``embedding_benchmark.py``) disagree on the same quantity — redshift recoverable
from the raw ``spender_I`` latents — because they differ in three ways at once:

===================  ==========================  ==========================
                     ivae_sweep.py               embedding_benchmark.py
===================  ==========================  ==========================
MLP probe            (64, 32), max_iter=300      (64, 64), max_iter=1000
train rows           40k, random subsample       200k, first rows after mask
row mask             every physical target       redshift validity only;
                     valid (43.1k test rows)     per-target mask at probe
                                                 time (52.8k test rows for z)
===================  ==========================  ==========================

Raw ``z`` reads 0.555 under the first and 0.711 [0.703, 0.718] under the second.
A weaker probe understates recoverable redshift in *every* row, so quoting ICA and
Flower on the first while the paper quotes the second is not a like-for-like
comparison.

This script runs the ICA/iVAE/Flower comparison on the paper's protocol exactly:
data loading, masking and row truncation are imported from ``embedding_benchmark``
(``load_run``), and the probes are its ``ARCHITECTURES["2-Layer"]``. The method
machinery — FastICA, iVAE, the residual constructions, the source-dropping sweep —
is imported unchanged from ``ivae_sweep``. Only the evaluation protocol differs
from that script.

Note the physical targets are masked per target, as in the benchmark, so ``z`` is
measured on more rows than ``logSFR``/``A_v``. That is the paper's convention;
``ivae_sweep_matched_probe.py`` is the variant restricted to common rows.

Run from this directory:

    python ivae_sweep_paper_eval.py [--spender spender_I] [--epochs 30] [--n-k 9]
"""

import argparse
import json
import os

import matplotlib as mpl

mpl.use("Agg")
import embedding_benchmark as eb
import ivae_sweep as isw
import numpy as np
import pandas as pd
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

# The paper's probe: embedding_benchmark.ARCHITECTURES["2-Layer"].
PROBE = {"hidden_layer_sizes": (64, 64), "max_iter": 1000}
RANDOM_STATE = 42
TARGETS = eb.PHYSICAL_TARGETS  # ["logM*", "logSFR", "A_v"]


def _r2(x_tr, x_te, y_tr, y_te, kind):
    reg = (
        LinearRegression()
        if kind == "linreg"
        else MLPRegressor(random_state=RANDOM_STATE, **PROBE)
    )
    reg.fit(x_tr, y_tr)
    return r2_score(y_te, reg.predict(x_te))


def evaluate(x_tr, x_te, z, targets):
    """z removal (linear + MLP) and per-target preservation (MLP).

    ``targets[name] = (y_train, y_test, mask_train, mask_test)`` — each physical
    target carries its own validity mask, as in ``embedding_benchmark``, so it is
    evaluated on its own row subset while ``z`` uses all rows.

    Every representation is standardised (scaler fit on train) immediately before
    probing. This is not cosmetic: iVAE sources are raw encoder ``mu`` values with
    per-dimension std spanning 2.9-98.7 and means as far as -37, on which
    ``MLPRegressor`` fails to converge erratically depending on which columns
    survive the drop sweep — producing R^2 curves that *rise* as coordinates are
    deleted, which is impossible for nested feature sets. FastICA sources are
    unit-variance by construction (``whiten="unit-variance"``) and the raw/Flower
    embeddings were already scaled, so this only changes the iVAE arm, and it makes
    every row read out on the same footing.
    """
    z_tr, z_te = z
    sc = StandardScaler().fit(x_tr)
    x_tr, x_te = sc.transform(x_tr), sc.transform(x_te)
    out = {
        "z_r2_linreg": _r2(x_tr, x_te, z_tr, z_te, "linreg"),
        "z_r2_mlp": _r2(x_tr, x_te, z_tr, z_te, "mlp"),
        "n_dims": x_tr.shape[1],
    }
    for name, (y_tr, y_te, m_tr, m_te) in targets.items():
        out[f"{name}_r2_mlp"] = _r2(x_tr[m_tr], x_te[m_te], y_tr, y_te, "mlp")
    return out


def check_monotonic(df, tol=0.02):
    """Sanity gate on the source-dropping sweep.

    ``drop_top_k_dependent`` ranks columns once and removes the top ``k``, so the
    kept sets are nested: ``keep(k+1) subset keep(k)``. Deleting a coordinate cannot
    add information, so R^2 must be non-increasing in ``k`` for every target. A rise
    beyond probe noise means the probe failed to fit, not that the representation
    improved — the failure mode that unstandardised iVAE sources produced.
    """
    cols = ["z_r2_linreg", "z_r2_mlp"] + [f"{t}_r2_mlp" for t in TARGETS]
    violations = []
    for src in df[df.method == "residA"].source.unique():
        a = df[(df.source == src) & (df.method == "residA")].sort_values("k")
        for col in cols:
            v = a[col].to_numpy()
            rises = np.where(np.diff(v) > tol)[0]
            for i in rises:
                violations.append(
                    f"  {src} {col}: k={a.k.iloc[i]} -> {a.k.iloc[i + 1]} "
                    f"rises {v[i]:.3f} -> {v[i + 1]:.3f}"
                )
    if violations:
        print(f"\n*** MONOTONICITY VIOLATIONS (tol={tol}) — probe fitting suspect ***")
        print("\n".join(violations))
    else:
        print(
            f"\nMonotonicity check passed (tol={tol}): all residA curves "
            "non-increasing in k."
        )
    return violations


def load_paper_protocol(spender, n_filter, n_train):
    """Load embeddings and targets exactly as ``embedding_benchmark`` does.

    ``load_run`` masks on ``isfinite(x).all(axis=1)``, which is computed from the
    embedding itself, so ``orig``/``cond``/``uncond`` need not keep the same rows.
    The condition and the targets are therefore re-aligned to *each* split's own
    mask, mirroring ``evaluate_split``, which is called per split for that reason.
    Sharing one split's ``z`` across all three would silently misalign rows.
    """
    load_dotenv()
    embed_path = f"{os.getenv('DATA_ROOT')}/{isw.SPENDER_MAP[spender]}"
    cat, (z_train, z_test), runs = eb.load_run(
        embed_path,
        splits=["orig", "cond", "uncond"],
        n_filter=n_filter,
        n_train=n_train,
        matched=False,
    )
    cat_train = {t: eb._clean(cat["train"][t])[:n_filter] for t in TARGETS}
    cat_test = {t: eb._clean(cat["test"][t]) for t in TARGETS}

    out = {}
    for split, (x_tr, x_te, mask_train, mask_test) in runs.items():
        targets = {}
        for t in TARGETS:
            y_tr = cat_train[t][mask_train][:n_train]
            y_te = cat_test[t][mask_test]
            m_tr, m_te = eb._valid(y_tr), eb._valid(y_te)
            targets[t] = (y_tr[m_tr], y_te[m_te], m_tr, m_te)
        out[split] = {
            "x": (x_tr, x_te),
            "z": (z_train[mask_train][:n_train], z_test[mask_test]),
            "targets": targets,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spender", default="spender_I", choices=list(isw.SPENDER_MAP))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=eb.N_TRAIN)
    parser.add_argument("--n-filter", type=int, default=eb.N_FILTER)
    parser.add_argument("--n-k", type=int, default=9)
    parser.add_argument("--outdir", default="ivae_sweep_paper_eval_results")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "params.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "protocol": "embedding_benchmark.load_run(matched=False)",
                "probe": {
                    "hidden_layer_sizes": list(PROBE["hidden_layer_sizes"]),
                    "max_iter": PROBE["max_iter"],
                },
                "embed_path": isw.SPENDER_MAP[args.spender],
                "ivae": "see ivae_sweep._dump_params",
            },
            f,
            indent=2,
        )

    data = load_paper_protocol(args.spender, args.n_filter, args.n_train)
    orig = data["orig"]
    z_tr, z_te = orig["z"]
    targets = orig["targets"]
    o_tr, o_te = orig["x"]
    for split, dat in data.items():
        print(f"{split}: train {dat['x'][0].shape}, test {dat['x'][1].shape}")
    for t, (y_tr, y_te, _, _) in targets.items():
        print(f"  {t}: {y_tr.shape[0]} train / {y_te.shape[0]} test valid")

    def scaled(train, test):
        sc = StandardScaler().fit(train)
        return (
            sc.transform(train).astype(np.float32),
            sc.transform(test).astype(np.float32),
        )

    x_tr, x_te = scaled(o_tr, o_te)
    u_tr = z_tr.reshape(-1, 1).astype(np.float32)
    u_te = z_te.reshape(-1, 1).astype(np.float32)
    d = x_tr.shape[1]

    def ev(xt, xe):
        return evaluate(xt, xe, (z_tr, z_te), targets)

    rows = [{"source": "Raw", "method": "none", "k": 0, **ev(x_tr, x_te)}]

    print("Fitting FastICA...")
    ica = FastICA(
        n_components=d, random_state=args.seed, max_iter=1000, whiten="unit-variance"
    )
    f_tr, f_te = ica.fit_transform(x_tr), ica.transform(x_te)
    fB_tr, coef = regression_residual(f_tr, z_tr)
    fB_te, _ = regression_residual(f_te, z_te, coef)
    rows.append({"source": "FastICA", "method": "residB", "k": 0, **ev(fB_tr, fB_te)})

    print(f"Training iVAE ({args.epochs} epochs)...")
    ivae = isw.train_ivae(x_tr, u_tr, epochs=args.epochs, seed=args.seed)
    s_tr, pm_tr = isw.encode(ivae, x_tr, u_tr)
    s_te, pm_te = isw.encode(ivae, x_te, u_te)
    iB_tr = conditional_prior_residual(s_tr, pm_tr)
    iB_te = conditional_prior_residual(s_te, pm_te)
    rows.append({"source": "iVAE", "method": "residB", "k": 0, **ev(iB_tr, iB_te)})

    k_grid = sorted(set(np.linspace(1, d - 1, args.n_k).round().astype(int).tolist()))
    for name, (src_tr, src_te) in {
        "FastICA": (f_tr, f_te),
        "iVAE": (s_tr, s_te),
    }.items():
        for k in k_grid:
            print(f"  {name} residA k={k}")
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

    resid_models = {
        "Resid-linear": LinearRegression(),
        "Resid-mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=args.seed
        ),
        "Resid-rf": RandomForestRegressor(n_estimators=100, random_state=args.seed),
    }
    for name, mdl in resid_models.items():
        print(f"Evaluating {name}...")
        r_tr, r_te = isw.model_residual(x_tr, z_tr, x_te, z_te, mdl)
        rows.append({"source": name, "method": "direct", "k": 0, **ev(r_tr, r_te)})

    # Each Flower split carries its own row mask, so it is evaluated against its
    # own z / targets rather than against `orig`'s.
    for label, split in {"Flower-cond": "cond", "Flower-uncond": "uncond"}.items():
        dat = data[split]
        xt, xe = scaled(*dat["x"])
        rows.append(
            {
                "source": label,
                "method": "embedding",
                "k": 0,
                **evaluate(xt, xe, dat["z"], dat["targets"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "results.csv"), index=False)

    cols = ["source", "method", "k", "n_dims", "z_r2_linreg", "z_r2_mlp"] + [
        f"{t}_r2_mlp" for t in TARGETS
    ]
    table = df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    header = (
        f"ICA baselines vs Flower on {args.spender} spectra, "
        "embedding_benchmark protocol (probe (64,64)/1000, z-validity mask)\n"
        "z R2: LOWER = better removal | logM*/logSFR/A_v R2: HIGHER = better "
        "preservation\n"
    )
    violations = check_monotonic(df)
    footer = (
        "\nMonotonicity check: FAILED — probe fitting suspect, do not quote:\n"
        + "\n".join(violations)
        if violations
        else "\nMonotonicity check: passed (residA curves non-increasing in k)."
    )
    with open(os.path.join(args.outdir, "summary.txt"), "w") as fh:
        fh.write(header + "\n" + table + "\n" + footer + "\n")
    print("\n" + header + "\n" + table)

    isw.make_tradeoff_plot(
        df, TARGETS, args.spender, os.path.join(args.outdir, "tradeoff.png")
    )
    print(f"\nSaved results.csv, summary.txt, tradeoff.png in {args.outdir}/")


if __name__ == "__main__":
    main()
