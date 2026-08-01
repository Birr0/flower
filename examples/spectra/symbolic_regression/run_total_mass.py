"""Run the symbolic-regression sweep for a second mass target, locally.

``train_logm.py`` is a SLURM entrypoint with the cluster's data paths hardcoded
(``/data/dtce-schmidt/...``) and no way to point it elsewhere. Rather than edit it --
which would risk the provenance of the published LGM_FIB_P50 fronts -- this runner
reuses the pieces that matter, unmodified:

- ``val_rescore.build_merged`` for the embeddings + astroML + galSpecExtra merge,
  which is already validated: re-scoring the stored fronts through it reproduces their
  recorded ``Test_R2`` for 305/320 equations at a median error of 3e-08.
- ``utils.fit_sym_fn`` and ``utils.fit_model_fn`` verbatim, so the search, the 10k
  subsample, the scaling and the MLP/LR references are identical to the published run.

Output goes to ``job_results_{FEATURE}_{EMBED}_seed{SEED}/`` in exactly the schema
``train_logm.py`` writes, so ``variable_additions.py``, ``latent_zero_limit.py`` and
``val_rescore.py`` all read it without changes.

**Why a second target.** The published fronts target ``LGM_FIB_P50``, a *fibre* mass.
A flux limit constrains *total* luminosity, so the selection-function argument invites
the reply that we have merely rediscovered the aperture. ``lgm_tot_p50`` is aperture-
corrected and settles it. It is also what the paper's Data section actually describes
(astroML supplies no fibre mass).

**Version caveat.** pyoperon here is 0.4.0 from PyPI; the published fronts were
produced on the cluster with an unrecorded build. Constant optimisation and search
behaviour can differ between builds, so a fibre-vs-total comparison is only strictly
like-for-like if *both* targets are run on the same pyoperon. Re-running LGM_FIB_P50
here too costs one more sweep and removes the confound -- see ``--feature``.

Usage:
    python run_total_mass.py --embed-type z --seed 42        # one cell, for timing
    python run_total_mass.py                                  # all 5 arms x 3 seeds
    python run_total_mass.py --feature LGM_FIB_P50            # fibre, same pyoperon
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from utils import fit_model_fn, fit_sym_fn
from val_rescore import build_merged

EMBED_TYPES = ["z", "cond", "cond+z", "uncond", "orig"]
SEEDS = [42, 43, 44]


def run_cell(
    feature: str, embed_type: str, seed: int, merged: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict]:
    """One (arm, seed) cell, mirroring train_logm.py's body."""
    started = time.time()
    results_sym = fit_sym_fn(feature, embed_type, merged, seed)
    model = results_sym["Model"]

    best_len, best_mse = None, float("inf")
    pareto = []
    for m in model.pareto_front_:
        tree = m["tree"]
        pred_test = model.evaluate_model(tree, results_sym["X_test"])
        pred_train = model.evaluate_model(tree, results_sym["X_train"])

        finite = np.isfinite(pred_test).all() and np.isfinite(pred_train).all()
        if finite:
            mse_test = mean_squared_error(results_sym["y_test"], pred_test)
            r2_test = r2_score(results_sym["y_test"], pred_test)
            mse_train = mean_squared_error(results_sym["y_train"], pred_train)
            r2_train = r2_score(results_sym["y_train"], pred_train)
        else:
            n_bad = int((~np.isfinite(pred_test)).sum())
            print(f"  [warn] length={tree.Length}: {n_bad} non-finite test preds")
            mse_test, r2_test = float("inf"), float("-inf")
            mse_train, r2_train = float("inf"), float("-inf")

        pareto.append(
            {
                "Feature": feature,
                "Embed_Type": embed_type,
                "Seed": seed,
                "Length": tree.Length,
                "Train_MSE": mse_train,
                "Train_R2": r2_train,
                "Test_MSE": mse_test,
                "Test_R2": r2_test,
                "Equation": m["model"],
                # Extra column, not in train_logm.py's schema: the same expression at
                # higher precision. It does NOT rescue the degenerate constant pairs --
                # pyoperon formats fixed-point, so a 1e-22 partner still prints as
                # zero -- but it costs nothing and narrows rounding on everything else.
                "Equation_p12": model.get_model_string(tree, precision=12),
            }
        )
        if mse_test < best_mse:
            best_mse, best_len = mse_test, tree.Length

    # Keyword, not positional: this sklearn's MLPRegressor takes `loss` first, so a
    # positional (64, 32) lands there and raises. train_logm.py:166 uses the keyword.
    mlp = fit_model_fn(
        feature,
        embed_type,
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=100, random_state=seed),
        merged,
    )
    linear = fit_model_fn(feature, embed_type, LinearRegression(), merged)

    summary = {
        "Feature": feature,
        "Embed_Type": embed_type,
        "Seed": seed,
        "MLP_Test_R2": mlp["Test_R2"],
        "MLP_Test_MSE": mlp["Test_MSE"],
        "LR_Test_R2": linear["Test_R2"],
        "LR_Test_MSE": linear["Test_MSE"],
        "SYM_Best_Test_MSE": best_mse,
    }
    elapsed = time.time() - started
    print(
        f"  {len(pareto)} front entries, best length {best_len}, "
        f"MLP {mlp['Test_R2']:.3f} / LR {linear['Test_R2']:.3f}"
        f"  [{elapsed / 60:.1f} min]"
    )
    return pd.DataFrame(pareto), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = os.environ.get("DATA_ROOT", "/home/birr0/local_data")
    parser.add_argument("--feature", default="lgm_tot_p50")
    parser.add_argument("--embed-type", default=None, choices=EMBED_TYPES)
    parser.add_argument("--seed", type=int, default=None, choices=SEEDS)
    parser.add_argument(
        "--embeddings",
        default=f"{data_root}/sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0",
    )
    parser.add_argument("--galspec", default=f"{data_root}/galSpecExtra-dr8.fits")
    parser.add_argument("--specgals-home", default=f"{data_root}/sdss")
    args = parser.parse_args()

    arms = [args.embed_type] if args.embed_type else EMBED_TYPES
    seeds = [args.seed] if args.seed else SEEDS

    print(f"target: {args.feature}")
    merged = build_merged(args.embeddings, args.galspec, args.specgals_home)

    for embed_type in arms:
        for seed in seeds:
            print(f"\n=== {args.feature} | {embed_type} | seed {seed} ===")
            front, summary = run_cell(args.feature, embed_type, seed, merged)

            job_dir = f"job_results_{args.feature}_{embed_type}_seed{seed}"
            os.makedirs(job_dir, exist_ok=True)
            front.to_csv(f"{job_dir}/pareto_fronts.csv", index=False)
            pd.DataFrame([summary]).to_csv(
                f"{job_dir}/summary_metrics.csv", index=False
            )
            print(f"  wrote {job_dir}/")


if __name__ == "__main__":
    main()
