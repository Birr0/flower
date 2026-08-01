"""Fit an ``orig+z`` arm — the missing control for the separability test (TODO item 3c).

``separability.py`` finds that **58 of 58** scored ``cond+z`` front equations are
additively separable in latent and redshift (median binned R² of z -> delta = 0.0046
against a 0.0009 permutation null). A result pinned at 100% carries no information on
its own: it could mean ``cond`` is factorised, or simply that this search on this target
produces separable equations whatever the latent is. The only way to tell is to run the
identical measurement on a representation where factorisation should **not** hold.

That arm does not exist. ``utils.fit_sym_fn`` branches on ``cond+z`` and ``z`` only; the
else-branch builds the design matrix from the embedding alone, so ``orig`` has never
carried a redshift feature.

**How this adds one without touching the published code path.** ``fit_sym_fn``'s
``cond+z`` branch reads the column *named* ``cond`` (``utils.py:39-40``), and
``val_rescore.build_matrices`` does the same (``base = "cond"`` when the arm is
``cond+z``). So overwriting that column with ``orig`` and asking for ``cond+z`` runs the
published code path verbatim on the unconditioned latent: same StandardScaler fit on the
same rows, same raw redshift appended after scaling, same 10,000-row subsample per seed,
same operator set and length budget. Nothing in ``utils.py`` is edited, so the published
fronts keep their provenance, and the control differs from the treatment in exactly one
respect -- which latent is in the matrix.

Output goes to ``job_results_origz_{FEATURE}_cond+z_seed{SEED}/``. Score it with
``python separability.py --feature origz_LGM_FIB_P50 --target LGM_FIB_P50
--swap-cond-for-orig``, which applies the same swap when rebuilding the design matrix.

**Reading the result.** If ``orig+z`` is also ~100% separable, the statistic measures
the regressor and item 3(c) cannot support the factorisation claim -- say so and drop
it. If ``orig+z`` is visibly mixed, the contrast is the claim demonstrated in the
paper's own idiom. Anything in between needs the per-equation distributions compared,
not just the headline fraction.

Usage:
    python run_orig_z.py                       # 3 seeds, LGM_FIB_P50, full sample
    python run_orig_z.py --seed 42             # one cell, for timing
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from run_total_mass import SEEDS, run_cell
from val_rescore import build_merged


def swap_cond_for_orig(
    merged: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Overwrite the ``cond`` column with ``orig``, per split.

    Copies rather than mutating in place so a caller holding the original frames is not
    silently affected.
    """
    out = {}
    for split, df in merged.items():
        frame = df.copy()
        if "orig" not in frame or "cond" not in frame:
            message = (
                f"split {split!r} lacks orig/cond columns; cannot build the control"
            )
            raise SystemExit(message)
        frame["cond"] = frame["orig"]
        out[split] = frame
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = os.environ.get("DATA_ROOT", "/home/birr0/local_data")
    parser.add_argument("--feature", default="LGM_FIB_P50")
    parser.add_argument("--seed", type=int, default=None, choices=SEEDS)
    parser.add_argument(
        "--embeddings",
        default=f"{data_root}/sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0",
    )
    parser.add_argument(
        "--cut",
        default=None,
        help="use the volume-limited embeddings at this cut instead of the full sample",
    )
    parser.add_argument("--galspec", default=f"{data_root}/galSpecExtra-dr8.fits")
    parser.add_argument("--specgals-home", default=f"{data_root}/sdss")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed else SEEDS

    root = args.embeddings
    tag = ""
    if args.cut:
        root = f"{data_root}/vol_limited_embeddings_7655991_0_allsplits/z={args.cut}"
        tag = f"vollim_z{args.cut}_"

    sample = f"volume-limited z <= {args.cut}" if args.cut else "full sample"
    print(f"target: {args.feature} | {sample} | control arm: orig+z (cond <- orig)")
    merged = build_merged(root, args.galspec, args.specgals_home)
    merged = swap_cond_for_orig(merged)

    for seed in seeds:
        print(f"\n=== {tag}{args.feature} | orig+z | seed {seed} ===")
        front, summary = run_cell(args.feature, "cond+z", seed, merged)
        # Relabel so nothing downstream mistakes these for real cond+z fronts.
        front["Embed_Type"] = "orig+z"

        job_dir = f"job_results_origz_{tag}{args.feature}_cond+z_seed{seed}"
        os.makedirs(job_dir, exist_ok=True)
        front.to_csv(f"{job_dir}/pareto_fronts.csv", index=False)
        pd.DataFrame([{**summary, "Embed_Type": "orig+z"}]).to_csv(
            f"{job_dir}/summary_metrics.csv", index=False
        )
        print(f"  wrote {job_dir}/")


if __name__ == "__main__":
    main()
