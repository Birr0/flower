"""Run the symbolic regression on a volume-limited cut (TODO item 4).

The paper asserts the recovered mass-redshift relation "traces the selection function"
and currently offers no evidence: Figure 2 shows the relation exists, not that it is
selection. Volume-limiting removes the flux limit that couples mass to redshift, so if
the relation is selection, the X=0 reduction of the ``cond+z`` fronts should flatten or
change shape relative to Figure 2b. If it survives unchanged, the claim is wrong and the
paper's lead has to move.

Thin wrapper: ``run_cell`` is imported verbatim from ``run_total_mass``, which in turn
uses ``utils.fit_sym_fn``/``fit_model_fn`` unchanged, so the search, the 10,000-row
subsample (``utils.N``, drawn per seed), the scaling and the MLP/LR references are
identical to the published sweep. **Only the embeddings root differs.** Every cut's
train split exceeds 10,000 rows after the catalogue merge, so the subsample size is the
same here as on the full sample -- the comparison is like-for-like on n.

Reads ``$DATA_ROOT/vol_limited_embeddings_7655991_0_allsplits/z={cut}``, built by
``examples/spectra/make_volume_limited.py``, which reproduces the stored volume-limited
test rows exactly (recall 1.0000, precision 1.0000 at all five cuts) while adding the
train and val splits the SR needs.

Output goes to ``job_results_vollim_z{cut}_{FEATURE}_{EMBED}_seed{SEED}/`` -- prefixed
so it cannot be confused with the full-sample fronts, and in ``train_logm.py``'s schema
so ``variable_additions.py``, ``latent_zero_limit.py`` and ``val_rescore.py`` read it
unchanged.

**Same pyoperon caveat as ``run_total_mass.py``**: 0.4.0 from PyPI here, an unrecorded
build on the cluster for the published fronts. The volume-limited vs full-sample
comparison should therefore be made against the ``lgm_tot_p50`` runs produced by
``run_total_mass.py`` on this same pyoperon, not against the published LGM_FIB_P50
fronts.

Usage:
    python run_volume_limited.py --embed-type cond+z --seed 42   # one cell, for timing
    python run_volume_limited.py                                  # all 5 arms x 3 seeds
    python run_volume_limited.py --cut 0.100                      # a different cut
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from run_total_mass import EMBED_TYPES, SEEDS, run_cell
from val_rescore import build_merged

Z_CUTS = ["0.050", "0.075", "0.100", "0.125", "0.150"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = os.environ.get("DATA_ROOT", "/home/birr0/local_data")
    parser.add_argument("--feature", default="lgm_tot_p50")
    parser.add_argument("--cut", default="0.150", choices=Z_CUTS)
    parser.add_argument("--embed-type", default=None, choices=EMBED_TYPES)
    parser.add_argument("--seed", type=int, default=None, choices=SEEDS)
    parser.add_argument(
        "--vol-root",
        default=f"{data_root}/vol_limited_embeddings_7655991_0_allsplits",
        help="output of make_volume_limited.py, one subdir per cut",
    )
    parser.add_argument("--galspec", default=f"{data_root}/galSpecExtra-dr8.fits")
    parser.add_argument("--specgals-home", default=f"{data_root}/sdss")
    args = parser.parse_args()

    arms = [args.embed_type] if args.embed_type else EMBED_TYPES
    seeds = [args.seed] if args.seed else SEEDS

    root = f"{args.vol_root}/z={args.cut}"
    if not os.path.isdir(root):
        message = (
            f"no volume-limited embeddings at {root!r}; "
            "run examples/spectra/make_volume_limited.py first"
        )
        raise SystemExit(message)

    print(f"target: {args.feature} | volume-limited z <= {args.cut}")
    merged = build_merged(root, args.galspec, args.specgals_home)

    for embed_type in arms:
        for seed in seeds:
            print(
                f"\n=== vollim z<={args.cut} | {args.feature} | "
                f"{embed_type} | seed {seed} ==="
            )
            front, summary = run_cell(args.feature, embed_type, seed, merged)
            front["Z_Cut"] = float(args.cut)

            job_dir = (
                f"job_results_vollim_z{args.cut}_{args.feature}_{embed_type}_seed{seed}"
            )
            os.makedirs(job_dir, exist_ok=True)
            front.to_csv(f"{job_dir}/pareto_fronts.csv", index=False)
            pd.DataFrame([{**summary, "Z_Cut": float(args.cut)}]).to_csv(
                f"{job_dir}/summary_metrics.csv", index=False
            )
            print(f"  wrote {job_dir}/")


if __name__ == "__main__":
    main()
