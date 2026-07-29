"""Benchmark one or more Flower spectra embedding runs side by side.

Written for **E1, the frozen-base ablation** (R2 Q1 / A1, R3 A4): a model trained
with the base fixed to N(0,I) — no learnable conditional base, and therefore no
KL term at all. This is *not* the β=0 ablation (Fig 7/8), which keeps the
learnable conditional base and only switches the KL off.

``flower_benchmark.py`` evaluates a single hard-coded run, one split per
invocation. This script runs the *same* protocol (same targets, scaling, probe
architectures and bootstrap) over any number of embedding directories and all
splits at once, so the frozen-base run can be compared directly against full
Flower, or one seed against another.

Label every run so the ablation is unambiguous in ``results.csv`` /
``summary.txt`` — e.g. ``--run frozen_base=<path>``: the ``Run`` column and the
output directory name are the only record of which model produced which
embeddings.

Probes: ``MLPRegressor`` (1- and 2-layer) on standard-scaled embeddings,
R² on the test split with a 1000-sample bootstrap CI. For the condition
``z`` (redshift) LOWER R² = better removal; for ``logM*``/``logSFR``/``A_v``
HIGHER R² = better preservation.

Targets come from the HF catalog ``Birr001/spectra_catalog``, index-aligned to
the embeddings. That alignment holds in ``datasets.load_dataset`` glob order —
do not pre-concatenate the parquet shards in numeric order, which is a different
ordering.

Run from this directory (needs ``DATA_ROOT``):

    EMB=sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings
    python embedding_benchmark.py \
        --run flower=$EMB/7526202_0 \
        --run frozen_base=$EMB/7655991_0 \
        --outdir flower_vs_frozen_base_results

Add ``--matched --factor-bound`` for the numbers cited in the rebuttal: it puts
the ``z`` residual and the physical-target R² on identical rows and adds the
correlation bound (``z`` predicted from the preserved factors alone). Each run
writes ``results.csv``, ``summary.txt`` and ``params.json`` (the full invocation)
into ``--outdir``. See ``review/neurips/residual_floor_analysis.md`` §Provenance
for the exact commands behind the cited tables.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from flower.evaluation.metrics import bootstrap_summary, print_bootstrap_stats

load_dotenv()

# --- Configuration (kept identical to flower_benchmark.py) ---
N_BOOT = 1000
N_FILTER = 300000
N_TRAIN = 200000
# Runs have different train-split lengths (7526202_0: 329,216 rows;
# 7655991_*: 422,481), and 7526202_0's last 38 shards are misordered relative to
# the catalog. Rows 0..309,759 are catalog-aligned in every run, so any
# --n-filter at or below that is comparable across runs; above it, 7526202_0's
# targets are scrambled (and it has no id column to rejoin on).
MAX_ALIGNED_TRAIN = 309760
TARGET_ATTRIBUTES = ["z", "logM*", "logSFR", "A_v"]
PHYSICAL_TARGETS = ["logM*", "logSFR", "A_v"]
ARCHITECTURES = {"1-Layer": (64,), "2-Layer": (64, 64)}
RANDOM_STATE = 42


def _clean(v):
    return pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=float)


def _valid(v):
    return np.isfinite(v) & (v != -99.0)


def load_run(embed_path, splits, n_filter=N_FILTER, n_train=N_TRAIN, matched=False):
    """Load embeddings + index-aligned catalog targets for one run.

    The condition ``z`` is read from the embeddings themselves (older runs name
    the column ``Z``, newer ones ``z``) — it is row-aligned by construction. The
    physical targets come from the catalog, which carries no plain ``z`` column.

    ``n_filter`` truncates the train split *before* masking, so passing the same
    value for every run gives each the same rows — the only way the arms are
    comparable when the runs have different train lengths.

    ``matched`` additionally requires every physical target to be valid, so the
    ``z`` probe and the physical-target probes run on *identical* rows. Without
    it, z uses ~52.8k test rows and the physical targets ~43.1k, and the two
    cannot be quoted against each other.
    """
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"{embed_path}/train/*.parquet",
            "test": f"{embed_path}/test/*.parquet",
        },
    )
    cat = load_dataset("Birr001/spectra_catalog")
    z_col = "z" if "z" in ds["train"].column_names else "Z"

    z_train = _clean(ds["train"][z_col])[:n_filter]
    z_test = _clean(ds["test"][z_col])

    phys_train = np.ones_like(z_train, dtype=bool)
    phys_test = np.ones_like(z_test, dtype=bool)
    if matched:
        for t in PHYSICAL_TARGETS:
            phys_train &= _valid(_clean(cat["train"][t])[:n_filter])
            phys_test &= _valid(_clean(cat["test"][t]))

    out = {}
    for split in splits:
        x_train_raw = np.array(ds["train"][split], dtype=float)[:n_filter]
        x_test_raw = np.array(ds["test"][split], dtype=float)
        # Redshift is the alignment key used to mask out invalid rows.
        mask_train = _valid(z_train) & np.isfinite(x_train_raw).all(axis=1) & phys_train
        mask_test = _valid(z_test) & np.isfinite(x_test_raw).all(axis=1) & phys_test
        out[split] = (
            x_train_raw[mask_train][:n_train],
            x_test_raw[mask_test],
            mask_train,
            mask_test,
        )
    return cat, (z_train, z_test), out


def evaluate_split(
    label, split, x_train, x_test, cat, z, mask_train, mask_test, n_filter, n_train
):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    z_train, z_test = z

    rows = []
    for attr in TARGET_ATTRIBUTES:
        if attr == "z":
            y_tr = z_train[mask_train][:n_train]
            y_te = z_test[mask_test]
        else:
            y_tr = _clean(cat["train"][attr])[:n_filter][mask_train][:n_train]
            y_te = _clean(cat["test"][attr])[mask_test]
        m_tr = np.isfinite(y_tr) & (y_tr != -99.0)
        m_te = np.isfinite(y_te) & (y_te != -99.0)

        for arch_name, layers in ARCHITECTURES.items():
            print(f"  [{label}/{split}] {attr} — {arch_name}")
            reg = MLPRegressor(
                hidden_layer_sizes=layers, max_iter=1000, random_state=RANDOM_STATE
            )
            reg.fit(x_train_scaled[m_tr], y_tr[m_tr])

            y_true = y_te[m_te]
            y_true = y_true.values if hasattr(y_true, "values") else y_true
            y_pred = reg.predict(x_test_scaled[m_te])

            boot = [
                r2_score(*resample(y_true, y_pred, random_state=i))
                for i in range(N_BOOT)
            ]
            stats = bootstrap_summary(boot)
            print_bootstrap_stats(f"{label}_{split}_{attr}_{arch_name}", stats)

            rows.append(
                {
                    "Run": label,
                    "Split": split,
                    "Attribute": attr,
                    "Layers": arch_name,
                    "N_train": int(m_tr.sum()),
                    "N_test": int(m_te.sum()),
                    "R2_Mean": round(stats["mean"], 4),
                    "R2_Median": round(stats["median"], 4),
                    "CI_95_Low": round(stats["ci_95"][0], 4),
                    "CI_95_High": round(stats["ci_95"][1], 4),
                    "Err_95": round(stats["err_95"], 4),
                }
            )
    return rows


def factor_bound(cat, z, mask_train, mask_test, n_filter, n_train):
    """How well ``z`` is predictable from the *preserved physical factors alone*.

    This is the reference the seed's residual has to be read against: because
    logM* correlates ~0.76 with z, an embedding that retained the physical
    factors intact would leak z at this level. A residual *below* this bound
    means the flow removed more of the condition than the factor correlations
    alone would imply.
    """
    z_train, z_test = z
    x_tr = np.column_stack(
        [
            _clean(cat["train"][t])[:n_filter][mask_train][:n_train]
            for t in PHYSICAL_TARGETS
        ]
    )
    x_te = np.column_stack(
        [_clean(cat["test"][t])[mask_test] for t in PHYSICAL_TARGETS]
    )
    y_tr = z_train[mask_train][:n_train]
    y_te = z_test[mask_test]

    sc = StandardScaler().fit(x_tr)
    x_tr, x_te = sc.transform(x_tr), sc.transform(x_te)

    rows = []
    for arch_name, layers in ARCHITECTURES.items():
        print(f"  [factor_bound] z from {PHYSICAL_TARGETS} — {arch_name}")
        reg = MLPRegressor(
            hidden_layer_sizes=layers, max_iter=1000, random_state=RANDOM_STATE
        )
        reg.fit(x_tr, y_tr)
        boot = [
            r2_score(*resample(y_te, reg.predict(x_te), random_state=i))
            for i in range(N_BOOT)
        ]
        stats = bootstrap_summary(boot)
        print_bootstrap_stats(f"factor_bound_z_{arch_name}", stats)
        rows.append(
            {
                "Run": "factor_bound",
                "Split": "physical_factors",
                "Attribute": "z",
                "Layers": arch_name,
                "N_train": len(y_tr),
                "N_test": len(y_te),
                "R2_Mean": round(stats["mean"], 4),
                "R2_Median": round(stats["median"], 4),
                "CI_95_Low": round(stats["ci_95"][0], 4),
                "CI_95_High": round(stats["ci_95"][1], 4),
                "Err_95": round(stats["err_95"], 4),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="run to evaluate; PATH is relative to DATA_ROOT. Repeatable.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["orig", "cond", "uncond"],
        help="embedding columns to evaluate",
    )
    parser.add_argument(
        "--n-filter",
        type=int,
        default=N_FILTER,
        help=(
            "truncate each run's train split to this many rows before masking, "
            f"so every run gets the same rows (max aligned: {MAX_ALIGNED_TRAIN})"
        ),
    )
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument(
        "--matched",
        action="store_true",
        help=(
            "restrict every probe to rows where all physical targets are valid, so "
            "the z residual and the physical-target R² are on identical rows"
        ),
    )
    parser.add_argument(
        "--factor-bound",
        action="store_true",
        help="also predict z from the physical factors alone (the leakage reference)",
    )
    parser.add_argument("--outdir", default="frozen_base_benchmark_results")
    args = parser.parse_args()

    if args.factor_bound and not args.matched:
        parser.error("--factor-bound requires --matched, or the rows won't line up")

    if args.n_filter > MAX_ALIGNED_TRAIN:
        parser.error(
            f"--n-filter {args.n_filter} exceeds {MAX_ALIGNED_TRAIN}, beyond which "
            "run 7526202_0's train rows are misordered relative to the catalog"
        )

    data_root = os.getenv("DATA_ROOT")
    runs = dict(spec.split("=", 1) for spec in args.run)
    os.makedirs(args.outdir, exist_ok=True)

    # Record how these numbers were produced, next to the numbers themselves —
    # the results are cited in the rebuttal and must stay reproducible.
    with open(f"{args.outdir}/params.json", "w") as fh:
        json.dump(
            {
                "command": " ".join(sys.argv),
                "args": vars(args),
                "runs": runs,
                "n_boot": N_BOOT,
                "random_state": RANDOM_STATE,
                "architectures": {k: list(v) for k, v in ARCHITECTURES.items()},
                "targets": TARGET_ATTRIBUTES,
                "physical_targets": PHYSICAL_TARGETS,
                "max_aligned_train": MAX_ALIGNED_TRAIN,
            },
            fh,
            indent=2,
        )

    rows = []
    for label, rel in runs.items():
        print(f"\n=== {label}: {rel} ===")
        cat, z, loaded = load_run(
            f"{data_root}/{rel}",
            args.splits,
            args.n_filter,
            args.n_train,
            args.matched,
        )
        for split, (x_tr, x_te, m_tr, m_te) in loaded.items():
            print(f"{label}/{split}: train {x_tr.shape}, test {x_te.shape}")
            rows += evaluate_split(
                label,
                split,
                x_tr,
                x_te,
                cat,
                z,
                m_tr,
                m_te,
                args.n_filter,
                args.n_train,
            )
        if args.factor_bound and not any(r["Run"] == "factor_bound" for r in rows):
            # Depends only on the catalog and the row mask, so compute it once.
            m_tr, m_te = next(iter(loaded.values()))[2:]
            rows += factor_bound(cat, z, m_tr, m_te, args.n_filter, args.n_train)

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.outdir}/results.csv", index=False)

    pivot = df.pivot_table(
        index=["Attribute", "Layers"], columns=["Run", "Split"], values="R2_Mean"
    )
    header = (
        "Spectra embedding benchmark — bootstrap-mean test R²\n"
        "z: LOWER = better removal | logM*/logSFR/A_v: HIGHER = better preservation\n"
        + "\n".join(f"  {label}: {rel}" for label, rel in runs.items())
    )
    summary = header + "\n\n" + pivot.to_string(float_format="%.4f") + "\n"
    with open(f"{args.outdir}/summary.txt", "w") as fh:
        fh.write(summary)
    print("\n" + summary)
    print(f"Saved to {args.outdir}/")


if __name__ == "__main__":
    main()
