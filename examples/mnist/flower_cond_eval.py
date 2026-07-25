"""Evaluate Flower's conditioned embedding on the three axes (issue #20 / E2).

Directly probes the pretrained flow embeddings — no training — on:
- **digit** classification (the condition; want removed, chance = 0.10),
- **colour b** regression (independent factor; want preserved),
- **rotation** regression (entangled-with-digit factor; want preserved),

for each embedding column: ``orig`` (the raw VAE latent = "Raw") and ``cond``
(Flower's condition-suppressed seed). This is the surgical counterpart to the
FastICA source-dropping sweep: can the conditional flow remove the digit while
keeping BOTH colour and rotation, where source-dropping had to sacrifice rotation?

Uses the same probe families and the cached rotation targets from
``compute_rotation.py``. Run from this directory (needs ``DATA_ROOT``):

    python flower_cond_eval.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from flower.evaluation.metrics import prepare_data

RANDOM_STATE = 42
DEFAULT_EMBED_SUBPATH = "rgbmnist/rgbmnist_Flow_cond_prior/embeddings/7518770_0"


def _digit_acc(x_tr, x_te, y_tr, y_te, kind):
    if kind == "logreg":
        clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE
        )
    clf.fit(x_tr, y_tr)
    return accuracy_score(y_te, clf.predict(x_te))


def _r2(x_tr, x_te, y_tr, y_te, kind):
    if kind == "linreg":
        reg = LinearRegression()
    else:
        reg = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE
        )
    reg.fit(x_tr, y_tr)
    return r2_score(y_te, reg.predict(x_te))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-subpath", type=str, default=DEFAULT_EMBED_SUBPATH)
    parser.add_argument("--embed-types", type=str, default="orig,cond")
    parser.add_argument("--rotation-dir", type=str, default=".")
    parser.add_argument("--outdir", type=str, default="flower_cond_results")
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
    _, dig_tr, _, dig_te = prepare_data(ds, "orig", "digit")
    _, b_tr, _, b_te = prepare_data(ds, "orig", "b")
    dig_tr, dig_te = dig_tr.astype(int), dig_te.astype(int)
    rot_tr = pd.read_csv(os.path.join(args.rotation_dir, "train_rotation_aligned.csv"))[
        "Rotation_Deg"
    ].to_numpy()
    rot_te = pd.read_csv(os.path.join(args.rotation_dir, "test_rotation_aligned.csv"))[
        "Rotation_Deg"
    ].to_numpy()

    label = {"orig": "Raw (orig)", "cond": "Flower cond", "uncond": "uncond"}
    rows = []
    for et in [e.strip() for e in args.embed_types.split(",") if e.strip()]:
        x_tr, _, x_te, _ = prepare_data(ds, et, "digit")
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr).astype(np.float32)
        x_te = scaler.transform(x_te).astype(np.float32)
        print(f"Evaluating {et} ({x_tr.shape})...")
        rows.append(
            {
                "embedding": label.get(et, et),
                "digit_acc_logreg": _digit_acc(x_tr, x_te, dig_tr, dig_te, "logreg"),
                "digit_acc_mlp": _digit_acc(x_tr, x_te, dig_tr, dig_te, "mlp"),
                "b_r2_linreg": _r2(x_tr, x_te, b_tr, b_te, "linreg"),
                "b_r2_mlp": _r2(x_tr, x_te, b_tr, b_te, "mlp"),
                "rot_r2_linreg": _r2(x_tr, x_te, rot_tr, rot_te, "linreg"),
                "rot_r2_mlp": _r2(x_tr, x_te, rot_tr, rot_te, "mlp"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "results.csv"), index=False)
    header = (
        "Flower cond vs Raw on RGB-MNIST (class-only; condition = digit)\n"
        "digit acc: lower = better removal (chance 0.10) | "
        "b / rotation R2: higher = better preservation\n"
    )
    table = df.to_string(index=False, float_format=lambda v: f"{v:.3f}")
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(header + "\n" + table + "\n")
    print("\n" + header + "\n" + table)
    print(f"\nSaved results.csv, summary.txt in {args.outdir}/")


if __name__ == "__main__":
    main()
