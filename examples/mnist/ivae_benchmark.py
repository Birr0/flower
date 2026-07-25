"""iVAE nonlinear-ICA baseline on RGB-MNIST embeddings (issue #20 / E2).

Parallels ``examples/2d_gaussians/ivae_benchmark.py`` but operates on the
pretrained VAE embedding (the ``orig`` column of the flow embeddings — the same
representation Flower's flow consumes and the other residual baselines
residualise), with the **digit** as the discrete condition to remove.

For both a linear-ICA floor (FastICA) and nonlinear ICA (iVAE), two
condition-suppressed residuals are produced and evaluated with the same
bootstrap-CI metrics as ``benchmark.ipynb``:

- **residual A** — drop the source(s) most dependent on the digit;
- **residual B** — subtract the digit's mean (empirical per-digit mean for
  FastICA; the learned conditional-prior mean ``lambda_mu(u)`` for iVAE).

Metrics: **digit classification** (condition suppression; chance = 0.1) and
**colour ``b`` regression** (preservation — ``b`` is not conditioned on).

Requires ``DATA_ROOT`` (in ``.env``) pointing at the flow embeddings. Run from
this directory:

    python ivae_benchmark.py [--epochs 30] [--embed-subpath ...] [--drop-k 1]
"""

import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from flower.evaluation.ica import (
    conditional_mean_residual,
    conditional_prior_residual,
    drop_top_k_dependent,
)
from flower.evaluation.metrics import (
    evaluate_embedding_classifier,
    evaluate_embedding_regressor,
    prepare_data,
)
from flower.models.ivae import IVAE, LightningIVAE

RANDOM_STATE = 42
N_DIGITS = 10
DEFAULT_EMBED_SUBPATH = "rgbmnist/rgbmnist_Flow_cond_prior/embeddings/7518770_0"

# Same probe families as benchmark.ipynb.
clf_models = {
    "log_regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "2-mlp": MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=1000, random_state=RANDOM_STATE
    ),
}
reg_models = {
    "lin_regression": LinearRegression(),
    "2-mlp": MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=1000, random_state=RANDOM_STATE
    ),
}


def _one_hot(labels, n):
    return np.eye(n, dtype=np.float32)[labels]


def train_ivae(x, u, *, epochs, lr=1e-2, batch_size=256, seed=RANDOM_STATE):
    """Train the iVAE with the shipped ELBO (Adam + StepLR decay)."""
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


def encode_features(ivae, x, u):
    with torch.no_grad():
        out = ivae(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(u, dtype=torch.float32),
        )
    return out["mu"].numpy(), out["prior_mu"].numpy()


def fastica_residuals(x_tr, x_te, y_tr, y_te, *, drop_k, seed=RANDOM_STATE):
    """Linear-ICA floor: FastICA components, then residual A (drop top-k) and
    residual B (empirical conditional-mean removal)."""
    ica = FastICA(
        n_components=x_tr.shape[1],
        random_state=seed,
        max_iter=1000,
        whiten="unit-variance",
    )
    s_tr = ica.fit_transform(x_tr)
    s_te = ica.transform(x_te)

    resB_tr, means = conditional_mean_residual(s_tr, y_tr)
    resB_te, _ = conditional_mean_residual(s_te, y_te, means)

    _, dropped = drop_top_k_dependent(s_tr, y_tr, k=drop_k)
    keep = np.setdiff1d(np.arange(s_tr.shape[1]), dropped)
    return (s_tr[:, keep], s_te[:, keep]), (resB_tr, resB_te), dropped


def evaluate_feature_set(name, x_tr, x_te, y_clf_tr, y_clf_te, y_reg_tr, y_reg_te):
    """Digit classification + colour-b regression, with bootstrap CIs."""
    print(f"\n################ {name} ################")
    print("--- digit classification (condition suppression; chance = 0.1) ---")
    for probe, clf in clf_models.items():
        print(f"[{probe}]")
        evaluate_embedding_classifier(
            X_train=x_tr, y_train=y_clf_tr, X_test=x_te, y_test=y_clf_te, model=clf
        )
    print("--- colour b regression (residual preservation) ---")
    for probe, reg in reg_models.items():
        print(f"[{probe}]")
        evaluate_embedding_regressor(
            X_train=x_tr, y_train=y_reg_tr, X_test=x_te, y_test=y_reg_te, model=reg
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embed-subpath", type=str, default=DEFAULT_EMBED_SUBPATH)
    parser.add_argument("--drop-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    load_dotenv()
    data_root = os.getenv("DATA_ROOT")
    embed_path = f"{data_root}/{args.embed_subpath}"
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"{embed_path}/train/*.parquet",
            "test": f"{embed_path}/test/*.parquet",
        },
    )

    # X = the original VAE embedding (pre-flow); condition = digit; preserve = b.
    x_train, digit_train, x_test, digit_test = prepare_data(ds, "orig", "digit")
    _, b_train, _, b_test = prepare_data(ds, "orig", "b")
    digit_train = digit_train.astype(int)
    digit_test = digit_test.astype(int)

    # Scale embeddings (fit on train) so the fixed decoder variance is well-posed.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    u_train = _one_hot(digit_train, N_DIGITS)
    u_test = _one_hot(digit_test, N_DIGITS)

    print(f"Training iVAE ({args.epochs} epochs) on {x_train.shape} embeddings...")
    ivae = train_ivae(x_train, u_train, epochs=args.epochs, seed=args.seed)
    s_tr, pmu_tr = encode_features(ivae, x_train, u_train)
    s_te, pmu_te = encode_features(ivae, x_test, u_test)

    # iVAE residuals.
    ivae_resB_tr = conditional_prior_residual(s_tr, pmu_tr)
    ivae_resB_te = conditional_prior_residual(s_te, pmu_te)
    _, dropped = drop_top_k_dependent(s_tr, digit_train, k=args.drop_k)
    keep = np.setdiff1d(np.arange(s_tr.shape[1]), dropped)
    ivae_resA_tr, ivae_resA_te = s_tr[:, keep], s_te[:, keep]

    # FastICA (linear floor) residuals.
    (fica_resA_tr, fica_resA_te), (fica_resB_tr, fica_resB_te), fica_dropped = (
        fastica_residuals(
            x_train, x_test, digit_train, digit_test, drop_k=args.drop_k, seed=args.seed
        )
    )
    print(
        f"dropped source idx for residual-A: iVAE {dropped.tolist()}, "
        f"FastICA {fica_dropped.tolist()}"
    )

    evaluate_feature_set(
        "Raw embedding (no removal)",
        x_train,
        x_test,
        digit_train,
        digit_test,
        b_train,
        b_test,
    )
    evaluate_feature_set(
        "FastICA residual A (drop top-k)",
        fica_resA_tr,
        fica_resA_te,
        digit_train,
        digit_test,
        b_train,
        b_test,
    )
    evaluate_feature_set(
        "FastICA residual B (conditional-mean)",
        fica_resB_tr,
        fica_resB_te,
        digit_train,
        digit_test,
        b_train,
        b_test,
    )
    evaluate_feature_set(
        "iVAE residual A (drop top-k)",
        ivae_resA_tr,
        ivae_resA_te,
        digit_train,
        digit_test,
        b_train,
        b_test,
    )
    evaluate_feature_set(
        "iVAE residual B (conditional-prior)",
        ivae_resB_tr,
        ivae_resB_te,
        digit_train,
        digit_test,
        b_train,
        b_test,
    )


if __name__ == "__main__":
    main()
