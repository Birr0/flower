"""iVAE nonlinear-ICA baseline on the 2D-Gaussians toy (issue #20 / E2).

Trains an identifiable VAE (Khemakhem et al. 2020; see ``flower.models.ivae``)
conditioned on the GMM mode ``u = one-hot(y)``, then evaluates two
condition-suppressed residual representations against the *same* metrics and
bootstrap CIs as ``benchmark.ipynb``:

- **mode classification** — how much of the condition survives (should drop
  towards chance = 0.25);
- **radial-distance regression** — how much condition-independent structure is
  preserved (the distance from the cluster centre; should stay high).

Two residual constructions (both are reported):

- **residual B** — the conditional-prior residual ``s - lambda_mu(u)``, the
  natural choice here since the mode is entirely in the per-cluster mean;
- **residual A** — drop the source(s) most dependent on the mode.

CIs come from ``evaluate_embedding_{classifier,regressor}`` (1000-sample
bootstrap), identical to the existing Gaussian benchmark.

Run from this directory (``examples/2d_gaussians``):

    python ivae_benchmark.py [--epochs 30] [--n-train 20000] [--n-test 5000]
"""

import argparse

import numpy as np
import torch
from data import generate_quad_gmm
from sklearn.decomposition import FastICA
from sklearn.neural_network import MLPClassifier, MLPRegressor

from flower.evaluation.ica import (
    compute_mcc,
    conditional_mean_residual,
    conditional_prior_residual,
    drop_top_k_dependent,
)
from flower.evaluation.metrics import (
    evaluate_embedding_classifier,
    evaluate_embedding_regressor,
)
from flower.models.ivae import IVAE, LightningIVAE

RANDOM_STATE = 42
N_MODES = 4

# Same probe families as benchmark.ipynb.
clf_models = {
    "1-mlp": MLPClassifier(
        hidden_layer_sizes=(64,), max_iter=1000, random_state=RANDOM_STATE
    ),
    "2-mlp": MLPClassifier(
        hidden_layer_sizes=(64, 64), max_iter=1000, random_state=RANDOM_STATE
    ),
}
reg_models = {
    "1-mlp": MLPRegressor(
        hidden_layer_sizes=(64,), max_iter=1000, random_state=RANDOM_STATE
    ),
    "2-mlp": MLPRegressor(
        hidden_layer_sizes=(64, 64), max_iter=1000, random_state=RANDOM_STATE
    ),
}


def _to_numpy(*tensors):
    return [t.detach().cpu().numpy() for t in tensors]


def _one_hot(labels, n):
    return np.eye(n, dtype=np.float32)[labels]


def train_ivae(x, u, *, epochs, lr=1e-2, batch_size=64, seed=RANDOM_STATE):
    """Train the iVAE with the shipped ELBO (Adam + StepLR decay)."""
    torch.manual_seed(seed)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    u_t = torch.as_tensor(u, dtype=torch.float32)
    ivae = IVAE(
        data_dim=x.shape[1],
        aux_dim=u.shape[1],
        latent_dim=x.shape[1],
        hidden_dim=64,
        n_layers=3,
        activation="xtanh",
        learn_prior_mean=True,  # the mode lives in the mean
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
    """Return the posterior-mean sources and the conditional-prior mean."""
    with torch.no_grad():
        out = ivae(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(u, dtype=torch.float32),
        )
    return out["mu"].numpy(), out["prior_mu"].numpy()


def fastica_residuals(x_tr, x_te, y_tr, y_te, *, seed=RANDOM_STATE):
    """Linear-ICA floor: FastICA independent components, then the same two
    condition-suppressed residuals as the iVAE — residual A (drop the top-k
    mode-dependent components) and residual B (empirical conditional-mean
    removal, the linear analogue of subtracting lambda_mu(u)).
    """
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

    _, dropped = drop_top_k_dependent(s_tr, y_tr, k=1)
    keep = np.setdiff1d(np.arange(s_tr.shape[1]), dropped)
    return (s_tr[:, keep], s_te[:, keep]), (resB_tr, resB_te), dropped


def evaluate_feature_set(name, x_tr, x_te, y_tr, y_te, yr_tr, yr_te):
    """Mode classification + radial-distance regression, with bootstrap CIs."""
    print(f"\n################ {name} ################")
    print("--- mode classification (condition suppression; chance = 0.25) ---")
    for probe, clf in clf_models.items():
        print(f"[{probe}]")
        evaluate_embedding_classifier(
            X_train=x_tr, y_train=y_tr, X_test=x_te, y_test=y_te, model=clf
        )
    print("--- radial-distance regression (residual preservation) ---")
    for probe, reg in reg_models.items():
        print(f"[{probe}]")
        evaluate_embedding_regressor(
            X_train=x_tr, y_train=yr_tr, X_test=x_te, y_test=yr_te, model=reg
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    # generate_quad_gmm -> samples, mode, radial-distance, diff(=seed)
    x_tr, y_tr, yr_tr, diff_tr = _to_numpy(*generate_quad_gmm(args.n_train))
    x_te, y_te, yr_te, diff_te = _to_numpy(*generate_quad_gmm(args.n_test))
    y_tr, y_te = y_tr.astype(int), y_te.astype(int)
    u_tr, u_te = _one_hot(y_tr, N_MODES), _one_hot(y_te, N_MODES)

    print(f"Training iVAE ({args.epochs} epochs) on {len(x_tr)} samples...")
    ivae = train_ivae(x_tr, u_tr, epochs=args.epochs, seed=args.seed)

    s_tr, pmu_tr = encode_features(ivae, x_tr, u_tr)
    s_te, pmu_te = encode_features(ivae, x_te, u_te)

    # Residual B: conditional-prior residual s - lambda_mu(u).
    resB_tr = conditional_prior_residual(s_tr, pmu_tr)
    resB_te = conditional_prior_residual(s_te, pmu_te)
    # Residual A: drop the source(s) most mode-dependent (rank on train, apply to both).
    _, dropped = drop_top_k_dependent(s_tr, y_tr, k=1)
    keep = np.setdiff1d(np.arange(s_tr.shape[1]), dropped)
    resA_tr, resA_te = s_tr[:, keep], s_te[:, keep]

    # Linear-ICA floor (FastICA), same two residual constructions.
    (fica_resA_tr, fica_resA_te), (fica_resB_tr, fica_resB_te), fica_dropped = (
        fastica_residuals(x_tr, x_te, y_tr, y_te, seed=args.seed)
    )

    print(
        f"\nSeed recovery MCC(diff, residual-B): "
        f"iVAE train={compute_mcc(diff_tr, resB_tr):.3f} "
        f"test={compute_mcc(diff_te, resB_te):.3f}  |  "
        f"FastICA train={compute_mcc(diff_tr, fica_resB_tr):.3f} "
        f"test={compute_mcc(diff_te, fica_resB_te):.3f}"
    )
    print(
        f"(dropped source idx for residual-A: iVAE {dropped.tolist()}, "
        f"FastICA {fica_dropped.tolist()})"
    )

    evaluate_feature_set("Raw X (no removal)", x_tr, x_te, y_tr, y_te, yr_tr, yr_te)
    # Linear ICA floor
    evaluate_feature_set(
        "FastICA residual A (drop top-k)",
        fica_resA_tr,
        fica_resA_te,
        y_tr,
        y_te,
        yr_tr,
        yr_te,
    )
    evaluate_feature_set(
        "FastICA residual B (conditional-mean)",
        fica_resB_tr,
        fica_resB_te,
        y_tr,
        y_te,
        yr_tr,
        yr_te,
    )
    # Nonlinear ICA (iVAE)
    evaluate_feature_set(
        "iVAE residual A (drop top-k)", resA_tr, resA_te, y_tr, y_te, yr_tr, yr_te
    )
    evaluate_feature_set(
        "iVAE residual B (conditional-prior)",
        resB_tr,
        resB_te,
        y_tr,
        y_te,
        yr_tr,
        yr_te,
    )


if __name__ == "__main__":
    main()
