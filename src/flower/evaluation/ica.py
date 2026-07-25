"""Nonlinear-ICA evaluation utilities: MCC and a synthetic benchmark.

Supports the iVAE baseline (issue #20 / E2). Ported to follow the reference
implementation https://github.com/ilkhem/iVAE (Khemakhem et al., AISTATS 2020):

- ``compute_mcc`` reproduces ``metrics.mean_corr_coef_np`` — the absolute Pearson
  correlation matrix between true and recovered sources, optimally matched with
  the Hungarian algorithm, then averaged. Invariant to permutation and
  sign/scale (the ICA equivalence class).
- ``sample_synthetic_nonlinear_ica`` reproduces ``data.data.generate_data`` (the
  ``repeat_linearity=True`` path used by their ``SyntheticDataset``): TCL-style
  segment-modulated sources, mixed by a random invertible MLP with **xtanh**
  nonlinearities and uniform, condition-controlled mixing matrices. This is the
  regime where a *linear* method (FastICA) cannot invert the mixing but a
  nonlinear model with access to the auxiliary variable ``u`` can.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


def compute_mcc(
    true: np.ndarray, recovered: np.ndarray, method: str = "pearson"
) -> float:
    """Mean Correlation Coefficient between true and recovered sources.

    Reproduces ``mean_corr_coef_np`` from the reference iVAE repo: build the
    absolute correlation matrix between the columns of ``true`` and
    ``recovered``, find the optimal one-to-one matching (Hungarian algorithm),
    and return the mean matched absolute correlation. 1.0 = perfect recovery up
    to permutation/sign/scale; ~0 = unrelated.

    Args:
        true: ``(n_samples, d_true)`` ground-truth sources.
        recovered: ``(n_samples, d_rec)`` estimated sources.
        method: ``"pearson"`` (default) or ``"spearman"``.
    """
    true = np.asarray(true, dtype=float)
    recovered = np.asarray(recovered, dtype=float)
    d = true.shape[1]

    if method == "pearson":
        cc = np.corrcoef(true, recovered, rowvar=False)[:d, d:]
    elif method == "spearman":
        cc = spearmanr(true, recovered)[0][:d, d:]
    else:
        msg = f"not a valid method: {method}"
        raise ValueError(msg)

    cc = np.abs(np.nan_to_num(cc))  # nan_to_num guards constant columns
    row, col = linear_sum_assignment(-cc)
    return float(cc[row, col].mean())


# ---------------------------------------------------------------------------
# Residual constructions for the iVAE baseline
# ---------------------------------------------------------------------------
def conditional_prior_residual(sources: np.ndarray, prior_mu: np.ndarray) -> np.ndarray:
    """Method B: the conditional-prior residual ``s - lambda_mu(u)``.

    Subtracts each source's ``u``-conditioned prior mean, removing the part of
    the representation explained by the condition and leaving the "innovation".
    Natural when the condition acts through the mean (e.g. the 2D-Gaussians toy,
    where the mode is the per-cluster centre).
    """
    return np.asarray(sources) - np.asarray(prior_mu)


def conditional_mean_residual(
    sources: np.ndarray, y: np.ndarray, means: dict | None = None
) -> tuple[np.ndarray, dict]:
    """Method B for models without a learned prior mean (e.g. FastICA): subtract
    the empirical per-condition (per-``y``-group) mean of each source.

    This is the linear/empirical analogue of :func:`conditional_prior_residual`,
    which subtracts a model-provided prior mean ``lambda_mu(u)``. Fit the group
    means on the training split (``means=None`` returns them) and pass them back
    to residualise a held-out split with the *training* means.

    Returns the residual and the ``{label: mean_vector}`` mapping used.
    """
    sources = np.asarray(sources, dtype=float)
    y = np.asarray(y).ravel()
    if means is None:
        means = {lab: sources[y == lab].mean(axis=0) for lab in np.unique(y)}
    residual = sources.copy()
    for lab, mean in means.items():
        residual[y == lab] -= mean
    return residual, means


def regression_residual(
    sources: np.ndarray, y: np.ndarray, coef: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Method B for a *continuous* condition: subtract ``E[sources | y]``
    estimated by (multi-output) linear regression on ``y``.

    The continuous analogue of :func:`conditional_mean_residual` (which handles a
    discrete condition via per-group means). Fit on the training split
    (``coef=None`` returns the fitted coefficients) and pass ``coef`` back to
    residualise a held-out split with the *training* fit. Uses a plain
    least-squares fit with an intercept (no sklearn dependency).

    Returns the residual and the coefficient matrix used.
    """
    sources = np.asarray(sources, dtype=float)
    y = np.asarray(y, dtype=float)
    y2 = y.reshape(len(y), -1)
    design = np.concatenate([np.ones((len(y2), 1)), y2], axis=1)  # [1, y]
    if coef is None:
        coef, *_ = np.linalg.lstsq(design, sources, rcond=None)
    return sources - design @ coef, coef


def _abs_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column |Pearson correlation| of features ``x`` with a continuous
    1-D ``y`` — a [0, 1] dependence score for a continuous condition."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float) - np.mean(y)
    xc = x - x.mean(axis=0)
    denom = np.sqrt((xc**2).sum(axis=0) * (y**2).sum())
    num = np.abs((xc * y[:, None]).sum(axis=0))
    return np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)


def _correlation_ratio(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-column correlation ratio eta^2 (between-group variance fraction) of
    features ``x`` w.r.t. discrete ``labels`` — a [0, 1] dependence score."""
    x = np.asarray(x, dtype=float)
    grand = x.mean(axis=0)
    ss_total = ((x - grand) ** 2).sum(axis=0)
    ss_between = np.zeros(x.shape[1])
    for lab in np.unique(labels):
        group = x[labels == lab]
        ss_between += group.shape[0] * (group.mean(axis=0) - grand) ** 2
    return np.divide(
        ss_between, ss_total, out=np.zeros_like(ss_between), where=ss_total > 0
    )


def drop_top_k_dependent(
    sources: np.ndarray, y: np.ndarray, k: int, dependence: str = "categorical"
) -> tuple[np.ndarray, np.ndarray]:
    """Method A: drop the ``k`` sources most dependent on the condition ``y``.

    Ranks source columns by their dependence on ``y`` and removes the top ``k``.
    ``dependence="categorical"`` uses the correlation ratio (eta^2) with a
    discrete label; ``dependence="continuous"`` uses |Pearson correlation| with a
    continuous condition. Returns the residual (kept columns, original order
    preserved) and the sorted indices of the dropped columns.
    """
    sources = np.asarray(sources)
    d = sources.shape[1]
    if k <= 0:
        return sources, np.array([], dtype=int)

    y = np.asarray(y).ravel()
    if dependence == "categorical":
        scores = _correlation_ratio(sources, y)
    elif dependence == "continuous":
        scores = _abs_correlation(sources, y)
    else:
        msg = f"invalid dependence: {dependence}"
        raise ValueError(msg)
    dropped = np.sort(np.argsort(scores)[::-1][:k])
    keep = np.setdiff1d(np.arange(d), dropped)
    return sources[:, keep], dropped


# ---------------------------------------------------------------------------
# Synthetic nonlinear-ICA data (faithful port of ilkhem/iVAE data.data)
# ---------------------------------------------------------------------------
def _xtanh(x: np.ndarray, slope: float) -> np.ndarray:
    """tanh plus a linear term — the reference ``xtanh`` activation."""
    return np.tanh(x) + slope * x


def _apply_activation(x: np.ndarray, activation: str, slope: float) -> np.ndarray:
    if activation == "xtanh":
        return _xtanh(x, slope)
    if activation == "lrelu":
        return np.where(x > 0, x, slope * x)
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if activation == "none":
        return x
    msg = f"incorrect nonlinearity: {activation}"
    raise ValueError(msg)


def _generate_mixing_matrix(
    d_sources: int, d_data: int, rng: np.random.Generator, n_iter_4_cond: float = 1e4
) -> np.ndarray:
    """Uniform, column-normalised mixing matrix with a controlled condition
    number (reference ``generate_mixing_matrix``, ``lin_type='uniform'``)."""

    def _gen() -> np.ndarray:
        a = rng.uniform(0, 2, (d_sources, d_data)) - 1.0
        a /= np.sqrt((a**2).sum(axis=0, keepdims=True))
        return a

    # Condition-number threshold = 25th percentile of random matrices' cond.
    cond_list = []
    for _ in range(int(n_iter_4_cond)):
        a = rng.uniform(-1, 1, (d_sources, d_data))
        a /= np.sqrt((a**2).sum(axis=0, keepdims=True))
        cond_list.append(np.linalg.cond(a))
    cond_thresh = np.percentile(cond_list, 25)

    a = _gen()
    while np.linalg.cond(a) > cond_thresh:
        a = _gen()
    return a


def sample_synthetic_nonlinear_ica(
    n_per_seg: int = 1000,
    n_seg: int = 40,
    d_sources: int = 5,
    d_data: int | None = None,
    n_layers: int = 3,
    prior: str = "gauss",
    activation: str = "xtanh",
    slope: float = 0.1,
    var_bounds: tuple[float, float] = (0.5, 3.0),
    uncentered: bool = False,
    n_iter_4_cond: float = 1e4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic nonlinear-ICA data with a segment auxiliary variable.

    Faithful port of ``generate_data(..., repeat_linearity=True)`` from
    ilkhem/iVAE. Sources are piecewise-stationary (per-segment variance, and
    optionally mean, modulation), then mixed by a random invertible nonlinear
    MLP: ``X = act(S A); X = act(X B); ... ; X = X B`` (last layer linear), with
    ``B = A`` when ``d_sources == d_data``.

    Returns:
        x: ``(n, d_data)`` observations, ``n = n_per_seg * n_seg``.
        u: ``(n, n_seg)`` one-hot auxiliary variable (segment label).
        s: ``(n, d_sources)`` ground-truth sources.
    """
    rng = np.random.default_rng(seed)
    if d_data is None:
        d_data = d_sources
    n = n_per_seg * n_seg
    var_lb, var_ub = var_bounds

    # --- segment-modulated (nonstationary) sources ---
    modulation = rng.uniform(var_lb, var_ub, (n_seg, d_sources))
    means = (
        rng.uniform(-5, 5, (n_seg, d_sources))
        if uncentered
        else np.zeros((n_seg, d_sources))
    )
    if prior == "gauss":
        s = rng.standard_normal((n, d_sources))
    elif prior == "lap":
        s = rng.laplace(0.0, 1.0 / np.sqrt(2.0), (n, d_sources))
    else:
        msg = f"incorrect prior: {prior}"
        raise ValueError(msg)

    labels = np.zeros(n, dtype=np.int64)
    for seg in range(n_seg):
        idx = slice(n_per_seg * seg, n_per_seg * (seg + 1))
        s[idx] = s[idx] * modulation[seg] + means[seg]
        labels[idx] = seg

    # --- invertible nonlinear mixing (repeat_linearity=True) ---
    a = _generate_mixing_matrix(d_sources, d_data, rng, n_iter_4_cond)
    x = _apply_activation(s @ a, activation, slope)
    if d_sources == d_data:
        b = a
    else:
        b = _generate_mixing_matrix(d_data, d_data, rng, n_iter_4_cond)
    for layer in range(1, n_layers):
        if layer == n_layers - 1:
            x = x @ b
        else:
            x = _apply_activation(x @ b, activation, slope)

    u = np.eye(n_seg)[labels]
    return x.astype(np.float32), u.astype(np.float32), s.astype(np.float32)
