"""Probe-free dependence metrics for residual embeddings (issue #20 / E2).

The removal/preservation numbers elsewhere in the repo are *probe* scores — fit a
classifier/regressor on an embedding and read off accuracy or R^2. Those are
aggregate and capacity-dependent: an MLP R^2 of 0.05 conflates "no information"
with "information this MLP did not find", and it moves when you change
``hidden_layer_sizes`` or ``max_iter``.

The metrics here are the complement: **per-dimension**, no fitting, no train/test
split and (except for dCor's subsample size) no hyperparameters. They answer "how
much does any single latent coordinate still track the condition", which is the
honest worst-case statement a probe cannot make. Report them *alongside* the
probes, not instead.

All but one are linear or monotone measures and will miss dependence a nonlinear
probe can still exploit — both failure modes are documented with real numbers in
the `correlation_metrics_note.md` files. :func:`distance_correlation` is the
exception: it is zero only under independence, at O(n^2) cost, so it is
subsampled.

Metric per variable type:

- continuous condition  -> :func:`abs_pearson` (linear), :func:`abs_spearman`
  (monotone; use this when the relation is monotone but curved, as embedding-vs-
  redshift is — plain Pearson understates the leak and flatters removal).
- categorical condition -> :func:`correlation_ratio` (eta^2). Note this only sees
  *mean* shifts between groups: a latent whose variance depends on the label but
  whose group means coincide scores ~0.
- angular / multi-column condition -> :func:`multiple_correlation`, e.g. against
  the ``[sin, cos]`` pair the dSprites catalog stores for orientation.
- confounded targets -> :func:`partial_correlation`, which removes a control
  variable from both sides first. This is what separates "preserved physics" from
  "the condition leaked back in through a correlated target".
- any dependence, any shape -> :func:`distance_correlation` (wraps the `dcor`
  package). Use it to check whether a near-zero score above is real independence
  or a blind spot; read it against the permutation ``null_level``, never zero.

Every score is in ``[0, 1]``, one per column of ``x``. Because they are bounded
below by chance rather than by zero, :func:`dependence_report` also returns the
``null_level`` — the value the metric takes on average for *independent* data at
this sample size — so a max can be read against chance instead of against 0.
"""

from __future__ import annotations

import dcor
import numpy as np
from scipy.stats import rankdata

__all__ = [
    "abs_pearson",
    "abs_spearman",
    "correlation_ratio",
    "dependence_report",
    "distance_correlation",
    "multiple_correlation",
    "partial_correlation",
]

_METRICS = (
    "pearson",
    "spearman",
    "eta",
    "multiple",
    "partial_pearson",
    "partial_spearman",
    "dcor",
)


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Coerce ``x`` to a float ``(n_samples, n_features)`` array."""
    x = np.asarray(x, dtype=float)
    return x if x.ndim == 2 else x.reshape(len(x), -1)


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Elementwise divide, yielding 0 where the denominator vanishes.

    Constant columns (zero variance) are genuinely uncorrelated with everything,
    so 0 is the right score rather than a nan that would poison a max.
    """
    num = np.asarray(num, dtype=float)
    denom = np.asarray(denom, dtype=float)
    return np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)


def _rank(x: np.ndarray) -> np.ndarray:
    """Column-wise average rank transform (ties share their mean rank)."""
    return rankdata(_as_2d(x), axis=0).astype(float)


def _residualize(x: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Return ``x`` with the least-squares fit on ``[1, control]`` removed."""
    x = _as_2d(x)
    control = _as_2d(control)
    design = np.concatenate([np.ones((len(control), 1)), control], axis=1)
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ coef


def abs_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column ``|Pearson r|`` of features ``x`` against a continuous 1-D ``y``.

    Args:
        x: ``(n_samples, n_features)`` features (1-D is treated as one feature).
        y: ``(n_samples,)`` continuous variable.

    Returns:
        ``(n_features,)`` scores in ``[0, 1]``.
    """
    x = _as_2d(x)
    y = np.asarray(y, dtype=float).ravel()
    yc = y - y.mean()
    xc = x - x.mean(axis=0)
    denom = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum())
    num = np.abs((xc * yc[:, None]).sum(axis=0))
    return _safe_divide(num, denom)


def abs_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column ``|Spearman rho|`` — :func:`abs_pearson` on average ranks.

    Catches any *monotone* relation, not just a linear one. Prefer this over
    :func:`abs_pearson` when the condition enters the embedding through a curved
    but order-preserving map.
    """
    return abs_pearson(_rank(x), _rank(np.asarray(y).ravel()).ravel())


def correlation_ratio(
    x: np.ndarray, labels: np.ndarray, *, squared: bool = True
) -> np.ndarray:
    """Per-column correlation ratio against a discrete ``labels`` vector.

    ``eta^2`` is the fraction of each feature's variance lying *between* label
    groups. It measures group-mean separation only — a feature whose spread
    depends on the label but whose group means agree scores ~0.

    Args:
        x: ``(n_samples, n_features)`` features.
        labels: ``(n_samples,)`` discrete labels of any comparable dtype.
        squared: return ``eta^2`` (default, the variance fraction). Pass ``False``
            for ``eta``, which shares the ``|Pearson r|`` scale and so is the
            comparable choice when a table mixes categorical and continuous
            conditions.

    Returns:
        ``(n_features,)`` scores in ``[0, 1]``.
    """
    x = _as_2d(x)
    labels = np.asarray(labels).ravel()
    grand = x.mean(axis=0)
    ss_total = ((x - grand) ** 2).sum(axis=0)
    ss_between = np.zeros(x.shape[1])
    for lab in np.unique(labels):
        group = x[labels == lab]
        ss_between += group.shape[0] * (group.mean(axis=0) - grand) ** 2
    eta_sq = _safe_divide(ss_between, ss_total)
    return eta_sq if squared else np.sqrt(eta_sq)


def multiple_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column multiple correlation ``R`` of each feature on *all* of ``y``.

    The several-columns-of-``y`` generalisation of :func:`abs_pearson`: the square
    root of the R^2 from regressing each feature on ``[1, y]``. With a single
    column of ``y`` it reduces exactly to ``|Pearson r|``.

    The motivating case is a circular variable stored as its ``[sin, cos]`` pair
    (dSprites orientation): correlating against either component alone is
    parameterisation-dependent, while ``R`` against both is not.

    Returns:
        ``(n_features,)`` scores in ``[0, 1]``.
    """
    x = _as_2d(x)
    resid = _residualize(x, y)
    ss_total = ((x - x.mean(axis=0)) ** 2).sum(axis=0)
    ss_resid = (resid**2).sum(axis=0)
    r_sq = np.clip(_safe_divide(ss_total - ss_resid, ss_total), 0.0, 1.0)
    return np.sqrt(r_sq)


def partial_correlation(
    x: np.ndarray, y: np.ndarray, control: np.ndarray, *, rank: bool = False
) -> np.ndarray:
    """Per-column ``|partial correlation|`` of ``x`` with ``y`` given ``control``.

    Least-squares removes ``control`` from both ``x`` and ``y``, then the two
    residuals are correlated. Use it when a preservation target is itself
    entangled with the condition — on SDSS spectra ``logM*``/``A_v`` correlate
    with redshift, so a raw correlation between a residual embedding and ``logM*``
    partly re-measures the redshift that was supposed to have been removed.
    Controlling for ``z`` reports only the target structure that is *not* redshift.

    Args:
        x: ``(n_samples, n_features)`` features.
        y: ``(n_samples,)`` target.
        control: ``(n_samples,)`` or ``(n_samples, n_controls)`` confounder(s).
            A *categorical* control must be one-hot encoded by the caller, since
            the removal here is linear.
        rank: rank-transform ``x``, ``y`` and ``control`` first, giving a partial
            Spearman — monotone rather than linear removal.

    Removal is only as good as the fit: with ``rank=False`` a *nonlinearly*
    acting control is barely removed at all (a ``z**3`` confound still reads ~1.0),
    and even ``rank=True`` fits linearly on ranks, so a residual can persist.
    Prefer ``rank=True`` whenever the control enters through a curved map — as
    redshift does on spectra — and read the result as an upper bound on how much
    of the confound was taken out.

    Returns:
        ``(n_features,)`` scores in ``[0, 1]``.
    """
    x = _as_2d(x)
    y = np.asarray(y, dtype=float).reshape(len(x), -1)
    control = _as_2d(control)
    if rank:
        x, y, control = _rank(x), _rank(y), _rank(control)
    return abs_pearson(_residualize(x, control), _residualize(y, control).ravel())


def _safe_dcor(a: np.ndarray, b: np.ndarray) -> float:
    """``dcor.distance_correlation`` with a guard for degenerate input.

    A constant column has no distance variance, so the statistic divides by zero;
    it is genuinely independent of everything, so 0 is the right score.
    """
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    value = float(dcor.distance_correlation(a, b))
    return 0.0 if not np.isfinite(value) else value


def _dcor_scores(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_subsample: int,
    n_replicates: int,
    random_state: int | None,
) -> tuple[np.ndarray, float]:
    """Per-column dCor plus its permutation null, averaged over subsamples.

    Returns ``(scores, null_level)``. The null is the same statistic with ``y``
    shuffled inside each subsample — the empirical chance floor, which matters
    here because dCor is positively biased at small ``n``.
    """
    x = _as_2d(x)
    y2 = _as_2d(y)
    n_samples, n_features = x.shape
    rng = np.random.default_rng(random_state)

    take = min(n_subsample, n_samples)
    if take >= n_samples:
        n_replicates = 1  # nothing left to resample over

    totals = np.zeros(n_features)
    null_total = 0.0
    for _ in range(n_replicates):
        idx = (
            np.arange(n_samples)
            if take >= n_samples
            else rng.choice(n_samples, take, replace=False)
        )
        y_sub = y2[idx]
        y_perm = y_sub[rng.permutation(take)]
        for j in range(n_features):
            x_col = x[idx, j : j + 1]
            totals[j] += _safe_dcor(x_col, y_sub)
            null_total += _safe_dcor(x_col, y_perm)

    denom = n_replicates * n_features
    return totals / n_replicates, (null_total / denom if denom else 0.0)


def distance_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_subsample: int = 2000,
    n_replicates: int = 5,
    random_state: int | None = None,
) -> np.ndarray:
    """Per-column distance correlation of ``x`` against ``y``, on a subsample.

    Wraps :func:`dcor.distance_correlation`. dCor is zero **only** under
    independence, so unlike everything else in this module it sees dependence of
    any shape — the ``y = x**2`` case that Pearson and Spearman both read as ~0,
    and the rotationally symmetric functions that defeat per-coordinate
    correlation entirely (see the 2D-Gaussians note).

    Cost is why this is subsampled. `dcor`'s ``AUTO`` method picks an O(n log n)
    AVL algorithm when **both** sides are univariate, but falls back to the naive
    O(n²) pairwise-distance algorithm as soon as ``y`` has more than one column —
    which is exactly the case we need for circular targets like MNIST rotation
    (``[sin 2θ, cos 2θ]``). Capping the rows keeps the multi-column path
    affordable and bounds the univariate path too; ``n_replicates`` independent
    draws are averaged to damp the sampling variance the cap introduces.

    Args:
        x: ``(n_samples, n_features)`` features.
        y: ``(n_samples,)`` or ``(n_samples, n_components)`` variable.
        n_subsample: rows per replicate. Raising it improves the estimate and
            lowers its chance floor, at up to quadratic cost. If it meets or
            exceeds ``n_samples`` every row is used and ``n_replicates``
            collapses to 1.
        n_replicates: independent subsamples to average.
        random_state: seed for the draws; pass one for a reproducible number.

    Returns:
        ``(n_features,)`` scores in ``[0, 1]``.

    dCor is **positively biased at small n**, so a raw 0.15 on a 200-row
    subsample can be pure noise. Prefer :func:`dependence_report` with
    ``metric="dcor"``, which returns the matching permutation ``null_level``,
    over reading these values against zero.
    """
    scores, _ = _dcor_scores(
        x,
        y,
        n_subsample=n_subsample,
        n_replicates=n_replicates,
        random_state=random_state,
    )
    return scores


def _null_level(metric: str, n_samples: int, n_aux: int) -> float:
    """Expected score for *independent* data — the floor a max is read against.

    Correlation-type statistics are bounded below by chance, not by zero, and that
    floor grows with the number of nuisance dimensions removed and shrinks with
    sample size. ``|r|`` under the null has standard error ``1/sqrt(n - 1)``;
    ``eta^2`` averages ``(n_groups - 1) / (n - 1)``; a multiple ``R^2`` on ``m``
    regressors averages ``m / (n - 1)``.
    """
    dof = max(n_samples - 1, 1)
    if metric in ("pearson", "spearman"):
        return float(np.sqrt(1.0 / dof))
    if metric == "eta":
        return float(np.sqrt(max(n_aux - 1, 0) / dof))
    if metric == "multiple":
        return float(np.sqrt(n_aux / dof))
    # Partial variants lose one degree of freedom per control column.
    return float(np.sqrt(1.0 / max(n_samples - n_aux - 1, 1)))


def dependence_report(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "pearson",
    *,
    control: np.ndarray | None = None,
    threshold: float = 0.1,
    n_subsample: int = 2000,
    n_replicates: int = 5,
    random_state: int | None = None,
) -> dict:
    """Score every column of ``x`` against ``y`` and summarise the result.

    A uniform wrapper over the metrics in this module, so a results table can be
    built by varying ``metric`` alone.

    Args:
        x: ``(n_samples, n_features)`` embedding or residual.
        y: the condition or target. 1-D for every metric except ``"multiple"``,
            which takes ``(n_samples, n_components)``.
        metric: one of ``"pearson"``, ``"spearman"``, ``"eta"``, ``"multiple"``,
            ``"partial_pearson"``, ``"partial_spearman"``.
        control: required by the ``partial_*`` metrics, ignored otherwise.
        threshold: cutoff for the ``n_above`` count.

    Returns:
        ``{metric, per_dim, max, mean, argmax, n_above, threshold, null_level,
        n_samples}``. ``max`` is the headline for a removal claim (the worst
        single-coordinate leak); ``mean`` describes how diffusely the condition is
        spread across coordinates. Compare ``max`` against ``null_level`` — at
        small ``n`` a max of 0.05 can be pure chance.
    """
    if metric not in _METRICS:
        msg = f"invalid metric: {metric!r} (expected one of {list(_METRICS)})"
        raise ValueError(msg)

    x = _as_2d(x)
    y_arr = np.asarray(y)
    n_samples = len(x)
    used_subsample = None

    if metric == "dcor":
        per_dim, dcor_null = _dcor_scores(
            x,
            y_arr,
            n_subsample=n_subsample,
            n_replicates=n_replicates,
            random_state=random_state,
        )
        used_subsample = min(n_subsample, n_samples)
        return {
            "metric": metric,
            "per_dim": per_dim,
            "max": float(per_dim.max()) if per_dim.size else 0.0,
            "mean": float(per_dim.mean()) if per_dim.size else 0.0,
            "argmax": int(per_dim.argmax()) if per_dim.size else -1,
            "n_above": int((per_dim > threshold).sum()),
            "threshold": threshold,
            # Empirical (permutation) floor, not analytic: dCor has no simple
            # closed-form null, and its small-n bias is large enough that
            # reading a score against 0 would overstate the dependence.
            "null_level": dcor_null,
            "n_samples": n_samples,
            "n_subsample": used_subsample,
            "n_replicates": n_replicates,
        }

    if metric.startswith("partial_"):
        if control is None:
            msg = f"metric {metric!r} requires `control`"
            raise ValueError(msg)
        per_dim = partial_correlation(
            x, y_arr, control, rank=metric.endswith("spearman")
        )
        n_aux = _as_2d(control).shape[1]
    elif metric == "pearson":
        per_dim, n_aux = abs_pearson(x, y_arr), 1
    elif metric == "spearman":
        per_dim, n_aux = abs_spearman(x, y_arr), 1
    elif metric == "eta":
        # eta, not eta^2, so the column shares the |Pearson r| scale.
        per_dim = correlation_ratio(x, y_arr, squared=False)
        n_aux = len(np.unique(y_arr))
    else:
        per_dim = multiple_correlation(x, y_arr)
        n_aux = _as_2d(y_arr).shape[1]

    return {
        "metric": metric,
        "per_dim": per_dim,
        "max": float(per_dim.max()) if per_dim.size else 0.0,
        "mean": float(per_dim.mean()) if per_dim.size else 0.0,
        "argmax": int(per_dim.argmax()) if per_dim.size else -1,
        "n_above": int((per_dim > threshold).sum()),
        "threshold": threshold,
        "null_level": _null_level(metric, n_samples, n_aux),
        "n_samples": n_samples,
    }
