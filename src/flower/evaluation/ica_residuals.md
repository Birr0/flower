# ICA baselines & condition-suppressed residuals

Reference notes for the ICA baselines in `flower.evaluation.ica` and the model in
`flower.models.ivae`. These are **baselines for Flower** (issue #20 / experiment
E2): given a representation and a *known factor* `y` (e.g. the cluster/mode, a
digit, a redshift), recover independent factors and produce a representation with
`y` removed but the rest preserved.

There are two orthogonal choices:

1. **How the factors are recovered** — linear ICA (FastICA) vs nonlinear ICA
   (iVAE).
2. **How the condition is deleted from those factors** — residual **A** (drop) vs
   residual **B** (subtract the mode-mean).

---

## Running example: the 2D-Gaussians toy

Each point is

```
X = center(mode) + diff
```

- **mode** — which of the 4 clusters (the known factor `y` to remove).
- **diff** — the wiggle around that cluster centre, `diff ~ N(0, 0.5 I)`: the
  condition-independent **seed** to keep.

Goal: a representation where the **mode is unrecoverable** but the **wiggle
survives**. Measured by mode-classification accuracy (should fall to chance =
0.25) and radial-distance regression R² (should stay high).

---

## Recovering the factors

### Linear ICA — FastICA
Assumes the data is a *linear* mix of independent sources: finds one unmixing
matrix `W` and sets `sources = W · X`. No training, no knowledge of `y`. Can only
undo linear mixing. This is the **linear floor**.

### Nonlinear ICA — iVAE (`flower.models.ivae`)
A small VAE that can undo *nonlinear* mixing and is **given the factor** as an
auxiliary input `u = one-hot(y)`. That side-information is what makes the sources
identifiable under nonlinear mixing (Khemakhem et al. 2020). It is trained; the
conditional prior `p(s|u)` carries the `y`-dependence.

> **Honest caveat.** On a *linear* toy (like the 2D Gaussians) FastICA is already
> optimal, so nonlinear iVAE offers no benefit and its nonlinearity can slightly
> hurt. iVAE's advantage appears only under **nonlinear** mixing — see the
> synthetic benchmark (`sample_synthetic_nonlinear_ica`), where iVAE reaches
> MCC ≈ 0.94 vs FastICA ≈ 0.65.

---

## Residual A — rank by dependence, then delete

`drop_top_k_dependent(sources, y, k)` in `ica.py`, using `_correlation_ratio`.

**Step 1 — score each source's dependence on the mode.** For each source column,
compute the **correlation ratio** η² — the fraction of that column's variance
explained by the mode label:

```
η²(col) = between-group variance / total variance
        = Σ_g n_g · (mean_g − grand_mean)²   /   Σ_i (x_i − grand_mean)²
```

with `g` over the modes, `n_g` the group size, `mean_g` the column mean within
mode `g`, `grand_mean` the overall mean. Interpretation:

- η² ≈ 0 → the mode's average is the same everywhere → the source carries **no**
  mode information.
- η² ≈ 1 → the source is essentially determined by the mode → **pure** mode
  information.

This is the ANOVA between-vs-total ratio per column. It is for **discrete**
conditions; a continuous factor would use `|correlation|` instead.

**Step 2 — delete the top-k.** Sort columns by η² descending, drop the `k`
highest, keep the rest in original order:

```python
dropped = np.sort(np.argsort(scores)[::-1][:k])
keep    = np.setdiff1d(np.arange(d), dropped)
return sources[:, keep], dropped
```

Properties:

- **Output has fewer columns** (`d − k`): a coordinate is thrown away, so any
  seed signal it carried is lost too (why A wrecks the distance-R² in the toy).
- **`k` is a hand-set knob** (we use `k=1`); the function only ranks and cuts.
- **Train/test consistency is the caller's job**: rank on **train** to fix
  `dropped`, then keep those same columns on test.

---

## Residual B — subtract the mode's mean

Same idea in both variants — *remove the part of each source the mode explains* —
differing only in where the mode-mean comes from. Unlike A, B **keeps every
column** (it shifts them), so it removes only the mode's average location and
preserves everything orthogonal to it (the seed).

### FastICA variant — empirical group means
`conditional_mean_residual(sources, y, means=None)` in `ica.py`:

```python
means = {mode: sources[y == mode].mean(axis=0) for mode in unique(y)}  # fit on TRAIN
residual[y == mode] -= means[mode]                                     # subtract per row
```

For each mode, average the source vectors of the training points in that mode,
then subtract it from every point of that mode → **re-centre each cluster onto
the origin.** Every mode now has zero mean, so its *location* is erased while each
point's within-cluster offset (the seed) is untouched. On the test split reuse
the **train** means (the optional `means` argument) to avoid leakage.

### iVAE variant — learned conditional-prior mean
`conditional_prior_residual(sources, prior_mu)` in `ica.py`:

```python
return sources - prior_mu       # prior_mu = λ_μ(u), from the model
```

Identical operation, but the subtracted mean is `λ_μ(u)` — the **conditional
prior mean the network learned as a function of `u`** — rather than an empirical
per-group table. For discrete one-hot `u` the two coincide; the learned version
matters for **continuous** conditions, where the network interpolates a mean for
unseen values that a per-group table cannot.

---

## Mental model

- **FastICA vs iVAE** = linear vs nonlinear way of recovering the factors.
- **Residual A vs B** = *deleting* the most mode-dependent coordinate vs
  *subtracting* each coordinate's mode-average.

On the linear 2D toy the linear method wins and Residual B cleanly beats A:

| Feature set | Seed MCC | Mode acc (↓, chance 0.25) | Distance R² (↑) |
|---|---|---|---|
| Raw X | — | 1.00 | 0.96–0.99 |
| FastICA residual A (drop top-k) | — | 0.52 | 0.49 |
| FastICA residual B (conditional-mean) | 1.00 | 0.26 | 1.00 |
| iVAE residual A (drop top-k) | — | 0.88 | 0.35 |
| iVAE residual B (conditional-prior) | 0.77 | 0.31 | 0.95 |

(Numbers from a short `examples/2d_gaussians/ivae_benchmark.py` run; the nonlinear
advantage of iVAE shows on nonlinearly-mixed data, not this linear toy.)
