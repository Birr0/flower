# Redshift removal vs physical-factor preservation (spender_I spectra)

Records the spectra run of the ICA-baseline sweep (`results.csv`, `summary.txt`,
`tradeoff.png`) and its interpretation. This is the **real science** version of
the RGB-MNIST digit/rotation experiment (`../../mnist/flower_cond_results/note.md`):
the same surgical-vs-blunt argument, with a **continuous** condition and genuine
physical factors.

- **Condition removed:** redshift `z` (continuous).
- **Factors probed for retention:** `logM*` (stellar mass), `logSFR` (star
  formation rate), `A_v` (dust attenuation) — physical properties entangled with
  redshift to *different* degrees.
- **Input:** the spender `orig` embedding (10-dim). Probes: linear + MLP, point
  estimates. Redshift from the embedding's `Z` column; physical targets from the
  HF catalog `Birr001/spectra_catalog` (index-aligned; Raw target R² are high, so
  alignment holds).

## The numbers (MLP probe unless noted)

| Method | z linreg | z **MLP** ↓ | logM* ↑ | logSFR ↑ | A_v ↑ |
|---|---|---|---|---|---|
| Raw (orig) | 0.404 | 0.555 | 0.809 | 0.659 | 0.573 |
| Resid-linear (regress-out z) | −0.000 | 0.607 | 0.730 | 0.663 | 0.566 |
| Resid-MLP (nonlinear regress-out z) | 0.001 | 0.602 | 0.693 | 0.663 | 0.571 |
| Resid-RF (random-forest regress-out z) | −0.000 | −0.405 | −0.214 | 0.022 | −0.040 |
| FastICA residual B (regress-out z) | −0.000 | 0.607 | 0.728 | 0.666 | 0.567 |
| iVAE residual B (conditional-prior) | 0.003 | 0.434 | 0.618 | 0.653 | 0.559 |
| FastICA residual A, k=7 | 0.002 | 0.029 | 0.146 | 0.106 | 0.212 |
| iVAE residual A, k=6 | 0.018 | 0.027 | 0.135 | 0.335 | 0.451 |
| **Flower cond** | **0.011** | **0.028** | 0.250 | 0.577 | 0.512 |
| Flower uncond | 0.404 | 0.534 | 0.797 | 0.648 | 0.567 |

## Two headline findings

**1. Flower removes redshift completely *and* consistently; the ICA baselines do
not.** Flower cond drives z to R² ≈ 0.03 with a linear-vs-MLP gap of ~0.02 — the
removal survives a stronger probe. FastICA residual B is the classic weak-probe
illusion: z R² = −0.00 to a *linear* probe but **0.607** to an MLP (even higher
than raw). iVAE residual B is better (0.434) but still far from removed. No
mean/regression-based residual actually removes a nonlinearly-encoded condition.

**2. At matched removal, Flower preserves the physics far better than blunt
source-dropping.** To reach Flower's z R² ≈ 0.03 the ICA baselines must drop 6–7
of 10 components, which guts the physical factors. At that matched removal:

| | logM* | logSFR | A_v |
|---|---|---|---|
| Flower cond | **0.250** | **0.577** | **0.512** |
| FastICA residA k=7 | 0.146 | 0.106 | 0.212 |
| iVAE residA k=6 | 0.135 | 0.335 | 0.451 |

Flower preserves all three better — *dramatically* on `logSFR` (0.58 vs 0.11) and
`A_v` (0.51 vs 0.21).

## Direct residualisation — including a *nonlinear* residualiser — also fails

To pre-empt "did you try a stronger residualiser?", we residualise the embedding
directly against `z` with three regressors (subtract the fitted `E[X|z]`):

| Residualiser | z linreg | z **MLP** ↓ |
|---|---|---|
| Resid-linear | −0.000 | 0.607 |
| Resid-MLP (nonlinear) | 0.001 | **0.602** |
| Resid-RF | −0.000 | −0.405 (degenerate) |

Three points:

1. **Resid-linear ≡ FastICA residual B** (both z MLP 0.607). Linear residualisation
   of the embedding is equivalent to residualising the ICA sources — it commutes
   with the linear unmixing, so ICA adds nothing to *linear* removal.
2. **The nonlinear (MLP) residualiser fails just as badly** — z MLP 0.602 ≈ linear.
   Mean-removal only subtracts `E[X|z]`; redshift here is encoded *structurally*
   (it warps the whole spectral manifold, not as an additive offset), so it
   survives in the residual's higher-order structure and an MLP reads it straight
   back. A more expressive *mean* model does not help, because the problem is not
   the mean. This is the key rebuttal point: no mean-based residualiser removes a
   structurally-encoded condition.
3. **Resid-RF is degenerate** (all R² negative). A random forest fitting
   `z (1-D) → X (10-D)` overfits and does not generalise, so the train/test
   residual distributions mismatch and it corrupts the embedding — its low z R² is
   an artefact, *not* removal.

Note `iVAE residual B` (z MLP 0.434) actually removes redshift *better* than any
direct residualiser (~0.60): its *learned* conditional prior captures more
z-structure than a fitted conditional mean. But it is still far from Flower
(0.028). **Only Flower actually removes the redshift.**

## Why this is the meaningful test (physical reading)

The `logM*` vs `logSFR`/`A_v` split is exactly the independent-vs-entangled
distinction from the MNIST note, now physical:

- **`logM*` (stellar mass) is *strongly* entangled with redshift** — the
  Malmquist / selection bias of a flux-limited sample means mass and redshift are
  correlated by construction. So even Flower retains only 0.25: much of `logM*` is
  *legitimately* redshift-correlated and goes with the redshift. This is the
  spectra analogue of "a factor genuinely associated with the label is removed to
  the extent of that association."
- **`logSFR`, `A_v` are *less* z-entangled** — they are distinct residual physics.
  A surgical remover keeps them; blunt source-dropping cannot, because the
  components it deletes to remove redshift also carry them (`logSFR` 0.58 → 0.11).

This is why preserving the entangled-but-distinct factors is the test that
matters: it is the difference between removing *redshift* and removing *everything
redshift touches* — and the latter would destroy the residual physics you set out
to study. Flower does the former; ICA source-dropping does the latter.

## Why redshift *does* reach ~chance here (unlike the MNIST digit)

Redshift is largely a **coherent, near-separable** axis of the spender
representation (spectra shift smoothly with `z`), so the flow can route it out
almost entirely (z R² → 0.03). The digit was woven into the geometry (shape) and
could only be suppressed to ~0.79. Same mechanism, different entanglement: how far
the condition can be driven toward chance is set by how separable it is from the
structure being kept — here it is highly separable, so removal is near-complete
while the less-entangled physics survives.

## One line

On real spectra, Flower removes redshift near-completely and consistently across
probes, and at matched removal retains the distinct physics (logSFR, A_v) that
blunt ICA source-dropping destroys — with the strongly z-entangled stellar mass
(logM*) legitimately going with the redshift.
