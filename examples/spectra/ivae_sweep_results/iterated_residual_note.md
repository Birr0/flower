# Iterated (boosting) residualisation — E4 (spender_I redshift)

Answers the reviewer suggestion (E4): "repeat the residualisation loop
twice/thrice — presumably less and less information about `y` can be retrieved."
Produced by `../iterated_residual.py`; figure `iterated_residual.png`, data
`iterated_residual.csv`.

Setup: residualise the (scaled) spender embedding against redshift `z` repeatedly
(each pass fits a regressor `z → X` and subtracts `E[X|z]`), with a **linear** and
an **MLP** regressor, 1×…5×. Track recoverable redshift (linear + MLP probe) and
mean physics preservation per iteration.

## Result — iterating does NOT help (mean-residualisation is idempotent)

| iterations | 0 (raw) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| linear — z R² MLP ↓ | 0.555 | 0.607 | 0.607 | 0.607 | 0.607 | 0.607 |
| MLP — z R² MLP ↓ | 0.555 | 0.602 | 0.607 | 0.608 | 0.607 | 0.606 |
| linear — mean physics R² | 0.680 | 0.653 | 0.653 | 0.653 | 0.653 | 0.653 |
| MLP — mean physics R² | 0.680 | 0.643 | 0.637 | 0.645 | 0.640 | 0.644 |

(linear z R² *linreg* is 0.404 → −0.00 after pass 1 and stays there.)

Recoverable redshift is **flat from iteration 1** for both residualisers, and if
anything the single pass slightly *raises* the MLP score (0.555 → 0.607) by
removing the linear component and exposing the nonlinear structure. Flower `cond`
sits at z R² **0.028** — the iterated baselines asymptote ~20× higher.

## Why (the informative part)

- One residualisation removes the conditional mean `E[X|z]`. The residual then has
  **zero conditional mean by construction**, so a second pass fits ≈0 and subtracts
  ≈nothing. Linear residualisation is *exactly* idempotent; the MLP converges after
  ~one pass. Boosting therefore adds no removal.
- The redshift that survives is not in the mean — it is in **higher-order
  structure** (variance / manifold geometry) that mean-subtraction of any order
  cannot reach. This is the same reason a single nonlinear (MLP/RF) residualiser
  already fails (see `note.md`).

## Takeaway for E4

Iterating the residualisation baseline 1×→5× leaves recoverable redshift flat at
z R² ≈ 0.61 (MLP), far above Flower's 0.03. The baseline's failure to remove
redshift is **structural** — the condition is encoded beyond the conditional mean
— not a matter of too few iterations. (Physics stays ~0.65 only because redshift
was never removed; at genuine removal the source-dropping baselines collapse the
physics, see the main results table.)
