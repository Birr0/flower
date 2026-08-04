# Iterated (boosting) residualisation — E4 (RGB-MNIST digit)

The cMNIST companion to the spectra E4 (`../../spectra/ivae_sweep_results/iterated_residual_note.md`).
Produced by `../iterated_residual.py`; figure `iterated_residual.png`, data
`iterated_residual.csv`. Residualise the VAE embedding against `one-hot(digit)`
repeatedly (1×…5×, linear and MLP regressor), tracking recoverable digit + colour
`b`/rotation preservation.

## Result — iterating does NOT help

| iterations | 0 (raw) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| linear — digit logreg ↓ | 0.947 | 0.114 | 0.114 | 0.114 | 0.114 | 0.114 |
| linear — digit **MLP** ↓ | 0.979 | 0.894 | 0.893 | 0.897 | 0.895 | 0.896 |
| MLP — digit MLP ↓ | 0.979 | 0.965 | 0.974 | 0.968 | 0.967 | 0.959 |
| linear — mean preserv (b, rot) | 0.957 | 0.890 | 0.889 | 0.892 | 0.893 | 0.897 |

Recoverable digit is **flat from iteration 1**. Linear residualisation removes the
digit from a *linear* probe (0.947 → 0.114) but leaves it MLP-recoverable at
**~0.89**, and further iterations do nothing (idempotent). Flower `cond` sits at
digit MLP **0.792** — below the iterated baseline's asymptote — with mean
preservation 0.785.

## Two notes

1. **Idempotence:** residualising against `one-hot(digit)` subtracts the per-digit
   mean; the residual then has zero per-digit mean, so a second pass subtracts
   nothing. The digit that survives is in higher-order structure that
   mean-removal cannot reach — so iterating is flat.
2. **The "MLP residualiser" is not stronger here:** for a *discrete* condition,
   linear regression on the one-hot is the *exact* per-group mean, so an MLP
   regressor is only a worse (underfit) mean estimator — it removes *less* of the
   digit (MLP-probe 0.96 vs the linear residualiser's 0.89), not more. A more
   expressive residualiser does not help because the problem is not the mean.

## Takeaway

Consistent with the spectra E4: iterating the residualisation baseline (1×→5×)
does not reduce recoverable condition information — it asymptotes from the first
iteration, and even its endpoint (digit MLP ~0.89) removes *less* than Flower
(0.792). The baseline's limit is structural, not iterative.
