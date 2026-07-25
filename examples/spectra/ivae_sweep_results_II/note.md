# spender_II robustness check

Repeats the redshift-removal sweep on the second spender model (`spender_II`,
6-dim embedding) with **identical hyperparameters** to spender_I (only
`--spender`/`--outdir` differ; see `../../ICA_HYPERPARAMETERS.md`). Full
interpretation is in the spender_I note (`../ivae_sweep_results/note.md`); this
records what reproduces and what shifts.

Condition removed = redshift `z` (continuous). Preserved = `logM*`, `logSFR`,
`A_v`. Probes: linear + MLP.

## Key numbers (MLP probe), spender_I vs spender_II

| Method | z MLP ↓ (I / II) | logSFR ↑ (I / II) | A_v ↑ (I / II) |
|---|---|---|---|
| Raw | 0.555 / 0.514 | 0.659 / 0.632 | 0.573 / 0.535 |
| Resid-linear | 0.607 / 0.764 | 0.663 / 0.640 | 0.566 / 0.545 |
| Resid-MLP (nonlinear) | 0.602 / 0.720 | 0.663 / 0.643 | 0.571 / 0.537 |
| Resid-RF | −0.405 / −0.454 (degenerate) | — | — |
| FastICA residB | 0.607 / 0.821 | 0.666 / 0.645 | 0.567 / 0.548 |
| iVAE residB | 0.434 / 0.602 | 0.653 / 0.655 | 0.559 / 0.486 |
| **Flower cond** | **0.028 / 0.208** | **0.577 / 0.620** | **0.512 / 0.503** |
| Flower uncond | 0.534 / 0.563 | 0.648 / 0.648 | 0.567 / 0.561 |

## Reproduces on spender_II

1. **No mean-based residualiser removes redshift.** Linear *and* nonlinear (MLP)
   residualisation leave z at MLP R² 0.72–0.82 (here even higher than raw). RF
   residualisation is degenerate (negative R²). Same conclusion as spender_I.
2. **Cross-probe illusion, even starker.** FastICA residB: z R² −0.00 (linear) →
   **0.821** (MLP).
3. **Source-dropping guts the physics.** Pushing z down collapses the physical
   targets (residual A, k=5: logSFR 0.11, A_v 0.29).
4. **Flower is the best performer.** Lowest z (0.208, cross-probe-consistent:
   0.059 ≈ 0.208) while preserving the physics near raw levels (logSFR 0.620 ≈
   raw 0.632; A_v 0.503 ≈ raw 0.535). At Flower's removal level the ICA baselines
   have already destroyed the physics.

## What shifts (model-dependent, honest)

- Flower removes redshift **less completely** on spender_II (z MLP 0.208 vs 0.028
  on spender_I). Its cross-probe gap is correspondingly larger (0.15 vs 0.02) but
  still far tighter than any residualiser (FastICA residB gap 0.82).
- Consistent with weaker removal, **more of the z-entangled stellar mass survives**
  (logM* 0.453 vs 0.250 on spender_I) — less redshift removed ⇒ more of the
  redshift-correlated mass retained.

So the **operating point is model-dependent, but the ranking and mechanism are
robust**: mean residualisation fails, source-dropping is blunt, and the
conditioned flow removes redshift most (and consistently) while best preserving
the distinct physics.

Outputs: `results.csv`, `summary.txt`, `tradeoff.png` in this directory.
