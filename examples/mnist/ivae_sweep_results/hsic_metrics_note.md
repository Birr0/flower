# HSIC independence test — RGB-MNIST (issue #22 / E3)

`python hsic_metrics.py --rotation-dir . --n-perm 2000 --n-subsample 2000`
→ `hsic_metrics.{csv,txt,log}`

## Question

W5 (R1) and R2's independence-test ask: probes measure *predictability under one
model class*, so weak prediction is not proof that information is gone. E3 was
planned as HSIC + distance correlation with permutation p-values, to test
statistical independence directly.

`dcor_metrics.py` ran first and failed (see `dcor_metrics_note.md`): distance
correlation reads `FastICA residB` as near-independent of the digit while an MLP
recovers it at 0.858. This note is the HSIC half.

## Setup

Normalized HSIC (equivalently CKA), Gaussian kernels, 2000 test rows shared
across all rows, 2000 permutations. Two comparability choices:

- **Fixed bandwidth.** Representations are standardized per dimension and the
  kernel width set to `sqrt(d)`, the scale implied by `E||a-b||^2 = 2d`. The
  median heuristic used in the first pass fits a *different* kernel per row, so
  effect-size differences could be bandwidth artefacts.
- **One shared subsample**, so row-to-row differences are not draw-to-draw noise.

Only `none`/`embedding`/`residB` rows are run; the residual-A sweep multiplies
the cost without adding to the argument.

## Result 1: nothing is independent, including Flower

| source | method | digit HSIC | null | ratio | p |
|---|---|---|---|---|---|
| Raw | none | 0.4158 | 0.0108 | 38.7 | 0.0005 |
| Flower-cond | embedding | 0.1731 | 0.0164 | 10.6 | 0.0005 |
| Flower-uncond | embedding | 0.2232 | 0.0157 | 14.2 | 0.0005 |
| FastICA | residB | 0.0310 | 0.0191 | 1.6 | 0.0005 |
| iVAE-cond | residB | 0.4115 | 0.0094 | 43.8 | 0.0005 |
| iVAE-fair | residB | 0.1848 | 0.0032 | 58.1 | 0.0005 |

**Every row rejects independence at the smallest attainable p (1/2001).** That
includes Flower's own embeddings. HSIC gives no independence certificate to any
method here, and the write-up must not imply otherwise.

The null column varies sixfold across rows even at fixed bandwidth, so raw HSIC
values are not comparable between representations — the `ratio` column is.

## Result 2: HSIC tracks the *strong* probe; dCor tracks the weak one

Spearman rank correlation of each probe-free statistic against each probe,
across the six rows:

| statistic | vs logreg | vs MLP |
|---|---|---|
| HSIC ratio | +0.657 | **+0.943** |
| dCor ratio | **+1.000** | +0.543 |

This is the substantive difference between the two tests. dCor's ordering is
*exactly* the linear probe's, because `conditional_mean_residual` equalises the
per-class means and dCor is essentially a within-vs-between-class distance
contrast. HSIC with a Gaussian kernel has power against covariance-only
dependence — verified directly on synthetic data where two classes share a mean
(max per-dim gap 0.076) and differ only in covariance: HSIC 0.0870 against a
0.0029 null, while an MLP reads 0.762 against 0.50 chance.

So HSIC is a genuine model-free corroboration of the cross-probe argument, not a
restatement of the linear probe.

## Result 3: the one disagreement, and why it does not rescue FastICA

| source | method | HSIC ratio | logreg | MLP |
|---|---|---|---|---|
| FastICA | residB | 1.6 | 0.114 | 0.858 |

`FastICA residB` is the single row where HSIC and the MLP disagree: HSIC ranks it
as the *least* dependent representation in the table, while the MLP recovers the
digit at 0.858 (chance 0.10). dCor said the same thing (ratio 1.19).

The asymmetry that resolves it: **recovery is a positive existence proof;
failure to detect is not evidence of absence.** An MLP reading the digit off a
representation at 0.858 demonstrates constructively that the digit is present.
Three independent statistics failing to see it constrains nothing — it bounds
their power, not the information content. A reviewer who argues "your own HSIC
says FastICA removed the most" is making exactly the inference W5 warns against.

This is the sharpest available form of the W5 answer, and it agrees with the
reviewer rather than resisting them: no test proves erasure, so the evidence that
carries weight is *positive recovery*, and by that standard the baselines'
residuals still contain the digit.

## What can and cannot be claimed

**Can:** HSIC rejects independence between the condition and every residual
tested, including Flower's; the *degree* of detected dependence tracks the
strongest probe rather than the weakest; by that measure the iVAE residuals
retain far more digit dependence (43.8x, 58.1x above null) than Flower's
conditional embedding (10.6x), consistent with their MLP scores (0.999 / 1.000
vs 0.792).

**Cannot:** that Flower achieves independence — it does not, p = 0.0005. Nor
that HSIC ranks removal reliably, given the FastICA row.

## Caveats

- n = 2000 subsample, one draw, `RANDOM_STATE = 42`; no repeat over draws.
- Normalized HSIC is a biased estimator; the permutation null absorbs the bias
  for the *test* but the effect sizes are not bias-corrected.
- The `b` and `rot` columns invert the reading (dependence is wanted), so their
  p-values are guaranteed significant and carry no information. Treat those
  columns as effect sizes only.
- Fixed `sqrt(d)` bandwidth is a defensible default, not a tuned choice; HSIC
  effect sizes are known to move with bandwidth.
