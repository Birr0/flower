# Distance correlation independence test — RGB-MNIST (issue #22 / E3)

## ⚠ This result is NOT USABLE for the claim it was run to support. Do not cite it.

`python dcor_metrics.py` → `dcor_metrics.{csv,txt}`.

The numbers below are recorded because they were measured, not because they are
citable. `hsic_metrics_note.md` refers to this file as the reason the HSIC half
was run; this is that reason, written up.

## Question

Probes measure *predictability under one model class*, so weak prediction is not
proof that information is gone (R1 W5, and the second half of R2's ask). E3 was
planned as distance correlation + HSIC with permutation nulls, to test
statistical independence directly rather than through a probe.

## Why it fails: the ordering inverts against the probes

Digit dependence, whole representation vs the label, read against each row's own
permutation null:

| source | method | digit dCor | null | ratio | digit acc, MLP probe |
|---|---|---|---|---|---|
| Raw | none | 0.641 | 0.083 | 7.7 | 0.979 |
| Flower-cond | embedding | 0.399 | 0.103 | 3.9 | **0.792** |
| FastICA | residB | **0.130** | 0.109 | **1.2** | **0.858** |

dCor calls `FastICA residB` almost independent of the digit — ratio 1.2 against
its null, the lowest of any row — while an MLP recovers the digit from it at
0.858. It simultaneously calls Flower's embedding *three times more dependent*,
though the MLP recovers less digit from Flower (0.792) than from FastICA residB.
The measure orders the two methods in the opposite direction to every probe.

**Cause.** `conditional_mean_residual` equalises the per-class means by
construction, and dCor against a one-hot label is essentially a within- versus
between-class distance contrast. So the statistic is blind to exactly the
structure that mean-subtraction leaves behind, and rewards exactly the
construction that removes class means and nothing else. It behaves as a proxy
for a linear probe, not as an independence test.

## Second defect: insensitive to the drop sweep

dCor is computed on the whole representation through pairwise distances, so it is
dominated by the highest-variance directions. Dropping low-variance components
barely moves the distance geometry:

- `iVAE-cond residA` reads **0.690 at k = 1, 7, 13 and 20** — identical to three
  decimals while 20 of 64 sources are deleted.
- `iVAE-fair residA` moves from 0.303 to 0.301 across k = 1 → 57.

A statistic that cannot see 20 deletions cannot be used to compare
representations that differ by deletions.

## What to use instead

- **HSIC** (`hsic_metrics_note.md`) — sound, but **one-sided**: every
  representation including Flower's rejects independence at p ≈ 0.0005, so it
  supports "information remains" and can never certify removal.
- **Cross-probe consistency** — the argument that actually carries: apparent
  removal that a stronger probe undoes is not removal.
- **Closed-form per-dimension measures** — `flower.evaluation.dependence`,
  reported beside probe scores by `correlation_metrics.py`.

## Status

E3 was **descoped on 2026-07-26**. No reviewer-facing text may promise
independence tests or permutation p-values. The principle that replaced it:
recovery is a positive existence proof, failure to detect is not evidence of
absence — which concedes the reviewer's point rather than resisting it.

Not done, and would be needed before any revival: a bandwidth-robustness check on
the HSIC effect sizes, and spectra / 2D-Gaussian runs of both measures.
