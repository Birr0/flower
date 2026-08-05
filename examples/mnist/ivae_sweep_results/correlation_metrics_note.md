# Probe-free correlation metrics — RGB-MNIST (issue #20 / E2)

`python correlation_metrics.py --rotation-dir .`
→ `correlation_metrics.{csv,txt,png,log}`

## Question

The MNIST counterpart of the spectra and 2D-Gaussians `correlation_metrics.py`.
Condition = **digit** (categorical), so removal is measured with the correlation
ratio η rather than a correlation coefficient. Preservation targets are colour `b`
and rotation.

Chance level for η at n = 10 000 is **0.0300**; chance digit accuracy is 0.10.

## Headline: η and the probe disagree by a mile

| source | method | digit_eta_max | n_above | digit_acc_mlp | b_raw_max | b_partial_max |
|---|---|---|---|---|---|---|
| Raw | none | 0.686 | 64 | 0.979 | 0.574 | 0.657 |
| FastICA | residB | **0.080** | **0** | **0.858** | 0.971 | 0.971 |
| iVAE-cond | residB | 0.836 | 40 | 0.999 | 0.510 | 0.510 |
| iVAE-fair | residB | 0.918 | 50 | 1.000 | 0.404 | 0.406 |

**`FastICA residB` is the sharpest counterexample in the whole study.** η = 0.080
(2.7× chance) with *zero* coordinates above the 0.1 threshold — by every
per-dimension measure the digit is gone. An MLP then classifies the digit from it
at **85.8% accuracy**, against 10% chance.

This is the same failure mode as spectra `FastICA residB` (max |ρ| 0.21, probe
R² 0.61) but far more extreme, and for the same reason: subtracting the per-digit
mean nulls each coordinate's *between-group means*, which is exactly and only what
η measures. The digit survives in the joint configuration of 64 coordinates, where
no single-coordinate statistic can reach it.

If a table reported η alone, it would claim FastICA-residB removes the digit. It
does not remove it at all in any usable sense.

**The iVAE residuals are worse than the raw embedding on η** (0.836 and 0.918 vs
0.686) at essentially perfect probe accuracy. `iVAE-cond` is expected to leak — its
encoder sees `u = one-hot(digit)` and can bake the digit back in — but `iVAE-fair`
(`condition_encoder=False`) is no better here. Subtracting the learned λ_μ(u)
concentrates the digit into individual coordinates rather than removing it.

## Rotation: the 180° wrap costs up to a third of the signal

`compute_rotation.py` returns a principal-axis orientation
(`0.5·atan2(2·μ11, μ20−μ02)`), which is periodic **mod 180°** — ±90° are the same
orientation. Correlating against raw degrees charges a representation for the wrap.

| source | method | rot_naive_max (wrapped) | rot_circular_max (correct) | understated by |
|---|---|---|---|---|
| Raw | none | 0.648 | 0.707 | 9% |
| FastICA | residB | 0.335 | 0.498 | **49%** |
| FastICA | residA(1) | 0.322 | 0.474 | **47%** |
| iVAE-cond | residB | 0.516 | 0.554 | 7% |
| iVAE-fair | residA(1) | 0.541 | 0.557 | 3% |

The correct measure is `multiple_correlation` against `[sin 2θ, cos 2θ]`, which
maps period-180 onto a full turn. The artefact is worst exactly where it matters —
the residual representations, where preserved rotation is the claim being made.

**This applies to the existing probe columns too.** `rot_r2_linreg` and
`rot_r2_mlp` in `ivae_sweep_results/results.csv` regress on raw degrees and share
the wrapped-target problem, so they understate preserved rotation by an unknown
amount of the same order. Any rotation-preservation claim sourced from those
columns should be re-derived against the doubled angle before being quoted.

## Colour `b` is digit-independent, as assumed

`b_partial_max` ≈ `b_raw_max` in every row (largest gap 0.083, at Raw). Partialling
the one-hot digit out of the colour association changes essentially nothing, which
is the expected result — RGB-MNIST assigns colour independently of the label. This
is a sanity check on the dataset construction, and it passes.

## Residual A: the preservation cliff

Sweeping `k` for FastICA, colour `b` holds at 0.969–0.971 up to k = 20, then
collapses to 0.035 at k = 26 and never recovers. Digit η falls smoothly
(0.720 → 0.029) but digit accuracy plateaus around 0.11–0.14 from k = 26 onward —
i.e. past the cliff the representation has lost the colour without buying further
digit removal. The useful operating range is k ≤ 20.

## Caveats

- **η sees group means only.** A coordinate whose *variance* depends on the digit
  but whose per-digit means coincide scores ~0. Combined with the distributed-leak
  result above, η should never be read as evidence of removal on its own.
- Single seed (`RANDOM_STATE = 42`) — see the multi-seed item in `TODO.md`.
- `residA` rows have varying `n_dims`, so their max is over fewer coordinates and
  is not directly comparable across `k`.
