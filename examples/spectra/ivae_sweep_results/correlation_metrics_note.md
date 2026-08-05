# Probe-free correlation metrics — spectra (issue #20 / E2)

`python correlation_metrics.py --spender spender_I`
→ `correlation_metrics.{csv,txt,png,log}`

## Question

Every removal/preservation number we quote on spectra is a probe R². That is
aggregate and capacity-dependent, and it cannot say "no *single* coordinate still
tracks redshift". This adds the per-dimension, closed-form, hyperparameter-free
family from `flower.evaluation.dependence` over the same representations as
`ivae_sweep.py`, so both families sit in one table.

Condition = redshift `z`. Chance level for |ρ| at n = 43 098 is **0.0048**.

## Headline: the two families disagree, in both directions

| source | method | z_pearson_max | z_spearman_max | z_r2_mlp |
|---|---|---|---|---|
| Raw | none | 0.489 | 0.529 | 0.555 |
| FastICA | residB | **0.010** | **0.212** | **0.607** |
| Resid-linear | direct | **0.011** | 0.123 | **0.607** |
| Resid-mlp | direct | 0.024 | 0.073 | 0.602 |
| iVAE | residB | 0.035 | 0.082 | 0.434 |
| Flower-cond | embedding | 0.062 | **0.042** | **0.028** |
| Flower-uncond | embedding | 0.367 | 0.378 | 0.534 |

Three things fall out:

1. **Pearson alone would certify linear residualisation as a complete success.**
   `FastICA residB` and `Resid-linear` both score `z_pearson_max ≈ 0.010` — twice
   the chance level of 0.0048. Spearman on the same representations is 0.212 and
   0.123, i.e. 10–20× chance. Both remove `E[X|z]` by least squares, so by
   construction they null the *linear* part and leave the curved part untouched.
   Any redshift table reporting Pearson only is reporting the artefact.

2. **A low per-dimension max does not mean the condition is gone.** `FastICA
   residB` has a worst coordinate of 0.212, yet an MLP recovers redshift from it
   at R² = 0.607 — *higher than the raw embedding's* 0.555. Redshift survives
   distributed across coordinates, where no single-coordinate statistic can see
   it. The probe is strictly stronger here.

3. **Flower-cond is the only representation where Spearman ≤ Pearson**
   (0.042 vs 0.062), with zero coordinates above the 0.1 threshold and a probe
   R² of 0.028 — an order of magnitude below every residualisation baseline
   (0.60). No hidden monotone leak, and nothing distributed either.

## The preservation confound goes the *opposite* way to what we assumed

Target entanglement with `z` (|Spearman|): **logM\* 0.788**, logSFR 0.132,
A_v 0.023. So logM\* is the only target where partialling out `z` should matter —
and it is the only one where it does (logSFR/A_v move by <0.01 everywhere).

But it moves **up**, not down:

| source | logM\*_raw_max | logM\*_partial_max |
|---|---|---|
| Raw | 0.726 | 0.675 |
| FastICA residB | 0.357 | **0.530** |
| iVAE residB | 0.406 | **0.645** |
| Resid-linear | 0.385 | **0.635** |
| Flower-cond | 0.288 | **0.435** |
| Flower-uncond | 0.507 | 0.395 |

For every *residual* representation the raw correlation **understates** how much
genuine stellar-mass structure survives. This is classical suppression: the
coordinate retains some `z`, `z` is strongly tied to logM\*, and the two
associations partly cancel until `z` is removed from both sides.

The direction reverses only for `Raw` and `Flower-uncond` — the two
representations that still carry a lot of redshift, where the confound inflates
the raw number as originally expected.

**Practical consequence:** raw correlations with logM\* are not a safe
preservation metric on this dataset in either direction. Report the partial.

## `Resid-rf` — the existing probe row is an artefact

`ivae_sweep_results/results.csv` reports `Resid-rf` at logM\* R² = −0.214,
logSFR 0.022, A_v −0.040, which reads as "removes z, destroys the physics". The
correlation columns disagree: logM\* partial = 0.542, logSFR 0.486, A_v 0.432 —
comparable to every other residual method.

The correlation columns are right. The random-forest residualiser overfits `z → X`
on train, so the residual it produces has **train std 0.437 vs test std 1.141**, a
2.6× distribution mismatch. Any probe *fitted on the train residual* is fitted on
a degenerate representation and cannot generalise; the negative R² measures that
mismatch, not the absence of physics. The correlation metrics are computed on the
test split alone with nothing fitted, so they are immune to it.

Do not cite the `Resid-rf` probe row as evidence about what that method preserves.
(It is not currently cited in `review/` or `papers/`.)

## Caveats

- **η-style blind spots apply here too.** These are linear/monotone measures. A
  low value bounds single-coordinate association only, never "information is
  absent" — see finding 2 above, and the radial-distance case in the
  2D-Gaussians note, where max correlation ≈ chance while MLP R² = 0.999.
- `partial_spearman` removes the control with a *linear* fit on ranks, so it is an
  upper bound on how much redshift was taken out, not a guarantee.
- `residA` rows have varying `n_dims`, so their max is taken over fewer
  coordinates and is not directly comparable across `k`.
- Single seed (`RANDOM_STATE = 42`), as with every ICA baseline here. The
  multi-seed repeat is tracked in `TODO.md`; the correlation metrics themselves
  are deterministic given an embedding, but FastICA and the iVAE upstream are not.
