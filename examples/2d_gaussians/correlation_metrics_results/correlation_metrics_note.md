# Probe-free correlation metrics — 2D-Gaussians toy (issue #20 / E2)

`python correlation_metrics.py`
→ `correlation_metrics.{csv,txt,png,log}`

## Question

This is the **calibration case** for the correlation metrics: the generator hands
back `diff`, the 2-D condition-independent seed, so identifiability can be checked
directly with MCC. If the probe-free metrics disagree with MCC here, they are not
to be trusted on spectra or MNIST where no ground truth exists.

Condition = GMM mode (4 clusters). Chance level for η at n = 5 000 is **0.0245**;
chance classification accuracy is 0.25.

## Result: on ground truth, every metric agrees

| source | method | n_dims | mode_eta_max | mode_acc_mlp | seed_multiple_max | seed_mcc |
|---|---|---|---|---|---|---|
| Raw | none | 2 | 0.974 | 1.000 | 0.228 | 0.213 |
| FastICA | residA | 1 | 0.974 | 0.493 | 0.198 | 0.198 |
| FastICA | residB | 2 | **0.044** | **0.250** | **1.000** | **1.000** |
| iVAE | residA | 1 | 0.956 | 0.880 | 0.293 | 0.261 |
| iVAE | residB | 2 | 0.052 | 0.263 | 0.999 | 0.865 |

`FastICA residB` is exact: MLP accuracy 0.250 is *at* the 0.25 chance level,
η = 0.044 is within 2× the 0.0245 chance level, and MCC = 1.000 — the seed is
recovered perfectly and the mode is gone. Probe, probe-free and identifiability
metrics agree on a case where the answer is known. That is the result this script
exists to establish.

Two secondary readings:

- **FastICA beats the iVAE here, as it should.** The toy is a linear mixture, so
  linear ICA is exactly the right model class; the iVAE is over-powered and lands
  at MCC 0.865. This is a sanity check on the iVAE baseline, not a defect.
- **Residual A cannot work at d = 2.** Dropping the single most mode-dependent of
  two sources leaves η at 0.974/0.956 and accuracy far above chance (0.493,
  0.880). The mode is not isolated in one coordinate. Both families agree.

## The blind spot, demonstrated cleanly

The radial distance is condition-independent by construction and is the
preservation target. Every correlation-style metric says it is absent:

| source | method | dist_raw_max | dist_r2_linreg | dist_r2_mlp |
|---|---|---|---|---|
| Raw | none | 0.011 | −0.001 | **0.998** |
| FastICA | residB | 0.009 | −0.000 | **1.000** |
| iVAE | residB | 0.012 | −0.000 | **0.994** |

`dist_raw_max` ≈ 0.01 is *below* the 0.0245 chance level, and a linear probe gets
R² ≈ 0 — yet an MLP recovers the distance at R² = 0.999.

The reason is structural, not statistical: the radial distance is a norm, a
rotationally symmetric function of the two coordinates. It correlates with neither
coordinate individually and is not linear in them, while being perfectly
determined by them jointly.

**This is the sharpest available statement of what the correlation family cannot
do.** A near-chance correlation is evidence about single-coordinate, monotone
association only. It is never evidence that a factor has been removed. Any use of
these columns to claim removal must be paired with a nonlinear probe — which is
exactly why `correlation_metrics.py` reports both.

This blind spot is *distinct* from the one in the spectra note: there, redshift
survived spread across coordinates (each individually weak); here, a single
nonlinear function of two coordinates is invisible to per-coordinate correlation.
Both defeat a per-dimension max, for different reasons.

## Reproducibility: the toy was unseeded

`generate_quad_gmm` takes no seed argument and `data.py` never calls
`torch.manual_seed`, so **the toy is redrawn on every run** — the `--seed` flag on
the ICA scripts controls FastICA, the iVAE and the probes, but not the data.

Two unseeded draws of this table gave:

| row | draw 1 | draw 2 |
|---|---|---|
| iVAE residB — mode_eta_max | 0.109 | 0.038 |
| iVAE residB — seed_mcc | 0.754 | 0.813 |
| iVAE residA — mode_acc_mlp | 0.778 | 0.843 |

The FastICA rows were stable (MCC 1.000 both times); the iVAE rows were not. This
script now calls `torch.manual_seed(args.seed)` before generating, and two runs at
the same seed produce **bit-identical** CSVs. The numbers in this note are from
the seeded run.

This is concrete evidence for the multi-seed item in `TODO.md`: the iVAE-vs-FastICA
gaps quoted anywhere from this toy are single-draw numbers, and at least on the
iVAE side the draw-to-draw spread is comparable to the effects being reported.
Other scripts in this directory (`ivae_benchmark.py`, `beta_sweep.py`,
`no_prior_ablation.py`) call `generate_quad_gmm` the same way and are affected
identically — they have not been changed.

## No Flower row here — deliberately

Unlike the spectra and MNIST tables, this one has no `Flower-cond`/`Flower-uncond`
row. The toy has no cached embeddings: its Flower model is trained in-process by
`train.py`/`model.py` (the self-contained sandbox that never imports the `flower`
package), so adding the row means training the toy flow inside this script or
teaching `train.py` to cache its embeddings.

**Decision: not needed.** The information plots already produced for this toy
cover what a Flower row would show, and this table's purpose is narrower — it is
the ground-truth calibration case for the *metrics*, establishing that η, MCC and
the probes agree where the answer is known. That job is done by the FastICA and
iVAE rows alone. Revisit only if the calibration argument itself is challenged.

## Caveats

- `d = 2`, so `max` and `mean` coincide for the residual-A rows and the max is
  over very few coordinates.
- Probes here are point estimates; `ivae_benchmark.py` reports the same
  quantities with 1000-sample bootstrap CIs.
- Seeded at `RANDOM_STATE = 42` only — one seed, now reproducible, but still one.
