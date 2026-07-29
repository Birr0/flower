# ICA-baseline experiments — hyperparameters (branch `feature/ICA`)

Authoritative record of every hyperparameter used in the iVAE / FastICA / direct
residualisation experiments and the Flower comparisons. Global seed
`RANDOM_STATE = 42` everywhere (`torch.manual_seed`, sklearn `random_state`,
numpy default_rng), unless noted.

---

## iVAE model (`flower.models.ivae.IVAE`)

| param | value | notes |
|---|---|---|
| `hidden_dim` | 128 | MNIST & spectra sweeps; 100 for the synthetic identifiability test; 64 for the 2D-Gaussians toy test |
| `n_layers` | 3 | linear layers per MLP |
| `activation` | `xtanh` | `tanh(x) + slope·x` |
| `slope` | 0.1 | activation slope |
| `decoder_var` | 0.01 | fixed Gaussian decoder variance |
| `learn_prior_mean` | `True` | MNIST/spectra/2D-Gaussians; `False` for the synthetic variance-modulated benchmark |
| `condition_encoder` | `True` | default; `False` = "fair" MNIST variant (encoder blind to `u`) |
| weight init | Xavier-uniform | `nn.init.xavier_uniform_` |

## iVAE training

| param | value | notes |
|---|---|---|
| optimizer | Adam | |
| `lr` | 1e-2 | |
| scheduler (sweeps) | `StepLR(step_size=max(1, epochs//3), gamma=0.3)` | manual training loop |
| scheduler (`configure_optimizers`) | `ReduceLROnPlateau(factor=0.1, patience=3)` | Lightning path |
| `batch_size` | 256 | MNIST/spectra; 64 for Gaussians/synthetic tests |
| `beta` (KL weight) | 1.0 | |
| `epochs` | 30 | default; **100** for the `iVAE-fair` re-run; 40 for the synthetic test |
| loss | recon NLL (fixed `decoder_var`) + `beta`·closed-form KL to `p(s|u)` | |

## FastICA (linear floor)

`sklearn.decomposition.FastICA(n_components=d, max_iter=1000, whiten="unit-variance", random_state=42)`
where `d` = embedding dimension (MNIST 64, spectra 10).

## Probes (evaluation)

Point-estimate sweeps use:
- classifier (MNIST digit): `LogisticRegression(max_iter=1000)`, `MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300)`
- regressor (colour/rotation/physics): `LinearRegression()`, `MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300)`

Bootstrap-CI benchmarks (`ivae_benchmark.py`) use `evaluate_embedding_{classifier,regressor}` with `n_iterations=1000`; MNIST probe families `{log_regression, 2-mlp=(64,32)}` / `{lin_regression, 2-mlp=(64,32)}`.

## Probe-free dependence metrics (`correlation_metrics.py`)

`flower.evaluation.dependence` — per-dimension, closed-form, **no fitted
hyperparameters**, so there is nothing here to tune or drift. Computed on the
**test split only** (no train/test split is needed: nothing is fitted). Scores are
scale-invariant, so the StandardScaler upstream does not move them. Reported as
the per-representation `max` over coordinates (the worst single-coordinate leak),
with `mean` and `n_above` (threshold `0.1`) also in the CSV.

| variable type | metric | used for |
|---|---|---|
| continuous condition | `abs_pearson`, `abs_spearman` | spectra redshift `z` |
| categorical condition | `correlation_ratio`, reported as `eta` (`squared=False`, so it shares the \|r\| scale) | MNIST digit, 2D-Gaussians mode |
| circular / multi-column | `multiple_correlation` | MNIST rotation, vs `[sin 2θ, cos 2θ]` |
| confounded target | `partial_correlation(rank=True)` | spectra logM\*/logSFR/A_v given `z`; MNIST colour/rotation given one-hot digit |

Two choices that matter for reproducing the numbers:

- **`partial_spearman`, not `partial_pearson`, for spectra.** Control removal is a
  least-squares fit, so a *linearly* removed control that acts nonlinearly is
  barely removed at all (a `z**3` confound still reads ~1.0; see
  `tests/test_evaluation_dependence.py`). Rank-partial is the default.
- **MNIST rotation is period-180.** `compute_rotation.py` returns a principal-axis
  orientation (`0.5·atan2`), so ±90° are the *same* orientation. The angle is
  doubled before forming `[sin, cos]`. The `rot_naive_max` column keeps the
  wrapped |Spearman| for comparison — and the existing `rot_r2_linreg`/`rot_r2_mlp`
  probe columns share the wrapped-target problem.

`null_level` in each CSV is the score expected for *independent* data at that `n`
(|r|: `1/√(n−1)`; `eta`: `√((G−1)/(n−1))`), since these statistics floor at chance
rather than zero.

2D-Gaussians point-estimate probes (`correlation_metrics.py` only; the bootstrap
variants live in `ivae_benchmark.py`): `LogisticRegression(max_iter=1000)`,
`MLPClassifier((64,64), max_iter=1000)`, `LinearRegression()`,
`MLPRegressor((64,64), max_iter=1000)`.

## Residual constructions

| method | function | setting |
|---|---|---|
| A (drop top-k) | `drop_top_k_dependent` | `dependence="categorical"` (MNIST digit) / `"continuous"` (spectra z); k-grid = `linspace(1, d-1, n_k)` rounded/unique |
| B — FastICA, discrete | `conditional_mean_residual` | per-group mean, fit on train |
| B — FastICA, continuous | `regression_residual` | linear least-squares (with intercept), fit on train |
| B — iVAE | `conditional_prior_residual` | subtract learned `λ_μ(u)` |

## Direct residualisation of the embedding (spectra only)

Fit `model` to predict the scaled embedding from `z` (train), subtract on both splits:
- `Resid-linear`: `LinearRegression()`
- `Resid-mlp`: `MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300)`
- `Resid-rf`: `RandomForestRegressor(n_estimators=100)`

---

## Dataset-specific settings

### RGB-MNIST — `examples/mnist/ivae_sweep.py`
- Embedding: `orig` VAE latent (64-dim), StandardScaler (fit on train).
- Condition removed: `digit` (10-class, one-hot `u`, `aux_dim=10`).
- Preserved factors: colour `b`; **rotation** (see below).
- Data: `$DATA_ROOT/rgbmnist/rgbmnist_Flow_cond_prior/embeddings/7518770_0` (train n=48000, test n=10000).
- Sweep: `--epochs 30`, `--n-k 32` (fair@100 run) / 11, `--drop-k 1` (benchmark), `--models FastICA,iVAE-cond,iVAE-fair`.
- Flower comparison: `orig`/`uncond`/`cond` columns, StandardScaler per column.

Rotation estimation — `examples/mnist/compute_rotation.py`
- VAE checkpoint `$DATA_ROOT/rgbmnist/rgbmnist_VAE/ckpts/5252446.ckpt`, `VAE(hidden_dim=64)`.
- Reconstruct via `projection_up → decoder`; angle `θ = 0.5·atan2(2·μ₁₁, μ₂₀−μ₀₂)` (deg) from central image moments (`skimage.measure`), channel 0, `batch_size=256`.
- Cached to `{train,test}_rotation_aligned.csv` (row-aligned to the embeddings).

### Spectra — `examples/spectra/ivae_sweep.py`
- Embedding: spender `orig` (10-dim), StandardScaler (fit on train).
- Condition removed: redshift `z` (continuous, embedding `Z` column, `aux_dim=1`).
- Preserved factors: `logM*`, `logSFR`, `A_v` (HF catalog `Birr001/spectra_catalog`, index-aligned).
- `--n-train 40000` (subsample of first `--n-filter 300000`), `--n-k 8`, `--epochs 30`.
- Masking: finite `z`, `z != -99.0`, finite embedding rows, finite/`!=-99` targets.
- Runs: `spender_I` (`sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7526202_0`) and `spender_II` (`sdss_II/spender_I_flow_v2/spender_II_flow_v2/embeddings/7527549_0`). Identical hyperparameters; only `--spender`/`--outdir` differ.

### 2D-Gaussians (tests) — `tests/test_ica_gaussians.py`
- 4-mode GMM, centres `(±3, ±3)`, isotropic cov 0.5; `IVAE(hidden_dim=64, learn_prior_mean=True)`; 30 epochs, batch 64, lr 1e-2, StepLR.

### Synthetic nonlinear-ICA identifiability (tests) — `flower.evaluation.ica.sample_synthetic_nonlinear_ica`
- Reference port of `ilkhem/iVAE` `generate_data(repeat_linearity=True)`.
- Headline test: `d_sources=2, d_data=6, n_seg=40, n_per_seg=500, n_layers=3, prior="gauss", activation="xtanh", slope=0.1, var_bounds=(0.5,3.0), uncentered=False, n_iter_4_cond=1e4`; iVAE `learn_prior_mean=False`, 30 epochs, batch 64, lr 1e-2, StepLR; FastICA floor as above.
