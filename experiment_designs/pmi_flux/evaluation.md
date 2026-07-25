↑ [Back to README](README.md)

# Evaluation

## 4. Benchmark Design

### 4.1 Datasets

| Dataset | Role | Ground truth |
|---|---|---|
| Synthetic tasks from `bmi` (see §4.6) | Calibration | Closed-form MI via the Czyż et al. 2023 40-task suite (`bmi.benchmark.BENCHMARK_TASKS`) — normal, Student-t, and diffeomorphically-transformed variants, dense and sparse correlation structure, dimensionality 1×1–25×25, MI up to ~2–5 nats. Requires a new minimal flow config (`src/conf/experiment/synthetic_bmi_Flow/`, new) trained on `bmi`-generated `(x, y)` pairs. |
| dSprites | Known-factor structure | No closed-form MI, but known generative factors (shape, scale, rotation, x, y — existing dataset/config) allow constructing both positive controls (y = a real factor) and negative controls (y = an independent/shuffled factor, expected MI ≈ 0). |
| RGBMNIST | Real-data scaling test | No closed-form ground truth by default; adopt the `mibenchmark` binary-symmetric-channel trick (§4.6) to construct a *known*, dialable ground-truth MI on MNIST-like pairs instead of relying on qualitative comparison only. |

### 4.2 Baseline estimators

- **KSG** (k-NN, non-parametric) — no training required, standard ground-truth-free reference.
- **MINE** (neural variational lower bound) — widely cited, known to underestimate at high MI.
- **InfoNCE/CPC** (contrastive lower bound) — common in representation-learning MI literature.

All three run on the *same* paired `(x, y)` samples as the flow-based estimator for direct
comparison. See §4.6 for existing packages that ship these estimators and their ground-truth
tasks, so §4.1's synthetic case does not need a bespoke implementation.

### 4.6 Reuse existing MI benchmark suites instead of building from scratch

Several maintained benchmark suites already provide exactly the ground-truth-known synthetic
tasks and reference-estimator implementations §4.1/§4.2 call for. Prefer these over hand-rolling
KSG/MINE/InfoNCE and the correlated-Gaussian sweep:

- **[`bmi`](https://github.com/cbg-ethz/bmi) (Benchmarking Mutual Information, CBG-ETH Zürich)**
  — PyPI package `benchmark-mi`, MIT licensed. Ships `KSGEnsembleFirstEstimator`, and
  JAX-implemented `DonskerVaradhanEstimator`, `MINEEstimator`, `InfoNCEEstimator`, `NWJEstimator`,
  plus CCA — i.e. all three baselines in §4.2 plus one extra (NWJ) for free. Ground-truth tasks
  are addressed by name with known MI attached:
  ```python
  import bmi
  task = bmi.benchmark.BENCHMARK_TASKS["1v1-normal-0.75"]
  ground_truth_mi = task.mutual_information
  x, y = task.sample(1000, seed=42)
  ```
  This implements the 40-task suite from Czyż et al. 2023, *"Beyond Normal: On the Evaluation of
  Mutual Information Estimators"* (arXiv:2306.11078) — bivariate and multivariate normal (dense
  and sparse correlation structure), multivariate Student-t at various degrees of freedom, and
  diffeomorphic transforms of each (Gaussian-CDF/uniform-margin, half-cube, asinh, spiral,
  "wiggly" non-uniform-lengthscale mappings) at dimensionality 1×1 up to 25×25, spanning MI up to
  ~2 nats in the main suite and up to 5 nats in a documented high-MI extension. This directly
  supersedes the bespoke synthetic-Gaussian sweep proposed in an earlier draft of §4.1 — reuse
  `bmi`'s tasks (including its sparse/transformed variants, which stress-test estimators harder
  than plain correlated Gaussians) rather than rebuilding a narrower version of the same thing.
  A relevant documented finding from the underlying paper: KSG degrades sharply on the *sparse*
  2-pair-interaction tasks even though it's accurate on dense multivariate normal — worth
  specifically including sparse tasks in our sweep, since that is exactly the regime where a
  flow-based estimator might differentiate itself.
- **[`mibenchmark`](https://github.com/kyungeun-lee/mibenchmark)** (Lee & Rhee, NeurIPS 2024,
  *"A Benchmark Suite for Evaluating Neural Mutual Information Estimators on Unstructured
  Datasets"*, arXiv:2410.10924) — extends ground-truth MI construction to *real* unstructured
  data (MNIST, CIFAR-10/100, IMDB/BERT text embeddings) via same-class positive pairing plus a
  binary-symmetric-channel trick that lets you dial the true MI to a chosen value even on real
  data. Relevant to our RGBMNIST track (§4.1): this gives a way to get a real, non-synthetic
  ground-truth MI benchmark on MNIST-like data, which our current design otherwise lacks (RGBMNIST
  in §4.1 has no ground truth at all today). Worth adopting the BSC-trick construction for
  RGBMNIST specifically, upgrading it from "no ground truth" to "known ground truth."
- **[`mutinfo`](https://github.com/VanessB/mutinfo)** (2025, *"Towards Diverse and Comprehensive
  Benchmarks for Mutual Information Estimation"*, arXiv:2607.03487, CC-BY-4.0) — the most recent
  suite, combining a copula-based synthetic generator (independently controllable MI,
  dimensionality, and marginal complexity) with a "marginals-first" real-image track built on the
  same-class-pairing idea. Its headline finding — *no estimator dominates uniformly; ranking
  flips by task category (non-parametric vs. discriminative vs. generative)* — is directly
  relevant to how we should report results in §4.3/§4.4: report performance broken out by task
  category rather than a single aggregate ranking, since that is the axis the field has converged
  on as the meaningful one.

**Revision to §4.1**: replace the bespoke "new `synthetic_gaussian_Flow` config, sweep ρ
manually" plan with training/evaluating the flow estimator directly on `bmi.benchmark.BENCHMARK_TASKS`
samples (still requires a `synthetic_*_Flow` Hydra config to train a flow on `bmi`-generated
`(x, y)` pairs, but the task definitions, ground-truth MI, and baseline-estimator implementations
come from the package rather than being reimplemented). Similarly, replace the ad hoc
KSG/MINE/InfoNCE implementations referenced in §4.2 with `bmi`'s estimator classes directly.

### 4.3 Success criteria

Comparative, not a fixed absolute threshold: on the synthetic Gaussian sweep, the flow-based
estimator should be **closer to the analytic ground truth than KSG/MINE/InfoNCE at matched
sample budget**, with particular attention to the high-true-MI end of the sweep (ρ ≥ 0.9), where
MINE/InfoNCE are documented to degrade. On dSprites/RGBMNIST (no ground truth), success is
cross-estimator agreement plus passing the sanity checks in §4.4.

### 4.4 Metrics & sanity checks

- Bias vs. true MI (Gaussian only).
- Variance across seeds/bootstrap resamples, at fixed sample budget.
- Sample efficiency: estimate quality as a function of `n` paired samples.
- Wall-clock / compute cost per estimate (the flow estimator pays ODE-integration cost per
  sample — expect this to be more expensive per-point than KSG, cheaper than training MINE/CPC
  from scratch).
- High-MI degradation behavior (does accuracy fall off the way MINE is known to, or differently?).
- Non-negativity: MI ≥ 0 always.
- Permutation null: shuffling the `(x, y)` pairing should collapse the estimate toward 0.
- Negative control on dSprites: y = an independent/irrelevant factor should give MI ≈ 0.

### 4.5 Interpretability evaluation (qualitative)

Plot `∇·v_cond(t) − ∇·v_uncond(t)` against `t` for individual `(x, y)` pairs on dSprites, across
different conditioning factors (e.g. rotation vs. scale), to check whether the time-resolved flux
profile differs meaningfully by factor (e.g. does a "sharper," more localized-in-time signal
correspond to a factor with cleaner generative structure?). This is a case-study/qualitative
track, not a numeric pass/fail benchmark — it supports the interpretability half of the
contribution claim in §7 and has no baseline to compare against by construction.

## 5. Experimental Protocol

- **Training**: reuse existing Hydra CFG-trained flow configs as-is (`rgbmnist_Flow`,
  `dsprites_Flow`, both existing under `src/conf/experiment/`); add one new
  `synthetic_bmi_Flow` config (§4.1/§4.6) for the calibration case.
- **Likelihood validation gate**: run the closed-form-density check from §3 before any MI number
  from this pipeline is trusted.
- **Held-out evaluation set**: MI is estimated on a held-out split, never on training data, to
  avoid optimistic bias from the flow having memorized training pairs.
- **Repeats**: each configuration (dataset × estimator × sample budget) run across multiple
  seeds to report variance, not just a point estimate.
