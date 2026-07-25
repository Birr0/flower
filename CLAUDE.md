# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Flower trains, loads, inspects, explains, and infers conditional generative models: a VAE
encoder combined with a conditional flow matching (CFM) model over the learned latent space.
Conditioning on known factors of variation lets you study the residual structure those
factors don't explain. Every experiment is a [Hydra](https://hydra.cc/) config, trained with
[PyTorch Lightning](https://lightning.ai/).

## Setup

```bash
uv sync
source .venv/bin/activate
```

Fill in `.env` with personal data directories and W&B credentials before running any training
or embedding job.

## Commands

Nothing is on `PATH` unless the venv is activated — prefix with `uv run` (`uv run pytest ...`,
`uv run ruff ...`) if you haven't sourced `.venv/bin/activate`.

```bash
# Full test suite (168 tests, no GPU/data required)
pytest tests/ -v

# Skip the `slow` marker (the iVAE/ICA tests that actually train a model)
pytest tests/ -v -m "not slow"

# Single test file / class / test
pytest tests/test_flow_matching.py -v
pytest tests/test_integration.py::TestDspritesFlowConfig -v
pytest tests/test_vae.py::TestLightningVAE::test_configure_optimizers -v

# Lint / format (CI runs these as a separate job from tests, so a lint failure
# does not hide test results). `src/` is clean; on the current branch `tests/` is
# not — RUF059 in test_ica_gaussians.py, plus ica.py/test_ica_gaussians.py unformatted.
ruff check src/ tests/
ruff format --check src/ tests/

# Pre-commit (ruff + ruff-format + pytest, runs on every commit if hooks are installed)
pre-commit install
pre-commit run --all-files
```

Train and embed are run as Hydra entrypoints, not via pytest:

```bash
cd src/flower/training
(srun) python train.py -cn "experiment/{experiment_name}/train" hydra/launcher={compute_config}

cd src/flower/inference
(srun) python embed.py -cn "experiment/{experiment_name}/embed" hydra/launcher={compute_config}
```

Use `srun` and an HPC/SLURM launcher config on a cluster; locally, drop `srun` and set
`hydra/launcher=local`.

Baseline/analysis scripts under `examples/{dataset}/` are plain `python` scripts, not Hydra
jobs — run them from their own directory (they `load_dotenv()` and read `DATA_ROOT`), e.g.
`python ivae_sweep.py --epochs 30 --outdir ivae_sweep_results`.

## Architecture

**Everything is driven by Hydra composition, not code.** `src/conf/experiment/{name}/` holds
one config tree per experiment — currently `dsprites_Flow`, `dsprites_VAE`, `rgbmnist_Flow`,
`rgbmnist_Flow_beta_ablation`, `rgbmnist_VAE`, `spender_I_flow`, `spender_II_flow`. Each
experiment directory composes `meta.yaml` (experiment identity/paths),
`model.yaml` (the `lightning_loader` + callbacks), `train.yaml`/`embed.yaml` (defaults lists
tying data/trainer/logger/model together), and `sweeps.yaml` (hydra multirun sweep params).
See `src/flower/README.md` for the full anatomy of these files and what gets written to disk
(`$DATA/{data_name}/{experiment_name}/{ckpts,embeddings,metrics,multiruns}`).

**Shared flow, per-dataset encoders.** `flower.models.modules` now owns the single shared
`VelocityField` and `LightningFlowMatching` base class (plus timestep/condition embedders,
AdaLN, conditional prior, classifier-free-guidance wrapper). Each dataset module —
`flower.models.dsprites`, `flower.models.rgbmnist`, `flower.models.spectra` — subclasses that
base. `rgbmnist` and `spectra` subclasses are now empty (`class LightningFlowMatching(
LightningFlowMatchingBase): pass`), `dsprites` still overrides with dSprites-specific
evaluation; configs target the per-dataset name directly (e.g.
`_target_: flower.models.rgbmnist.LightningFlowMatching`) so keep the subclass even when empty.
What stays genuinely per-dataset is the encoder/decoder stack: `VAE`/`BetaVAE`/`LightningVAE`/
`LightningBetaVAE` in `rgbmnist`/`dsprites`, and `PretrainedSpender` (a frozen `spender` encoder)
in `spectra`. This consolidation (issues #7/#9) has landed on the current branch; on `main` the
flow classes may still be duplicated per file.

**Encoder output contract.** Encoders return a dict `{z, mu, logvar}` (mu/logvar optional — a
non-VAE encoder like `PretrainedSpender.encode` returns just `{z: ...}`). The flow model consumes
`z`; don't assume a bare tensor. `n_layers` for the velocity field is set in each experiment's
`model.yaml`, not hardcoded.

**`flower.training.lightning_loaders` was removed** (dead code, unreferenced by any config) —
if you see it mentioned in an old plan, PR, or stale doc, it no longer exists; the real Lightning
modules live in the per-dataset `flower.models.*` files above.

**Data flow:** `flower.data.modules.FlowerDataset`/`FlowerDataLoader` wrap the raw per-source
`Dataset` classes (`flower.data.dsprites.dSprites`, `flower.data.rgbmnist.RGBMNIST`,
`flower.data.sdss.SDSS`) and attach a `y_catalog` — a dict describing each conditioning variable
(name, size, whether continuous, which variables to drop). The catalog shape drives conditional
dimensionality throughout the models (`get_conditional_len`, `get_no_of_continuous_variables` in
`flower.models.modules`), so when adding a dataset, the catalog contract matters more than the
Dataset implementation itself.

**ICA baselines are deliberately outside the Hydra pipeline.** `flower.models.ivae` (a port of
Khemakhem et al.'s reference iVAE: xtanh MLPs, fixed `decoder_var`, conditional prior
`p(s|u)`) and `flower.evaluation.ica` (MCC via Hungarian matching, the synthetic nonlinear-ICA
generator, and the three residual constructions — `conditional_prior_residual`,
`conditional_mean_residual`, `regression_residual`, plus `drop_top_k_dependent`) exist as
comparison baselines for Flower, not as part of the main model. They operate on flat
*pretrained embeddings*, so they have no experiment config; they're driven by the standalone
scripts in `examples/mnist/`, `examples/spectra/` and `examples/2d_gaussians/`
(`ivae_sweep.py`, `ivae_benchmark.py`,
`flower_cond_eval.py`, `plot_*frontier.py`), each writing `results.csv` / `summary.txt` /
`tradeoff.png` into a `*_results/` directory. `examples/ICA_HYPERPARAMETERS.md` is the
authoritative record of every hyperparameter used in those runs — update it when you change
one, and read it before reproducing or comparing numbers. `flower.models.ivae` still honours
the encoder contract (`encode` returns `{z, mu, logvar}`).

**`examples/2d_gaussians/` is a self-contained toy pipeline, not a Flower experiment.** Its
`data.py`/`model.py`/`train.py` import `flow_matching` and each other directly and never touch
the `flower` package, so its scripts must be run from that directory. Treat it as an
illustrative sandbox; changes to `flower.models` won't propagate to it.

**Pipeline stages** (`flower.training`, `flower.inference`, `flower.evaluation`,
`flower.explainability`, `flower.outliers`): data prep → train/test → embed → stat tests /
anomaly detection / visualization. `train.py` and `embed.py` are the two Hydra `@hydra.main`
entrypoints; both instantiate `cfg.data.loader`, `cfg.lightning_loader`, and `cfg.trainer` via
`hydra.utils.instantiate` and fail loudly (wrapped in explicit try/except that re-raise with
context) if any stage can't be built — preserve that pattern rather than silently swallowing
instantiation errors.

## Testing strategy

Tests live in `tests/`, one file per `flower` module plus `tests/test_integration.py`. Two
tiers:

1. **Unit tests** (`test_vae.py`, `test_flow_matching.py`, `test_models_modules.py`,
   `test_data_modules.py`, `test_augmentations.py`, `test_evaluation_metrics.py`,
   `test_inference_modules.py`, `test_utilities.py`) — construct models/data directly with
   small dummy tensors and fixtures from `tests/conftest.py` (dummy catalogs, dummy batches,
   minimal Hydra-style config dicts). No real datasets, no GPU, no checkpoints.
2. **Integration tests** (`test_integration.py`) — load real Hydra configs from `src/conf/`
   via `initialize_config_dir`/`compose`, override to CPU/tiny batch sizes, and instantiate the
   actual `lightning_loader` graph end-to-end (including a couple of mini training-step tests).
   This is what catches config/code drift (e.g. a `_target_` pointing at a class that no longer
   exists) that unit tests can't.
3. **`slow`-marked tests** (`test_ivae.py`, `test_ica_gaussians.py`) — the identifiability
   checks that genuinely train an iVAE on synthetic nonlinear-ICA data and assert MCC beats a
   linear baseline. Everything else in those files is fast; the marker is declared in
   `pyproject.toml`.

`pyproject.toml` sets `filterwarnings = ["error", ...]`, so an unfiltered warning fails the
test. When new library warnings appear, add a targeted ignore there rather than loosening the
rule.

When adding a new experiment config or renaming a model class, prefer extending the integration
layer over adding another isolated unit test — the point (see issue #2) is behavior coverage
per pipeline stage without excess fragmentation, plus config verification, not one test per
function.

## Commit conventions

Every commit is prefixed per `CONTRIBUTING.md`: `[Feature]`, `[Fix]`, `[Hotfix]`, `[Refactor]`,
`[Style]`, `[Docs]`, `[Test]`, `[Chore]`, `[Perf]`, `[Break]`, and situationally `[Revert]`,
`[WIP]`, `[Config]`, `[Rename]`. Pick the prefix that matches the change that matters most (a
commit touching both `[Feature]` and `[Style]` is `[Feature]`); skip the prefix entirely for
merge commits or trivial fixes.

## Other docs worth knowing

- `src/flower/README.md` — full anatomy of the Hydra config tree and on-disk output layout.
- `examples/ICA_HYPERPARAMETERS.md` — every hyperparameter for the ICA/iVAE baseline runs.
- `DEBUGGING_METHODOLOGY.md` — checklist for silent hangs / no-traceback bugs in the
  Hydra + Lightning + submitit stack (includes the `faulthandler` trick for getting a stack
  trace out of a stuck job without `sudo`/`py-spy`).
- `experiment_designs/` — design docs for planned experiments, one directory per proposal.
- `tests/README.md` and `.github/workflows/README.md` — notes on the test tiers and CI jobs.

`papers/`, `review/`, `solutions/`, `data/` and the `.gitkeep`-only `analysis/`, `docs/` are
local scratch/working dirs — untracked but *not* gitignored, so don't `git add -A` them in.
