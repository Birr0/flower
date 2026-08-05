# Testing

## What we test, and why it is shaped this way

The suite is organised by **pipeline stage and behaviour**, not by source file. There is no
ambition to reach one test per function: a test earns its place by pinning behaviour that
something downstream depends on, and a suite of narrow unit tests would fragment without
catching the failures this project actually has.

The failure mode that motivates the shape is **configuration drift**. Flower is driven by
Hydra composition rather than code — an experiment is a tree of YAML under `src/conf/`, and
a `_target_` string is a dotted path resolved at runtime. Nothing in a unit test reads those
files. So a class can be renamed or relocated, every unit test still passes, and the
experiment is dead on the next run. That has happened more than once. Config is therefore
treated as code and given its own verification tier.

The second principle is that **real data is opt-in and never default**. The suite must run
on a fresh clone with no `DATA_ROOT`, no checkpoints, no GPU and no network. Anything
needing those is marked and excluded by default.

## Tiers

| Tier | Files | Needs | Default |
|---|---|---|---|
| **Unit** | `test_vae.py`, `test_flow_matching.py`, `test_models_modules.py`, `test_data_modules.py`, `test_augmentations.py`, `test_evaluation_metrics.py`, `test_evaluation_dependence.py`, `test_inference_modules.py`, `test_utilities.py` | nothing | runs |
| **Integration** | `test_integration.py` | nothing | runs |
| **Slow** | `test_ivae.py`, `test_ica_gaussians.py` | CPU time | runs |
| **Smoke** | `test_smoke.py` | `DATA_ROOT`, populated `local_data/` | **excluded** |

**Unit** — models and data built directly from small dummy tensors, using the fixtures in
`conftest.py` (dummy catalogs, dummy batches, minimal Hydra-style config dicts). No real
datasets, no GPU, no checkpoints.

**Integration** — loads the *real* configs from `src/conf/` via
`initialize_config_dir`/`compose`, overrides to CPU and tiny batch sizes, and instantiates
the actual `lightning_loader` graph, including a couple of mini training steps. This is the
tier that catches config/code drift, and where new coverage belongs when an experiment
config or a model class changes.

Config verification lives here as its own classes. `TestDataConfigTargets` and
`TestExperimentConfigTargets` assert that every `_target_` under `src/conf/data/` and
`src/conf/experiment/` imports, and that each `lightning_loader._target_` really names a
`LightningModule` subclass — resolving the dotted path is not enough, since a target that
still imports but no longer names a trainable module would pass and then fail at
`trainer.fit`.

**Slow** — the identifiability checks that genuinely train an iVAE on synthetic
nonlinear-ICA data and assert MCC beats a linear baseline. Everything else in those files
is fast; only the training tests carry the marker.

**Smoke** — end-to-end training against real data and checkpoints. Excluded by default via
`-m "not smoke"` in `addopts`, because it needs a populated data directory.

## Two settings that will bite you

`filterwarnings = ["error", ...]` — an unfiltered warning **fails** the test. When a library
starts emitting a new one, add a targeted ignore to `pyproject.toml` rather than loosening
the rule.

`--strict-markers` — an unregistered marker is an error, not a warning. Both `slow` and
`smoke` must stay registered in `pyproject.toml`; dropping either breaks the tests using it.

## Running

```bash
# Everything except smoke (the default)
pytest tests/ -v

# Skip the training-based identifiability tests too
pytest tests/ -v -m "not slow"

# Smoke tests only — needs DATA_ROOT and local_data populated
pytest tests/ -m smoke

# One file / class / test
pytest tests/test_flow_matching.py -v
pytest tests/test_integration.py::TestDspritesFlowConfig -v
pytest tests/test_models_modules.py::TestWrappedModel::test_cfg_scale_gt1_guidance -v

# Coverage
pytest tests/ -v --cov=src/flower --cov-report=term-missing
```

Nothing is on `PATH` unless the venv is active — prefix with `uv run` if you have not
sourced `.venv/bin/activate`.

Note that `pre-commit` runs the **full** suite on every commit, slow markers included, so
commits take minutes.

## Adding tests

When you add an experiment config or rename a model class, extend the **integration** tier
rather than adding an isolated unit test. The point is behaviour coverage per pipeline
stage plus config verification, not one test per function.
