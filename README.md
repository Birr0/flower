# 🌸 Flower

Welcome to **Flower**, a framework to train, load, inspect, explain, and infer conditional
generative models so you can explore what your data — and your models — might be missing.

Flower combines a VAE encoder with a conditional flow matching (CFM) model over the learned
latent space, letting you condition generation on known factors of variation and study the
residual structure that those factors don't explain. The framework is built around
[PyTorch Lightning](https://lightning.ai/) for training and [Hydra](https://hydra.cc/) for
experiment configuration, so every experiment is reproducible from a config file.

## What's in this repository

- `src/flower/` — the core library: data modules, models (VAE + CFM), training, inference,
  evaluation, explainability, and outlier-detection code. See
  [`src/flower/README.md`](src/flower/README.md) for how experiments are configured and run.
- `src/conf/` — Hydra configs for datasets, models, training, and experiments.
- `examples/` — worked examples and paper-reproduction notebooks for each dataset:
  - [`2d_gaussians`](examples/2d_gaussians/README.md) — toy 2D Gaussian mixtures
  - [`mnist`](examples/mnist/README.md) — coloured MNIST (cMNIST)
  - [`dsprites`](examples/dsprites/README.md) — dSprites
  - [`spectra`](examples/spectra/README.md) — SDSS galaxy spectra
- `analysis/`, `data/`, `docs/` — scratch space for analysis notebooks, local datasets, and
  additional documentation.

## Getting started 

### Installation
Packages are installable using uv.

```
uv sync
source .venv/bin/activate
```

### Environment variables
Fill out the .env with personal data directories and logging information. If running models, ensure that the wandb information is filled out for model logging.

## Experiments
We use hydra configs to coordinate experiments. The parameters for each experiment can be found in the src/conf/experiments folder. 

Instructions on reproducing the experiments can be found in the examples folder in this directory.

### Train
To train a model from any experiment run the following code:

```
cd src/flower/train

(srun) python train.py -cn "experiment/{experiment_name}/train" hydra/launcher={compute_config}
```

If running on a SLURM-based cluster append srun and use the htc config. If running locally, do not use srun and set hydra/launcher=local.

### Embedding
To embed the a model after it has been trained, run
```
cd src/flower/embed

(srun) python embed.py -cn "experiment/{experiment_name}/embed" hydra/launcher={compute_config}
```

If running on a SLURM-based cluster append srun and use the htc config. If running locally, do not use srun and set hydra/launcher=local.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setting up a
development environment and pre-commit hooks.