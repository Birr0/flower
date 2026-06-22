# Flower

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