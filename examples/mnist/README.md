# cMNIST

## Data download 

Ensure MNIST has been downloaded to the DATA_ROOT directory specified in the .env file.
MNIST can be downloaded using TorchVision: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html.

## Training

First, move to the training directory of the repository:

```
cd /path/to/flower/src/flower/training
```

### VAE
The VAE for cMNIST can be trained using the following command

```
srun python train.py -cn "experiment/rgbmnist_VAE/train" hydra/launcher=hpc
```

If running locally, remove srun and use `hydra/launcher=local`.

### Conditional Flow Matching Model
The conditional flow matching model can be trained by first specifying the directory of the VAE in the meta.yaml file
of the experiment config. Then run 

```
srun python train.py -cn "experiment/rgbmnist_Flow/train" hydra/launcher=hpc
```

If running locally, remove srun and use `hydra/launcher=local`.

### Beta ablation

Similarly, the beta ablation can be run using
```
srun python train.py -cn "experiment/rgbmnist_Flow_beta_ablation/train" hydra/launcher=hpc
```

## Embed the MNIST data 

Any of the models trained above can be used to embed the cMNIST data. Change into the inference folder

```
cd /path/to/flower/src/flower/inference
```

Then run the following

```
srun python embed.py -cn "experiment/{experiment_name}/embed" hydra/launcher=hpc
```

Note, if one sweep is ran, ensure that you prepend a '_0' to the end of the model number (i.e. 7534343.ckpt should become 7534343_0.ckpt.) 
This is a byproduct of the sweep system used and will be fixed for future releases.

## Results from paper

In order to re-create the results from the paper, note the model number and experiment name. After training and embedding, the embedded data can be loaded with the following:
```
from datasets import load_dataset
ds = load_dataset(
    "parquet",
    data_files={
        "train": f"{DATA_ROOT}/rgbmnist/{experiment_name}/embeddings/{model_number}/train/*.parquet",
        "test": f"{DATA_ROOT}/rgbmnist/{experiment_name}/embeddings/{model_number}/test/*.parquet"
    }
)
```

A handy tsne embedding script is contained in `tsne_embed.py`. If using SLURM, add the venv filepath to submit_tsne.sh and run

```
sbatch submit_tsne.sh
```

- The benchmarks found in Table 2 can be recreated by running benchmark.ipynb.

- The thumbnail embedding plot in fig 6.a can be recreated by running tsne_image_plots.ipynb 

- The rotation plot in fig 6.b and found in the appendices can be recreated by running rotation.ipynb.

- The beta ablation plots can be recreated by running beta_ablation.ipynb.

- The red and green control ablation study in the appendix can be recreated using r_g_ablation.ipynb 


- The similarity metrics and style transfer plots can be recreated by running style_transfer.ipynb. Note: LPIPS can conflict with more recent versions of torch.

