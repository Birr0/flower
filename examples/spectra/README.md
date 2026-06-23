# Galaxy Spectra

Reproduces the SDSS galaxy spectra experiments: training Flower's conditional flow matching
model on SPENDER-encoded spectra and comparing against the paper's benchmarks.

## Data installation

Follow the Multimodal Universe instructions on downloading the SDSS I/II spectra data, found here: https://github.com/MultimodalUniverse/MultimodalUniverse. 

Also download the GSWLC from here: https://salims.pages.iu.edu/gswlc/

We have positionally matched the GSWLC and SDSS spectra here to one arcsecond. The catalogues can be linked using astropy positional matching tools: https://learn.astropy.org/tutorials/4_Coordinates-Crossmatch.html. We use an internal dataset in this code that has already linked the raw spectra and catalog; we cannot make this available anonymously but it will be released along with the paper.

## Training
### Conditional Flow Matching Model
Run the following command to train the flow models

```
srun python train.py -cn "experiment/spender_I_flow/train" hydra/launcher=hpc
```

Replace spender_I_flow with spender_II_flow to train the spender II version of the base model.

If running locally, remove srun and use `hydra/launcher=local`.

### Embed the SDSS data 

Any of the models trained above can be used to embed the cMNIST data. Change into the inference folder

```
cd /path/to/flower/src/flower/inference
```

Then run the following

```
srun python embed.py -cn "experiment/{experiment_name}/embed" hydra/launcher=hpc
```

Note, if one sweep is ran, ensure that you prepend a '_0' to the end of the model number (i.e. 7534343.ckpt should become 7534343_0.ckpt.). This is a byproduct of the sweep system used and will be fixed for future releases.

### Recreating results from the paper

In order to recreate the results from the paper, in the following python files, replace the mapping to the trained flow models with your model number:

```
SPENDER_MAP = {
    "spender_I": "spender_I_flow_v2/embeddings/7526202_0",
    "spender_II": "spender_II_flow_v2/embeddings/7527549_0"
}
```

Run the following batch scripts:

```
sbatch submit_flower.sh
```

```
sbatch submit_resid.sh
```

```
sbatch submit_umap.sh
```

Then to recreate the results in the Table 3, run benchmark.ipynb.
