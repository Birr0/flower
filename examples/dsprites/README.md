# dSprites

Reproduces the dSprites experiments, training Flower on procedurally generated 2D shapes
with known, disentangled factors of variation (shape, scale, rotation, position).

## Data Access
The dsprites dataset can be accessed and downloaded using the following Hugging Face Repository:

```
ds = load_dataset("eurecom-ds/dsprites", split="train")
```

Or by running the prep_dsprites_subsample(...) function in src/flower/data/dsprites.py 

