import os
import argparse
import numpy as np
from datasets import load_dataset
from sklearn.manifold import TSNE
from dotenv import load_dotenv

# 1. Setup Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--key", type=str, choices=["vae", "uncond", "cond"], required=True)
args = parser.parse_args()

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

# 2. Data Loading
data_files = {
    #"train": f"{DATA_ROOT}/rgbmnist/rgbmnist_Flow_cond_prior/embeddings/7518770_0/train/*.parquet",
    "test": f"{DATA_ROOT}/rgbmnist/rgbmnist_Flow_cond_prior/embeddings/7518770_0/test/*.parquet"
}
w1 = load_dataset("parquet", data_files=data_files)

# Mapping the key to the specific dataset column
key_map = {
    "vae": "orig",
    "uncond": "uncond",
    "cond": "cond"
}

# 3. Setup t-SNE
tsne_params = {
    "n_components": 2,
    "perplexity": 1000,
    "max_iter": 5000,
    "random_state": 42,
    "learning_rate": "auto",
    "init": "pca",
}

# 4. Run t-SNE on the specific slice
data_slice = np.array(w1["test"][key_map[args.key]])
tsne = TSNE(**tsne_params)
result = tsne.fit_transform(data_slice)

# 5. Save individually (to avoid write-collisions)
np.save(f"./tsne_embeddings/tsne_result_test_{args.key}.npy", result)
print(f"Finished processing {args.key}")
