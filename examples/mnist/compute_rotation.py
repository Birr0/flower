"""Estimate per-sample digit rotation from RGB-MNIST VAE embeddings.

Ported from ``rotation.ipynb``: reconstruct each image from its ``orig`` VAE
latent (``projection_up -> decoder``) and measure the principal-axis orientation
from second-order central image moments,
``theta = 0.5 * atan2(2 * mu_11, mu_20 - mu_02)`` (degrees, in [-90, 90]).

Rotation is a random augmentation independent of the digit and is *not* in the
conditioning set, so it serves as a second **preservation** target (alongside
colour ``b``) for the ICA baselines — the digit-removal residual should retain it.

Caches ``{train,test}_rotation_aligned.csv`` in ``--outdir`` (default this dir),
row-aligned to the parquet embedding order used by ``ivae_sweep.py``.

Run from this directory (needs ``DATA_ROOT``):

    python compute_rotation.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from skimage.measure import moments, moments_central
from tqdm import tqdm

from flower.models.rgbmnist import VAE

DEFAULT_EMBED_SUBPATH = "rgbmnist/rgbmnist_Flow_cond_prior/embeddings/7518770_0"
DEFAULT_VAE_CKPT = "rgbmnist/rgbmnist_VAE/ckpts/5252446.ckpt"


def load_vae(ckpt_path, hidden_dim=64):
    vae = VAE(hidden_dim=hidden_dim)
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck.get("state_dict", ck)
    state = {k.replace("vae.", "", 1): v for k, v in state.items()}
    info = vae.load_state_dict(state, strict=False)
    print(
        f"VAE loaded from {os.path.basename(ckpt_path)}: "
        f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}"
    )
    vae.eval()
    return vae


def _angle_deg(img):
    """Principal-axis orientation (deg) of an image via central moments."""
    m = moments(img, order=1)
    mass = m[0, 0]
    if mass <= 0:
        return 0.0
    centroid = (m[1, 0] / mass, m[0, 1] / mass)
    mu = moments_central(img, center=centroid, order=2)
    return float(0.5 * np.degrees(np.arctan2(2 * mu[1, 1], mu[2, 0] - mu[0, 2])))


def rotation_for_latents(vae, z, batch_size=256):
    z = torch.as_tensor(np.asarray(z), dtype=torch.float32)
    angles = []
    with torch.no_grad():
        for i in tqdm(range(0, len(z), batch_size)):
            z_up = vae.projection_up(z[i : i + batch_size])
            imgs = vae.decoder(z_up).cpu().numpy()
            for j in range(len(imgs)):
                img = imgs[j, 0] if imgs.ndim == 4 else imgs[j]
                angles.append(_angle_deg(img))
    return np.array(angles, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-subpath", type=str, default=DEFAULT_EMBED_SUBPATH)
    parser.add_argument("--vae-ckpt", type=str, default=DEFAULT_VAE_CKPT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    load_dotenv()
    root = os.getenv("DATA_ROOT")
    embed_path = f"{root}/{args.embed_subpath}"
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"{embed_path}/train/*.parquet",
            "test": f"{embed_path}/test/*.parquet",
        },
    )
    vae = load_vae(f"{root}/{args.vae_ckpt}")

    os.makedirs(args.outdir, exist_ok=True)
    for split in ("train", "test"):
        print(f"Reconstructing + measuring rotation for {split}...")
        angles = rotation_for_latents(vae, ds[split]["orig"])
        out = os.path.join(args.outdir, f"{split}_rotation_aligned.csv")
        pd.DataFrame({"Rotation_Deg": angles}).to_csv(out, index=False)
        print(
            f"  saved {out}  (n={len(angles)}, "
            f"range [{angles.min():.1f}, {angles.max():.1f}] deg)"
        )


if __name__ == "__main__":
    main()
