"""Drop state_dict keys from a Lightning checkpoint that no longer match the
current model architecture, and save the result under a new filename.

Written for `rgbmnist_VAE/ckpts/5252446.ckpt`, whose decoder has an extra
`block0` stage (1024ch) not present in any version of `Decoder` in
`flower.models.rgbmnist` -- the checkpoint predates the repo history. The
encoder and decoder block1-4 match the current architecture exactly, so
dropping the orphaned keys (rather than remapping them) lets the checkpoint
load with `strict=True` again.

Usage:
    python -m flower.models.strip_legacy_ckpt_keys \
        --src /path/to/5252446.ckpt \
        --dst /path/to/5252446_0.ckpt \
        --key-substring decoder.block0.
"""

import argparse

import torch


def strip_ckpt_keys(src: str, dst: str, key_substring: str) -> None:
    ckpt = torch.load(src, map_location="cpu")

    dropped = [k for k in ckpt["state_dict"] if key_substring in k]
    print(f"Dropping {len(dropped)} keys matching {key_substring!r}:")
    for k in dropped:
        print(f"  {k}")

    ckpt["state_dict"] = {
        k: v for k, v in ckpt["state_dict"].items() if k not in dropped
    }

    torch.save(ckpt, dst)
    print(f"Saved {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Path to the source checkpoint.")
    parser.add_argument(
        "--dst", required=True, help="Path to save the stripped checkpoint to."
    )
    parser.add_argument(
        "--key-substring",
        default="decoder.block0.",
        help="state_dict keys containing this substring are dropped.",
    )
    args = parser.parse_args()
    strip_ckpt_keys(args.src, args.dst, args.key_substring)


if __name__ == "__main__":
    main()
