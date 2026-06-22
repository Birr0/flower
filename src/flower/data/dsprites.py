import os
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset, load_from_disk, DatasetDict
from torch.utils.data import Dataset
import torch 
from torchvision.transforms import functional as F
import math 

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")

class dSprites(Dataset):
    def __init__(self, x_ds, y_catalog, split):
        self.x_ds = x_ds
        self.y_catalog = y_catalog

        self.data = load_from_disk(
            f"{DATA_ROOT}/dsprites-dataset"
        )[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # k: torch.as_tensor(data[k]).unsqueeze(0) for k in self.y_catalog.variables.keys()

        data = self.data[idx]
        pos_x = (data["value_x_position"] * 2) - 1
        pos_y = (data["value_y_position"] * 2) - 1
        
        orient = data["label_orientation"] / 40.0

        angle_rad = (data["label_orientation"] / 40.0) * 2 * math.pi

        # 2. Represent as 2D coordinates
        orient_sin = math.sin(angle_rad)
        orient_cos = math.cos(angle_rad)
        
        # Scale: [0.5, 1.0] -> [-1, 1]
        scale = (data["value_scale"] - 0.75) / 0.25

        return {
            "X": F.to_tensor(data["image"]),
            "catalog": {
                "label_shape": torch.tensor(data["label_shape"], dtype=torch.float32).unsqueeze(0),
                "value_x_position": torch.tensor(pos_x, dtype=torch.float32).unsqueeze(0),
                "value_y_position": torch.tensor(pos_y, dtype=torch.float32).unsqueeze(0),
                "value_orientation_sin": torch.tensor(orient_sin, dtype=torch.float32).unsqueeze(0),
                "value_orientation_cos": torch.tensor(orient_cos, dtype=torch.float32).unsqueeze(0),
                "value_scale": torch.tensor(scale, dtype=torch.float32).unsqueeze(0),
            }
        }

def prep_dsprites_subsample(
    n_train=100000,
    n_val=25000,
    n_test=25000,
    seed=42
):
    n_total = n_train + n_val + n_test
    # 1. Load the full dataset 
    full_ds = load_dataset("eurecom-ds/dsprites", split="train")
    # 2. Shuffle and take the 150,000 subset
    subset_ds = full_ds.shuffle(seed=seed).select(range(n_total))

    # 3. Create the 100k Train / 50k Temp split
    train_tmp = subset_ds.train_test_split(test_size=n_total-n_train, seed=seed)

    # 4. Split the 50k Temp into 25k Val / 25k Test
    val_test = train_tmp['test'].train_test_split(test_size=n_val/(n_val+n_test), seed=seed)

    # Final Dictionary of Dataset objects
    ds_splits = {
        "train": train_tmp['train'],
        "val": val_test['train'],
        "test": val_test['test']
    }

    ds_splits_dict = DatasetDict(ds_splits)
    # 2. Save to a folder
    ds_splits_dict.save_to_disk(f"{DATA_ROOT}/dsprites-dataset")

if __name__ == "__main__":
    #prep_dsprites_subsample()
    if not Path(f"{DATA_ROOT}/dsprites-dataset").exists():
        print("Data not found. Running prep_dsprites_subsample...")
        prep_dsprites_subsample()

    # 2. Test the Splits
    for split_name in ["train", "val", "test"]:
        print(f"\n{'='*10} Testing Split: {split_name.upper()} {'='*10}")
        
        # Instantiate your dSprites class
        # (Passing None for x_ds/y_catalog since your current __init__ doesn't use them yet)
        ds = dSprites(x_ds=None, y_catalog=None, split=split_name)
        
        print(f"Total samples: {len(ds)}")
        
        # Get the first item
        sample = ds[0]
        
        print(f"Keys returned: {list(sample.keys())}")
        
        # Check the Image
        img = sample['X']
        print(f"Image Type: {type(img)}") 
        # Before your preprocessing logic, this will be <PIL.PngImagePlugin.PngImageFile>
        
        if hasattr(img, 'size'):
            print(f"Image Dimensions: {img.size}")

        # Check a few labels (using the column names from eurecom-ds/dsprites)
        print(f"Shape Label: {sample["catalog"]['label_shape']}")
        attrs = ["value_x_position", "value_y_position", 
            "value_orientation", "value_scale"
        ]
        for attr in attrs:
            print(sample["catalog"][attr].shape)
