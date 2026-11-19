"""Download the public Kaggle pneumonia dataset.

Run after configuring Kaggle credentials as described in README.md.
"""

from pathlib import Path

import kagglehub


destination = Path("data")
destination.mkdir(exist_ok=True)
downloaded = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print(f"Dataset downloaded to: {downloaded}")
print("Copy or link its train, val, and test folders into data/chest_xray/.")
