"""
Single entry point for loading split data and computing class weights.
All models import from here — never read CSVs directly.

    from src.data import load_splits, class_weights
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

SPLIT_DIR = Path(__file__).parent.parent / "data" / "splits"
LABEL_MAP = {"minimum": 0, "mild": 1, "moderate": 2, "severe": 3}
NUM_CLASSES = 4


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) loaded from data/splits/."""
    missing = [f for f in ("train.csv", "val.csv", "test.csv")
               if not (SPLIT_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Split files not found: {missing}\n"
            f"Run `python src/splits.py` first, then share the files in data/splits/ "
            f"with your teammates before they start training."
        )

    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val   = pd.read_csv(SPLIT_DIR / "val.csv")
    test  = pd.read_csv(SPLIT_DIR / "test.csv")
    return train, val, test


def class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute balanced class weights from the training set.
    Pass the result to nn.CrossEntropyLoss(weight=...) to counter class imbalance.

        weights = class_weights(train_df)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    """
    labels  = np.array([0, 1, 2, 3])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=labels,
        y=train_df["label_id"].values,
    )
    return torch.tensor(weights, dtype=torch.float)
