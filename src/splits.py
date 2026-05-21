"""
Stratified 80/10/10 train/val/test split on combined_dataset.csv.
Stratifies on (label_id x source) so both class and platform balance are preserved.
Saves to data/splits/{train,val,test}.csv.

Run once — do not re-run after splits are shared with teammates.
The test set is frozen from this point forward; no model tuning against it.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

HERE      = Path(__file__).parent.parent
DATA_DIR  = HERE / "data"
SPLIT_DIR = DATA_DIR / "splits"

COMBINED  = DATA_DIR / "combined_dataset.csv"
SEED      = 42


def make_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(COMBINED)
    print(f"Loaded {len(df):,} rows from {COMBINED.name}")

    # Joint stratify key: label_id x source (8 strata)
    df["_stratum"] = df["label_id"].astype(str) + "_" + df["source"]

    # Step 1: carve out 20% for val+test together (stratified)
    train, temp = train_test_split(
        df, test_size=0.20, stratify=df["_stratum"], random_state=SEED
    )
    # Step 2: split the 20% evenly into val and test (stratified)
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["_stratum"], random_state=SEED
    )

    # Drop the helper column before saving
    for split in (train, val, test):
        split.drop(columns=["_stratum"], inplace=True)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(train, val, test) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLIT_DIR / "train.csv", index=False)
    val.to_csv(SPLIT_DIR   / "val.csv",   index=False)
    test.to_csv(SPLIT_DIR  / "test.csv",  index=False)
    print(f"Saved splits to {SPLIT_DIR}/")


def print_balance(name: str, df: pd.DataFrame) -> None:
    total = len(df)
    print(f"\n{name} ({total:,} rows)")
    print(f"  {'label':<12} {'n':>6}  {'%':>6}   {'twitter':>8}  {'reddit':>7}")
    for label_id in sorted(df["label_id"].unique()):
        subset = df[df["label_id"] == label_id]
        label  = subset["label"].iloc[0]
        n      = len(subset)
        tw     = (subset["source"] == "twitter").sum()
        rd     = (subset["source"] == "reddit").sum()
        print(f"  {label:<12} {n:>6}  {100*n/total:>5.1f}%   {tw:>8}  {rd:>7}")


if __name__ == "__main__":
    train, val, test = make_splits()

    print_balance("Train", train)
    print_balance("Val",   val)
    print_balance("Test",  test)

    # Sanity check: no overlap between splits (compare on text content)
    train_texts = set(train["text"])
    val_texts   = set(val["text"])
    test_texts  = set(test["text"])
    assert len(train_texts & val_texts) == 0,  "Train/val overlap!"
    assert len(train_texts & test_texts) == 0, "Train/test overlap!"
    assert len(val_texts & test_texts) == 0,   "Val/test overlap!"
    print("\nSanity checks passed — no overlap between splits.")

    save_splits(train, val, test)
    print("\nDone. Freeze these files — do not re-generate after sharing with teammates.")
