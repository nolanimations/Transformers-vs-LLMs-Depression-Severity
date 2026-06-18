"""
Dataset preprocessing pipeline for MentalBERT fine-tuning.
Combines DEPTWEET (Twitter) and Reddit depression datasets.

Outputs:
  - twitter_processed.csv   : cleaned Twitter data only
  - reddit_processed.csv    : cleaned Reddit data only
  - combined_dataset.csv    : merged, shuffled, balanced dataset
"""

import pandas as pd
import emoji
import re
import html
from pathlib import Path

HERE          = Path(__file__).parent
# Raw DEPTWEET CSV — not committed (data-use-agreement restricted, see .gitignore).
# Place it at data/deptweet_dataset.csv (shared with the team via Discord).
TWITTER_PATH  = HERE / "data" / "deptweet_dataset.csv"
REDDIT_PATH   = HERE / "data" / "Reddit_depression_dataset_clean.csv"
OUT_TWITTER   = HERE / "data" / "twitter_processed.csv"
OUT_REDDIT    = HERE / "data" / "reddit_processed.csv"
OUT_COMBINED  = HERE / "data" / "combined_dataset.csv"

# ── Label mapping ─────────────────────────────────────────────────────────────
# Unify "non-depressed" (Twitter) and "minimum" (Reddit) → "minimum"
# Final string labels kept alongside integer IDs for readability
LABEL_UNIFY = {"non-depressed": "minimum"}
LABEL_MAP   = {"minimum": 0, "mild": 1, "moderate": 2, "severe": 3}

# ── Text cleaning ─────────────────────────────────────────────────────────────
_STANDARD_EMOJI_NAMES = set(emoji.EMOJI_DATA.keys())   # set of ":name:" strings

def _is_standard_emoji_token(token: str) -> bool:
    """Return True if :token: is a recognised standard emoji name."""
    return token in _STANDARD_EMOJI_NAMES


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    text = text.replace("�", "")

    text = emoji.demojize(text)

    # Remove :custom_token: patterns that are NOT standard emoji names
    #    (e.g. :ezra:, :chef_kiss: was already converted above if it's real)
    def _remove_non_standard(m):
        token = m.group(0)          # includes the colons, e.g. ":ezra:"
        return token if _is_standard_emoji_token(token) else " "
    text = re.sub(r":[a-zA-Z0-9_\-]+:", _remove_non_standard, text)

    # Remove URLs (t.co, http, www)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Anonymise @mentions → @user
    text = re.sub(r"@\w+", "@user", text)

    # Collapse excess whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("Loading Twitter (DEPTWEET) dataset …")
tw = pd.read_csv(TWITTER_PATH)
print(f"  Raw shape: {tw.shape}")

tw = tw[["tweet", "label", "confidence_score"]].rename(columns={"tweet": "text"})

# drop low-confidence annotations.
# Distribution: min=0.643, Q1=0.670, median=1.0
# Threshold 0.67 removes the bottom ~2% while keeping all high-confidence rows.
CONFIDENCE_THRESHOLD = 0.67
before = len(tw)
tw = tw[tw["confidence_score"] >= CONFIDENCE_THRESHOLD].copy()
print(f"  After confidence filter (>={CONFIDENCE_THRESHOLD}): {len(tw)} rows removed {before - len(tw)}")
tw = tw.drop(columns=["confidence_score"])

# Unify labels
tw["label"] = tw["label"].replace(LABEL_UNIFY)

# Add source tag
tw["source"] = "twitter"

# Clean text
print("  Cleaning text …")
tw["text"] = tw["text"].apply(clean_text)

# Remove anything that became too short after cleaning (< 10 chars)
tw = tw[tw["text"].str.len() >= 10]

# Deduplicate
before = len(tw)
tw = tw.drop_duplicates(subset=["text"])
print(f"  Duplicates removed: {before - len(tw)}")

print(f"  Final Twitter shape: {tw.shape}")
print(f"  Label distribution:\n{tw['label'].value_counts()}\n")


# ── Load & clean Reddit ───────────────────────────────────────────────────────
print("Loading Reddit dataset …")
rd = pd.read_csv(REDDIT_PATH)
print(f"  Raw shape: {rd.shape}")

rd["source"] = "reddit"

print("  Cleaning text …")
rd["text"] = rd["text"].apply(clean_text)

rd = rd[rd["text"].str.len() >= 10]

before = len(rd)
rd = rd.drop_duplicates(subset=["text"])
print(f"  Duplicates removed: {before - len(rd)}")

print(f"  Final Reddit shape: {rd.shape}")
print(f"  Label distribution:\n{rd['label'].value_counts()}\n")


# ── Encode labels ─────────────────────────────────────────────────────────────
for df in (tw, rd):
    df["label_id"] = df["label"].map(LABEL_MAP)


# ── Save individual datasets ──────────────────────────────────────────────────
tw.to_csv(OUT_TWITTER, index=False)
print(f"Saved: {OUT_TWITTER}")

rd.to_csv(OUT_REDDIT, index=False)
print(f"Saved: {OUT_REDDIT}\n")


CAP_RATIO = 2.0   # minimum class will be at most 2× the mild class

combined_raw = pd.concat([tw, rd], ignore_index=True)

mild_count    = (combined_raw["label"] == "mild").sum()
minimum_cap   = int(mild_count * CAP_RATIO)
minimum_rows  = combined_raw[combined_raw["label"] == "minimum"]
other_rows    = combined_raw[combined_raw["label"] != "minimum"]

if len(minimum_rows) > minimum_cap:
    minimum_rows = minimum_rows.sample(n=minimum_cap, random_state=42)
    print(f"Undersampled 'minimum' class: {len(combined_raw[combined_raw['label']=='minimum'])} -> {minimum_cap}")

combined = pd.concat([minimum_rows, other_rows], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCombined dataset shape: {combined.shape}")
print("Label distribution after balancing:")
vc = combined["label"].value_counts()
for label, count in vc.items():
    pct = 100 * count / len(combined)
    print(f"  {label:<15} {count:>6}  ({pct:.1f}%)")

print(f"\nSource split:")
print(combined["source"].value_counts().to_string())

combined.to_csv(OUT_COMBINED, index=False)
print(f"\nSaved: {OUT_COMBINED}")
print("\nDone.")
