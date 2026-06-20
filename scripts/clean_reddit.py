"""
Stage 0 (Reddit only): basic text cleanup of the raw Reddit depression-severity
dataset, run once before build_combined_dataset.py.

Input  : data/Reddit_depression_dataset.csv
Output : data/Reddit_depression_dataset_clean.csv
"""

# Data Handling
import pandas as pd
import html
import unicodedata
import emoji
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def preprocess_text(text: str):
    text = html.unescape(text)

    text = emoji.demojize(text, delimiters=(':', ':'))

    text = reduce_lengthening(text)
  
    text = re.sub(r'[/\\=\-_*|?~^!\[\]]{6,}', ' ', text)

    text = ''.join(c for c in text if unicodedata.category(c) != 'Cf')

    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

depression_df = pd.read_csv(DATA_DIR / "Reddit_depression_dataset.csv")

depression_df['text'] = depression_df['text'].apply(preprocess_text)
depression_df.to_csv(DATA_DIR / "Reddit_depression_dataset_clean.csv", index=False)
print(f"Saved: {DATA_DIR / 'Reddit_depression_dataset_clean.csv'}")
