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
import string
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def reduce_lengthening(text):
    # This regex looks for any character repeated 3 or more times
    # and replaces it with just 2 of that character.
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)



def preprocess_text(text: str):
    
    # 1303
    # Unescape HTML entities (converts &#x200B; to actual zero-width space)
    text = html.unescape(text)
    
    #107
    # Remove emoji's with text description
    text = emoji.demojize(text, delimiters=(':', ':'))
    
    #539
    # Compact lengthening of characters (e.g. "soooo" -> "soo")
    text = reduce_lengthening(text)
    
    #486
    # Remove any remaining mixed/repeating punctuation sequences (e.g. "!!!???" or "///\\\")    
    text = re.sub(r'[/\\=\-_*|?~^!\[\]]{6,}', ' ', text)
    
    # Remove format characters (zero-width spaces, etc) but preserve original text form
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cf')
    
    # Final whitespace cleanup: collapse multiple spaces and trim edges
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

depression_df = pd.read_csv(DATA_DIR / "Reddit_depression_dataset.csv")

depression_df['text'] = depression_df['text'].apply(preprocess_text)
depression_df.to_csv(DATA_DIR / "Reddit_depression_dataset_clean.csv", index=False)
print(f"Saved: {DATA_DIR / 'Reddit_depression_dataset_clean.csv'}")

