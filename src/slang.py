"""
Per-post informality score for H2 (slang/register sensitivity analysis).

Informality score = (# emoji tokens + # all-caps tokens + # repeated-char tokens
                     [+ # OOV-vs-GloVe tokens if GloVe is available])
                    / token_count

Each token is one whitespace-separated chunk — this deliberately keeps emoji,
punctuation, and slang intact rather than sub-word tokenising.

Usage
-----
    from src.slang import add_informality_score

    # Without GloVe (3 components):
    test_df = add_informality_score(test_df)

    # With GloVe (4 components, richer signal):
    test_df = add_informality_score(test_df, glove_path="data/glove.6B.300d.txt")

The function adds two columns to the DataFrame:
    informality_score  float  0.0–∞ (usually 0.0–0.5 in practice)
    informality_bin    str    "low" / "mid" / "high"  (equal-size tertiles)

Pass include_components=True to also get individual count columns for deeper
analysis in notebook 09.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)

# ── Regex patterns ─────────────────────────────────────────────────────────────

# Colon-style emoji produced by our preprocessor  (:heart:, :sob:, :cry_face:)
_EMOJI_COLONS = re.compile(r':[a-z0-9_]+:')

# Unicode emoji blocks (covers most common emoji ranges)
_UNICODE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emotes
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # regional flags
    "\U00002600-\U000026FF"   # misc symbols
    "\U00002700-\U000027BF"   # dingbats
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"   # chess / symbols extended-A
    "]+",
    flags=re.UNICODE,
)

# Any character repeated 3+ times in a row  (soooo, nooo, !!!, ???, pleaaase)
_REPEATED_CHARS = re.compile(r'(.)\1{2,}')

# All-caps word: 2 or more uppercase letters, no lowercase  (HELP, IM, WTF)
_ALL_CAPS = re.compile(r'^[A-Z]{2,}$')


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Whitespace split
    Preserves emoji, punctuation, and slang tokens as-is.
    """
    return str(text).split()


def _is_emoji(token: str) -> bool:
    return bool(_EMOJI_COLONS.search(token) or _UNICODE_EMOJI.search(token))


def _score_tokens(tokens: list[str], glove_vocab: set | None) -> dict:
    """Return raw counts for each informality component."""
    n = max(len(tokens), 1)

    n_emoji    = sum(1 for t in tokens if _is_emoji(t))
    n_allcaps  = sum(1 for t in tokens if _ALL_CAPS.match(t))
    n_repeated = sum(1 for t in tokens if _REPEATED_CHARS.search(t))
    n_oov      = (
        sum(1 for t in tokens if t.lower() not in glove_vocab)
        if glove_vocab is not None else None
    )

    informal = n_emoji + n_allcaps + n_repeated + (n_oov if n_oov is not None else 0)

    return {
        "informality_score": informal / n,
        "n_tokens":   n,
        "n_emoji":    n_emoji,
        "n_allcaps":  n_allcaps,
        "n_repeated": n_repeated,
        "n_oov":      n_oov,   # None when GloVe not loaded
    }


def _load_glove_vocab(glove_path: str | Path) -> set[str] | None:
    """
    Load GloVe vocabulary.
    Returns None if the file doesn't exist so callers can degrade gracefully.
    """
    glove_path = Path(glove_path)
    if not glove_path.exists():
        log.warning(
            f"GloVe file not found at {glove_path} — "
            f"OOV component will be skipped. "
            f"Score will be based on 3 components instead of 4."
        )
        return None

    log.info(f"Loading GloVe vocabulary from {glove_path} ...")
    vocab: set[str] = set()
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            word = line.split(" ", 1)[0]
            vocab.add(word)
    log.info(f"GloVe vocabulary size: {len(vocab):,}")
    return vocab


# ── Main API ───────────────────────────────────────────────────────────────────

def add_informality_score(
    df: pd.DataFrame,
    glove_path: str | Path | None = None,
    include_components: bool = False,
) -> pd.DataFrame:
    """
    Add informality_score and informality_bin columns to df.

    Parameters
    ----------
    df                 : DataFrame with a 'text' column
    glove_path         : path to glove.6B.300d.txt (optional).
                         If None, defaults to data/glove.6B.300d.txt relative
                         to the repo root. If the file doesn't exist, the OOV
                         component is silently dropped.
    include_components : if True, also add per-component count columns:
                         n_emoji, n_allcaps, n_repeated, n_oov, n_tokens

    Returns
    -------
    Copy of df with new columns added (original df is not modified).
    """
    # Resolve GloVe path
    if glove_path is None:
        repo_root  = Path(__file__).parent.parent
        glove_path = repo_root / "data" / "glove.6B.300d.txt"

    glove_vocab = _load_glove_vocab(glove_path)

    log.info(f"Computing informality scores for {len(df):,} posts ...")

    scores = [
        _score_tokens(_tokenize(text), glove_vocab)
        for text in df["text"]
    ]

    result = df.copy()
    result["informality_score"] = [s["informality_score"] for s in scores]

    if include_components:
        result["n_tokens"]   = [s["n_tokens"]   for s in scores]
        result["n_emoji"]    = [s["n_emoji"]    for s in scores]
        result["n_allcaps"]  = [s["n_allcaps"]  for s in scores]
        result["n_repeated"] = [s["n_repeated"] for s in scores]
        result["n_oov"]      = [s["n_oov"]      for s in scores]  # None if no GloVe

    # Assign tertile bins (equal-count, not equal-width)
    result["informality_bin"] = pd.qcut(
        result["informality_score"],
        q=3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    ).astype(str)

    n_low  = (result["informality_bin"] == "low").sum()
    n_mid  = (result["informality_bin"] == "mid").sum()
    n_high = (result["informality_bin"] == "high").sum()
    log.info(
        f"Informality bins — low: {n_low}  mid: {n_mid}  high: {n_high}  "
        f"(score range {result['informality_score'].min():.3f}–"
        f"{result['informality_score'].max():.3f})"
    )

    return result


def bin_macro_f1(
    df_with_scores: pd.DataFrame,
    preds: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """
    Compute macro-F1 per informality bin for one model's predictions.

    Parameters
    ----------
    df_with_scores : test_df after add_informality_score()
    preds          : int array of predicted label_ids, aligned with df rows
    model_name     : string label used in the returned DataFrame

    Returns
    -------
    DataFrame with columns: model, bin, n, macro_f1
    """
    from sklearn.metrics import f1_score

    preds  = np.asarray(preds)
    labels = df_with_scores["label_id"].values
    rows   = []

    for bin_label in ["low", "mid", "high"]:
        mask = df_with_scores["informality_bin"].values == bin_label
        if mask.sum() == 0:
            continue
        f1 = f1_score(labels[mask], preds[mask], average="macro", zero_division=0)
        rows.append({
            "model":    model_name,
            "bin":      bin_label,
            "n":        int(mask.sum()),
            "macro_f1": round(float(f1), 4),
        })

    return pd.DataFrame(rows)
