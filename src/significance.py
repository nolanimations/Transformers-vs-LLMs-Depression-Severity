"""
Paired significance tests for H1 (fine-tuned encoders vs. best LLM variant).

The pre-registered H1 verdict rule (PLAN.md) requires both:
  1. The 95% CI on Delta = macro-F1(encoder) - macro-F1(LLM) excludes 0
     (paired bootstrap, same resample of the test set used for both models).
  2. McNemar's test on prediction correctness is significant (p < 0.05).

Usage: python -m src.significance
Output: results/significance_h1.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from src.data import load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
RUNS_DIR = REPO_ROOT / "results" / "runs"
LLM_LOGS_DIR = REPO_ROOT / "results" / "llm_logs"
OUTPUT_PATH = REPO_ROOT / "results" / "significance_h1.json"

LABEL2ID = {"minimum": 0, "mild": 1, "moderate": 2, "severe": 3}

N_BOOTSTRAP = 1000
SEED = 42

PAIRS = [
    ("RoBERTa", "GPT-5.4-mini CoT"),
    ("MentalBERT", "GPT-5.4-mini CoT"),
    ("RoBERTa", "Gemini 3 Flash CoT"),
    ("MentalBERT", "Gemini 3 Flash CoT"),
]


def load_encoder_preds(run_name: str) -> np.ndarray:
    df = pd.read_csv(RUNS_DIR / run_name / "test_predictions.csv")
    return df["pred_id"].values


def load_llm_cot_preds(log_dir: str, prefix: str, n: int) -> np.ndarray:
    """Reconstruct per-index predictions from chain_of_thought logs (canonical + fills)."""
    files = sorted((LLM_LOGS_DIR / log_dir).glob(f"{prefix}_chain_of_thought_*.jsonl"))
    preds = np.full(n, -1, dtype=int)
    for fp in files:
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("index")
            if idx is not None and 0 <= idx < n:
                preds[idx] = LABEL2ID.get(obj.get("pred_label", ""), -1)
    return preds


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact two-sided McNemar test on the discordant-pair counts."""
    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n = b + c
    p = 1.0 if n == 0 else binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "n_discordant": n, "p_value": p}


def paired_bootstrap_delta(y_true, preds_a, preds_b, n_boot=N_BOOTSTRAP, seed=SEED) -> dict:
    """95% CI on Delta = macro-F1(A) - macro-F1(B), resampling indices jointly."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        f1_a = f1_score(y_true[idx], preds_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], preds_b[idx], average="macro", zero_division=0)
        deltas[i] = f1_a - f1_b
    return {
        "delta_point": float(f1_score(y_true, preds_a, average="macro", zero_division=0)
                              - f1_score(y_true, preds_b, average="macro", zero_division=0)),
        "delta_ci_low": float(np.percentile(deltas, 2.5)),
        "delta_ci_high": float(np.percentile(deltas, 97.5)),
    }


def main():
    _, _, test_df = load_splits()
    y_true = test_df["label_id"].values
    n = len(y_true)
    log.info(f"Test set: {n} rows")

    preds = {
        "RoBERTa": load_encoder_preds("roberta"),
        "MentalBERT": load_encoder_preds("mentalbert"),
        "GPT-5.4-mini CoT": load_llm_cot_preds("gpt5", "gpt", n),
        "Gemini 3 Flash CoT": load_llm_cot_preds("gemini3", "gemini", n),
    }

    for name, p in preds.items():
        n_invalid = int((p < 0).sum())
        if n_invalid:
            log.warning(f"{name}: {n_invalid} unmatched predictions (treated as wrong)")

    macro_f1 = {name: float(f1_score(y_true, p, average="macro", zero_division=0))
                 for name, p in preds.items()}
    for name, f1 in macro_f1.items():
        log.info(f"{name:<22} macro-F1 = {f1:.4f}")

    results = {"n_test": n, "macro_f1": macro_f1, "pairs": {}}
    for a, b in PAIRS:
        correct_a = preds[a] == y_true
        correct_b = preds[b] == y_true
        mc = mcnemar_exact(correct_a, correct_b)
        bs = paired_bootstrap_delta(y_true, preds[a], preds[b])
        key = f"{a} vs {b}"
        results["pairs"][key] = {"mcnemar": mc, "bootstrap_delta": bs}
        log.info(f"{key}: Delta={bs['delta_point']:+.4f} "
                 f"CI=[{bs['delta_ci_low']:+.4f}, {bs['delta_ci_high']:+.4f}]  "
                 f"McNemar p={mc['p_value']:.2e} (n_discordant={mc['n_discordant']})")

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    log.info(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
