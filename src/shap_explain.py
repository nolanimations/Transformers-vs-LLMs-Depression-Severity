"""
Token-level SHAP explanations for fine-tuned encoder models.

Computes attributions on a curated subset of test posts and saves
paper-ready matplotlib figures to results/figures/.

    from src.shap_explain import load_model, get_predictions, select_examples
    from src.shap_explain import make_predict_fn, run_shap, save_figures

Typical usage: see notebooks/07_shap_examples.ipynb
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import shap.maskers
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import get_logger, set_seed

log = get_logger(__name__)

ID2LABEL = {0: "minimum", 1: "mild", 2: "moderate", 3: "severe"}
LABEL_COLOURS = {
    "minimum": "#4e9a8c",
    "mild":     "#e6a817",
    "moderate": "#d4612a",
    "severe":   "#b02020",
}


# ── Model loading ─────────────────────────────────────────────────────────────
def find_best_checkpoint(run_dir: str | Path) -> Path:
    """
    Return the path to the best model checkpoint saved by HuggingFace Trainer.

    Reads trainer_state.json (written by Trainer) to find the checkpoint with
    the highest eval_macro_f1. Falls back to the highest-numbered checkpoint
    if trainer_state.json is missing.
    """
    run_dir = Path(run_dir)
    state_file = run_dir / "trainer_state.json"

    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        best = state.get("best_model_checkpoint")
        if best and Path(best).exists():
            log.info(f"Best checkpoint from trainer_state.json: {best}")
            return Path(best)

    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {run_dir}.\n"
            f"Make sure you're running this on the machine where training happened."
        )
    log.warning(f"trainer_state.json not found — using last checkpoint: {checkpoints[-1]}")
    return checkpoints[-1]


def load_model(run_dir: str | Path, model_name: str, device: torch.device | None = None):
    """
    Load the best fine-tuned checkpoint and its tokenizer.

    Parameters
    ----------
    run_dir    : output_dir from training (e.g. "results/runs/mentalbert")
    model_name : HuggingFace model name used during training (e.g. "mental/mental-bert-base-uncased")
                 — needed to load the correct tokenizer
    device     : torch device; defaults to CUDA if available

    Returns
    -------
    (model, tokenizer, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = find_best_checkpoint(run_dir)
    log.info(f"Loading checkpoint: {checkpoint}  →  device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    model.to(device)

    return model, tokenizer, device


# ── Inference ─────────────────────────────────────────────────────────────────
def get_predictions(
    model, tokenizer, texts: list[str], device, max_length: int = 256, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a list of texts.

    Returns
    -------
    preds  : int array of shape (n,)  — predicted label_ids
    probs  : float array of shape (n, 4) — softmax probabilities
    """
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

    probs = np.vstack(all_probs)
    preds = np.argmax(probs, axis=1)
    return preds, probs


# ── Example selection ─────────────────────────────────────────────────────────
def select_examples(
    test_df,
    preds: np.ndarray,
    probs: np.ndarray,
    n_correct: int = 6,
    n_wrong:   int = 2,
) -> dict:
    """
    Pick a curated subset of test examples for SHAP explanation.

    For each class:
      - n_correct high-confidence correctly classified examples (clear signal)
      - n_wrong   misclassified borderline examples (interesting failures)

    Returns
    -------
    dict mapping label_id → list of row dicts with keys:
        text, true_label, pred_label, confidence, correct
    """
    import pandas as pd
    labels = test_df["label_id"].values
    selected = {}

    for cls in range(4):
        cls_mask  = labels == cls
        correct   = (preds == cls) & cls_mask
        wrong     = (preds != cls) & cls_mask
        confidence = probs[np.arange(len(probs)), preds]

        correct_idx = np.where(correct)[0]
        correct_idx = correct_idx[np.argsort(-confidence[correct_idx])][:n_correct]

        wrong_idx = np.where(wrong)[0]
        wrong_idx = wrong_idx[np.argsort(confidence[wrong_idx])][:n_wrong]

        rows = []
        for idx in np.concatenate([correct_idx, wrong_idx]):
            rows.append({
                "idx":        int(idx),
                "text":       test_df["text"].iloc[idx],
                "true_label": ID2LABEL[int(labels[idx])],
                "pred_label": ID2LABEL[int(preds[idx])],
                "confidence": float(confidence[idx]),
                "correct":    bool(preds[idx] == labels[idx]),
                "source":     test_df["source"].iloc[idx],
            })

        selected[cls] = rows
        log.info(f"  {ID2LABEL[cls]}: {len(correct_idx)} correct + {len(wrong_idx)} wrong selected")

    return selected


# ── SHAP ──────────────────────────────────────────────────────────────────────
def make_predict_fn(model, tokenizer, device, max_length: int = 256, batch_size: int = 8):
    """
    Return a function suitable for shap.Explainer:
        f(texts: array-like[str]) -> np.ndarray of shape (n, 4)

    batch_size controls how many texts are sent to the model at once during
    SHAP's masking iterations. Keep this small (4–8) to avoid OOM.
    """
    def predict(texts):
        texts = list(texts)
        all_probs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(all_probs)

    return predict


def run_shap(texts: list[str], predict_fn, tokenizer, seed: int = 42):
    """
    Run SHAP Partition explainer on a list of texts.

    Returns a shap.Explanation object. This can take several minutes
    for 30+ texts — a progress bar is printed automatically.

    Note: shap.Explainer with a Text masker uses the tokenizer's mask token
    to occlude input tokens, making attributions meaningful for BERT/RoBERTa.
    """
    set_seed(seed)
    masker   = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_fn, masker, output_names=list(ID2LABEL.values()))
    log.info(f"Running SHAP on {len(texts)} texts (this may take several minutes) ...")
    shap_values = explainer(texts)
    return shap_values


# ── Figures ───────────────────────────────────────────────────────────────────
def save_figures(
    shap_values,
    examples: list[dict],
    output_dir: str | Path,
    top_n_tokens: int = 15,
) -> None:
    """
    Save one matplotlib figure per example to output_dir.

    Each figure shows:
      - The post text (truncated to 200 chars)
      - A horizontal bar chart of the top_n_tokens tokens by |SHAP value|
        for the predicted class, coloured by sign (positive/negative)
      - True vs predicted label in the title
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, example in enumerate(examples):
        pred_cls   = list(ID2LABEL.values()).index(example["pred_label"])
        sv         = shap_values[i, :, pred_cls]
        tokens     = shap_values.data[i]

        keep = [
            j for j, t in enumerate(tokens)
            if t not in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]")
        ]
        sv_clean     = np.array([sv.values[j] for j in keep])
        tokens_clean = [tokens[j]              for j in keep]

        top_idx = np.argsort(np.abs(sv_clean))[-top_n_tokens:][::-1]
        top_sv     = sv_clean[top_idx]
        top_tokens = [tokens_clean[j] for j in top_idx]

        fig, ax = plt.subplots(figsize=(8, max(4, top_n_tokens * 0.35)))
        colours = ["#c0392b" if v > 0 else "#2980b9" for v in top_sv]
        ax.barh(range(len(top_sv)), top_sv[::-1], color=colours[::-1])
        ax.set_yticks(range(len(top_tokens)))
        ax.set_yticklabels(top_tokens[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"SHAP value  (→ {example['pred_label']})")

        status = "✓" if example["correct"] else "✗"
        ax.set_title(
            f"{status}  true={example['true_label']}  pred={example['pred_label']}  "
            f"conf={example['confidence']:.2f}\n"
            f"\"{example['text'][:120]}{'…' if len(example['text']) > 120 else ''}\"",
            fontsize=8, loc="left",
        )
        plt.tight_layout()

        fname = f"{i:03d}_{example['true_label']}_pred{example['pred_label']}.png"
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)

    log.info(f"Saved {len(examples)} figures → {output_dir}")
