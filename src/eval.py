"""
Shared evaluation harness used by every model.

Computes macro-F1 (primary metric) with bootstrap 95% CIs, per-class P/R/F1,
overall accuracy, per-platform slices (all / twitter / reddit), and saves a
confusion-matrix PNG and metrics.json to the run output directory.

    from src.eval import evaluate

    metrics = evaluate(preds, labels, df, run_name="mentalbert", output_dir="results/runs/mentalbert")
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
)

LABEL_NAMES = ["minimum", "mild", "moderate", "severe"]


# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def _bootstrap_macro_f1(
    preds: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) 95% CI for macro-F1 via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(preds)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(
            f1_score(labels[idx], preds[idx], average="macro", zero_division=0)
        )
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ── Per-source slice ─────────────────────────────────────────────────────────
def _slice_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


# ── Confusion matrix plot ─────────────────────────────────────────────────────
def _save_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    run_name: str,
    output_dir: Path,
) -> None:
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        cmap="Blues", ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{run_name} — confusion matrix (colour = row recall)")
    plt.tight_layout()
    path = output_dir / f"{run_name}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {path}")


# ── Main entry point ─────────────────────────────────────────────────────────
def evaluate(
    preds,
    labels,
    df,
    run_name: str,
    output_dir: str | Path | None = None,
    split: str = "test",
) -> dict:
    """
    Evaluate predictions against ground-truth labels.

    Parameters
    ----------
    preds       : array-like of predicted label_ids  (0–3)
    labels      : array-like of true label_ids       (0–3)
    df          : DataFrame aligned with preds/labels; must contain 'source' column
    run_name    : identifier used in filenames and the saved JSON
    output_dir  : where to write metrics.json and the confusion-matrix PNG
                  (if None, files are not saved — useful for val-time logging)
    split       : "val" or "test" — recorded in the JSON for traceability

    Returns
    -------
    dict  — all metrics; safe to log directly or pass to pandas
    """
    preds  = np.asarray(preds)
    labels = np.asarray(labels)

    # ── Overall metrics ───────────────────────────────────────────────────────
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    ci_low, ci_high = _bootstrap_macro_f1(preds, labels)

    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        mask = labels == i
        per_class[name] = {
            "precision": float(precision_score(labels, preds, labels=[i], average="micro", zero_division=0)),
            "recall":    float(recall_score(   labels, preds, labels=[i], average="micro", zero_division=0)),
            "f1":        float(f1_score(       labels, preds, labels=[i], average="micro", zero_division=0)),
            "support":   int(mask.sum()),
        }

    # ── Per-source slices ─────────────────────────────────────────────────────
    sources = df["source"].values if "source" in df.columns else None
    by_source: dict = {"all": _slice_metrics(preds, labels)}
    if sources is not None:
        for src in ("twitter", "reddit"):
            mask = sources == src
            if mask.sum() > 0:
                by_source[src] = _slice_metrics(preds[mask], labels[mask])

    metrics = {
        "run_name":         run_name,
        "split":            split,
        "n_samples":        int(len(labels)),
        "accuracy":         float(accuracy_score(labels, preds)),
        "macro_f1":         macro_f1,
        "macro_f1_ci_low":  ci_low,
        "macro_f1_ci_high": ci_high,
        "per_class":        per_class,
        "by_source":        by_source,
    }

    # ── Save to disk ─────────────────────────────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics      → {json_path}")
        _save_confusion_matrix(preds, labels, run_name, output_dir)

    # ── Pretty-print summary ──────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  {run_name}  [{split}]")
    print(f"{'─'*50}")
    print(f"  Macro-F1 : {macro_f1:.4f}  (95% CI {ci_low:.4f}–{ci_high:.4f})")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  {'Class':<12} {'P':>7} {'R':>7} {'F1':>7} {'N':>6}")
    for name, m in per_class.items():
        print(f"  {name:<12} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['support']:>6}")
    if sources is not None:
        print(f"\n  By source:")
        for src, m in by_source.items():
            print(f"    {src:<10}  macro-F1={m['macro_f1']:.4f}  n={m['n']}")
    print(f"{'─'*50}\n")

    return metrics
