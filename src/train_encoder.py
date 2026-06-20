"""
Fine-tuning driver for MentalBERT and RoBERTa-base.
Config-driven via configs/mentalbert.yaml or configs/roberta.yaml.
Uses HuggingFace Trainer with a custom weighted CrossEntropyLoss.

Usage:
    python src/train_encoder.py --config configs/mentalbert.yaml
    python src/train_encoder.py --config configs/roberta.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import Dataset as HFDataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import class_weights as compute_class_weights, load_splits
from src.eval import evaluate
from src.utils import get_logger, set_seed

log = get_logger(__name__)

ID2LABEL = {0: "minimum", 1: "mild", 2: "moderate", 3: "severe"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ── Weighted Trainer ──────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    """
    HuggingFace Trainer with class-weighted CrossEntropyLoss.

    The standard Trainer uses unweighted loss, which under-learns rare classes
    even when the dataset is already partially balanced. Injecting class weights
    here is the cleanest way to fix this without touching anything else.
    """

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred) -> dict:
    """
    Called by Trainer after each eval step.
    Returns macro-F1, which is used to select the best checkpoint.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0))
    }


# ── Dataset ───────────────────────────────────────────────────────────────────
def build_hf_dataset(df, tokenizer, max_length: int) -> HFDataset:
    """
    Tokenize a split DataFrame and return a HuggingFace Dataset.

    Uses truncation only (no padding here) — DataCollatorWithPadding handles
    padding dynamically per batch, saving memory vs. padding everything to 256.
    """
    ds = HFDataset.from_dict({
        "text":   df["text"].tolist(),
        "labels": df["label_id"].tolist(),
    })

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


# ── Main training function ────────────────────────────────────────────────────
def train(cfg: dict) -> dict:
    """
    Fine-tune a pre-trained encoder model on the depression severity task.

    Parameters
    ----------
    cfg : dict loaded from a YAML config file (configs/mentalbert.yaml or roberta.yaml)

    Returns
    -------
    dict — full test-set metrics from src.eval.evaluate()
    """
    set_seed(cfg["seed"])

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_splits()
    weights = compute_class_weights(train_df)
    log.info(f"Class weights: { {ID2LABEL[i]: round(w, 4) for i, w in enumerate(weights.tolist())} }")

    # ── Model + tokenizer ─────────────────────────────────────────────────────
    log.info(f"Loading '{cfg['model_name']}' ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels    = 4,
        id2label      = ID2LABEL,
        label2id      = LABEL2ID,
        ignore_mismatched_sizes = True,
    )

    # ── Tokenise splits ───────────────────────────────────────────────────────
    max_length = cfg["max_length"]
    log.info(f"Tokenising splits (max_length={max_length}) ...")
    train_ds = build_hf_dataset(train_df, tokenizer, max_length)
    val_ds   = build_hf_dataset(val_df,   tokenizer, max_length)
    test_ds  = build_hf_dataset(test_df,  tokenizer, max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = cfg["output_dir"],
        num_train_epochs            = cfg["num_epochs"],
        per_device_train_batch_size = cfg["batch_size"],
        per_device_eval_batch_size  = cfg["batch_size"],
        learning_rate               = cfg["learning_rate"],
        weight_decay                = cfg["weight_decay"],
        warmup_ratio                = cfg["warmup_ratio"],
        fp16                        = cfg["fp16"],
        eval_strategy               = cfg["eval_strategy"],
        save_strategy               = cfg["save_strategy"],
        load_best_model_at_end      = cfg["load_best_model_at_end"],
        metric_for_best_model       = cfg["metric_for_best_model"],
        greater_is_better           = True,
        logging_steps               = cfg["logging_steps"],
        seed                        = cfg["seed"],
        report_to                   = "none",
        save_total_limit            = 2,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = WeightedTrainer(
        class_weights     = weights,
        model             = model,
        args              = training_args,
        train_dataset     = train_ds,
        eval_dataset      = val_ds,
        processing_class  = tokenizer,
        data_collator     = data_collator,
        compute_metrics   = compute_metrics,
    )

    log.info("Starting training ...")
    trainer.train()

    # ── Test evaluation ───────────────────────────────────────────────────────
    log.info("Evaluating best checkpoint on test set ...")
    prediction_output = trainer.predict(test_ds)
    test_preds = np.argmax(prediction_output.predictions, axis=1)

    metrics = evaluate(
        preds      = test_preds,
        labels     = test_df["label_id"].values,
        df         = test_df,
        run_name   = cfg.get("run_name", "encoder"),
        output_dir = cfg["output_dir"],
        split      = "test",
    )

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MentalBERT or RoBERTa")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
