"""
H4 cost-benefit ledger: $ / 1k predictions, latency, and a rough CO2 estimate
for every model in the comparison (GPT-5.4-mini, Gemini 3 Flash, MentalBERT,
RoBERTa), plus the on-prem-deployability column for the headline H4 table.

LLM numbers are aggregated from the call-by-call logs in results/llm_logs/.
Encoder numbers come from a small local-inference benchmark (optional, since
it needs the checkpoints + a GPU/CPU run) and are otherwise reported as
effectively free / on-prem.

Usage
-----
    python -m src.cost_ledger                     # LLM ledger only (fast)
    python -m src.cost_ledger --benchmark-encoders  # also time MentalBERT/RoBERTa

Outputs
-------
    results/cost_ledger.csv          full per (model, variant) breakdown
    results/cost_ledger_summary.md   headline H4 table (best variant per model)
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)

REPO_ROOT   = Path(__file__).parent.parent
LLM_LOGS_DIR = REPO_ROOT / "results" / "llm_logs"
RUNS_DIR     = REPO_ROOT / "results" / "runs"
OUTPUT_DIR   = REPO_ROOT / "results"

# Maps the directory under results/llm_logs/ to (display name, file prefix,
# the corresponding results/runs/<run_name> for each prompt variant).
LLM_MODELS = {
    "gpt5": {
        "display": "GPT-5.4-mini",
        "prefix":  "gpt",
        "runs": {
            "zero_shot":        "gpt_zero_shot",
            "few_shot":         "gpt_few_shot",
            "chain_of_thought": "gpt_chain_of_thought",
        },
    },
    "gemini3": {
        "display": "Gemini 3 Flash",
        "prefix":  "gemini",
        "runs": {
            "zero_shot":        "gemini_zero_shot",
            "few_shot":         "gemini_few_shot",
            "chain_of_thought": "gemini_chain_of_thought",
        },
    },
}

VARIANT_ORDER = ["zero_shot", "few_shot", "chain_of_thought"]

# ── Rough CO2 constants (order-of-magnitude only — see notes in summary) ──────
# Energy per token: assume ~0.3 Wh per 1k tokens for transformer inference on
# data-centre accelerators. Same order of magnitude as recent per-query LLM
# inference measurements (~0.3 Wh/query for GPT-4o-class models; Google reports
# ~0.24 Wh per median Gemini text prompt). Treated as an assumption, not a
# precise figure. Methodology follows Strubell et al. (2019) and Lacoste et al.
# (2019), arXiv:1910.09700 (the ML CO2 Impact accounting approach).
KWH_PER_1K_TOKENS = 0.0003
# Global average grid carbon intensity (kg CO2 / kWh). IEA Emissions Factors put
# the global average at ~0.44-0.48; we use a round, conservative 0.4.
# https://www.iea.org/data-and-statistics/data-product/emissions-factors-2024
CARBON_INTENSITY_KG_PER_KWH = 0.4
# Assumed power draw (W) for the local encoder benchmark, by device type.
# GPU figure is the rated board power (TDP) of the RTX 3080 used for all
# training/inference on the team's training machine:
# https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3080-3080ti/
ASSUMED_GPU_POWER_W = 320
ASSUMED_CPU_POWER_W = 65


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


_FILE_RE = re.compile(
    r"^(?P<prefix>gpt|gemini)_(?P<variant>zero_shot|few_shot|chain_of_thought)"
    r"_\d{8}T\d{6}Z(?P<fill>_fill)?\.jsonl$"
)


def _group_files(model_dir: Path) -> dict:
    """
    Group jsonl files in a model's log directory by prompt variant.

    Returns {variant: {"canonical": Path, "fills": [Path, ...], "all": [Path, ...]}}
    The canonical file is the one with the most records (the full test-set
    sweep); smaller files are aborted/dev runs and are excluded from the
    per-variant ledger but still counted in the grand-total spend.
    """
    by_variant: dict[str, dict] = {}
    for path in sorted(model_dir.glob("*.jsonl")):
        m = _FILE_RE.match(path.name)
        if not m:
            log.warning(f"Skipping unrecognised log file: {path}")
            continue
        variant = m.group("variant")
        is_fill = m.group("fill") is not None
        entry = by_variant.setdefault(variant, {"candidates": [], "fills": []})
        if is_fill:
            entry["fills"].append(path)
        else:
            entry["candidates"].append(path)

    grouped = {}
    for variant, entry in by_variant.items():
        if not entry["candidates"]:
            continue
        canonical = max(entry["candidates"], key=lambda p: sum(1 for _ in open(p, encoding="utf-8")))
        others = [p for p in entry["candidates"] if p != canonical]
        grouped[variant] = {
            "canonical": canonical,
            "fills": entry["fills"],
            "discarded": others,
        }
    return grouped


def _median_latency_s(records: list[dict]) -> float | None:
    """Median wall-clock gap between consecutive logged calls, in seconds."""
    timestamps = sorted(
        datetime.fromisoformat(r["timestamp"]) for r in records if r.get("timestamp")
    )
    if len(timestamps) < 2:
        return None
    diffs = [
        (b - a).total_seconds()
        for a, b in zip(timestamps, timestamps[1:])
    ]
    diffs = [d for d in diffs if d >= 0]
    return float(np.median(diffs)) if diffs else None


def _co2_g_per_1k_from_tokens(avg_tokens_per_call: float) -> float:
    """Rough CO2 estimate (grams) per 1,000 predictions, from token volume."""
    co2_g_per_call = (avg_tokens_per_call / 1000) * KWH_PER_1K_TOKENS * CARBON_INTENSITY_KG_PER_KWH * 1000
    return co2_g_per_call * 1000


# ── LLM ledger ─────────────────────────────────────────────────────────────────
def build_llm_ledger() -> tuple[pd.DataFrame, float]:
    """
    Returns (per-variant ledger DataFrame, grand_total_cost_usd_across_all_logs).
    """
    rows = []
    grand_total = 0.0

    for model_key, cfg in LLM_MODELS.items():
        model_dir = LLM_LOGS_DIR / model_key
        if not model_dir.exists():
            log.warning(f"No log directory for {model_key} at {model_dir}")
            continue

        for path in model_dir.glob("*.jsonl"):
            for r in _load_jsonl(path):
                grand_total += float(r.get("cost_usd") or 0.0)

        grouped = _group_files(model_dir)
        for variant in VARIANT_ORDER:
            if variant not in grouped:
                continue
            info = grouped[variant]
            records = _load_jsonl(info["canonical"])
            for fill_path in info["fills"]:
                records += _load_jsonl(fill_path)

            n_predictions = len({r["index"] for r in records if "index" in r})
            total_cost = sum(float(r.get("cost_usd") or 0.0) for r in records)
            tokens_in  = sum(int(r.get("tokens_in") or 0) for r in records)
            tokens_out = sum(int(r.get("tokens_out") or 0) for r in records)
            n_calls    = len(records)
            avg_tokens = (tokens_in + tokens_out) / n_calls if n_calls else 0.0

            median_latency = _median_latency_s(_load_jsonl(info["canonical"]))

            cost_per_1k = (total_cost / n_predictions * 1000) if n_predictions else None

            rows.append({
                "model":               cfg["display"],
                "variant":             variant,
                "n_predictions":       n_predictions,
                "n_api_calls":         n_calls,
                "total_cost_usd":      round(total_cost, 6),
                "cost_per_1k_usd":     round(cost_per_1k, 4) if cost_per_1k is not None else None,
                "avg_tokens_in":       round(tokens_in / n_calls, 1) if n_calls else None,
                "avg_tokens_out":      round(tokens_out / n_calls, 1) if n_calls else None,
                "median_latency_s":    round(median_latency, 3) if median_latency is not None else None,
                "est_co2_g_per_1k":    round(_co2_g_per_1k_from_tokens(avg_tokens), 3),
                "deployable_on_prem":  "No (cloud API)",
                "macro_f1":            _read_macro_f1(cfg["runs"][variant]),
            })

    return pd.DataFrame(rows), grand_total


def _read_macro_f1(run_name: str) -> float | None:
    metrics_path = RUNS_DIR / run_name / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f).get("macro_f1")


# ── Local encoder benchmark ─────────────────────────────────────────────────────
def benchmark_encoders(batch_size: int = 16, max_length: int = 256, n_samples: int = 200) -> pd.DataFrame:
    """
    Time inference for MentalBERT and RoBERTa on a sample of the test set.
    Marginal $ cost is treated as 0 (local GPU/CPU, already-owned hardware);
    CO2 is estimated from wall time x an assumed GPU power draw.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.data import load_splits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Benchmarking encoders on device={device}")
    if device.type != "cuda":
        log.warning(
            "No CUDA device available here — this is NOT the RTX 3080 used for "
            "training/inference. CPU numbers are still useful (they already show "
            "a >10x latency advantage over the LLM APIs), but for the figure that "
            "goes in the paper, re-run `python -m src.cost_ledger --benchmark-encoders` "
            "on the training machine for a representative GPU number."
        )

    _, _, test_df = load_splits()
    texts = test_df["text"].astype(str).tolist()[:n_samples]

    rows = []
    for run_name, display in [("mentalbert", "MentalBERT"), ("roberta", "RoBERTa")]:
        ckpt_dir = RUNS_DIR / run_name / "checkpoint-2084"
        if not ckpt_dir.exists():
            log.warning(f"Checkpoint not found for {run_name}: {ckpt_dir} — skipping benchmark")
            continue

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
        model.eval()

        warm = tokenizer(texts[:batch_size], padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**warm)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(device)
                model(**enc)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        latency_s = elapsed / len(texts)
        power_w = ASSUMED_GPU_POWER_W if device.type == "cuda" else ASSUMED_CPU_POWER_W
        co2_g_per_1k = (
            (elapsed / len(texts) * 1000)
            / 3600 * power_w / 1000
            * CARBON_INTENSITY_KG_PER_KWH * 1000
        )

        rows.append({
            "model":              display,
            "variant":            f"local_inference ({device.type}, batch={batch_size})",
            "n_predictions":      len(texts),
            "n_api_calls":        None,
            "total_cost_usd":     0.0,
            "cost_per_1k_usd":    0.0,
            "avg_tokens_in":      None,
            "avg_tokens_out":     None,
            "median_latency_s":   round(latency_s, 5),
            "est_co2_g_per_1k":   round(co2_g_per_1k, 3),
            "deployable_on_prem": "Yes (local hardware)",
            "macro_f1":           _read_macro_f1(run_name),
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


# ── Headline H4 table ────────────────────────────────────────────────────────────
def build_headline_table(ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Pick, per model, the prompt variant used as that model's main result
    (highest macro-F1), and assemble the Model x {macro-F1, $/1k, latency,
    CO2/1k, on-prem} table for the paper.
    """
    headline_rows = []
    for model in ledger["model"].unique():
        sub = ledger[ledger["model"] == model]
        if sub["macro_f1"].notna().any():
            best = sub.loc[sub["macro_f1"].idxmax()]
        else:
            best = sub.iloc[0]
        headline_rows.append(best)
    return pd.DataFrame(headline_rows).sort_values("macro_f1", ascending=False)


def _to_markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            val = row[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cells.append("--")
            elif isinstance(val, float):
                cells.append(f"{val:.4g}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────────
def main(benchmark: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm_ledger, grand_total = build_llm_ledger()

    if benchmark:
        encoder_ledger = benchmark_encoders()
    else:
        encoder_ledger = pd.DataFrame([
            {
                "model": "MentalBERT", "variant": "local_inference (not benchmarked)",
                "n_predictions": None, "n_api_calls": None,
                "total_cost_usd": 0.0, "cost_per_1k_usd": 0.0,
                "avg_tokens_in": None, "avg_tokens_out": None,
                "median_latency_s": None, "est_co2_g_per_1k": None,
                "deployable_on_prem": "Yes (local hardware)",
                "macro_f1": _read_macro_f1("mentalbert"),
            },
            {
                "model": "RoBERTa", "variant": "local_inference (not benchmarked)",
                "n_predictions": None, "n_api_calls": None,
                "total_cost_usd": 0.0, "cost_per_1k_usd": 0.0,
                "avg_tokens_in": None, "avg_tokens_out": None,
                "median_latency_s": None, "est_co2_g_per_1k": None,
                "deployable_on_prem": "Yes (local hardware)",
                "macro_f1": _read_macro_f1("roberta"),
            },
        ])

    full_ledger = pd.concat([llm_ledger, encoder_ledger], ignore_index=True)
    csv_path = OUTPUT_DIR / "cost_ledger.csv"
    full_ledger.to_csv(csv_path, index=False)
    log.info(f"Saved full ledger -> {csv_path}")

    headline = build_headline_table(full_ledger)
    headline_cols = ["model", "macro_f1", "cost_per_1k_usd", "median_latency_s",
                      "est_co2_g_per_1k", "deployable_on_prem"]
    headline_headers = ["Model", "Macro-F1", "$ / 1k predictions", "Median latency (s/post)",
                         "Est. CO2 (g / 1k predictions)", "On-prem deployable"]
    md_table = _to_markdown_table(headline, headline_cols, headline_headers)

    summary_lines = [
        "# H4 cost-benefit ledger\n",
        f"Generated from {LLM_LOGS_DIR.relative_to(REPO_ROOT)} "
        f"(canonical full test-set runs; 'fill' re-runs added to cost totals).\n",
        "## Headline comparison\n",
        md_table,
        "\n## Notes\n",
        "- `$ / 1k predictions`: actual logged `cost_usd` from the API responses, "
        "scaled to 1,000 predictions.",
        "- `median_latency_s`: median wall-clock gap between consecutive logged "
        "calls in the canonical run (includes any client-side rate limiting).",
        "- `est_co2_g_per_1k`: rough order-of-magnitude estimate only — "
        f"{KWH_PER_1K_TOKENS*1000:.2f} Wh / 1k tokens "
        f"x {CARBON_INTENSITY_KG_PER_KWH} kg CO2/kWh (global grid average) for the LLMs; "
        f"{ASSUMED_GPU_POWER_W} W (GPU) / {ASSUMED_CPU_POWER_W} W (CPU) assumed draw "
        "x measured wall time for the local encoders, depending on the benchmark device. "
        "Treat as comparable orders of magnitude, not precise figures.",
        f"- Grand total LLM spend across **all** logged calls "
        f"(incl. prompt-iteration / aborted runs): **${grand_total:.4f}** "
        f"(budget cap was ~$40-50).",
        "- MentalBERT / RoBERTa run on local hardware already owned by the team: "
        "marginal $ cost per prediction is effectively $0.",
    ]
    if not benchmark:
        summary_lines.append(
            "- Encoder latency/CO2 not measured in this run — "
            "re-run with `python -m src.cost_ledger --benchmark-encoders`."
        )

    md_path = OUTPUT_DIR / "cost_ledger_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    log.info(f"Saved headline summary -> {md_path}")

    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the H4 cost-benefit ledger.")
    parser.add_argument("--benchmark-encoders", action="store_true",
                         help="Time MentalBERT/RoBERTa local inference (loads checkpoints).")
    args = parser.parse_args()
    main(benchmark=args.benchmark_encoders)
