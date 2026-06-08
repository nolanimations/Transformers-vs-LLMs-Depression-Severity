"""
One-shot utility: re-run GPT chain_of_thought for posts where reasoning is null.

Finds all indices with null reasoning in existing JSONL logs, re-calls the GPT
client (with summary="concise" to force a visible reasoning trace), and writes
results to a new timestamped JSONL. The existing load_cot_rationales() merger
automatically picks up the new file and the updated records overwrite the nulls.

Usage:
    python -m src.llm.fill_reasoning
"""

import json
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.data import load_splits
from src.llm.prompts import build_prompt
from src.llm.openai_client import classify as classify_gpt


def find_null_indices(log_dir: Path) -> list[int]:
    """Return sorted list of test-set indices where reasoning is null."""
    records: dict[int, dict] = {}
    for fp in sorted(log_dir.glob("*chain_of_thought*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    idx = obj.get("index")
                    if idx is not None:
                        records[int(idx)] = obj
                except json.JSONDecodeError:
                    continue
    return sorted(idx for idx, rec in records.items() if not rec.get("reasoning"))


def main() -> None:
    cfg_path = Path("configs/llm.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg           = cfg["gpt"]
    model_name          = model_cfg["model"]
    max_tokens          = model_cfg.get("max_tokens", 1600)
    requests_per_minute = int(model_cfg.get("requests_per_minute", 450))
    sleep_between       = 60.0 / max(1, requests_per_minute)
    log_dir             = Path(model_cfg.get("output_dir", "results/llm_logs/gpt5"))

    null_indices = find_null_indices(log_dir)
    if not null_indices:
        print("No null-reasoning posts found — nothing to do.")
        return
    print(f"Found {len(null_indices)} posts with null reasoning. Re-running...")

    # Load test set and few-shot examples
    train_df, _, test_df = load_splits()
    n_per_class = int(cfg.get("few_shot_examples_per_class", 2))
    few_shot_examples = (
        train_df.groupby("label_id")
        .apply(lambda g: g.sample(min(len(g), n_per_class), random_state=42))
        .reset_index(drop=True)[["text", "label"]]
        .to_dict("records")
    )

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path  = log_dir / f"gpt_chain_of_thought_{timestamp}_fill.jsonl"
    log_dir.mkdir(parents=True, exist_ok=True)

    backoffs = [2, 4, 8]

    for i, idx in enumerate(null_indices, 1):
        row    = test_df.iloc[idx]
        prompt = build_prompt(
            variant="chain_of_thought",
            post=row["text"],
            few_shot_examples=few_shot_examples,
        )

        attempt  = 0
        last_exc = None
        res      = None
        while True:
            try:
                res = classify_gpt(prompt, config={
                    "model":     model_name,
                    "max_tokens": max_tokens,
                    "variant":   "chain_of_thought",
                })
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= len(backoffs):
                    break
                print(f"  Error: {exc}. Retrying in {backoffs[attempt]}s...")
                time.sleep(backoffs[attempt])
                attempt += 1

        if res is None:
            print(f"  [{i}/{len(null_indices)}] idx={idx} FAILED: {last_exc}")
            time.sleep(sleep_between)
            continue

        record = {
            "timestamp":        datetime.utcnow().isoformat(),
            "index":            idx,
            "true_label":       row.get("label"),
            "prompt_preview":   (row.get("text") or "")[:500],
            "model":            model_name,
            "variant":          "chain_of_thought",
            "max_tokens":       max_tokens,
            "pred_label":       res["label"],
            "reasoning":        res["reasoning"],
            "tokens_in":        res.get("tokens_in", 0),
            "tokens_out":       res.get("tokens_out", 0),
            "cost_usd":         res.get("cost_usd", 0.0),
        }

        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        has_reasoning = bool(res.get("reasoning"))
        print(
            f"  [{i}/{len(null_indices)}] idx={idx:4d}  "
            f"label={res['label']:<10}  "
            f"reasoning={'yes' if has_reasoning else 'STILL NULL'}  "
            f"cost=${res.get('cost_usd', 0):.5f}"
        )
        time.sleep(sleep_between)

    print(f"\nDone. Written to {out_path}")
    print("Re-run `python -m src.interp_rubric generate` to rebuild the rating sheet.")


if __name__ == "__main__":
    main()
