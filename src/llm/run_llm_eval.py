"""Batched LLM evaluation runner.

Iterates over the test split, calls the configured model+prompt variant,
logs every call to results/llm_logs/<model>_<variant>.jsonl and stops when
cumulative estimated cost exceeds `cost_cap_usd` from configs/llm.yaml.

Usage examples:
    python src/llm/run_llm_eval.py --model gemini --variant chain_of_thought --limit 20

Notes:
- Uses `src.llm.prompts.build_prompt` to construct prompts.
- Uses `src.llm.gemini_client.classify` or `src.llm.openai_client.classify` depending on `--model`.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.data import load_splits
from src.eval import evaluate
from src.llm.prompts import build_prompt

LABEL2ID = {"minimum": 0, "mild": 1, "moderate": 2, "severe": 3}

try:
    from src.llm.gemini_client import classify as classify_gemini
except Exception:
    classify_gemini = None

try:
    from src.llm.openai_client import classify as classify_gpt
except Exception:
    classify_gpt = None


def _write_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemini", "gpt"], default="gemini")
    parser.add_argument("--variant", choices=["zero_shot", "few_shot", "chain_of_thought"], default="chain_of_thought")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of examples to run")
    parser.add_argument("--start", type=int, default=0, help="Zero-based start index into the dataset")
    parser.add_argument("--dry_run", action="store_true", help="Build prompts but don't call the LLM")
    args = parser.parse_args()

    cfg_path = Path("configs/llm.yaml")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    model_key = args.model
    model_cfg = cfg.get(model_key, {})
    cost_cap = float(cfg.get("cost_cap_usd", 40.0))

    model_name = model_cfg.get("model")
    temperature = model_cfg.get("temperature", 0.0)
    max_tokens = model_cfg.get("max_tokens", 800)
    requests_per_minute = int(model_cfg.get("requests_per_minute", 60))
    output_dir = Path(model_cfg.get("output_dir", f"results/llm_logs/{model_key}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"{model_key}_{args.variant}_{timestamp}.jsonl"

    if model_key == "gemini":
        if classify_gemini is None:
            raise RuntimeError("Gemini client is not importable. Check environment and dependencies.")
        classify_fn = classify_gemini
    else:
        if classify_gpt is None:
            raise RuntimeError("OpenAI client is not importable. Check environment and dependencies.")
        classify_fn = classify_gpt

    train_df, val_df, test_df = load_splits()
    df = test_df
    total = len(df)
    start = args.start
    limit = args.limit or total

    few_shot_examples = None
    if args.variant in {"few_shot", "chain_of_thought"}:
        n_per_class = int(cfg.get("few_shot_examples_per_class", 2))
        few_shot_examples = (
            train_df.groupby("label_id")
            .apply(lambda g: g.sample(min(len(g), n_per_class), random_state=42))
            .reset_index(drop=True)[["text", "label"]]
            .to_dict("records")
        )

    sleep_between = 60.0 / max(1, requests_per_minute)

    cumulative_cost = 0.0
    processed = 0
    preds_list: list[int] = []
    cost_cap_hit = False

    print(f"Running LLM eval: model={model_key} ({model_name}), variant={args.variant}, output={out_path}")

    for i, (_, row) in enumerate(df.iloc[start : start + limit].iterrows(), start=start):
        if args.limit and processed >= limit:
            break

        prompt = build_prompt(variant=args.variant, post=row["text"], few_shot_examples=few_shot_examples)

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "index": int(i),
            "true_label": row.get("label"),
            "prompt_preview": (row.get("text") or "")[:500],
            "model": model_name,
            "variant": args.variant,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if args.dry_run:
            record.update({"label": None, "reasoning": None, "error": "dry_run"})
            _write_jsonl(out_path, record)
            processed += 1
            print(f"[DRY] {i+1-start}/{limit} prompt prepared")
            time.sleep(sleep_between)
            continue

        attempt = 0
        backoffs = [2, 4, 8]
        last_exc = None
        while True:
            try:
                res = classify_fn(prompt, config={"model": model_name, "temperature": temperature, "max_tokens": max_tokens, "timeout": 30, "variant": args.variant})
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= len(backoffs):
                    break
                time.sleep(backoffs[attempt])
                attempt += 1

        if last_exc is not None and 'res' not in locals():
            record.update({"label": "unknown", "reasoning": None, "error": str(last_exc)})
            _write_jsonl(out_path, record)
            print(f"Error on index {i}: {last_exc}")
            processed += 1
            time.sleep(sleep_between)
            continue

        label = res.get("label")
        reasoning = res.get("reasoning")
        tokens_in = res.get("tokens_in", 0)
        tokens_out = res.get("tokens_out", 0)
        cost_usd = float(res.get("cost_usd", 0.0))

        cumulative_cost += cost_usd

        pred_id = LABEL2ID.get(label, -1) if label else -1
        preds_list.append(pred_id)

        record.update({
            "pred_label": label,
            "reasoning": reasoning,
            "tokens_in": int(tokens_in or 0),
            "tokens_out": int(tokens_out or 0),
            "cost_usd": cost_usd,
            "cumulative_cost_usd": cumulative_cost,
        })

        _write_jsonl(out_path, record)
        processed += 1

        print(f"{processed}/{min(limit, total)}: idx={i} label={label} cost=${cost_usd:.6f} cum=${cumulative_cost:.2f}")

        if cumulative_cost >= cost_cap:
            print(f"Reached cost cap ${cost_cap:.2f}; stopping early.")
            cost_cap_hit = True
            break

        time.sleep(sleep_between)

    print(f"Finished. Wrote {processed} records to {out_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    # Only run evaluate() on a complete pass (no --limit, no cost-cap hit)
    full_run = (start == 0) and (args.limit is None or args.limit >= total) and not cost_cap_hit
    if full_run and len(preds_list) == len(test_df):
        run_name = f"{model_key}_{args.variant}"
        runs_dir = Path("results/runs") / run_name
        print(f"\nEvaluating {run_name} on full test set ...")
        evaluate(
            preds      = np.array(preds_list),
            labels     = test_df["label_id"].values,
            df         = test_df,
            run_name   = run_name,
            output_dir = str(runs_dir),
            split      = "test",
        )
    else:
        print(f"\nSkipping evaluate() — partial run (processed {processed}/{total}, cap_hit={cost_cap_hit}).")


if __name__ == "__main__":
    main()
