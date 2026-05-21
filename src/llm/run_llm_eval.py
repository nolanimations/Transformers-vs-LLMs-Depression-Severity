"""
Batched LLM evaluation runner.
Iterates over the test split, calls the configured model+prompt variant,
logs every call to results/llm_logs/<model>_<variant>.jsonl.
Hard-stops if cumulative cost exceeds cost_cap_usd from configs/llm.yaml.

Usage:
    python src/llm/run_llm_eval.py --model gpt --variant zero_shot
    python src/llm/run_llm_eval.py --model gemini --variant chain_of_thought
"""

# TODO: implement
