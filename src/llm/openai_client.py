"""
GPT-5 API caller with exponential-backoff retries, JSON response parsing,
and per-call cost + token logging.
Reads OPENAI_API_KEY from .env.

    from src.llm.openai_client import classify
"""

# TODO: implement classify(prompt: str, config: dict) -> dict (label, reasoning, tokens, cost_usd)
