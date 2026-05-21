"""
Gemini 3 API caller with exponential-backoff retries, JSON response parsing,
and per-call cost + token logging.
Reads GOOGLE_API_KEY from .env.

    from src.llm.gemini_client import classify
"""

# TODO: implement classify(prompt: str, config: dict) -> dict (label, reasoning, tokens, cost_usd)
