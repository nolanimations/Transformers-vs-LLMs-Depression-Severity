"""
GPT-5 API caller with exponential-backoff retries, JSON response parsing,
and per-call cost + token logging.
Reads OPENAI_API_KEY from .env.

    from src.llm.openai_client import classify
"""

import json
import os
import re
import time

from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise EnvironmentError("OPENAI_API_KEY is not set in the environment")
openai.api_key = OPENAI_API_KEY

LABELS = {"minimum", "mild", "moderate", "severe"}

OPENAI_PRICING = {
    # Estimated prices per 1,000 tokens; adjust if real billing differs.
    "gpt-5": {"prompt": 0.0015, "completion": 0.0025},
}


def _normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return "unknown"
    label = label.strip().lower()
    return label if label in LABELS else "unknown"


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def _parse_label(text: str) -> str:
    parsed = _parse_json_response(text)
    label = parsed.get("label")
    if isinstance(label, str):
        return _normalize_label(label)
    match = re.search(r'"label"\s*:\s*"([a-zA-Z]+)"', text)
    if match:
        return _normalize_label(match.group(1))
    return "unknown"


def _parse_reasoning(text: str) -> str | None:
    parsed = _parse_json_response(text)
    reasoning = parsed.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning.strip()
    match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text)
    return match.group(1).strip() if match else None


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = OPENAI_PRICING.get(model, {})
    prompt_rate = prices.get("prompt", 0.0)
    completion_rate = prices.get("completion", 0.0)
    return round((prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1000.0, 8)


def classify(prompt: str, config: dict) -> dict:
    """Classify a prompt using the OpenAI GPT-5 model.

    Returns a standardized result dict with label, reasoning, token counts, and cost.
    """
    model = config.get("model")
    temperature = config.get("temperature", 0.0)
    max_tokens = config.get("max_tokens", 400)

    messages = [{"role": "user", "content": prompt}]

    attempt = 0
    backoffs = [2, 4, 8]
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message["content"]
            break
        except Exception as exc:
            if attempt >= len(backoffs):
                raise
            time.sleep(backoffs[attempt])
            attempt += 1

    label = _parse_label(text)
    reasoning = _parse_reasoning(text)
    usage = getattr(response, "usage", {})
    tokens_in = int(getattr(usage, "prompt_tokens", usage.get("prompt_tokens", 0)))
    tokens_out = int(getattr(usage, "completion_tokens", usage.get("completion_tokens", 0)))
    cost_usd = _estimate_cost(model, tokens_in, tokens_out)

    return {
        "label": label,
        "reasoning": reasoning,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
    }

