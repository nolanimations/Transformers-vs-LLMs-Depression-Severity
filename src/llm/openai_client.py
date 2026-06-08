"""
GPT client using the OpenAI Responses API with native reasoning support.
Reads OPENAI_API_KEY from .env.

Reasoning behaviour mirrors the Gemini client:
  - zero_shot / few_shot  : reasoning effort "none"  (fast, cheap, no thinking tokens)
  - chain_of_thought      : reasoning effort "medium" + summary "auto"
                            (API-level reasoning, summary captured as `reasoning` field)

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

client = openai.OpenAI(api_key=OPENAI_API_KEY)

LABELS = {"minimum", "mild", "moderate", "severe"}

# Pricing per 1,000 tokens  ($0.75 input / $4.50 output per million)
OPENAI_PRICING = {
    "gpt-5.4-mini": {"input": 0.00075, "output": 0.0045},
}


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return "unknown"
    label = label.strip().lower()
    return label if label in LABELS else "unknown"


def _parse_label(text: str) -> str:
    text = text.strip()
    try:
        return _normalize_label(json.loads(text).get("label", ""))
    except Exception:
        pass
    match = re.search(r'"label"\s*:\s*"([a-zA-Z]+)"', text)
    if match:
        return _normalize_label(match.group(1))
    return "unknown"


def _extract_reasoning_summary(response) -> str | None:
    """Pull reasoning summary text from the Responses API output array."""
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "reasoning":
            for s in getattr(item, "summary", []) or []:
                text = getattr(s, "text", None)
                if text:
                    return text.strip()
    return None


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    prices = OPENAI_PRICING.get(model, {})
    return round(
        (tokens_in * prices.get("input", 0.0) +
         tokens_out * prices.get("output", 0.0)) / 1000.0,
        8,
    )


# ── Main classify function ─────────────────────────────────────────────────────

def classify(prompt: str, config: dict) -> dict:
    """
    Classify a post using the OpenAI Responses API.

    Reasoning is configured per variant:
      - zero_shot / few_shot  → effort "none"   (no reasoning tokens)
      - chain_of_thought      → effort "medium", summary "auto"
                                (reasoning summary captured and returned)

    Returns
    -------
    dict with keys: label, reasoning, tokens_in, tokens_out, cost_usd
    """
    model    = config.get("model", "gpt-5.4-mini")
    variant  = config.get("variant", "zero_shot")
    max_tokens = config.get("max_tokens", 1000)

    # Reasoning config — mirrors Gemini's ThinkingConfig logic
    use_reasoning = variant == "chain_of_thought"
    reasoning_cfg = (
        {"effort": "medium", "summary": "detailed"}
        if use_reasoning
        else {"effort": "none"}
    )

    # System message keeps the model focused on JSON output
    input_messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical classification assistant. "
                "Always respond with valid JSON only, matching the requested schema exactly."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    attempt = 0
    backoffs = [2, 4, 8]
    while True:
        try:
            response = client.responses.create(
                model=model,
                input=input_messages,
                reasoning=reasoning_cfg,
                text={"format": {"type": "json_object"}},
                max_output_tokens=max_tokens,
            )
            break
        except Exception as exc:
            if attempt >= len(backoffs):
                raise
            print(f"OpenAI error: {exc}. Retrying in {backoffs[attempt]}s...")
            time.sleep(backoffs[attempt])
            attempt += 1

    # Check for incomplete response (ran out of tokens during reasoning)
    if getattr(response, "status", None) == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", "unknown")
        raise RuntimeError(
            f"Response incomplete ({reason}). "
            f"Increase max_tokens in configs/llm.yaml (currently {max_tokens}). "
            f"Recommended minimum: 1500 for chain_of_thought."
        )

    text = getattr(response, "output_text", None) or ""
    label = _parse_label(text)

    # Reasoning summary (chain_of_thought only)
    reasoning = _extract_reasoning_summary(response) if use_reasoning else None

    # Token counts — Responses API uses input_tokens / output_tokens
    usage = getattr(response, "usage", None)
    tokens_in  = int(getattr(usage, "input_tokens",  0) or 0)
    tokens_out = int(getattr(usage, "output_tokens", 0) or 0)

    return {
        "label":     label,
        "reasoning": reasoning,
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
        "cost_usd":  _estimate_cost(model, tokens_in, tokens_out),
    }
