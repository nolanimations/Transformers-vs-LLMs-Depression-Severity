"""
Gemini 3 API caller with exponential-backoff retries, JSON response parsing,
and per-call cost + token logging.
Reads GOOGLE_API_KEY from .env.

    from src.llm.gemini_client import classify
"""

import json
import os
import re
from google.genai import types

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from typing import Literal, Optional

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise EnvironmentError("GOOGLE_API_KEY is not set in the environment")
genai.api_key = GOOGLE_API_KEY

LABELS = {"minimum", "mild", "moderate", "severe"}

GEMINI_PRICING = {
    "gemini-3-flash-preview": {"prompt": 0.0012, "completion": 0.0012},
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
    prices = GEMINI_PRICING.get(model, {})
    prompt_rate = prices.get("prompt", 0.0)
    completion_rate = prices.get("completion", 0.0)
    return round((prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1000.0, 8)


def _safe_int(value, default=0) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _extract_label_reasoning_from_response(response) -> tuple[str, str | None]:
    """
    Robustly extract label and reasoning from a Gemini SDK response.

    Attempt order:
      1. response.parsed  — Pydantic model returned when response_schema is set (most reliable)
      2. response.text    — raw JSON text, parse manually
      3. candidate content parts (non-thought) joined and parsed
      4. Regex fallback on any available text
    Reasoning is separately collected from thought parts (ThinkingConfig).
    """
    label    = "unknown"
    reasoning = None

    # ── 1. response.parsed (Pydantic instance from response_schema) ──────────
    parsed_obj = getattr(response, "parsed", None)
    if parsed_obj is not None:
        raw_label = getattr(parsed_obj, "label", None)
        label = _normalize_label(raw_label)
        raw_r = getattr(parsed_obj, "reasoning", None)
        reasoning = raw_r.strip() if isinstance(raw_r, str) and raw_r.strip() else None

    # ── 2. response.text as JSON ──────────────────────────────────────────────
    if label == "unknown":
        raw = getattr(response, "text", None) or ""
        if raw:
            label = _parse_label(raw)
            if reasoning is None:
                reasoning = _parse_reasoning(raw)

    # ── 3. Candidate content parts (non-thought only) ────────────────────────
    if label == "unknown":
        try:
            cand    = response.candidates[0]
            content = getattr(cand, "content", None)
            parts   = getattr(content, "parts", []) or []
            text_parts = [
                getattr(p, "text", "")
                for p in parts
                if not getattr(p, "thought", False) and getattr(p, "text", None)
            ]
            raw = "".join(text_parts)
            if raw:
                label = _parse_label(raw)
                if reasoning is None:
                    reasoning = _parse_reasoning(raw)
        except Exception:
            pass

    # ── 4. Extract reasoning from thought parts (ThinkingConfig) ─────────────
    if reasoning is None:
        try:
            cand    = response.candidates[0]
            content = getattr(cand, "content", None)
            parts   = getattr(content, "parts", []) or []
            thoughts = [
                getattr(p, "text", "")
                for p in parts
                if getattr(p, "thought", False) and getattr(p, "text", None)
            ]
            if thoughts:
                reasoning = " ".join(thoughts).strip()
        except Exception:
            pass

    return label, reasoning


def classify(prompt: str, config: dict) -> dict:
    """Classify a prompt using the Gemini 3 Flash model.

    Returns a standardized result dict with label, reasoning, token counts, and cost.
    """
    model = config.get("model")
    temperature = config.get("temperature", 0.0)
    max_tokens = config.get("max_tokens", 1000)

    client = genai.Client(api_key=GOOGLE_API_KEY)

    model = model or "gemini-3.5-flash"

    class LabelResponse(BaseModel):
        label: Literal["minimum", "mild", "moderate", "severe"] = Field(
            description="Depression severity label for the post."
        )
        reasoning: Optional[str] = Field(
            default=None,
            description="Brief reasoning for the classification.",
        )

    use_thinking = config.get("variant") == "chain_of_thought"
    config_kwargs = dict(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        response_schema=LabelResponse,
    )
    if use_thinking:
        config_kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception:
        raise
    label, reasoning = _extract_label_reasoning_from_response(response)
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = _safe_int(getattr(usage, "prompt_token_count", None))
    completion_tokens = _safe_int(getattr(usage, "candidates_token_count", None))
    cost_usd = _estimate_cost(model, prompt_tokens, completion_tokens)

    return {
        "label": label,
        "reasoning": reasoning,
        "tokens_in": prompt_tokens,
        "tokens_out": completion_tokens,
        "cost_usd": cost_usd,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model": model,
    }
