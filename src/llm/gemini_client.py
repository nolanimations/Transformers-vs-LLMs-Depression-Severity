"""
Gemini 3 API caller with exponential-backoff retries, JSON response parsing,
and per-call cost + token logging.
Reads GOOGLE_API_KEY from .env.

    from src.llm.gemini_client import classify
"""

import json
import os
import re
import time
from google.genai import types

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise EnvironmentError("GOOGLE_API_KEY is not set in the environment")
genai.api_key = GOOGLE_API_KEY

LABELS = {"minimum", "mild", "moderate", "severe"}

GEMINI_PRICING = {
    # Estimated prices per 1,000 tokens; adjust if real billing differs.
    "gemini-3-flash": {"prompt": 0.0012, "completion": 0.0012},
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
    # Robustly extract a structured `label` and `reasoning` from a Gemini SDK response.

    # Order of attempts:
    # 1. Use `candidate.structured` or `response.output` if present (validated JSON object).
    # 2. Try `response.text` as JSON.
    # 3. Join `candidate.content` parts into text and parse JSON.
    # 4. Regex-extract first {...} and parse.
    # 5. Fallback: try to use thought parts as `reasoning` and text parts for label.

    try:
        cand = response.candidates[0]
    except Exception:
        return "unknown", None

    # 1) structured object from SDK
    structured = getattr(cand, "structured", None) or getattr(response, "output", None)
    parsed = {}
    if isinstance(structured, dict):
        parsed = structured
    else:
        # 2) try response.text
        raw = getattr(response, "text", None) or ""
        # 3) build raw from content parts if needed
        if not raw:
            try:
                parts = getattr(cand, "content", []) or []
                raw = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None))
            except Exception:
                raw = ""

        # 4) parse JSON or extract {...}
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, flags=re.S)
                if m:
                    try:
                        parsed = json.loads(m.group())
                    except Exception:
                        parsed = {}

    # Extract fields if parsed is a dict
    label = _normalize_label(parsed.get("label") if isinstance(parsed, dict) else None)
    reasoning = parsed.get("reasoning") if isinstance(parsed, dict) else None
    if isinstance(reasoning, str):
        reasoning = reasoning.strip()
    else:
        reasoning = None

    # 5) fallback: use thought parts for reasoning if nothing parsed
    if not reasoning:
        try:
            parts = getattr(cand, "content", []) or []
            thought_texts = [getattr(p, "text", "") for p in parts if getattr(p, "thought", False) and getattr(p, "text", None)]
            if thought_texts:
                reasoning = " ".join(thought_texts).strip()
        except Exception:
            pass

    return label, reasoning


# def classify(prompt: str, config: dict) -> dict:
#     """Classify a prompt using the Gemini 3 Flash model.

#     Returns a standardized result dict with label, reasoning, token counts, and cost.
#     """
#     model = config.get("model")
#     temperature = config.get("temperature", 0.0)
#     max_tokens = config.get("max_tokens", 400)

#     client = genai.Client(api_key=GOOGLE_API_KEY)
#     chat = client.chats.create(
#         model=model,
#         config={"temperature": temperature, "max_output_tokens": max_tokens},
#     )

#     attempt = 0
#     backoffs = [2, 4, 8]
#     while True:
#         try:
#             response = chat.send_message(prompt)
#             text = response.text or ""
#             break
#         except Exception:
#             if attempt >= len(backoffs):
#                 raise
#             time.sleep(backoffs[attempt])
#             attempt += 1

#     label = text
#     reasoning = _parse_reasoning(text)
#     usage = getattr(response, "usage_metadata", None)
#     prompt_tokens = _safe_int(getattr(usage, "prompt_token_count", None))
#     completion_tokens = _safe_int(getattr(usage, "candidates_token_count", None))
#     cost_usd = _estimate_cost(model, prompt_tokens, completion_tokens)

#     return {
#         "label": label,
#         "reasoning": reasoning,
#         "tokens_in": prompt_tokens,
#         "tokens_out": completion_tokens,
#         "cost_usd": cost_usd,
#     }

def classify(prompt: str, config: dict) -> dict:
    """Classify a prompt using the Gemini 3 Flash model.

    Returns a standardized result dict with label, reasoning, token counts, and cost.
    """
    model = config.get("model")
    temperature = config.get("temperature", 0.0)
    max_tokens = config.get("max_tokens", 400)

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Ensure a model is set; default to a reasonable Gemini variant
    model = model or "gemini-3.5-flash"

    # Class to base response on
    class LabelResponse(BaseModel):
        label: str = Field(description="The predicted label for the input post, one of 'minimum', 'mild', 'moderate', 'severe', or 'unknown' if parsing fails.")
        reasoning: Optional[str] = Field(description="The reasoning behind the predicted label, if available.")

    # Only enable thinking for chain_of_thought — zero/few-shot don't need it
    # and it wastes tokens (= cost) when unused
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
    # chat = client.chats.create(
    #     model=model,
    #     config={"temperature": temperature, "max_output_tokens": max_tokens},
    # )

    # attempt = 0
    # backoffs = [2, 4, 8]
    # while True:
    #     try:
    #         response = chat.send_message(prompt)
    #         text = response.text or ""
    #         break
    #     except Exception:
    #         if attempt >= len(backoffs):
    #             raise
    #         time.sleep(backoffs[attempt])
    #         attempt += 1
    # Extract label and reasoning using robust extractor
    label, reasoning = _extract_label_reasoning_from_response(response)
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = _safe_int(getattr(usage, "prompt_token_count", None))
    completion_tokens = _safe_int(getattr(usage, "candidates_token_count", None))
    cost_usd = _estimate_cost(model, prompt_tokens, completion_tokens)

    return {
        "label": label,          # parsed label string, e.g. "mild"
        "reasoning": reasoning,
        "tokens_in": prompt_tokens,
        "tokens_out": completion_tokens,
        "cost_usd": cost_usd,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model": model,
    }


