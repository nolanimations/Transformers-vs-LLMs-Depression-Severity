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

    # Call generate_content with temperature and max_output_tokens forwarded
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            ),
        )
    except Exception:
        # Let the caller handle network/SDK errors; raise after any cleanup if needed
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
    text = None
    reasoning = None

    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            reasoning = part.text
        else:
            text = part.text

    label = _parse_label(text) if text else "unknown"
    reasoning = reasoning.strip() if reasoning else None
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
