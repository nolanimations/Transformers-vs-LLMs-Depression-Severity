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

OPENAI_PRICING = {
    "gpt-5": {"prompt": 0.0015, "completion": 0.0025},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.010}, # Ter referentie
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
    """Classify a prompt using the OpenAI API (GPT-5 Optimized)."""
    model = config.get("model", "gpt-5")
    max_tokens = config.get("max_tokens", 400)
    
    temperature = config.get("temperature", 0.0)

    messages = [
        {"role": "system", "content": "You are a clinical assistant. Output your classification in valid JSON format."},
        {"role": "user", "content": prompt}
    ]

    # Stel de parameters direct correct in voor GPT-5
    api_params = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_tokens
    }

    if not any(m in model.lower() for m in ["gpt-5", "o1"]):
        api_params["temperature"] = temperature

    attempt = 0
    backoffs = [2, 4, 8]
    
    while True:
        try:
            response = client.chat.completions.create(**api_params)
            text = response.choices[0].message.content
            break 
        except Exception as exc:
            if attempt >= len(backoffs):
                raise exc
            print(f"OpenAI error: {exc}. Retrying in {backoffs[attempt]}s...")
            time.sleep(backoffs[attempt])
            attempt += 1

    # Verwerking en metadata
    usage = response.usage
    tokens_in = usage.prompt_tokens
    tokens_out = usage.completion_tokens

    return {
        "label": _parse_label(text),
        "reasoning": _parse_reasoning(text),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": _estimate_cost(model, tokens_in, tokens_out),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model": model,
    }