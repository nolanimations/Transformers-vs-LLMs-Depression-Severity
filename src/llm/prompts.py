"""
Shared prompt templates for GPT-5 and Gemini 3.
Three variants: zero_shot, few_shot, chain_of_thought.
Templates are identical across both models so any difference is attributable
to the model, not the prompt design.

    from src.llm.prompts import build_prompt
"""

# TODO: implement build_prompt(variant, post, few_shot_examples=None) -> str

from dotenv import load_dotenv
import os
from pathlib import Path
import pandas as pd
import random

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

def build_prompt(variant: str, post: str, few_shot_examples=None) -> str:
    base_instruction = (
        "You are an assistant that classifies social media posts based on the severity of depression symptoms.\n"
        "Classify the following post into one of four categories based on the described symptoms:\n"
        "Choose one label that best fits the symptoms described in the post:\n"
        "- minimum: no or minimal symptoms\n"
        "- mild: mild symptoms, still functions mostly normally\n"
        "- moderate: clear symptoms, noticeably reduced functioning\n"
        "- severe: severe symptoms, significant suffering or dysfunction\n\n"
    )

    if variant == "zero_shot":
        return (
            base_instruction +
            'Provide your answer as JSON: {"label": "<choice>"}\n\n'
            f'Post: "{post}"'
        )

    if variant in {"few_shot", "chain_of_thought"}:
        if few_shot_examples is None:
            repo_root = Path(__file__).resolve().parents[2]
            train_path = repo_root / "data" / "splits" / "train.csv"
            if not train_path.exists():
                raise FileNotFoundError(f"Training file not found at {train_path}")

            df = pd.read_csv(train_path)
            # Ensure consistent sampling
            random_state = 42
            few_shot_examples = (
                df.groupby("label_id")
                .apply(lambda g: g.sample(2, random_state=random_state))
                .reset_index(drop=True)[["text", "label"]]
                .to_dict("records")
            )

        examples_block = "\n\n".join(
            
            f'Post: "{example["text"]}"\nLabel: "{example["label"]}"'
            for example in few_shot_examples
        )

        if variant == "few_shot":
            return (
                base_instruction +
                f"{examples_block}\n\n"
                'Provide your answer as JSON: {"label": "<choice>"}\n\n'
                f'Post: "{post}"'
            )

        return (
            base_instruction +
            f"{examples_block}\n\n"
            'Please provide just your answer and provide it as JSON:\n'
            '{"reasoning": "<your reasoning>", "label": "<choice>"}\n\n'
            f'Post: "{post}"'
        )

    raise ValueError("Invalid variant. Choose from 'zero_shot', 'few_shot', or 'chain_of_thought'.")

