"""
Shared prompt templates for GPT-5 and Gemini 3.
Three variants: zero_shot, few_shot, chain_of_thought.
Templates are identical across both models so any difference is attributable
to the model, not the prompt design.

    from src.llm.prompts import build_prompt
"""

# TODO: implement build_prompt(variant, post, few_shot_examples=None) -> str
