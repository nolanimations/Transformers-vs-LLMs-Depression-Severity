"""
Per-post informality score for H2 (slang/register sensitivity analysis).
Score = (# emoji tokens + # OOV tokens + # all-caps tokens + # repeated-char tokens)
        / token_count

    from src.slang import add_informality_score
"""

# TODO: implement add_informality_score(df) -> df with 'informality_score' column
