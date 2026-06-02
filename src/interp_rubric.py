"""
Interpretability rubric (H3) — blinded human rating of SHAP token highlights
vs LLM chain-of-thought rationales.

H3 verdict rule (pre-registered in the project plan):
    LLM CoT wins if mean Likert on ≥2/3 criteria is strictly higher than SHAP
    AND Krippendorff's α ≥ 0.6 (acceptable agreement) for those criteria.
    Otherwise: "no clear advantage" / "SHAP comparable".

Criteria (1 = strongly disagree, 5 = strongly agree):
  1. faithfulness       — explanation only references things actually in the post
  2. clinical_relevance — non-expert can understand WHY this severity was predicted
  3. actionability      — points to a concrete cue a clinician could independently verify

Usage
-----
Step 1 — run notebook 07_shap_examples.ipynb (generates examples_meta.json)
Step 2 — run full LLM eval, chain_of_thought variant for both models
Step 3 — generate rating sheet:
    python src/interp_rubric.py generate
Step 4 — team fills in CSVs independently (one per person)
Step 5 — analyze:
    python src/interp_rubric.py analyze
"""

import argparse
import base64
import csv
import html
import json
import random
from pathlib import Path

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

CRITERIA = ["faithfulness", "clinical_relevance", "actionability"]

CRITERION_DESC = {
    "faithfulness":
        "Does the explanation <em>only</em> reference things actually present in the post text? "
        "(1 = heavily hallucinated / fabricated; 5 = every cited cue is verbatim in the post)",
    "clinical_relevance":
        "Would a non-expert reader understand <em>why</em> this specific severity label was assigned? "
        "(1 = completely opaque; 5 = crystal-clear causal link between cues and label)",
    "actionability":
        "Does the explanation point to a concrete linguistic cue a clinician could independently verify? "
        "(1 = vague / circular; 5 = specific, verifiable cue named explicitly)",
}

RATER_NAMES = ["nolan", "xander", "vincent", "efraim"]

# Explanation slots shown to raters
EXPL_IDS = ["SHAP", "CoT_A", "CoT_B"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_examples_meta(shap_dir: str | Path) -> list[dict]:
    """
    Load the curated SHAP examples metadata saved by notebook 07.

    Returns a list of dicts with keys:
        fig_index, figure_file, idx, text, true_label, pred_label,
        confidence, correct, source, shap_top_tokens
    """
    meta_path = Path(shap_dir) / "examples_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"examples_meta.json not found at {meta_path}.\n"
            "Run the 'Save metadata' cell in notebooks/07_shap_examples.ipynb first."
        )
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def load_cot_rationales(log_dir: str | Path, model_key: str) -> dict[int, str]:
    """
    Load chain_of_thought rationales from all JSONL log files in log_dir.

    If multiple files exist (partial/resumed runs), they are merged;
    later timestamps overwrite earlier entries for the same test-set index.

    Parameters
    ----------
    log_dir   : directory containing *chain_of_thought*.jsonl files
    model_key : "gpt" or "gemini" (used only in the error message)

    Returns
    -------
    dict : test_set_index (int) -> reasoning text (str)
           Entries where reasoning is None/empty are excluded.
    """
    log_dir = Path(log_dir)
    files   = sorted(log_dir.glob("*chain_of_thought*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No chain_of_thought JSONL found in {log_dir}.\n"
            f"Run: python src/llm/run_llm_eval.py --model {model_key} --variant chain_of_thought"
        )

    records: dict[int, dict] = {}
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    idx = obj.get("index")
                    if idx is not None:
                        records[int(idx)] = obj
                except json.JSONDecodeError:
                    continue

    return {
        idx: rec["reasoning"]
        for idx, rec in records.items()
        if rec.get("reasoning")
    }


# ── HTML helpers ───────────────────────────────────────────────────────────────

def _embed_png(png_path: Path) -> str:
    """Return an <img> tag with the PNG embedded as base64 (self-contained HTML)."""
    if not png_path.exists():
        return f'<p style="color:#c0392b;">&#9888; Figure not found: {png_path.name}</p>'
    data = base64.b64encode(png_path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;border-radius:4px;" />'


def _cot_to_html(text: str) -> str:
    """Safely convert CoT rationale text to HTML (escape entities, preserve newlines)."""
    return html.escape(text).replace("\n", "<br>")


_HTML_STYLE = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  max-width: 1450px; margin: 24px auto; padding: 0 20px; color: #222;
}
h1 { border-bottom: 3px solid #4e9a8c; padding-bottom: 8px; }
h2 { color: #4e9a8c; }
.instructions {
  background: #f0f8f5; border: 1px solid #4e9a8c; border-radius: 8px;
  padding: 18px 22px; margin-bottom: 36px;
}
.instructions ol { margin: 8px 0 0 0; }
.instructions li { margin-bottom: 6px; }
.post-block {
  border: 1px solid #ccc; border-radius: 8px; padding: 20px 24px;
  margin-bottom: 44px; background: #fafafa;
}
.post-block h3 { margin: 0 0 12px 0; font-size: 1.05em; }
blockquote.post-text {
  border-left: 4px solid #4e9a8c; margin: 0 0 18px 0;
  padding: 8px 14px; background: white; font-style: italic;
  font-size: 0.95em; line-height: 1.55;
}
.expls {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
}
.expl {
  border: 1px solid #ddd; border-radius: 6px; padding: 14px;
  background: white; min-height: 160px;
}
.expl h4 { margin: 0 0 10px 0; font-size: 0.95em; color: #4e9a8c; }
.expl-body { font-size: 0.88em; line-height: 1.6; }
.label {
  background: #e8f4f0; padding: 2px 7px; border-radius: 3px;
  font-weight: bold; font-size: 0.9em;
}
.correct { color: #277a27; }
.wrong   { color: #aa2020; }
.csv-hint { font-size: 0.78em; color: #999; margin-top: 14px; }
@media (max-width: 960px) { .expls { grid-template-columns: 1fr; } }
"""


def _build_html(posts: list[dict]) -> str:
    """Render the complete rating-sheet HTML from the posts list."""

    criteria_items = "".join(
        f'<li><strong>{c.replace("_", " ").title()}</strong> &mdash; '
        f'<span style="color:#555">{CRITERION_DESC[c]}</span></li>'
        for c in CRITERIA
    )

    post_blocks = []
    for p in posts:
        correct_cls = "correct" if p["correct"] else "wrong"
        status      = "&#10003;" if p["correct"] else "&#10007;"
        truncated   = p["text"][:320] + ("&hellip;" if len(p["text"]) > 320 else "")
        post_text   = html.escape(truncated).replace("&amp;hellip;", "&hellip;")

        expls_html = []
        for eid, body_html in [
            ("SHAP &mdash; MentalBERT token attributions", p["shap_html"]),
            ("CoT Rationale A",                            p["cot_a_html"]),
            ("CoT Rationale B",                            p["cot_b_html"]),
        ]:
            expls_html.append(
                f'<div class="expl">'
                f'<h4>{eid}</h4>'
                f'<div class="expl-body">{body_html}</div>'
                f'</div>'
            )

        start_row    = (p["post_id"] - 1) * 3 + 2   # +2: header row + 1-indexing
        expls_joined = "\n".join(expls_html)

        post_blocks.append(
            f'<div class="post-block" id="post-{p["post_id"]}">\n'
            f'  <h3>Post {p["post_id"]}&thinsp;/&thinsp;{len(posts)}'
            f' &nbsp;&mdash;&nbsp; True label: <span class="label">{p["true_label"]}</span>'
            f' &nbsp;&mdash;&nbsp; SHAP predicted: '
            f'<span class="label {correct_cls}">{p["pred_label"]} {status}</span>'
            f' &nbsp;&mdash;&nbsp; confidence: {p["confidence"]:.2f}</h3>\n'
            f'  <blockquote class="post-text">{post_text}</blockquote>\n'
            f'  <div class="expls">\n{expls_joined}\n  </div>\n'
            f'  <p class="csv-hint">'
            f'In your CSV: Post {p["post_id"]} = rows {start_row}&ndash;{start_row + 2}'
            f' &nbsp;|&nbsp; SHAP / CoT A / CoT B</p>\n'
            f'</div>'
        )

    all_blocks = "\n\n".join(post_blocks)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        "<title>Interpretability Rating &mdash; Depression Severity</title>\n"
        f"<style>{_HTML_STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>Depression Severity &mdash; Interpretability Rating Sheet</h1>\n\n"
        '<div class="instructions">\n'
        '  <h2 style="margin-top:0">How to rate</h2>\n'
        f"  <p>For each of the {len(posts)} posts below you will see three explanations: "
        "the <strong>SHAP token attribution</strong> from MentalBERT (bar chart) "
        "and two <strong>chain-of-thought rationales</strong> from two different LLMs "
        "(which model is A and which is B is revealed after all ratings are collected).</p>\n"
        "  <p>Rate each explanation on three criteria using a <strong>1&ndash;5 Likert scale</strong>:</p>\n"
        f"  <ol>{criteria_items}</ol>\n"
        "  <p>Enter your scores in your personal CSV: "
        "<code>results/interp_rubric/ratings/rater_YOUR_NAME.csv</code></p>\n"
        '  <p style="color:#aa2020;font-weight:bold;">'
        "&#9888; Do NOT open <code>blind_key.json</code> until all four team members "
        "have submitted their CSVs.</p>\n"
        "  <p>Rate <em>independently</em> &mdash; do not discuss with teammates until all "
        "ratings are in. After submitting, run: "
        "<code>python src/interp_rubric.py analyze</code></p>\n"
        "</div>\n\n"
        + all_blocks
        + "\n</body>\n</html>"
    )


# ── generate() ────────────────────────────────────────────────────────────────

def generate(
    output_dir:     str | Path = "results/interp_rubric",
    shap_dir:       str | Path = "results/figures/shap",
    gpt_log_dir:    str | Path = "results/llm_logs/gpt5",
    gemini_log_dir: str | Path = "results/llm_logs/gemini3",
    seed:           int        = 42,
) -> None:
    """
    Generate the blinded HTML rating sheet and per-rater CSV templates.

    Outputs
    -------
    results/interp_rubric/
        rating_sheet.html          open in any browser; shows all posts + explanations
        blind_key.json             DO NOT open until all ratings are submitted
        ratings/
            rater_nolan.csv
            rater_xander.csv       pre-filled post/expl rows; fill in the Likert scores
            rater_vincent.csv
            rater_efraim.csv
    """
    output_dir  = Path(output_dir)
    ratings_dir = output_dir / "ratings"
    ratings_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # ── Load SHAP examples metadata ──────────────────────────────────────────
    examples = load_examples_meta(shap_dir)
    shap_dir = Path(shap_dir)

    # ── Load CoT rationales (warn but continue if LLM eval not done yet) ─────
    gpt_rationales:    dict[int, str] = {}
    gemini_rationales: dict[int, str] = {}

    try:
        gpt_rationales = load_cot_rationales(gpt_log_dir, "gpt")
        print(f"  GPT rationales loaded:    {len(gpt_rationales)} posts")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")

    try:
        gemini_rationales = load_cot_rationales(gemini_log_dir, "gemini")
        print(f"  Gemini rationales loaded: {len(gemini_rationales)} posts")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")

    # ── Build posts, assign blind slots ─────────────────────────────────────
    posts:     list[dict] = []
    blind_key: dict       = {}   # post_id (str) -> {"CoT_A": "gpt"/"gemini", "CoT_B": ...}
    skipped = 0

    for ex in examples:
        idx         = ex["idx"]
        gpt_text    = gpt_rationales.get(idx)
        gemini_text = gemini_rationales.get(idx)

        # Skip posts where BOTH rationales are missing
        if gpt_text is None and gemini_text is None:
            skipped += 1
            continue

        # Graceful fallback when one model failed on this post
        gpt_text    = gpt_text    or "[GPT rationale unavailable — LLM call failed for this post]"
        gemini_text = gemini_text or "[Gemini rationale unavailable — LLM call failed for this post]"

        # Randomly swap GPT/Gemini into slots A and B (same shuffle for all raters)
        if rng.random() < 0.5:
            cot_a_text, cot_b_text = gpt_text,    gemini_text
            assignment = {"CoT_A": "gpt", "CoT_B": "gemini"}
        else:
            cot_a_text, cot_b_text = gemini_text, gpt_text
            assignment = {"CoT_A": "gemini", "CoT_B": "gpt"}

        post_id = len(posts) + 1
        blind_key[str(post_id)] = assignment

        fig_name  = ex.get(
            "figure_file",
            f"{ex['fig_index']:03d}_{ex['true_label']}_pred{ex['pred_label']}.png",
        )
        shap_html = _embed_png(shap_dir / fig_name)

        posts.append({
            "post_id":    post_id,
            "idx":        idx,
            "text":       ex["text"],
            "true_label": ex["true_label"],
            "pred_label": ex["pred_label"],
            "confidence": ex["confidence"],
            "correct":    ex["correct"],
            "shap_html":  shap_html,
            "cot_a_html": _cot_to_html(cot_a_text),
            "cot_b_html": _cot_to_html(cot_b_text),
        })

    print(f"\n{len(posts)} posts included, {skipped} skipped (no rationales found).")

    if not posts:
        print(
            "[ERROR] No posts to rate. Run LLM eval (chain_of_thought) for at least one "
            "model first, then re-run `generate`."
        )
        return

    # ── Write blind key (secret until after rating) ──────────────────────────
    key_path = output_dir / "blind_key.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(blind_key, f, indent=2)
    print(f"Blind key  ->  {key_path}  (keep secret until all ratings are in!)")

    # ── Write HTML ───────────────────────────────────────────────────────────
    html_path = output_dir / "rating_sheet.html"
    html_path.write_text(_build_html(posts), encoding="utf-8")
    print(f"HTML sheet ->  {html_path}")

    # ── Write per-rater CSV templates ────────────────────────────────────────
    header = ["post_id", "true_label", "pred_label", "explanation_id"] + CRITERIA
    for name in RATER_NAMES:
        csv_path = ratings_dir / f"rater_{name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for post in posts:
                for eid in EXPL_IDS:
                    writer.writerow([
                        post["post_id"],
                        post["true_label"],
                        post["pred_label"],
                        eid,
                        "",   # faithfulness
                        "",   # clinical_relevance
                        "",   # actionability
                    ])
        print(f"CSV template -> {csv_path}")

    print(
        "\n── Next steps ────────────────────────────────────────────────────────\n"
        "  1. Open rating_sheet.html in a browser\n"
        "  2. Each team member fills in their OWN rater_NAME.csv independently\n"
        "  3. Once all 4 CSVs are done: python src/interp_rubric.py analyze"
    )


# ── analyze() ─────────────────────────────────────────────────────────────────

def _load_ratings(output_dir: Path) -> list[dict]:
    """Load all rater_*.csv files. Rows with any blank criterion are skipped."""
    ratings_dir = output_dir / "ratings"
    all_ratings: list[dict] = []

    csv_files = sorted(ratings_dir.glob("rater_*.csv"))
    if not csv_files:
        print(f"[WARNING] No CSV files found in {ratings_dir}")
        return []

    for csv_path in csv_files:
        rater = csv_path.stem.replace("rater_", "")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    r: dict = {
                        "rater":          rater,
                        "post_id":        int(row["post_id"]),
                        "true_label":     row.get("true_label", ""),
                        "explanation_id": row["explanation_id"].strip(),
                    }
                    complete = True
                    for c in CRITERIA:
                        val = row.get(c, "").strip()
                        if val == "":
                            complete = False
                            break
                        score = int(val)
                        if not (1 <= score <= 5):
                            raise ValueError(f"Score {score} out of range 1–5")
                        r[c] = score
                    if complete:
                        all_ratings.append(r)
                except (ValueError, KeyError):
                    continue

    return all_ratings


def _unblind(ratings: list[dict], blind_key: dict) -> list[dict]:
    """Replace CoT_A / CoT_B labels with actual model names using blind_key."""
    result = []
    for r in ratings:
        r = dict(r)
        eid = r["explanation_id"]
        if eid in ("CoT_A", "CoT_B"):
            mapping = blind_key.get(str(r["post_id"]), {})
            r["explanation_id"] = mapping.get(eid, eid)
        result.append(r)
    return result


def _krippendorff_alpha(reliability_data: np.ndarray) -> float:
    """
    Ordinal Krippendorff's alpha.
    Uses the `krippendorff` package if available; otherwise falls back to a
    manual implementation (Krippendorff 2011 ordinal metric).

    Parameters
    ----------
    reliability_data : (n_raters, n_items) float array, NaN for missing values
    """
    try:
        import krippendorff as kd
        return float(kd.alpha(reliability_data, level_of_measurement="ordinal"))
    except ImportError:
        pass

    # Manual fallback — d(v,u)^2 = (v-u)^2 for ordinal
    data = np.array(reliability_data, dtype=float)
    _, n_c = data.shape

    # Observed disagreement
    do_sum, do_n = 0.0, 0
    for col in range(n_c):
        vals = data[:, col]
        vals = vals[~np.isnan(vals)]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                do_sum += (vals[i] - vals[j]) ** 2
                do_n   += 1
    if do_n == 0:
        return float("nan")
    d_o = do_sum / do_n

    # Expected disagreement (all values pooled)
    all_v = data[~np.isnan(data)]
    n_all = len(all_v)
    if n_all < 2:
        return float("nan")
    de_sum = sum(
        (all_v[i] - all_v[j]) ** 2
        for i in range(n_all)
        for j in range(i + 1, n_all)
    )
    d_e = de_sum / (n_all * (n_all - 1) / 2)

    return 1.0 if d_e == 0 else float(1.0 - d_o / d_e)


def analyze(output_dir: str | Path = "results/interp_rubric") -> None:
    """
    Load filled rater CSVs, unblind model identities, compute mean Likert ± SD
    and Krippendorff's alpha per criterion, then print the pre-registered H3 verdict.

    Writes a summary to results/interp_rubric/h3_results.json.
    """
    output_dir = Path(output_dir)

    key_path = output_dir / "blind_key.json"
    if not key_path.exists():
        raise FileNotFoundError(
            f"blind_key.json not found in {output_dir}. Run `generate` first."
        )
    with open(key_path) as f:
        blind_key = json.load(f)

    raw_ratings = _load_ratings(output_dir)
    if not raw_ratings:
        print("No completed ratings found. Fill in the rater CSV files first.")
        return

    n_raters = len(set(r["rater"]   for r in raw_ratings))
    n_posts  = len(set(r["post_id"] for r in raw_ratings))
    print(f"Loaded {len(raw_ratings)} rating rows  |  {n_raters} raters  |  {n_posts} posts\n")

    ratings = _unblind(raw_ratings, blind_key)
    models  = ["SHAP", "gpt", "gemini"]

    # ── Mean ± SD per model per criterion ────────────────────────────────────
    col_w = 24
    print(f"{'── Mean Likert scores (1–5)':<50}")
    print(f"{'Model':<12}" + "".join(f"  {c[:col_w]:<{col_w}}" for c in CRITERIA))

    means: dict[str, dict[str, float]] = {}
    for model in models:
        rows = [r for r in ratings if r["explanation_id"] == model]
        means[model] = {}
        line = f"{model:<12}"
        for c in CRITERIA:
            vals = [r[c] for r in rows if c in r]
            if vals:
                m, s = float(np.mean(vals)), float(np.std(vals))
                means[model][c] = m
                cell = f"{m:.2f} ± {s:.2f}"
            else:
                means[model][c] = float("nan")
                cell = "n/a"
            line += f"  {cell:<{col_w}}"
        print(line)

    # ── Krippendorff's alpha per model per criterion ──────────────────────────
    print(f"\n{'── Krippendorff alpha (ordinal)'}")
    alphas: dict[str, dict[str, float]] = {m: {} for m in models}

    for model in models:
        rows        = [r for r in ratings if r["explanation_id"] == model]
        rater_names = sorted(set(r["rater"]   for r in rows))
        post_ids    = sorted(set(r["post_id"] for r in rows))
        print(f"\n  {model}  (raters={len(rater_names)}, posts={len(post_ids)})")

        for c in CRITERIA:
            matrix = np.full((len(rater_names), len(post_ids)), np.nan)
            for r in rows:
                if c not in r:
                    continue
                ri = rater_names.index(r["rater"])
                pi = post_ids.index(r["post_id"])
                matrix[ri, pi] = r[c]

            alpha = _krippendorff_alpha(matrix)
            alphas[model][c] = alpha
            level = "acceptable" if alpha >= 0.6 else ("low" if alpha >= 0.2 else "poor")
            print(f"    {c:<22}  alpha = {alpha:+.3f}  ({level})")

    # ── H3 verdict (pre-registered rule) ─────────────────────────────────────
    print(
        "\n── H3 Verdict (pre-registered) ──────────────────────────────────────\n"
        "Rule: LLM wins if mean > SHAP on >=2/3 criteria AND alpha >= 0.6 for those criteria\n"
    )

    verdicts: dict[str, str] = {}
    for llm in ["gpt", "gemini"]:
        wins = valid = 0
        for c in CRITERIA:
            shap_m = means.get("SHAP", {}).get(c, float("nan"))
            llm_m  = means.get(llm,    {}).get(c, float("nan"))
            alpha  = alphas.get(llm,   {}).get(c, 0.0)
            if np.isnan(shap_m) or np.isnan(llm_m):
                continue
            valid += 1
            if llm_m > shap_m and alpha >= 0.6:
                wins += 1

        if valid == 0:
            v = "INSUFFICIENT DATA"
        elif wins >= 2:
            v = f"LLM WINS — better on {wins}/3 criteria with alpha >= 0.6"
        else:
            v = f"No clear advantage — only {wins}/3 criteria meet the pre-registered rule"

        verdicts[llm] = v
        print(f"  {llm.upper():<10}: {v}")

    # ── Save ─────────────────────────────────────────────────────────────────
    results = {
        "n_raters": n_raters,
        "n_posts":  n_posts,
        "means":    means,
        "alphas":   alphas,
        "verdicts": verdicts,
    }
    out_path = output_dir / "h3_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved -> {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpretability rubric (H3): generate rating sheet or analyze results."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate HTML rating sheet + CSV templates")
    g.add_argument("--output_dir",     default="results/interp_rubric")
    g.add_argument("--shap_dir",       default="results/figures/shap")
    g.add_argument("--gpt_log_dir",    default="results/llm_logs/gpt5")
    g.add_argument("--gemini_log_dir", default="results/llm_logs/gemini3")
    g.add_argument("--seed",           type=int, default=42)

    a = sub.add_parser("analyze", help="Compute means + Krippendorff's alpha from filled CSVs")
    a.add_argument("--output_dir", default="results/interp_rubric")

    args = parser.parse_args()

    if args.command == "generate":
        generate(
            output_dir=args.output_dir,
            shap_dir=args.shap_dir,
            gpt_log_dir=args.gpt_log_dir,
            gemini_log_dir=args.gemini_log_dir,
            seed=args.seed,
        )
    else:
        analyze(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
