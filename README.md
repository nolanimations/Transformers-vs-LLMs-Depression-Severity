# Depression-Severity Classification on Social Media Text

Fine-tuning Transformer encoders vs. prompting large language models for
**four-class depression-severity classification** (*minimum*, *mild*,
*moderate*, *severe*) on combined Twitter and Reddit data.

This repository accompanies the paper *"Fine-Tuning Transformers and Prompting
LLMs for Depression Severity Classification on Social Media Text"* (Hogeschool Rotterdam). It contains the
full pipeline: preprocessing, stratified splitting, five model families, a shared
evaluation harness, interpretability analysis, and a cost/CO₂ ledger.

## Overview

We run a **matched comparison** in which the model is the only factor that varies
across conditions — identical data splits, identical evaluation code, and identical
prompt templates across both LLMs. Five models are compared:

| Model | Type | Macro-F1 (test) |
|---|---|---|
| **RoBERTa-base** | fine-tuned encoder | **0.679** |
| **MentalBERT** | fine-tuned encoder (domain-adapted) | **0.678** |
| GPT-5.4-mini (chain-of-thought) | prompted LLM | 0.605 |
| Gemini 3 Flash (few-shot) | prompted LLM | 0.568 |
| BiLSTM + GloVe | baseline (floor) | 0.525 |

## Repository structure

```
Capstone-Group-3/
├── scripts/
│   ├── clean_reddit.py            # stage 1: raw Reddit text cleanup
│   └── build_combined_dataset.py  # stage 2: merge + filter → combined_dataset.csv
├── src/
│   ├── data.py                    # load frozen splits, class weights (single source of truth)
│   ├── splits.py                  # stratified 80/10/10 split on (label_id × source), seed 42
│   ├── eval.py                    # macro-F1, per-class, per-platform, bootstrap CIs, confusion matrices
│   ├── utils.py                   # seeding + logging
│   ├── lstm_baseline.py           # BiLSTM + GloVe baseline
│   ├── train_encoder.py           # MentalBERT / RoBERTa fine-tuning (HF Trainer)
│   ├── slang.py                   # per-post informality score (H2)
│   ├── shap_explain.py            # token-level SHAP attributions for encoders
│   ├── interp_rubric.py           # blinded interpretability rubric (H3)
│   ├── significance.py            # H1 paired bootstrap + McNemar tests
│   ├── cost_ledger.py             # H4 cost / latency / CO₂ ledger
│   └── llm/
│       ├── prompts.py             # zero-shot / few-shot / chain-of-thought templates (shared)
│       ├── openai_client.py       # GPT-5.4-mini caller
│       ├── gemini_client.py       # Gemini 3 Flash caller
│       ├── run_llm_eval.py        # batched LLM evaluation
│       └── fill_reasoning.py      # re-query posts with missing rationales
├── configs/                       # YAML hyperparameter configs (one per run)
├── notebooks/                     # analysis only (training is via src/ CLIs)
│   ├── 01_eda.ipynb
│   ├── 02_llm_prompt_design.ipynb
│   ├── 03_shap_examples.ipynb
│   ├── 04_results_aggregation.ipynb
│   └── 05_error_analysis.ipynb
├── results/                       # metrics, logs, figures (see Data & artifacts)
├── paper/                         # LaTeX manuscript + figures
├── environment.yml                # conda env (GPU machines)
├── environment-cpu.yml            # conda env (no GPU)
└── requirements.txt               # pip alternative (CPU PyTorch)
```

## Setup

The project targets **Python 3.11**. The recommended setup is a **conda
environment**; a pip route via `requirements.txt` is provided as an alternative.

### Option A: conda (recommended)

**Without an NVIDIA GPU** (analysis / CPU inference):

```bash
conda env create -f environment-cpu.yml
conda activate capstone
python -m ipykernel install --user --name capstone --display-name "Capstone (Python 3.11)"
```

**With an NVIDIA GPU** (required for training in reasonable time):

```bash
conda env create -f environment.yml
conda activate capstone
```

Conda installs PyTorch without CUDA, so after creating the env, reinstall the
GPU wheels:

```bash
pip install --force-reinstall --no-deps torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
```

> If you have an older NVIDIA driver (check the "CUDA Version" in `nvidia-smi`),
> replace `cu126` with `cu124` or `cu121`.

Then register the Jupyter kernel:

```bash
python -m ipykernel install --user --name capstone --display-name "Capstone (Python 3.11)"
```

Verify the install (should print `CUDA: True` on a GPU machine, `False` otherwise):

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Option B: pip (alternative)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Note: `requirements.txt` installs the **CPU** build of PyTorch. On a GPU machine,
> either use the conda route above or reinstall the CUDA wheels with the
> `--force-reinstall` command shown in Option A.

### API keys (LLM evaluation only)

Copy `.env.example` to `.env` in the repository root and add your keys:

```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...
```

## Data & artifacts

The datasets are **not redistributed** in this repository:

- **DEPTWEET** (Twitter) — Kabir et al., 2023; obtain from the [original authors](https://github.com/mohsinulkabir14/DEPTWEET) under their data-use agreement.
- **Reddit Depression Severity** — Naseem et al., 2022; obtain from the [original source](https://github.com/usmaann/Depression_Severity_Dataset).

Place the prepared data as follows (the combined corpus and splits are produced by
the preprocessing scripts; the splits are **frozen**, generated with seed 42):

```
data/
├── combined_dataset.csv
├── splits/
│   ├── train.csv      # 8,323 posts
│   ├── val.csv        # 1,040 posts
│   └── test.csv       # 1,041 posts (frozen, used once)
└── glove.6B.300d.txt  # for the LSTM — download separately (see below)
```

The BiLSTM baseline needs GloVe embeddings. Download
[`glove.6B.zip`](https://nlp.stanford.edu/data/glove.6B.zip) (~820 MB), extract
`glove.6B.300d.txt`, and place it at `data/glove.6B.300d.txt`.

Run outputs are written under `results/` (`runs/<model>/metrics.json` and
predictions, `llm_logs/` raw API responses, `figures/` plots). Logs that contain
verbatim post text (`results/llm_logs/`) are gitignored.

## Reproducing the results

All commands are run from the repository root with the `capstone` env active.

```bash
# 0. rebuild the dataset from raw sources
python scripts/clean_reddit.py            # raw Reddit → cleaned Reddit
python scripts/build_combined_dataset.py  # merge + confidence filter → data/combined_dataset.csv

# 1. regenerate the frozen splits (deterministic, seed 42)
python -m src.splits

# 2. BiLSTM baseline
python -m src.lstm_baseline --config configs/lstm.yaml

# 3. Encoder fine-tuning (best configs use lr 2e-5; *_lr1e5 / *_lr3e5 reproduce the sweep)
python -m src.train_encoder --config configs/mentalbert.yaml
python -m src.train_encoder --config configs/roberta.yaml

# 4. LLM evaluation (requires .env keys) — both models × three prompt variants
python -m src.llm.run_llm_eval --model gpt    --variant zero_shot
python -m src.llm.run_llm_eval --model gpt    --variant few_shot
python -m src.llm.run_llm_eval --model gpt    --variant chain_of_thought
python -m src.llm.run_llm_eval --model gemini --variant zero_shot
python -m src.llm.run_llm_eval --model gemini --variant few_shot
python -m src.llm.run_llm_eval --model gemini --variant chain_of_thought
python -m src.llm.fill_reasoning          # backfill any missing CoT rationales

# 5. Statistical tests, interpretability, and cost ledger
python -m src.significance                # H1: bootstrap Δ + McNemar
python -m src.interp_rubric generate      # H3: build blinded rating sheets
python -m src.interp_rubric analyze       # H3: means + Krippendorff's alpha
python -m src.cost_ledger --benchmark-encoders   # H4: $/latency/CO₂ table
```

Tips: `run_llm_eval` accepts `--limit N` and `--dry_run` to test prompts cheaply
before a full run; a hard budget cap lives in `configs/llm.yaml` (`cost_cap_usd`).

Figures and the aggregated comparison tables are produced by the analysis
notebooks: `01_eda` (data), `03_shap_examples` (SHAP gallery), `04_results_aggregation`
(main tables, ROC/PR), and `05_error_analysis` (informality + error deep-dive).

## Configuration

Each run is driven by a YAML file in `configs/`: `lstm.yaml`, `mentalbert.yaml`,
`roberta.yaml` (plus `*_lr1e5` / `*_lr3e5` variants for the learning-rate sweep),
and `llm.yaml` for the LLM evaluation (models, token limits, few-shot count,
budget cap). All runs use a fixed seed of 42.

## Ethics & responsible use

This is a **screening / decision-support** tool, not an autonomous diagnostic
system, and should only be deployed with a human in the loop. Routing
clinical-adjacent text through third-party LLM APIs raises consent, privacy, and
GDPR/HIPAA concerns; the locally hosted encoders avoid that third-party data flow
entirely. See the Ethics section of the paper for the full discussion of data
privacy, environmental cost, and misuse risks.

## Citation

If you use this work, please cite:

```bibtex
@article{capstone2026depression,
  title  = {Fine-Tuning Transformers and Prompting LLMs for Depression Severity
            Classification on Social Media Text},
  author = {Bos, Xander and van den Hoek, Efra\"{i}m and Stoel, Vincent and Zijdemans, Nolan},
  year   = {2026},
  note   = {Hogeschool Rotterdam. Publication details TBD.}
}
```

> Publication venue and DOI to be added once available.

## License

Code in this repository is released under the **MIT License** (see `LICENSE`).
The datasets are governed by their original licenses / data-use agreements and
are **not** covered by this license.
