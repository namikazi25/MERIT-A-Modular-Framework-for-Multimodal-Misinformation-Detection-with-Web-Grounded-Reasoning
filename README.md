# Misinformation Reasoning with Agentic Generative Evaluation (MIRAGE)

Abstract
Misinformation spreads across web platforms through billions of daily multimodal posts that combine text and images, overwhelming manual fact-checking capacity; it is not possible to manually fact-check them all. Supervised detection models struggle at this scale since they require domain-specific training data and often fail to generalize across diverse manipulation tactics emerging on social media. We present MIRAGE, an inference-time, model-pluggable agentic framework that decomposes claim–image verification into three distinguishable steps: (i) cross-modal consistency analysis, (ii) visual veracity assessment, and (iii) retrieval-augmented factual checking, and then aggregates these signals with an LLM-as-judge. MIRAGE orchestrates LVLM reasoning with targeted web evidence and outputs structured, citation-linked rationales that support each decision. On MMFakeBench (n=10,000), MIRAGE with GPT-4o-mini achieves 79.6% accuracy, compared to 61.0% for the strongest zero-shot baseline (GPT-4V with MMD-Agent prompting), representing an 18.6 percentage point improvement. Beyond accuracy, MIRAGE yields transparent, auditable reasoning traces suitable for fact-checking and content-moderation pipelines. Our results demonstrate that decomposed agentic reasoning with web retrieval can match or exceed supervised detector performance without domain-specific training, opening pathways for studying misinformation across languages, platforms, and modalities where labeled data remains scarce.

This repository contains MIRAGE, the agentic evaluation pipeline described in the paper, together with explicit steps for reproducing every reported result.
## 1. Environment and Dependencies

- **Python**: 3.10 (the experiments were run with Python 3.10.10).
- **Virtual environment**: create a fresh environment to avoid solver drift.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All third-party calls (OpenAI, Google, DeepInfra, OpenRouter, Brave, DuckDuckGo) are
abstracted behind environment variables. Duplicate `.env.sample` to `.env` and populate
only the keys you intend to use:

```bash
cp .env.sample .env
# edit .env with OPENAI_API_KEY / GOOGLE_API_KEY / DEEPINFRA_API_KEY / OPENROUTER_API_KEY
# set SEARCH_PROVIDER=brave or duckduckgo, etc.
```

The per-run metadata file stores both CLI arguments and an enumerated subset of these
environment variables, enabling exact reconstruction of the configuration used at
runtime.

## 2. Dataset Preparation and Stratification

We evaluate on the 10k-sample `data/MMFakeBench_test` split. Unpack the provided ZIP if
the folder is missing:

```bash
python scripts/unzip_data.py --zip MMFakeBench_test.zip --dest data
```

The loader now applies **stratified sampling** to this 10k set so that every prefix of
the run (1k, 5k, etc.) mirrors the global class distribution. This is achieved by
setting `PIPELINE_DATASET_STRATIFY=fake_cls` (or `--dataset-stratify fake_cls`), which is
the default in `main.py`. Internally the sampler uses a deterministic random generator
with seed **42**; the same seed powers both the bucket shuffling and the interleaving
logic. Consequently, the first *N* items of any run are a stable, representative subset
of the 10k records, and two runs with identical seeds, limits, and stratification fields
will see the exact same samples.

If you need to stratify on multiple fields (e.g., `fake_cls,image_source`), adjust the
`PIPELINE_DATASET_STRATIFY` environment variable or pass the CLI flag explicitly. All
effective fields are recorded inside `results/run-*.metadata.json`.

## 3. Core Pipeline Execution

Use the main entry point to generate the structured outputs reported in the paper:

```bash
python main.py \
  --max-samples 5000 \
  --checkpoint-size 200 \
  --dataset-root data/MMFakeBench_test \
  --dataset-stratify fake_cls
```

Key reproducibility hooks:

- `max-samples` defines the prefix length (consistent across ablations to ensure
  identical subsets thanks to stratification).
- `checkpoint-size` controls persistence cadence; choose any value ≥1, but keep it
  consistent for resumability.
- `--dataset-stratify` is optional when using the default `fake_cls`, yet passing it
  makes scripts self-documenting.
- Every run writes `results/run-<timestamp>.jsonl`, `results/run-<timestamp>.html`, and
  `results/run-<timestamp>.metadata.json`. The metadata file includes the resolved seed,
  stratification fields, git commit, and sanitized environment snapshot.

## 4. Checkpointing and Resuming

Longer experiments (>10k samples when combining splits) rely on checkpoints stored in
`results/checkpoints/<run-id>/`. Resume with:

```bash
python main.py \
  --max-samples 10000 \
  --checkpoint-size 200 \
  --resume results/checkpoints/run-YYYYMMDD-HHMMSS/run-YYYYMMDD-HHMMSS.upto00200.json
```

Use the same stratification settings and seed to maintain identical sampling order.
When resuming, the pipeline sanity-checks that the JSONL path, stratify fields, and
other critical arguments match the original run to prevent data leakage.

## 5. Evaluation Protocol

After each run, compute the metrics referenced in the paper using `scripts.evaluate`:

```bash
python -m scripts.evaluate \
  --outputs results/run-YYYYMMDD-HHMMSS.jsonl \
  --dataset-json data/MMFakeBench_test/source/MMFakeBench_test.json \
  --image-root data/MMFakeBench_test \
  --save-report results/run-YYYYMMDD-HHMMSS.metrics.json \
  --save-csv results/run-YYYYMMDD-HHMMSS.metrics.csv
```

The evaluator emits accuracy, precision/recall/F1, per-`fake_cls` breakdown, and
calibration diagnostics (ECE, Brier score). These statistics map directly to the tables
and plots in the submission. The CSV export is the recommended source for figure
reproduction scripts.

## 6. Reproducibility Checklist

- [x] Versioned code (git commit recorded in metadata).
- [x] Dependency lock via `requirements.txt`.
- [x] Deterministic sampling with documented seed (=42) and stratified fields.
- [x] Dataset provenance (`MMFakeBench_test` ZIP shipped alongside repository).
- [x] Command templates for training/evaluation.
- [x] Output artifacts archived under `results/` with reproducible metadata snapshots.

## 7. Additional Notes

- To run ablations, toggle modules with `--disable-relevancy`, `--disable-visual`,
  `--disable-questions`, and `--disable-judge`. Keep the stratification settings and
  seeds constant across ablations for fair comparisons.
- Web-search answering (`--answer-questions`) depends on external APIs; record the
  provider choice (Brave or DuckDuckGo) in the paper as latency and coverage differ.
- Older runs (pre-September 2025) used uniform class balancing rather than stratified
  ordering. Confirm that `PIPELINE_DATASET_STRATIFY` appears in the metadata before
  mixing results.

Please cite the accompanying paper when using this codebase. For questions or clarifying
experiments, contact the corresponding author listed in the submission.