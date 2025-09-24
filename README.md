# MMFakeBench Pipeline

End-to-end framework for auditing multimodal misinformation with MMFakeBench. The
pipeline loads image/headline pairs, runs a sequence of reasoning heads (text/image
alignment, AI-image detection, investigative Q/A, and an AI judge), and emits JSONL
and HTML reports ready for analysis. Optional checkpoints, module toggles, and
evaluation tooling make it suitable for large research runs and ablation studies.

## 1. Quick Start

```bash
# 1) Create a virtual environment (recommended) and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Prepare secrets / configuration
cp .env.sample .env
# Edit .env with provider API keys (OPENAI_API_KEY / GOOGLE_API_KEY / DEEPINFRA_API_KEY)

# 3) Unpack the dataset (if needed)
python scripts/unzip_data.py --zip MMFakeBench_test.zip --dest data

# 4) Run the pipeline over N samples (writes JSONL + HTML + metadata)
python main.py --max-samples 1000 --checkpoint-size 100

# 5) Evaluate outputs against ground truth
python -m scripts.evaluate \
  --outputs results/run-YYYYMMDD-HHMMSS.jsonl \
  --dataset-json data/MMFakeBench_test/MMFakeBench_test.json \
  --image-root data/MMFakeBench_test \
  --save-report results/run-YYYYMMDD-HHMMSS.metrics.json \
  --save-csv results/run-YYYYMMDD-HHMMSS.metrics.csv
```


## 2. Configuration

- **Providers & models**: set `ALIGN_PROVIDER`, `ALIGN_MODEL`, and provider API keys in
  `.env`. The same provider configuration is reused for relevancy, visual checks,
  question answering, and the judge.
- **Search**: choose `SEARCH_PROVIDER` (brave or duckduckgo). Control minimum spacing
  for requests with `BRAVE_SEARCH_MIN_INTERVAL` or `DUCKDUCKGO_SEARCH_MIN_INTERVAL` to
  stay under rate limits.
- **Pipeline length**: `PIPELINE_MAX_SAMPLES` limits how many dataset entries are
  processed by default. Override per run with `--max-samples`.
- **Module toggles (ablations)**: set to `1` in `.env` or pass the CLI flag to skip a
  stage without editing code:
  - `PIPELINE_DISABLE_RELEVANCY` / `--disable-relevancy`
  - `PIPELINE_DISABLE_VISUAL` / `--disable-visual`
  - `PIPELINE_DISABLE_QUESTIONS` / `--disable-questions`
  - `PIPELINE_DISABLE_JUDGE` / `--disable-judge`
  - `--answer-questions` controls web-question answering; it is automatically disabled
    if `--disable-questions` is active.

All effective settings are recorded in the per-run metadata file (see below).

## 3. Running Long Experiments

### Checkpoints & Resume

Large runs periodically persist checkpoints in `results/checkpoints/<run-id>/` (default
every 100 samples). Resume from the most recent checkpoint with:

```bash
python main.py --resume results/checkpoints/run-YYYYMMDD-HHMMSS/run-YYYYMMDD-HHMMSS.upto00100.json
```

- Keep all other flags identical between the original run and the resume command.
- The pipeline restores `--save-jsonl` / `--html-report` from the checkpoint unless you
  override them explicitly.

### Outputs

Each run produces:

- `results/run-<timestamp>.jsonl` – per-sample structured outputs (relevancy, visual
  veracity, selected Q/A, judge verdict, token usage, module configuration).
- `results/run-<timestamp>.html` (+ optional `*.tokens.csv`) – human-readable report and
  token accounting.
- `results/run-<timestamp>.metadata.json` – provenance record (git commit, sanitized
  CLI/env args, module toggles, checkpoints, timestamps, progress). This is useful for
  reproducibility sections in a paper.

## 4. Evaluation & Analysis

Run the evaluator after each experiment to score the AI judge and the visual veracity
head against ground truth. Besides accuracy / F1, the tool now reports:

- **Confidence calibration**: Brier score, expected calibration error (ECE), and per-bin
  stats for the judge’s confidence values.
- **CSV export** (`--save-csv`): structured metrics for tables/plots.

Example:

```bash
python -m scripts.evaluate \
  --outputs results/run-20250917-135855.jsonl \
  --dataset-json data/MMFakeBench_test/MMFakeBench_test.json \
  --image-root data/MMFakeBench_test \
  --save-report results/run-20250917-135855.metrics.json \
  --save-csv results/run-20250917-135855.metrics.csv
```

## 5. Repository Highlights

- `main.py` – orchestrates the end-to-end pipeline, module toggles, checkpointing, and
  run metadata.
- `scripts/checkpoint.py` – checkpoint manager used by the pipeline.
- `scripts/evaluate.py` – computes metrics, calibration statistics, and CSV/JSON
  summaries.
- `scripts/unzip_data.py` – safe helper to unpack dataset archives.
- `requirements.txt` – minimal dependencies (Pillow, Torch, provider SDKs, etc.).

## 6. Suggested Ablation Workflow

1. Baseline run with all modules enabled.
2. Disable one module at a time (e.g., `--disable-visual`) and rerun with the same
   `--max-samples`/checkpoint settings.
3. Evaluate each JSONL and compare the exported CSV metrics to quantify contribution.
4. Use the metadata files to document the exact settings per experiment in your paper.

## 7. Housekeeping

- Remove or archive legacy result files produced before the ground-truth scrub (Sept‑2025).
- Respect API rate limits; adjust `DUCKDUCKGO_SEARCH_MIN_INTERVAL`/`BRAVE_SEARCH_MIN_INTERVAL`
  if you encounter throttling.
- Commit or stash changes between runs so the metadata’s `git.dirty` flag stays
  meaningful.

Happy experimenting!
