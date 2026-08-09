# Phase 0 — Confound Report (Task 0.3)

**Answer: the local qwen3.6-35b-a3b run and the gpt-4o-mini run did NOT share retrieval — each run performed its own web search on its own generated questions. The 200-sample head-to-head therefore conflates retrieval and judge-model effects, which Phase 1 fixes by freezing gpt-4o-mini evidence for both judges.**

## 1. Retrieval independence (evidence)

Files compared:
- `results/lmstudio-200.jsonl` (local run, provider=lmstudio, model=qwen/qwen3.6-35b-a3b)
- `results/gpt4omini-200.jsonl` (first 200 of `results/run-5k-test-balanced-02-10-25.jsonl`, provider=openai, model=gpt-4o-mini-2024-07-18)

Both runs use the identical seed-42 `fake_cls`-stratified sample set: the `image_path` lists are equal, in the same order (verified programmatically).

Per-sample overlap of retrieval artifacts (computed over the 200 samples):

| Artifact | Shared | gpt-4o-mini only | local only |
|---|---|---|---|
| Generated questions (lowercased) | 2 | 350 | 209 |
| Citation URLs (union 1326) | 36 (2.7%) | — | — |

Interpretation: the two pipelines generated almost disjoint question sets (the question generator is model-driven), hence nearly disjoint web searches and retrieved sources. Concrete example, sample `1021` (headline: "A campaign poster of incumbent President Teodoro Obiang Ngue..."):
- gpt-4o-mini questions (`results/gpt4omini-200.jsonl` → `best_qa_per_chain[].question`): "What happened in Malabo April 2016?", "What happened in Malabo April 23 2016?", "Is this a real campaign poster for Obiang?"
- local questions (`results/lmstudio-200.jsonl`, same field): "Teodoro Obiang 2016 campaign poster Malabo"
- citation URLs: gpt-4o-mini cites ethanzuckerman.com, gettyimages.com, en.wikipedia.org/wiki/Teodoro_Obiang...; local cites en.wikipedia.org/wiki/2016_Equatorial_Guinean_presidential_election, thepeninsulaqatar.com — no overlap.

## 2. Visual-module false positives (verification)

Source: `results/lmstudio-200.metrics.json` → `visual_veracity_ai_detection_all`:

- Flagged as AI-generated: **62** (= TP 38 + FP 24)
- **False positives: 24** — confirmed (brief's expectation of ~24 was correct)
- Confusion: TP=38, FP=24, FN=0, TN=138; GT AI-generated images in the 200: 38 (all flagged, recall 1.0)
- Coverage: `visual_veracity_gt_known=200`, `pred_covered=200`, `coverage_rate=1.0`

**Which F1 the 76.0 is:** the reported 0.76 F1 is the **binary F1 with positive class = "AI-generated"**, computed over **all 200 samples** (TN included): precision = 38/62 = 0.6129, recall = 38/38 = 1.0, F1 = 2·P·R/(P+R) = 0.76, accuracy = (38+138)/200 = 0.88. The "filtered" and "all-in" variants in the evaluator coincide here because coverage is 100% (no missing visual GT/predictions).

## 3. Consequence for the pilot findings

Because retrieval was not shared, the reported local-vs-gpt-4o-mini accuracy gap (74.5 vs 76.5) and the visual-module advantage are **not cleanly attributable to the judge model alone**. Phase 1 isolates the judge effect by running both judge models on the same frozen gpt-4o-mini evidence (Task 1.1). If the visual-module advantage survives on frozen evidence it is a real model property; if not, it was a retrieval artifact — the Phase 1 report will flag whichever occurs.
