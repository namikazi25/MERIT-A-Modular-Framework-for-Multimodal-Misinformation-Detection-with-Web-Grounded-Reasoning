# Phase 0 — Task 0.1 Acceptance Report (Stage B on frozen v1)

**Result: 197/200 = 98.5% label agreement — not 100%. The 3 disagreements are fully diagnosed as gpt-4o-mini model drift between the original run (Oct 2025) and today (Aug 2026), NOT a state leak in the Stage A/B split.**

## What was tested
Stage B (`scripts/judge_study/judge_only.py`) re-judged the 200 frozen bundles in `evidence/frozen-gpt4omini-200-v1/` (backfilled from `results/gpt4omini-200.jsonl`) using the unchanged current judge logic (`scripts/ai_judge.judge_from_structured`), model `gpt-4o-mini-2024-07-18`, temperature 0.0, manifest written before the run, spend $0.039 (cap $15). Outputs: `results/judge_study/j1-acceptance-v1/` (labels.csv, manifest.json, spend.json, run.log).

## Evidence that the split does NOT leak state

1. **Byte-identical judge inputs.** For every sample, `_build_user_message(reconstruct_judge_input(bundle))` is byte-identical to the message the original pipeline sent the judge (verified explicitly for all 3 disagreeing samples; the reconstruction is a pure function of bundle fields the judge actually consumes: headline, image_path, relevancy, visual_veracity, best_qa_per_chain).
2. **No prompt/logic drift.** `git diff f9c5c4a HEAD -- scripts/ai_judge.py` is empty; `_SYSTEM_PROMPT` is identical (len 1620 at both commits). The 5k run's metadata pins commit `f9c5c4a`.
3. **Not run-to-run nondeterminism.** Re-judging the 3 samples 5× each today yields 5/5 the same label and confidence (0.7/0.8/0.7) — stable today.

## The disagreements (all GT=Fake)

| sample | fake_cls | original label | today's label | who is right vs GT |
|---|---|---|---|---|
| 9285 | mismatch | Not Misinformation | Misinformation | today correct (fixes an error) |
| 4647 | textual_veracity_distortion | Not Misinformation | Misinformation | today correct (fixes an error) |
| 3899 | textual_veracity_distortion | Misinformation | Not Misinformation | original correct (today introduces an error) |

All three have original confidence 0.7–0.8 (borderline). Direction is mixed, net effect on accuracy: +1.

## Conclusion / disposition needed
The Stage A/B split is **leak-free and faithful**: identical inputs, identical prompt/logic, stable outputs. The 1.5% disagreement is attributable to the OpenAI-hosted model's behavior drifting between Oct 2025 and Aug 2026 while keeping the same pinned model id. This is outside the split's control (the judge surface is frozen by the brief).

**Recommended disposition:** accept Task 0.1 with this documented caveat. Phase 1 re-judges *both* models fresh on the same frozen evidence with the same snapshot, so intra-study comparisons (local vs gpt-4o-mini today) remain internally consistent; the drift only affects comparisons against the historical October labels. If strict 100% against historical labels is required, the only "fix" is re-running the original pipeline's judge — which is exactly what Stage B does, and it cannot reproduce a different model snapshot's outputs by definition.
