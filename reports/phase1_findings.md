# Phase 1 — Kill-or-Confirm the Pilot Findings (frozen v1, 200 samples)

**Summary: on identical frozen evidence judged today, there is NO detectable judge-model difference at n=200.** Local qwen3.6-35b-a3b: 73.5% accuracy / 0.787 F1; gpt-4o-mini: 77.0% / 0.826 — 3.5pp gap, McNemar p=0.210, CIs overlap. The pilot's per-class "weaker at semantic manipulation" narrative does not survive shared evidence (its per-class deltas were retrieval-driven). The real finding is the **PR crossing**: neither judge reaches the other's operating point at any threshold — the two models have genuinely different score distributions, and J1's hardcoded thresholds pick a different operating point on each. That is the empirical motivation for the Phase 2 bake-off. Separately, the pilot's **visual-module result is clean, not confounded** (the visual stage takes only the image; retrieval never touches it), and a paired McNemar on the same 200 images shows the local model is **significantly better at AI-generated-image detection** (p=0.029) — see the dedicated section below.

## Method (per study rules)

- Evidence: `evidence/frozen-gpt4omini-200-v1/` (200 bundles, seed-42 stratified subset; documents now enriched with citation re-fetch extracted text — unique URLs 780: 759 fetched OK, 21 failed (dead links/timeouts), 748 with extracted text; see `refetch_manifest.json`).
- **All comparisons use today's re-judged baselines only** (user rule): judge runs executed 2026-08-08 on the same snapshot, temperature 0.0. Historical (Oct 2025) labels are appendix material (`results/gpt4omini-200.jsonl` original `judgement`).
- Judges: J1 current rule, run via `scripts/judge_study/judge_only.py` (no vision, no search; manifests written pre-run).
  - `results/judge_study/j1-local-v1/` — local qwen3.6-35b-a3b (LM Studio, MLX), $0.00
  - `results/judge_study/j1-acceptance-v1/` — gpt-4o-mini-2024-07-18 (also the Task 0.1 acceptance run), $0.039
- GT: `gt_answers` (Fake → Misinformation) from `results/gpt4omini-200.jsonl` sample_details.
- Stats: 95% bootstrap CIs (1000 resamples, seed 42); exact two-sided McNemar on paired labels. All artifacts in `results/judge_study/phase1/`.

## Task 1.1 — Local judge vs gpt-4o-mini judge on identical frozen evidence

| judge (today) | accuracy [CI] | F1 [CI] | precision | recall | CM (TP/FP/FN/TN) |
|---|---|---|---|---|---|
| local qwen3.6-35b-a3b | **0.735** [0.670, 0.795] | 0.787 [0.726, 0.840] | **0.923** | 0.686 | 96/8/44/52 |
| gpt-4o-mini-2024-07-18 | **0.770** [0.710, 0.825] | 0.826 [0.772, 0.871] | 0.879 | **0.779** | 109/15/31/45 |

- McNemar (local vs gpt, paired): b=15 (local wrong, gpt right), c=8 (local right, gpt wrong), n=23, chi2=1.57, **p=0.210 → not significant** at α=0.05.
- CIs overlap for both accuracy and F1; the 3.5pp gap is within sampling noise at n=200.
- Per-fake_cls: `results/judge_study/phase1/per_fake_cls.csv`. Local's precision edge (0.923 vs 0.879) and recall deficit (0.686 vs 0.779) mirror the pilot's direction, now with retrieval fully controlled.

## Task 1.2 — Threshold sweep / PR curves (overlaid plot: `results/judge_study/phase1/pr_overlay.png`)

Scores: signed judge confidence (label=Misinformation → +conf; Not → −conf); threshold 0 = the judge's own decision; sweep θ ∈ [−1, 1].

**Question: can the local judge reach gpt-4o-mini's operating point at some threshold? Answer: No.**

- gpt-4o-mini operating point (θ=0): precision 0.879, recall 0.779.
- Local curve: **no threshold achieves both precision ≥ 0.879 and recall ≥ 0.779** (0 dominating points). Local's nearest point to gpt's operating point is (precision 0.880, recall 0.736) at θ=−0.8 — close on precision, short on recall; at gpt's recall (0.779) local's precision drops to ≈0.75.
- Symmetrically, at local's own recall (0.686), gpt-4o-mini still reaches precision 0.881 vs local's 0.923 — the curves **cross**: local is better at high precision/low recall, gpt-4o-mini better at high recall.
- Confidence granularity is comparable (13 vs 12 unique scores), so the comparison is fair.

## Verdict — "local is more conservative": model property or threshold artifact?

**The honest claim: no detectable judge-model difference on shared evidence at n=200 (McNemar p=0.210).** What remains is a *score-distribution* difference: the PR curves cross, so neither judge can reach the other's operating point at any threshold. A fixed threshold (J1's θ=0, itself a hardcoded design choice) therefore lands each model on a different precision-recall point — local high-precision/low-recall (0.923 / 0.686), gpt-4o-mini lower-precision/higher-recall (0.879 / 0.779). This is not "local is worse"; it is evidence that **J1's threshold encodes a model-specific operating choice that does not transfer across models** — the empirical motivation for the Phase 2 judge redesign.

**Deployment framing:** in misinformation detection the high-precision regime is the deployment-relevant one — false accusations of authentic content are the costly error and are this pipeline's documented failure mode (the pilot's FP analysis). The local judge dominating the high-precision region of its PR curve (precision 0.92 at recall 0.69; 0.88 at recall 0.74) is a point in favor of the open-weight judge for cost-sensitive moderation, not a consolation prize.

## Surprises / deviations flagged (per brief)

1. **Citation re-fetch (previously ordered, silently dropped — acknowledged and fixed this session):** v1 bundles now carry extracted text for cited URLs (`refetch_manifest.json`; 781 OK / 24 failed — dead links, paywalls, read timeouts). No J1 behavior changes: the judge consumes only url/title/citation counts.
2. **v2 live re-gather (for Phase 2 J2/J5 full evidence text) crashed once** at sample 64 on an unhandled DuckDuckGo `TimeoutException`. Fixed: search retries with exponential backoff (logged, rule 6), per-question error tolerance in gather, `--resume`. Re-gather completed 200/200 → `evidence/live200-v2/` (gpt-4o-mini gather, workers=4, DDG; cost ≈ $3–4). J1-on-v2 vs J1-on-v1 agreement will be checked before Phase 2 as ordered (flag disagreements, don't reconcile silently).
3. **Correction (previous version of this report was wrong):** the pilot's visual-module result is NOT confounded. The confound report established only that *retrieval* was not shared; the visual veracity module consumes only the image, and both pilot runs processed the identical 200 images. The paired comparison is valid as-is and is reported below (McNemar p=0.029, significant). What frozen v1 cannot do is *re-test* the visual module (v1 carries gpt-4o-mini's visual outputs) — but it does not need re-testing; the existing paired outputs are the test.

## Visual module — paired comparison on the identical 200 images (zero-spend, from pilot artifacts)

The visual veracity stage takes only the image (no retrieval), and both pilot runs (`results/lmstudio-200.jsonl`, `results/gpt4omini-200.jsonl`) processed the same 200 images in the same order. Per-sample `visual_veracity.ai_generated` vs GT (`image_source == "AI-generated Image"`, n=38/200). Data: `results/judge_study/phase1/visual_paired.csv`.

| model | accuracy [CI] | F1 [CI] | precision | recall | CM (TP/FP/FN/TN) |
|---|---|---|---|---|---|
| local qwen3.6-35b-a3b | **0.880** [0.835, 0.920] | **0.760** [0.660, 0.843] | 0.613 | **1.000** (38/38) | 38/24/0/138 |
| gpt-4o-mini | 0.810 [0.755, 0.865] | 0.548 [0.418, 0.667] | 0.500 | 0.605 (23/38) | 23/23/15/139 |

**McNemar (paired): b=11, c=25, n=36, chi2=4.69, p=0.029 → significant.** The local model correctly flags all 38 AI-generated images (recall 1.0) vs 23 for gpt-4o-mini, with a comparable false-positive count (24 vs 23). Local's F1 advantage (0.760 vs 0.548) is driven entirely by recall; precision is similar (0.61 vs 0.50). This is the strongest single result in favor of the open-weight model and it required no new compute.

## Appendix: historical labels (Oct 2025, for reference only)

Original run labels in `results/gpt4omini-200.jsonl` (judgement): accuracy 0.765, F1 0.821 on the same 200 (see `results/gpt4omini-200.metrics.json`). Not used for any Phase 1 comparison (user rule: today's re-judged baselines only).
