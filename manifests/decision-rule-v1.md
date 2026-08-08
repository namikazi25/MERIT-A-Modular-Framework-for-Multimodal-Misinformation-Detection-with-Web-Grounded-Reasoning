# Decision Rule v1 — Scale-up to Full MMFakeBench (pre-registered)

**Rule: the study extends from the 500-sample analysis to full MMFakeBench (10k) ONLY if criterion (a) OR (b) below holds. If neither holds, the 500-sample analysis IS the paper's evidence; no scale-up.**

Applies to: the winning judge configuration emerging from Task B (judge search) and Task C (retrieval hygiene), evaluated on the sealed 150-sample holdout (manifests/split-500-v1.json).

## Criterion (a) — Judge beats J1 on the sealed holdout, on BOTH models

A candidate judge counts as "beating J1" iff, on the sealed 150-sample holdout (evaluated **exactly once**, by the final top-3 at the end of Task B), for **both** judge models (local qwen3.6-35b-a3b AND gpt-4o-mini-2024-07-18):
- the candidate's 95% bootstrap CI (1000 resamples, per-sample) **excludes J1's point estimate** on the same samples, and
- the candidate's point estimate is above J1's (direction of improvement).

This is the Task B.3 CI criterion applied to the holdout, per model. (a) holds iff both models pass.

## Criterion (b) — Phase 3 cross-tab: FP concentration on evidence-insufficient samples

From reports/phase3_autopsy.md contrast (1): among **J1's** decisions on the 500 (dev + holdout; the test itself uses only dev samples — the holdout is never used for development):
- Let FP_suff = FP rate on samples labeled evidence-sufficient (Y), FP_insuff = FP rate on evidence-insufficient (N).
- (b) holds iff: FP_insuff > FP_suff, two-sided Fisher's exact test p < 0.05, **and** rate ratio FP_insuff / FP_suff >= 2.0 (a "clear effect").

Rationale: false accusations of authentic content are the costly error (documented deployment framing, Phase 1 report). The rule targets exactly that.

## Timing and application

1. (a) is evaluated after Task B final holdout results (both models, once).
2. (b) is evaluated after Phase 3 human labels return (Task F) — at the earliest, in parallel with (a).
3. The rule is applied mechanically from this document. No post-hoc reinterpretation, no additional criteria, no "borderline" calls.
4. If (a) or (b) holds: the winning configuration may be scaled to full MMFakeBench under a separate, explicit authorization (the 1k/10k confirmation run remains out of scope until then).
5. If neither holds: **no scale-up; the 500-sample analysis is the paper.**

## Pre-registration pin

- File: manifests/decision-rule-v1.md
- Created: 2026-08-08T19:22:58.025440+00:00
- Git commit at creation: 73aaa8f9fd696b1ded06e6ab765e28aa9a2a6fda (dirty: True)
- Content SHA256: e624ca15965c2a7bb628e97ea414a1d2c555cef26353402567e73d007e168cbc
- Status: PRE-REGISTERED (binding)
