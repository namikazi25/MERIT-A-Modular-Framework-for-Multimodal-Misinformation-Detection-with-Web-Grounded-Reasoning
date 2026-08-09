# Task B — Judge Auto-Search (in progress)

## Candidate-space accounting (reviewer note: "cap 50", found 44)

- The cap of 50 in the brief is a **ceiling, not a target**. The space was **not truncated by the cap**; it was expanded from an initial principled set of 29 to 44 before Round 1 scored them.
- **29 → 44 expansion**: the initial space covered single-axis sweeps (evidence budget, citation, reasoning, confidence) + 12 combos + 3 abstain variants. Before scoring, it was extended with 15 further combos (fuller cross-products of the four axes, more verbal-confidence variants, 3 additional abstain variants) → **44 candidates, 6 abstain variants**.
- What was deliberately NOT included: the full Cartesian product 4 evidence × 2 citation × 3 reasoning × 2 confidence × 2 abstain = 96 (would exceed the cap and is largely redundant), and a logprob-derived confidence axis (the local judge model exposes no logprobs; flagged in each candidate manifest, excluded from local Round 1; gpt transfer checks could add it in a future round if a top candidate justifies it).
- Every candidate prompt is SHA256-pinned in `manifests/candidates-50-v1.json`; per-candidate scored-run manifests record axes + concurrency 1 + active interventions (C1, C2).

## Abstain-variant trigger (reviewer note: degenerate confidence distribution)

- The 6 abstain variants (AB-1..AB-6) do **not** abstain on the raw confidence signal (known degenerate: 199/200 confidences >= 0.8 in Phase 2). Their prompts require an explicit **evidence-sufficiency rating** (`"evidence_sufficient": true|false`) and instruct: abstain=true only when evidence_sufficient=false AND signals are genuinely insufficient; never on confidence. The rating is captured per sample (candidate judge output `extra.evidence_sufficient`) and feeds the Phase 3 cross-tab contrast (abstention as a retrieval-failure detector).
- This was a pre-registration change made before any abstain candidate was scored (Round 1 order puts AB-* last).

## Protocol reminder
- Round 1: all 44 on dev-core (100), local, temp 0, concurrency 1 (B.1). Top-10 by macro F1 → Round 2 (full 350 dev). Top-3 → Round 3 (150 holdout, once, local + gpt-4o-mini). Beats-J1 = 95% CI excludes J1's point estimate on the same samples. B.4 transfer checks: after every 10 candidates, top-3 to gpt on dev-core; Spearman < 0.5 → pause and flag.
- J1 baseline: `results/judge_study/taskC/labels_none.csv` (J1-local on dev-core, 0.700 acc).

## B.4 transfer check #1 — FLAG (search paused for review)

After the first 10 candidates scored (Round 1), the top-3 (EV-top3_300, CF-verbal-top5_500, CB-2) were run on gpt-4o-mini on the dev core:

| candidate | local macroF1 | gpt macroF1 |
|---|---|---|
| EV-top3_300 | 0.6667 | 0.5496 |
| CF-verbal-top5_500 | 0.6486 | 0.5785 |
| CB-2 | 0.6486 | 0.5785 |

Spearman rho = -0.50 < 0.5 → **ranking inverted; local-only optimization is not predictive of gpt ordering.** Per B.4, Round 2 (local, 10 candidates on dev-350) was paused at candidate 0. Under review: switch Round 2 to gpt-4o-mini as the primary judge model (final Round 3 requires both models anyway; ~$1.75), and/or widen the transfer check (n=3 Spearman is degenerate — only 5 attainable values). No scored run from the paused Round 2 is used.


---

## B.4 transfer check #2 (n=10) — inversion CONFIRMED (standalone finding)

Widened per review: full Round-1 top-10 scored on gpt-4o-mini, dev core (100), ~$0.50.

| candidate | local macroF1 (rank) | gpt macroF1 (rank) |
|---|---|---|
| EV-top3_300 | 0.6667 (1) | 0.5593 (6) |
| CF-verbal-top5_500 | 0.6486 (2) | 0.5785 (2) |
| CB-2 | 0.6486 (3) | 0.5679 (5) |
| CB-19 | 0.6486 (4) | 0.5785 (3) |
| CT-free | 0.6485 (5) | 0.5300 (10) |
| CF-verbal-top3_300 | 0.6461 (6) | 0.5593 (7) |
| AB-2 | 0.6461 (7) | 0.5987 (1) |
| CB-17 | 0.6461 (8) | 0.5689 (4) |
| CB-7 | 0.6440 (9) | 0.5383 (9) |
| BASE | 0.6434 (10) | 0.5593 (8) |

**Spearman rho (n=10) = 0.370 < 0.5 → inversion confirmed.** Local-optimized judge rankings do NOT transfer to the API model. Pattern: the abstain-capable AB-2 (sufficiency-gated) is gpt's best but only local's 7th; CT-free (no citation requirement) collapses on gpt (5th → 10th); the verbal-confidence variants are the most rank-stable across models.

**Consequence:** Round 2 advances on **gpt-4o-mini ranking** (primary judge model); local runs in parallel as a sensitivity column only. Reported per the B.4 protocol; not a footnote.


---

## Round 2 (gpt-primary, 350 dev) + confidence-elicitation findings

Round 2 scored the Round-1 top-10 on the full 350 dev with gpt-4o-mini (primary; local sensitivity column running in parallel). J1-gpt dev baseline acc = 0.7514.

| rank | candidate | macroF1 [CI] | acc | beats J1 (CI) |
|---|---|---|---|---|
| 1 | AB-2 (sufficiency-gated, abstain) | 0.6174 [0.569, 0.669] | 0.622 | no |
| 2 | CB-2 | 0.5993 [0.548, 0.648] | 0.603 | no |
| 3 | BASE (J2B-like) | 0.5988 [0.548, 0.648] | 0.603 | no |
| 4 | CB-19 | 0.5962 | 0.600 | no |
| 5 | CF-verbal-top3_300 | 0.5947 | 0.597 | no |
| 6 | CF-verbal-top5_500 | 0.5909 | 0.594 | no |
| 7 | CB-17 | 0.5893 | 0.591 | no |
| 8 | CT-free | 0.5822 | 0.583 | no |
| 9 | EV-top3_300 | 0.5812 | 0.583 | no |
| 10 | CB-7 | 0.5782 | 0.580 | no |

**No candidate beats J1 on gpt; all sit far below J1's accuracy.** Advance to Round 3: **AB-2, CB-2, BASE** (gpt ranking) + **H2** (hybrid clause: best dev-core gpt score among hybrids, 0.730 acc / 0.821 F1 5-fold CV — supervised, NOT training-free; did not beat J1 on dev-core either, p=1.0; included as 4th finalist, flagged).

### Confidence elicitation (the prioritized axis) — negative result

Per-candidate ECE / Brier on gpt dev-350 (histograms: `results/judge_study/taskB/r2gpt_conf_histograms.png`):

| candidate | ECE | Brier | %conf>=0.8 |
|---|---|---|---|
| J1-gpt (baseline) | **0.126** | **0.192** | 83.7% |
| AB-2 | 0.225 | 0.269 | 86.9% |
| CB-2 | 0.259 | 0.300 | 87.4% |
| BASE | 0.225 | 0.274 | 82.9% |
| CF-verbal-top3_300 | 0.251 | 0.286 | 82.6% |
| CF-verbal-top5_500 | 0.254 | 0.288 | 82.9% |
| CT-free | 0.272 | 0.312 | 96.3% |
| CB-7 | 0.286 | 0.320 | 88.6% |

**Verbal elicitation does NOT de-degenerate the confidence distribution**: the model answers "high" almost always, so the verbal variants keep 82–96% of confidences at >=0.8 (raw candidates: 81–96%). All candidates are ~2x worse calibrated than J1. **The confidence channel is not a usable abstention signal from any candidate, gpt or local** — abstention must ride the explicit evidence-sufficiency mechanism (AB-2's design), not confidence thresholds.
