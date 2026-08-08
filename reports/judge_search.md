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
