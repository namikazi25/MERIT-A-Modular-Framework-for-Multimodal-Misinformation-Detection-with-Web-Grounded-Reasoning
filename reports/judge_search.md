# Task B — Judge Auto-Search: final report

**Honest framing (first line): the search found NO judge that beats J1 on the sealed holdout. After 44 candidates, successive halving, transfer checks, and three hybrids, the simple current-rule judge (J1) stands; the scale-up decision rule (criterion a) is NOT met, and per the pre-registered rule the 500-sample analysis is the paper unless criterion (b) triggers from the Phase 3 cross-tab.**

## Search trajectory
1. **Space**: 44 candidates (cap 50 = ceiling, not target; 96-combo Cartesian product deliberately excluded as redundant; logprob-confidence axis unavailable on local). Axes: evidence budget (none/top3_300/top5_500/full) x citation (free/must) x reasoning (direct/rationale_first/checklist) x confidence (raw/verbal) + 6 abstain variants (sufficiency-triggered, NOT confidence-triggered). Prompts SHA256-pinned (`manifests/candidates-50-v1.json`).
2. **B.1**: concurrency 1 pinned for scored runs (outputs not byte-identical across c1/c2/c4; c2=1.65x/c4=2.25x exploration-only).
3. **Round 1** (all 44, dev-core 100, local): full scores below; top-10 by macro F1 advanced. No candidate beat J1 (CI criterion).
4. **B.4 transfer checks**: #1 (n=3) Spearman -0.50 → pause; #2 (n=10) Spearman 0.370 → inversion CONFIRMED (full table above in this report): **local-optimized judge rankings do not transfer to gpt-4o-mini** (standalone finding). Round 2 advanced on gpt ranking; local kept as sensitivity.
5. **Round 2** (top-10, 350 dev, gpt-primary): AB-2 > CB-2 > BASE; none beat J1 (J1-gpt dev acc 0.751). Local sensitivity column runs in parallel (results/judge_study/taskB/r2local).
6. **Hybrids H1-H3** (gpt dev-core): sufficiency ratings show **64% of dev-core evidence rated insufficient** by gpt. H1 (rule+gate) F1=0.000 (64 abstentions); H2 (LR+evidence features, 5-fold CV, supervised) 0.730/0.821 ties J1; H3 (LLM+gate) 0.639/0.581 (64 abstentions). No hybrid beat J1. H2 advanced to Round 3 via the hybrid clause (flagged: supervised).
7. **Round 3** (150 sealed holdout, ONCE, both models):

| finalist | gpt acc [CI] | gpt McNemar vs J1 | local acc [CI] | local McNemar vs J1 |
|---|---|---|---|---|
| J1 (baseline) | 0.747 [0.680, 0.820] | — | 0.767 [0.693, 0.833] | — |
| AB-2 | 0.633 [0.553, 0.713] | p=0.0137 SIG (worse) | 0.730 [0.649, 0.797] | p=0.308 n.s. |
| CB-2 | 0.667 [0.593, 0.740] | p=0.081 n.s. | 0.705 [0.631, 0.779] | p=0.078 n.s. |
| BASE | 0.673 [0.600, 0.747] | p=0.108 n.s. | 0.733 [0.660, 0.807] | p=0.405 n.s. |
| H2 | 0.767 [0.693, 0.833] | p=0.728 n.s. | (LR, model-agnostic) | — |

**No candidate's CI excludes J1's point estimate on the holdout on either model → criterion (a) NOT met.** Notably, AB-2 — the search's own top candidate on gpt dev — is significantly WORSE than J1 on the holdout (p=0.014): the dev-core signal did not transfer to the held-out set, validating the halving/CI discipline.

## Round-1 dev-core scores (all 44, local)
| id | macroF1 | acc | pf |
| EV-top3_300 | 0.6667 | 0.680 | 0 |
| CF-verbal-top5_500 | 0.6486 | 0.660 | 0 |
| CB-2 | 0.6486 | 0.660 | 0 |
| CB-19 | 0.6486 | 0.660 | 0 |
| CT-free | 0.6485 | 0.657 | 1 |
| CF-verbal-top3_300 | 0.6461 | 0.657 | 1 |
| AB-2 | 0.6461 | 0.657 | 1 |
| CB-17 | 0.6461 | 0.657 | 1 |
| CB-7 | 0.6440 | 0.650 | 0 |
| BASE | 0.6434 | 0.657 | 1 |
| CB-1 | 0.6396 | 0.650 | 0 |
| AB-1 | 0.6396 | 0.650 | 0 |
| CB-14 | 0.6396 | 0.650 | 0 |
| CB-8 | 0.6393 | 0.646 | 1 |
| AB-3 | 0.6393 | 0.646 | 0 |
| AB-6 | 0.6393 | 0.646 | 0 |
| CF-verbal-none | 0.6380 | 0.653 | 2 |
| CB-15 | 0.6380 | 0.653 | 2 |
| EV-none | 0.6369 | 0.650 | 0 |
| CB-9 | 0.6347 | 0.640 | 0 |
| CF-verbal-full | 0.6344 | 0.646 | 1 |
| CB-22 | 0.6344 | 0.646 | 1 |
| EV-full | 0.6279 | 0.640 | 0 |
| RS-rationale_first | 0.6236 | 0.630 | 0 |
| CF-verbal-none-A | 0.6236 | 0.630 | 0 |
| AB-4 | 0.6225 | 0.636 | 0 |
| CB-3 | 0.6215 | 0.630 | 0 |
| CB-11 | 0.6215 | 0.630 | 0 |
| CB-18 | 0.6207 | 0.626 | 1 |
| CB-10 | 0.6186 | 0.626 | 1 |
| CB-16 | 0.6176 | 0.622 | 2 |
| J1REF | 0.6144 | 0.620 | 0 |
| CB-4 | 0.6144 | 0.620 | 0 |
| CB-12 | 0.6144 | 0.620 | 0 |
| CF-raw-none | 0.6144 | 0.620 | 0 |
| CB-6 | 0.6124 | 0.620 | 0 |
| CB-5 | 0.6094 | 0.616 | 1 |
| CB-21 | 0.6094 | 0.616 | 1 |
| AB-5 | 0.6081 | 0.610 | 0 |
| RS-checklist | 0.6033 | 0.610 | 0 |
| CB-23 | 0.6002 | 0.606 | 1 |
| CB-24 | 0.5970 | 0.602 | 2 |
| CB-20 | 0.5960 | 0.600 | 0 |
| CB-13 | 0.5867 | 0.590 | 0 |

Full artifacts: `results/judge_study/taskB/r1/` (labels + per-candidate manifests), `r2gpt-*/`, `r3gpt/`, `r3local/`, `r3h2/`, `hybrids/`; calibration: `r2gpt_calibration.json`, `r2gpt_conf_histograms.png`.

## Confidence-elicitation axis (prioritized): negative result
Verbal elicitation does not de-degenerate the confidence pile-up (82-96% of confidences >= 0.8 for verbal variants; raw 81-96%); all candidates ~2x worse calibrated than J1 (ECE 0.22-0.29 vs 0.126; Brier 0.27-0.32 vs 0.192). Confidence is not a usable abstention signal from any candidate; abstention must use the explicit evidence-sufficiency mechanism (AB-2's design).

## Compute / spend
Total Task B spend: dev-core rounds $0 (local) + transfer checks ~$0.75 + Round 2 gpt ~$1.75 + Round 3 gpt ~$0.35 + hybrids/sufficiency ~$0.2 + local sensitivity $0. Cumulative study spend ~$9-10 of $15 cap.
