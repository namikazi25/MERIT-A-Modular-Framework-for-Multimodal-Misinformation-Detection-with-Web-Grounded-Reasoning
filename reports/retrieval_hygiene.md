# Task C — Retrieval Hygiene A/B (dev core, n=100)

**Summary: none of the interventions hurt the pipeline. J1 (the rule judge) is evidence-invariant — it never reads the document text, so C1/C2/C4 change zero labels (p=1.000). On an evidence-reading judge (J2B), C1 (near-duplicate removal) improves accuracy directionally (+2.0pp, n.s.), C2 (boilerplate stripping) is a no-op on this evidence. Adoption: C1 adopted as default for Task B rounds 2–3; C2 adopted (neutral, trivial cost); C4 not adopted by default (no demonstrated benefit, adds local compute); C3 deferred (DuckDuckGo instability, see below).**

## Method

- Dev core: `manifests/devcore-100-v1.json` (100 samples, fake_cls-stratified).
- A/B: J1 (local qwen3.6-35b-a3b, $0) on dev-core bundles with/without each intervention; supplementary A/B on J2B (gpt-4o-mini, evidence-reading) so the interventions' effect on evidence consumers is measured. 95% bootstrap CIs; exact McNemar vs baseline. Artifacts: `results/judge_study/taskC/` (J1) and `results/judge_study/taskC-j2b/` (J2B).

## J1 A/B (as specified in the brief)

| intervention | acc [CI] | F1 [CI] | vs none |
|---|---|---|---|
| none | 0.700 [0.610, 0.790] | 0.754 [0.667, 0.836] | — |
| C1 dedup | 0.700 [0.610, 0.790] | 0.754 [0.667, 0.836] | p=1.000 n.s. |
| C2 boilerplate | 0.700 [0.610, 0.790] | 0.754 [0.667, 0.836] | p=1.000 n.s. |
| C4 re-rank | 0.700 [0.610, 0.790] | 0.754 [0.667, 0.836] | p=1.000 n.s. |

Identical labels on all 100 samples: **J1 consumes only relevancy/visual-veracity/QA-confidence/citation-counts; the document text never reaches it.** The interventions therefore cannot hurt J1 by construction. (C3: see deferral note.)

## Supplementary J2B A/B (evidence-reading judge, gpt-4o-mini)

| intervention | acc [CI] | F1 [CI] | vs none |
|---|---|---|---|
| none | 0.700 [0.610, 0.790] | 0.803 [0.730, 0.864] | — |
| C1 dedup | **0.720** [0.630, 0.800] | **0.818** [0.750, 0.878] | p=0.500 n.s. (+2.0pp acc) |
| C2 boilerplate | 0.700 [0.610, 0.790] | 0.803 [0.730, 0.864] | p=1.000 n.s. |

## Intervention stats (dev core, n=100)

- **C1**: 295 near-duplicate documents removed of 3,915 (~7.5% dedup rate), aggregated across questions.
- **C2**: 18,939 chars (~4,724 tokens) stripped (~47 tokens/sample); DDG snippets contain little boilerplate → near no-op.
- **C4**: 774 rank changes across 392 questions (~2.0 positions/question) by local-Qwen relevance scoring.

## Adoption decisions (per brief: adopt if no significant degradation)

| intervention | adopt? | note |
|---|---|---|
| C1 dedup | **YES** | no degradation; directionally better on J2B; cheap (no extra calls) |
| C2 boilerplate | **YES** | neutral, trivial cost; keeps future evidence cleaner |
| C4 re-rank | no (optional) | no demonstrated benefit; adds a local call per question |
| C3 query rewrite | **DEFERRED** | requires fresh DDG searches; DDG timed out under rewritten-query load (ConnectTimeout on repeated attempts, logged backoff). Retry in a calmer window; if still blocked, report as not-adopted. Manifest note: interventions active per scored run. |

C1 and C2 become defaults for Task B rounds 2–3; every scored run's manifest records which interventions were active.
