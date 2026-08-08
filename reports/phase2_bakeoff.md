# Phase 2 — Judge Bake-off (v2 evidence, 200 samples, five judges)

**Summary: the redesign hypothesis fails in its current form — none of the LLM-based redesigns beat the simple rule. J1 (current rule) 0.750 acc / 0.816 F1; the evidence-grounded J2 (0.665, p=0.002 vs J1) and structured-debate J3 (0.580, p<0.001) are significantly WORSE; the abstaining J5 (0.705, n.s.) never abstains with gpt-4o-mini. The only design that improves on J1 is the learned aggregator J4 (0.765 / 0.835, 5-fold CV) — which is not training-free and is flagged as such. Local-model repeats: J1-local ≈ J1-gpt (0.760, p=0.85), but local J2/J5 are significantly worse than their gpt counterparts — the evidence-heavy designs amplify the model gap.**

## Setup

- Evidence: `evidence/live200-v2/` (gpt-4o-mini gather, full extracted text, 2026-08-08). J1-on-v2 is the bake-off baseline (v1 remains the Phase 1 frozen evidence).
- Judges J1–J5 implemented in `scripts/judge_study/judges.py`; run via `scripts/judge_study/judge_bakeoff.py` (J4 via `j4_aggregator.py`). Every run has a manifest (model, prompt SHA256s, evidence version, git), temp 0.0, retries with backoff.
- Judge models: gpt-4o-mini-2024-07-18 (all judges) + local qwen3.6-35b-a3b (J1, J2, J5 repeats, $0).
- All outputs in `results/judge_study/phase2/`. Stats: 95% bootstrap CIs (1000 resamples), exact McNemar.

## Master table (n=200; decided / abstained; CI = accuracy and F1)

| judge | model | acc [CI] | F1 [CI] | macroF1 | balacc | ECE | Brier | dec | abst | parse-fail |
|---|---|---|---|---|---|---|---|---|---|---|
| **J4** | gpt (features) | **0.765** [0.705, 0.820] | **0.835** [0.784, 0.877] | 0.713 | 0.708 | 0.127 | 0.190 | 200 | 0 | 0 |
| **J1** | gpt-4o-mini | 0.750 [0.690, 0.805] | 0.816 [0.764, 0.861] | 0.713 | 0.721 | **0.110** | **0.185** | 200 | 0 | 0 |
| **J1** | local | 0.760 [0.705, 0.815] | 0.806 [0.752, 0.857] | **0.745** | **0.790** | 0.121 | 0.179 | 200 | 0 | 0 |
| J5 | gpt-4o-mini | 0.705 [0.640, 0.765] | 0.777 [0.720, 0.829] | 0.670 | 0.684 | 0.197 | 0.241 | 200 | 0 | 0 |
| J2 | gpt-4o-mini | 0.665 [0.605, 0.725] | 0.737 [0.678, 0.793] | 0.638 | 0.661 | 0.227 | 0.263 | 200 | 0 | 0 |
| J2 | local | 0.635 [0.565, 0.695] | 0.681 [0.609, 0.748] | 0.627 | 0.687 | 0.277 | 0.295 | 200 | 0 | 0 |
| J5 | local | 0.634 [0.569, 0.701] | 0.673 [0.601, 0.737] | 0.629 | 0.695 | 0.289 | 0.305 | **197** | **3** | 0 |
| J3 | gpt-4o-mini | 0.580 [0.505, 0.650] | 0.667 [0.594, 0.731] | 0.549 | 0.567 | 0.260 | 0.297 | 200 | 0 | 0 |

Parse failures: **0/200 for every judge** (no constrained decoding or repair pass needed).

## McNemar highlights (full matrix in `results/judge_study/phase2/report/master_table.json`)

- J1-gpt vs J2-gpt: p=0.0023 (b=6, c=23) → **evidence-grounded is significantly worse**
- J1-gpt vs J3-gpt: p<0.001 (b=12, c=46) → **debate significantly worse**
- J1-gpt vs J4: p=0.755 (n.s.); J4 vs J3: p<0.001; J4 vs J2: p=0.011 → J4 clearly above J2/J3, statistically tied with J1
- J1-gpt vs J1-local: p=0.851 (n.s.) — replicates Phase 1's no-detectable-judge-difference
- J5-gpt vs J5-local: p=0.044 (SIG); J2-gpt vs J2-local: p=0.405 (n.s.) — the evidence-heavy designs hurt local more

## J5 risk-coverage (`report/j5_risk_coverage.png`)

- gpt-4o-mini: **0 abstentions** (200/200 coverage); local: **3 abstentions** (197/200) — all 3 on GT=Fake samples (J1-v1 confidences 0.9, 0.8, 0.6; classes textual_veracity_distortion ×2, mismatch). Abstention never misfires on real content (0/3 errors), consistent with "aim at the fragile band" — but sample size is tiny.
- Confidence-based selective prediction is degenerate: judges' confidences concentrate at ≥0.8 (199/200), so the risk-coverage curve is a near-step (risk 0.30 at coverage 1.0 → 0.30 at 0.995). The judge confidence channel does not support fine-grained selective prediction as-is.

## Per-class accuracy vs J1-gpt (baseline)

| fake_cls | J1-gpt | J2-gpt | J3-gpt | J4 | J5-gpt | J1-local |
|---|---|---|---|---|---|---|
| mismatch | 0.633 | 0.467 | 0.517 | **0.800** | 0.600 | 0.567 |
| original | 0.650 | 0.650 | 0.533 | 0.567 | 0.633 | **0.867** |
| textual_veracity_distortion | **0.900** | 0.783 | 0.717 | 0.850 | 0.800 | 0.783 |
| visual_veracity_distortion | 0.950 | 0.950 | **0.500** | **1.000** | 0.950 | 0.950 |

- J4's win is concentrated on **mismatch** (0.800 vs 0.633) — the learned aggregator handles the headline-image-mismatch class that the rule misses.
- J3's debate format collapses on visual (0.500) and original (0.533).
- J1-local's best class is **original** (0.867) — consistent with the Phase 1 high-precision/conservative theme (fewest false accusations of real content).

## Cost accounting (per 200 samples)

| judge | model | tokens | USD |
|---|---|---|---|
| J1 | gpt-4o-mini | ~258k | $0.045 |
| J2 | gpt-4o-mini | 549k | $0.099 |
| J3 | gpt-4o-mini | 1,663k | $0.305 |
| J5 | gpt-4o-mini | 558k | $0.101 |
| J4 | numpy | 0 | $0 |
| J1/J2/J5 | local | 258–619k | $0 |

Total Phase 2 gpt spend ≈ **$0.55**; cumulative study spend ≈ $3.9 of the $15 cap (v2 gather ~$3.3 dominates).

## Surprises / deviations (flagged)

1. **The headline surprise: J2/J3 underperform the rule.** The empirical motivation for redesigning the judge (Phase 1's PR crossing) does not transfer to these prompt-based designs — giving the judge full retrieval text made it *worse* (likely evidence overload: noisy/off-topic DDG snippets dilute the decisive compact signals that J1's prompt isolates). J4 shows the headroom is real but requires supervision (not training-free, flagged).
2. **J5's abstention is almost never triggered** (0 gpt / 3 local), and all abstentions are on GT=Fake — good direction, insufficient volume. The confidence channel is too coarse for selective prediction.
3. **Evidence-heavy designs amplify the model gap**: local J5 is significantly worse than gpt J5 (p=0.044) even though J1-local ≈ J1-gpt. Model differences show up where the prompt pushes reasoning over retrieval text.
4. No parse failures anywhere; no retries needed (0 retried tokens); determinism: temp 0.0.
5. Local J2/J5 ran with full v2 evidence text; early log lines showing "prompt=338k" were cumulative-usage displays, not per-sample (verified per-sample usage is ~3–6k tokens).

## Verdict for the paper

- The **rule-based J1 baseline stands** (and is model-robust: local ≈ gpt).
- **J4 is the only design that beats J1** (0.765/0.835 vs 0.750/0.816; n.s. by McNemar but directionally +1.5pp acc / +1.9pp F1, and +16.7pp on the mismatch class) — a supervised headroom result, explicitly not training-free.
- J2/J3/J5 as specified do not improve the pipeline on this evidence; their failure modes (evidence overload, no abstention signal) are themselves the paper's negative-result content.
- Adaptive follow-up judge (J6, Phase 4) remains gated on approval; given J2's degradation, J6's additional retrieval is unlikely to help without first fixing evidence selection — recommendation: do not start Phase 4.
