# J1 on v1 vs v2 evidence — agreement breakdown (pre-Phase-2 check)

Ordered by the brief: J1 (gpt-4o-mini, today) run on both evidence versions; disagreements reported
by class and confidence band, not reconciled.

- v1: `evidence/frozen-gpt4omini-200-v1/` (backfilled from the Oct 2025 gpt-4o-mini run; citations re-fetched, 759/780 URLs OK)
- v2: `evidence/live200-v2/` (fresh gpt-4o-mini gather, full extracted text, 2026-08-08)
- Judges: `results/judge_study/j1-acceptance-v1/labels.csv` and `results/judge_study/j1-gpt-v2/labels.csv` (same model/snapshot, temp 0)

## Headline: 168/200 = 84.0% label agreement; 32 disagreements

## By fake_cls
| fake_cls | agree | rate |
|---|---|---|
| mismatch | 45/60 | 75.0% |
| original | 48/60 | 80.0% |
| textual_veracity_distortion | 56/60 | 93.3% |
| visual_veracity_distortion | 19/20 | 95.0% |

Disagreements concentrate in the classes where the QA/evidence channel is decisive (mismatch, original).

## By v1-confidence band
| band | disagree | rate |
|---|---|---|
| <0.7 | 2/12 | 16.7% |
| **0.7-0.8** | **11/21** | **52.4%** |
| 0.8-0.9 | 7/69 | 10.1% |
| >=0.9 | 12/98 | 12.2% |

The 0.7-0.8 band is where 52% of samples flip — the same band where the 0.1 model-drift disagreements clustered (both on GT=Fake). Two independent sources of instability (model snapshot, evidence version) concentrate in the same borderline band. **Implication: J5's abstention should be targeted at this band** (see Phase 2 risk-coverage analysis).

## Accuracy by evidence version (same judge, today)
| evidence | acc [CI] | F1 [CI] |
|---|---|---|
| v1 | 0.770 [0.710, 0.825] | 0.826 [0.772, 0.871] |
| v2 | 0.750 [0.690, 0.805] | 0.816 [0.764, 0.861] |

McNemar v1-vs-v2 labels: b=14, c=18, n=32, p=0.597 → **no significant accuracy difference**. The evidence version is a source of label noise in the fragile band, not a systematic accuracy shift.

## Consequence for Phase 2
Phase 2 runs on v2 (full evidence text, needed by J2/J5). J1-on-v2 is the J1 baseline for the bake-off master table (0.750 acc / 0.816 F1). v1 remains the Phase 1 frozen evidence per the brief.
