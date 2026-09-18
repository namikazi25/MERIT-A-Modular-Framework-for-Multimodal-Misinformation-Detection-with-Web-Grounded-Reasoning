# Development checkpoint — 18 September 2026

Work continues on `codex/merit-redesign-preflight`. The initial reviewed engineering checkpoint is `effc9f6`. Generated registries, raw evidence, annotation packets, downloaded models and private execution records are preserved locally and excluded from this branch's new commits.

| Component | Verified status | Remaining gate |
|---|---|---|
| Preflight 0.2.1 | Four reported defects corrected; current real-registry validation passes | Checksums establish recorded consistency, not malicious-rewrite resistance or complete semantic non-leakage |
| GLM-5.3-Flash and Jev | Previous synthetic live access and bounded five-case work recorded | Larger experiments require useful evidence and a complete-batch budget bound |
| Retrieval pacing | Shared search/extraction leases, intervals, `Retry-After`, persistent engine/host stops tested offline | Updated controls have not been exercised against live services |
| SearXNG | Pinned update returned relevant official results in three spaced public DDG queries | Sustainable access is unproven; later direct DDGS challenge stops further DDG requests pending review |
| Firecrawl | Prior self-hosted extraction smoke succeeded | Currently stopped; internal fan-out and effective server limits remain to be verified |
| Astra annotation | Five provisional AI evidence-quality reviews saved and coordinator-reviewed | Human labels remain blank; no annotation gold or calibration claims |
| PROBE-DINOv2 | Official metadata pinned; preprocessing/aggregation tested | Licence, exact backbone config, loader and real inference pending |
| Controlled ablations | Seven conditions previously executed on the same five examples with frozen evidence | Small, weak-evidence diagnostic; no generalisation or superiority claim |

The original initial baseline answered three of five cases: two correct misinformation answers and one authentic false positive; two cases had query failures. In the later frozen-evidence comparison C1/C2/C3 all abstained. A2 produced one binary answer, subsequently identified as ungrounded despite matching the benchmark label. The original result remains preserved; an added verdict-consistency guard was checked offline, without claiming a new live result. Detector results in those historical conditions refer to the earlier ViT, never PROBE.

All five Astra packets were judged to have insufficient supplied evidence: three had irrelevant sources and two had no supplied evidence. This is a provisional AI assessment of those packets, not five independently established truth labels. Historical annotation packets lack reconstructed objective source passages and must not treat generated QA as ground evidence. Human review remains necessary.

The final offline suite for this checkpoint passed **184 tests**, with zero audited Python network or environment-file operations. Run it with:

```sh
.venv/bin/python -B scripts/run_offline_tests.py
```

The first new traffic run exposed a zero-wait admission bug (173 passed, one failed). Admission now allows an immediately ready request even when the allowed wait is zero; the concurrency regression passes. Tests also cover cross-instance spacing, cross-process exclusion, interrupted leases, persistent blocks, API backoff, target-host exclusion, replay, malformed state and membership revalidation after waits. PROBE fixtures cover repeat padding, grid order, discarded edges, aggregation, malformed logits, resource limits and identity checks. Mocks do not establish live access or checkpoint accuracy.

Local validation found no changes to 4,199 protected artifacts, 429 original run artifacts or either v2 registry file. Registry counts remain 10,000 records, 9,665 groups and 4,187 reserve records; the 100-example core is unchanged. No reserved examples were opened or run. No new experimental GLM/Jev, search or extraction requests were made in this continuation. Public upstream documentation/metadata were inspected and Astra used the existing Codex allowance. New rated third-party spending remains $0.050643452 for the whole session, with no pending reservations; the stricter persisted $15 combined session cap leaves $14.949356548. Provider invoice reconciliation remains separate.

Next work depends on the [PROBE gates](probe_detector.md), supervised retrieval access review and actual human annotation. Direct DDGS repair remains deferred. Prime dispatch remains blocked because an enforceable per-dispatch paid-use bound has not been established. No alternate coding-provider route was used. Stage 1 and final evaluation are not marked complete.

See [reproducibility requirements](reproducibility.md) before attempting a clean-clone paper reproduction. This code checkpoint is not a complete dataset/evidence release.

## Subsequent offline engineering continuation

The remaining portability work was completed separately: **189 full local tests** and **185 portable tests** pass. The portable profile was actually run in a source-only export without data, registries, model files, environment files or private audit artifacts. Its four excluded historical-artifact checks are named explicitly. Before the correction, the same export failed three tests and skipped two.

A standard-library-only verifier reproduces **22 recorded condition metrics for the existing five cases** from a minimal evaluation-only local bundle. The synthetic example, exporter/verifier, regression tests and 18-package version snapshot are versionable; real labels/predictions remain local pending release review. No new experimental requests or paid use were required. A fresh dependency installation and full paper/evidence reproduction remain unvalidated. See [commands and exact limits](offline_reproduction.md).


## Latest bounded live continuation — 18 September 2026

The repaired SearXNG-to-DuckDuckGo route passed one public and five development queries, at least 30 seconds apart. Ten page-fetch attempts produced three accepted pages; the page-augmented condition remained blocked for two cases. Both exact model versions passed synthetic image tests. A preregistered five-case comparison then used identical cached search snippets in both arms:

| Model | Answered | Accuracy, all five | Macro-F1, answered | Authentic false positives | Rated pair-arm cost |
|---|---:|---:|---:|---:|---:|
| GPT-4o-mini-2024-07-18 | 5/5 | 60% | 0.583 | 0 | $0.14675295 |
| GLM-5.3-Flash | 5/5 | 60% | 0.375 | 1 | $0.01932820 |

This is a fixed-evidence backbone diagnostic on five reused core cases, not the larger old-dataset comparison or evidence of superiority. Previously GLM-generated queries, provider decoding defaults, snippet limitations and missing human annotation remain explicit. No PROBE experiment occurred. Full/portable offline suites passed 193/189 tests; 4,199 protected historical artifacts and both registries remained unchanged. New rated use was $0.16740290, cumulative $0.218046352, with no pending charges and $14.781953648 remaining under the unchanged cap. The two-hour window ends at 13:22:32 UTC. Direct DDGS remains blocked; pre-existing SearXNG remains running, and run-started Firecrawl containers are stopped.

The source adds a pinned GPT-mini comparison client and audited, budget-preserving session renewal. Local raw evidence, protocols, predictions, usage and acceptance records are retained under the ignored `reports/autonomous/20260917T215500Z/paired_baseline_20260918_v1/` directory; a fresh clone does not contain them. No credentials or raw cases belong in the code checkpoint. Stage 1 remains incomplete.
