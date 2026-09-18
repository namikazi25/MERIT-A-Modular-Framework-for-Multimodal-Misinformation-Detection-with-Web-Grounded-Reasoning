# Offline tests and metric reproduction

These commands separate synthetic implementation checks from reproduction of recorded metrics. Neither mode invokes models or retrieval. The real evaluation-only bundle remains local pending release review; the committed example contains invented records only.

## Synthetic source-only tests

From the repository root, with the recorded test dependencies installed:

```sh
python3 -B scripts/run_offline_tests.py --profile portable
```

This profile passed **185 tests** in a source-only export with no benchmark data, registry, audit directory, model files or environment files. Four historical-artifact checks are explicitly excluded and printed by name: two preservation checks and two pre-fix contrast checks. All synthetic integrity, exclusion, alias, nested-input, budget, retrieval and detector-preparation checks remain included. The overwrite-protection test now creates its own temporary v1 fixture and verifies its bytes remain unchanged.

The default/full profile retains those four checks:

```sh
python3 -B scripts/run_offline_tests.py --profile full
```

It passed **189 tests** in the original workspace. It requires preserved local artifacts; a clean source export cannot claim to have verified their preservation. The original source-only run actually failed three tests and skipped two before this separation, so the portable result is not inferred from the working tree.

Both profiles install Python network and environment-file guards before discovery. This is not an OS sandbox for arbitrary native/subprocess code. Tests use dummy configuration and mocks; they do not establish live service access.

## Recorded dependency versions

`config/offline_test_environment.json` records CPython/platform information and the 18 installed dependency versions used for these synthetic tests. `config/offline_test_requirements.txt` lists the same exact versions, including dependencies selected by the current platform's metadata markers. Optional live-search, full-model and detector-runtime dependencies are outside this test snapshot.

```sh
python3 -B scripts/check_offline_environment.py
```

This compares package versions and the platform with the recorded environment and returns nonzero on differences. It reads package metadata only, without installing anything or inspecting package URLs, credentials or environment files. The clean-source validation reused the installed interpreter dependencies. **A fresh dependency installation, wheel hashes and cross-platform compatibility have not been validated.** These files are a tested-version snapshot, not a universal lockfile.

## Standard-library-only metrics

From the repository root:

```sh
python3 -S -B -m scripts.redesign.reproduce verify examples/redesign/metrics_synthetic_v1.json
```

`-S` disables third-party site packages. The invented five-case example has one correct answer, one authentic false positive, one abstention, one failed request and one missing prediction. Expected coverage is 2/5, all-case accuracy 1/5 and answered accuracy 1/2. Failed and missing cases remain in the denominators.

To verify a separately supplied, reviewed real bundle:

```sh
python3 -S -B -m scripts.redesign.reproduce verify /path/to/metrics_bundle.json --output /path/to/new_metrics.json
```

The actual local bundle reproduced **22 recorded condition metrics across three batches for the same five development cases**, both in the working repository and in the clean source export. No model, search, extraction, image or original run directory is needed for verification. The verifier checks the schema, exact identities, group bindings, evaluator hash, bundle checksum and recomputed expected metrics. It rejects duplicate JSON keys, non-finite values, unknown identities, duplicate predictions and unexpected metadata. Checksums support consistency with recorded references, not authenticity against someone able to replace all references.

The bundle contains evaluation labels and predictions, so it is **evaluation-only** and must never be passed to a model. It omits claims, images, source passages, rationales, raw provider records and failure-message text. Export normalizes failed statuses to `ERROR` and malformed predictions to `null`, and checks that this projection leaves the original metrics unchanged. Metric reproduction does not reproduce the original inference, establish evidence quality, regenerate paired uncertainty estimates or calibrate Jev/PROBE.

## Local export for the completed run

Export requires the real registry and original records. This command was executed with a new output path under the ignored continuation directory:

```sh
python3 -B -m scripts.redesign.reproduce export \
  --run-dir reports/autonomous/20260917T215500Z \
  --output reports/autonomous/20260917T215500Z/continuation_02/metrics_bundle.json
```

The exporter admits only the five IDs in the existing initial protocol through the normal development guard, checks each against the approved core, verifies case/prediction identity and condition completeness, and compares recomputed metrics with the stored analysis before writing. Existing outputs cannot be overwritten; use an explicit new version if exporting again. It is intentionally limited to the completed three-batch run and does not authorize additional data selection or reserve evaluation.

The local bundle, export provenance, first failing test log and successful acceptance logs are preserved under `reports/autonomous/20260917T215500Z/continuation_02/`. This directory is ignored and not included in a fresh clone. Distributing the bundle remains subject to dataset/split review. A full paper release still needs permitted evidence artifacts, a portable pinned service configuration, a validated installation and the separately gated experiments and annotations.
