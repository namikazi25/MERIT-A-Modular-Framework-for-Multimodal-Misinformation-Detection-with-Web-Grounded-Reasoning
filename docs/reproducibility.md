# Reproducibility and repository boundaries

This records the release plan and current status. The paper reproduction package is not complete. Direct DDGS repair is deferred; fresh experiments remain stopped at their recorded access and annotation gates. Engineering continuation is on `codex/merit-redesign-preflight`.

## Search access

Self-hosting SearXNG does not establish an unlimited DuckDuckGo allowance. We have not verified a guaranteed requests-per-minute quota for automated DuckDuckGo access. SearXNG's inbound limiter and upstream-engine suspension are separate controls.

The guarded redesign adapters now enforce one in-flight search, at least 30 seconds between logical searches, and a shared persisted stop after HTTP 429, access denial or CAPTCHA. Extraction allows one in-flight request, at least 6 seconds globally and 30 seconds per original target host. These are conservative project ceilings, not proven sustainable provider rates. Keep first-page results, query deduplication and bounded batches. Report a provider block separately from an empty evidence finding. Do not rotate identities or switch engines silently.

`scripts/redesign/traffic.py` stores leases, spacing and engine/host blocks in `.runtime/redesign-retrieval/traffic.json`, shared across adapter instances and run directories. File locking prevents concurrent admission. Interrupted leases and corrupt state fail closed. API `Retry-After` values extend the minimum backoff of 3,600 seconds for rate failures or 86,400 seconds for access challenges. Elapsed backoff never automatically clears a block. Frozen snapshot replay dispatches nothing and does not consume a live lease. Membership and budget checks run again after any pacing wait. Do not delete state to bypass a block; reconcile an interrupted request or review access before any supervised state change.

This controls calls made through the guarded redesign adapters. Legacy runners, manual HTTP calls and other users of Docker are outside this control. SearXNG can fan out to multiple configured engines; Firecrawl may perform internal retries, page-resource loads and redirects that the client cannot count or pace. The target-host interval applies to the submitted host, not every downstream resource host. A client ceiling does not prove upstream quota compliance. The implementation adds no automatic retry. Hidden upstream fan-out and deployment settings must be reviewed before sustained collection.

The inspected Firecrawl service is self-hosted. Its source contains rate-limit defaults, but the active authentication/limiter path was not established. [Firecrawl Cloud limits](https://docs.firecrawl.dev/rate-limits) are not a verified quota for this deployment. The project ceilings above apply regardless.

SearXNG documents default engine suspensions of 3,600 seconds for HTTP 429 and 86,400 seconds for CAPTCHA/access denial. These are local backoff settings, not promises of upstream recovery. The inspected deployment instead had 180 seconds for HTTP 429/access denial and 3,600 seconds for CAPTCHA. Review these shorter settings before sustained live work; the research adapter additionally preserves a stop record requiring supervised review. No suspension settings were changed during repository cleanup.

Sources: [suspension settings](https://docs.searxng.org/admin/settings/settings_search.html), [inbound limiter](https://docs.searxng.org/admin/searx.limiter.html), [DuckDuckGo engine documentation](https://docs.searxng.org/dev/engines/online/duckduckgo.html).

## Three reproduction modes

1. **Offline result reproduction:** distribute permitted predictions, evaluation labels and split identifiers separately from model inputs, plus the exact evaluator and expected metrics. Recompute published metrics without API keys, search access or new inference.
2. **Frozen-evidence rerun:** provide a reviewed, versioned evidence bundle containing search queries, engine settings, ordered results, retrieved passages, URLs, timestamps, statuses and hashes. Run the pipeline against these recorded artifacts. Hosted-model responses can still vary with pinned names and settings; retain original outputs and usage for comparison.
3. **Fresh web replication:** publish the search/extraction procedure, pinned service/dependency versions, budgets, pacing and failure handling. Ranking, pages, availability and access restrictions can change. New searches must not silently replace the frozen evidence used for reported results.

`scripts/redesign/retrieval.py` implements hashed snapshot storage and configuration/query-checked replay. The [offline reproduction tools](offline_reproduction.md) now support portable synthetic tests and standard-library-only metric verification. They were checked in a source-only export; the real local bundle reproduces 22 condition metrics for five cases. A distributable evidence bundle, validated dependency installation/lock, portable live experiment runner and full paper reproduction validation are still required. Passing mocks or preparing annotation sheets does not satisfy those release requirements.

The source-tree offline test entry point is:

```sh
.venv/bin/python -B scripts/run_offline_tests.py
```

For a source-only checkout without private audit artifacts, explicitly select `--profile portable`. This retains synthetic safeguard tests and lists the four historical-artifact checks it excludes. The default full profile still requires the private preservation and pre-fix artifacts. See the linked offline guide for the exact tested dependency snapshot and metric-only commands.

It resolves the repository independently of the working directory and writes results to stdout, without requiring an ignored historical run directory. Python network and environment-file guards are installed before discovery; these guards are not an OS sandbox for arbitrary native/subprocess code. The existing installed dependencies are required. This tests the implementation; it does not regenerate paper results or establish live service access.

Review redistribution rights, personal information and dataset split restrictions before packaging evidence. When passages cannot be redistributed, record URLs, timestamps/hashes and acquisition procedures, and state the resulting reproduction limitation. Do not release reserved examples for convenience. Checksums establish consistency with recorded references, not authenticity against an actor able to rewrite all references.

## Verified service versions

On 2026-09-18 UTC, the local SearXNG deployment was updated from `2026.3.9-d4954a064` to `2026.9.17-274b63b67`, pinned as:

```text
docker.io/searxng/searxng@sha256:ba0a344c566ecc6e4429e81d02d93a01fa05c80e9fb6d08f0a1e57a729aa6d87
```

Three spaced DuckDuckGo-only public queries returned relevant official IANA, Python and NASA sources. An isolated direct DDGS 9.16.0 test then received HTTP 202 with a challenge page; all further searches stopped. The SearXNG route worked at those timestamps, while direct DDGS remained blocked. Neither establishes sustained access. The main Python environment remains on DDGS 9.14.4. The older historical requirement was unpinned, so its installed version cannot be reconstructed from that file alone.

The isolated 9.16.0 wheel has SHA256 `175d9198c958a263f51a06a54368ba0b41294942c0cd29f4aa71be03dbcd5f4a`. Its [release notes](https://github.com/deedy5/ddgs/releases/tag/v9.16.0) include a DuckDuckGo client fix, but it did not restore the tested direct route. Keep experimental provider identities explicit: modern DDGS can use automatic metasearch by default.

The full local diagnostic and rollback instructions remain under `reports/retrieval_diagnostic_20260918_v1/`. That directory is ignored and not included in a fresh clone. The pinned Docker deployment is also outside the repository. A portable service definition with non-secret example settings remains a release prerequisite.

## Repository contents and checkpoints

Keep implementation, synthetic fixtures/tests, prompts, non-secret provider configuration, dependency declarations and reviewed methodology documentation versionable. Keep credentials in local environment files.

The root `.gitignore` excludes new environment files, downloaded runtimes/model weights, editor state, audit backups, generated results, raw autonomous/diagnostic runs and generated registries. These remain on disk. Ignoring them does not back them up or make them available to collaborators. Required, permitted reproduction assets need a separately versioned release bundle with hashes and acquisition instructions.

Do not blanket-ignore all JSON/JSONL or all reports: that would hide legitimate fixtures/configuration and reviewed documents. Preflight reports/manifests remain visible for deliberate review before staging. Generated registries include local paths and protected split/exposure metadata; a fresh clone will not contain them under these rules. The integrity guard correctly requires its dependencies, so a portable acquisition/build procedure is a release prerequisite rather than a check to bypass.

At cleanup, Git already tracked 578 historical result files, including one approximately 22.8 MB JSONL file, and `.env.sample`. These names were inventoried without reading the environment template. Ignore rules do not remove tracked files or scrub history. Existing tracked artifacts were preserved. Publication review must determine what is suitable for distribution; `*.tokens.csv` usage-accounting filenames are not themselves evidence of exposed credentials.

Before a code checkpoint, review the explicit file list and diff and scan the proposed non-environment payload for secrets and private artifacts. Avoid blanket staging. Before publication, test a clean checkout with only documented release assets and complete historical-artifact review. No commit or push was made during this cleanup.

Subsequent user authorization allowed a code checkpoint and push to `codex/merit-redesign-preflight` (initial commit `effc9f6`). Raw run data, annotation packets, registry artifacts and local model files were excluded. New source and synthetic tests are versioned; the ignored evidence is preserved locally and is not backed up by that push. See [PROBE preparation](probe_detector.md) for its separate licence and inference gates.
