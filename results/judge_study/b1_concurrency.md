# B.1 Concurrency benchmark (local qwen3.6-35b-a3b, 30 realistic J1 judge calls)

| concurrency | wall (30 calls) | tok/s | vs c1 | swap after |
|---|---|---|---|---|
| 1 | 93.0s | 409.8 | 1.00x | 5.64GB used |
| 2 | 56.3s | 677.2 | 1.65x | 5.64GB used |
| 4 | 41.2s | 923.3 | 2.25x | 5.63GB used |

- **Outputs NOT byte-identical across concurrency settings (30/30 samples differ).**
- Per B.1 rule: **concurrency 1 is pinned for all scored runs**; c2/c4 allowed only for throwaway exploration (marked as such in manifests).
- Note: consistent with Brief-1 rule-7 finding that LM Studio temp-0 is not bit-deterministic across time; the concurrency comparison is strict (different settings => different outputs), so scored runs stay at c1.
