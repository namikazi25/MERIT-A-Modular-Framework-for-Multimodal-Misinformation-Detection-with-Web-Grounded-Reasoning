# Bundle structural equivalence check (pre-split, requested)

Question: are the new-300 bundles structurally equivalent to the v2-200, or systematically thinner
(which would confound any evidence-reading judge across the dev/holdout split)?

Method: per-bundle question count (documents keys) and estimated evidence tokens (description chars/4),
old-200 (`evidence/live200-v2`, n=200) vs new-300 (`evidence/live300-v2`, n=142 at check time).
Two-sided Mann-Whitney U (tie-corrected, normal approximation).

| metric | old-200 median (IQR) | new-300 median (IQR) | MWU p |
|---|---|---|---|
| n_questions | 3 (3-4) | 3 (3-4) | 0.2303 |
| evidence tokens | 1948 (1574-2503) | 2038 (1608-2725) | 0.5232 |

- n_answers mirrors n_questions (mean 3.7 vs 3.9); n_docs mean 37.1 vs 39.2.
- Field schema identical; all bundles have claim/relevancy/visual_veracity; questions_complete=True.
- The "3-5 questions per sample" is the gather's normal behavior (original 200-v2 mean 3.7), not a new-run degradation.

Verdict: **structurally equivalent; no evidence-density confound.** Split may proceed once A.2 completes.
