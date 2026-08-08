
"""Statistics for the judge study (hard rule 4).

- 95% bootstrap CI: 1000 resamples, per-sample pairing preserved.
- McNemar's test (exact binomial, two-sided) for paired model/judge comparisons.
"""
from __future__ import annotations

import math
import random
from typing import Callable, List, Sequence, Tuple

POSITIVE = "Misinformation"


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    n = len(y_true)
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / n if n else 0.0


def binary_f1(y_true: Sequence, y_pred: Sequence, positive: str = POSITIVE) -> float:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == positive and b == positive)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a != positive and b == positive)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == positive and b != positive)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric: Callable[[Sequence, Sequence], float],
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (point_estimate, lo, hi) of the 95% bootstrap CI (percentile method)."""
    n = len(y_true)
    rng = random.Random(seed)
    point = metric(y_true, y_pred)
    vals: List[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(metric([y_true[i] for i in idx], [y_pred[i] for i in idx]))
    vals.sort()
    lo = vals[int(n_resamples * alpha / 2)]
    hi = vals[int(n_resamples * (1 - alpha / 2)) - 1]
    return point, lo, hi


def mcnemar(
    y_true: Sequence, y_pred_a: Sequence, y_pred_b: Sequence
) -> dict:
    """Exact two-sided McNemar on paired labels.

    b = samples where A wrong & B right; c = samples where A right & B wrong.
    Returns dict with b, c, n, chi2 (continuity-corrected), p (exact binomial), significant.
    """
    b = sum(1 for t, a, p in zip(y_true, y_pred_a, y_pred_b) if a != t and p == t)
    c = sum(1 for t, a, p in zip(y_true, y_pred_a, y_pred_b) if a == t and p != t)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n": 0, "chi2": 0.0, "p": 1.0, "significant": False}
    # exact binomial two-sided
    p_val = 0.0
    for k in range(0, min(b, c) + 1):
        p_val += math.comb(n, k) * (0.5 ** n)
    p_val = min(1.0, 2.0 * p_val)
    chi2 = ((abs(b - c) - 1) ** 2) / n
    return {"b": b, "c": c, "n": n, "chi2": round(chi2, 4), "p": round(p_val, 6), "significant": p_val < 0.05}
