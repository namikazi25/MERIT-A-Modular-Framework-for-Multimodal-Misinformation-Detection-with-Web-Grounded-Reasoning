
"""API spend tracking + spend cap enforcement (hard rules 5 & 6).

Cap is read from MAX_API_SPEND_USD in .env. If the cap is undefined, paid
(API) runs are refused; only zero-cost local-provider runs are allowed.
Retries are logged and their tokens counted (rule 6).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

# per-1M-token (input, output) USD; extend as needed
PRICES_PER_1M: Dict[str, tuple] = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o": (2.50, 10.00),
}
LOCAL_PROVIDERS = {"lmstudio"}


class SpendTracker:
    def __init__(self, env_path: str = ".env", prices: Optional[Dict[str, tuple]] = None):
        self.prices = prices if prices is not None else dict(PRICES_PER_1M)
        self.tokens: Dict[str, Dict[str, int]] = {}  # model -> {prompt, completion, retried}
        self.cap = self._read_cap(env_path)
        self.cap_undefined = self.cap is None

    @staticmethod
    def _read_cap(env_path: str) -> Optional[float]:
        if os.path.exists(env_path):
            for line in open(env_path):
                line = line.strip()
                if line.startswith("MAX_API_SPEND_USD="):
                    try:
                        return float(line.split("=", 1)[1].strip())
                    except ValueError:
                        return None
        return None

    def add(self, model: str, prompt: int = 0, completion: int = 0, retried_prompt: int = 0, retried_completion: int = 0) -> None:
        bucket = self.tokens.setdefault(model, {"prompt": 0, "completion": 0, "retried": 0})
        bucket["prompt"] += int(prompt or 0)
        bucket["completion"] += int(completion or 0)
        bucket["retried"] += int(retried_prompt or 0) + int(retried_completion or 0)

    def model_cost(self, model: str) -> float:
        if model not in self.prices:
            return 0.0  # unknown/local: not billed in this tracker
        pin, pout = self.prices[model]
        t = self.tokens.get(model, {"prompt": 0, "completion": 0, "retried": 0})
        return (t["prompt"] + t["retried"]) / 1e6 * pin + t["completion"] / 1e6 * pout

    def total_cost(self) -> float:
        return sum(self.model_cost(m) for m in self.tokens)

    def check_cap(self) -> dict:
        """Return {'ok': bool, 'cost': float, 'cap': float|None, 'message': str}."""
        cost = self.total_cost()
        if self.cap_undefined:
            return {
                "ok": False,
                "cost": cost,
                "cap": None,
                "message": "MAX_API_SPEND_USD is not defined in .env; paid runs are refused (hard rule 5).",
            }
        if cost > self.cap:
            return {
                "ok": False,
                "cost": cost,
                "cap": self.cap,
                "message": f"Spend ${cost:.4f} exceeds cap ${self.cap:.4f}; halting (hard rule 5).",
            }
        return {"ok": True, "cost": cost, "cap": self.cap, "message": "within cap"}

    def would_exceed(self, extra_cost: float) -> bool:
        if self.cap_undefined:
            return True
        return self.total_cost() + extra_cost > self.cap

    def summary(self) -> str:
        parts = []
        for m, t in self.tokens.items():
            parts.append(f"{m}: prompt={t['prompt']} comp={t['completion']} retried={t['retried']} ~${self.model_cost(m):.4f}")
        return "; ".join(parts) if parts else "no tokens recorded"
