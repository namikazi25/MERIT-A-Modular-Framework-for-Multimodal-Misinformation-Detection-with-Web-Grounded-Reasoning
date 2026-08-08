
"""Persistent disk-backed web-search cache (Task 0.2).

Keyed by normalized query string + provider. Raw results stored as JSON.
Lives outside per-run directories (evidence/search_cache/) and persists
across runs. `--no-cache` bypasses reads AND writes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

from scripts.search_provider import web_search


class SearchCache:
    def __init__(self, cache_dir: str = "evidence/search_cache", enabled: bool = True, verbose: bool = False):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.verbose = verbose

    @staticmethod
    def normalize(query: str) -> str:
        return re.sub(r"\s+", " ", (query or "").strip().lower())

    def _path(self, query: str, provider: str) -> str:
        key = hashlib.sha256(f"{provider}::{self.normalize(query)}".encode("utf-8")).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, query: str, provider: str) -> Optional[Dict[str, Any]]:
        """Return cached raw payload, or None."""
        if not self.enabled:
            return None
        p = self._path(query, provider)
        if not os.path.exists(p):
            return None
        try:
            d = json.load(open(p))
            if d.get("provider") == provider and d.get("normalized") == self.normalize(query):
                return d.get("result")
        except Exception:
            return None
        return None

    def put(self, query: str, provider: str, result: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        payload = {
            "query": query,
            "normalized": self.normalize(query),
            "provider": provider,
            "ts": time.time(),
            "result": result,
        }
        with open(self._path(query, provider), "w") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    def search(self, query: str, *, provider: Optional[str] = None,
               max_attempts: int = 3, backoff_base: float = 1.0, **params: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Cached wrapper around web_search (rule 6: retries with backoff, logged).

        Returns (payload, meta). Raises after max_attempts consecutive failures.
        """
        prov = provider or "duckduckgo"
        cached = self.get(query, prov)
        if cached is not None:
            if self.verbose:
                print(f"[cache HIT ] {query[:70]}")
            return cached, {"cache_hit": True, "provider": prov, "attempts": 1}
        import time as _time
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                payload = web_search(query, provider=prov, **params)
                self.put(query, prov, payload)
                if self.verbose:
                    print(f"[cache MISS ] {query[:70]} (stored, attempt {attempt})")
                return payload, {"cache_hit": False, "provider": prov, "attempts": attempt}
            except Exception as e:
                last_err = e
                if self.verbose or True:
                    print(f"[search RETRY] {query[:60]} attempt {attempt} failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
                if attempt < max_attempts:
                    _time.sleep(backoff_base * (2 ** (attempt - 1)))
        raise RuntimeError(f"web_search failed after {max_attempts} attempts for {query!r}: {last_err!r}")
