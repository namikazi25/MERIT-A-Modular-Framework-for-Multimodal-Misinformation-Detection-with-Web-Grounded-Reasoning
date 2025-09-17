from __future__ import annotations

"""Unified web search wrapper supporting Brave and DuckDuckGo (DDGS)."""

import os
import threading
import time
from typing import Any, Dict, List, Optional

from scripts.brave_search import brave_web_search, extract_web_results

try:  # Prefer the modern package name first
    from ddgs import DDGS  # type: ignore
except ImportError:  # Fall back to the legacy import path
    try:
        from duckduckgo_search import DDGS  # type: ignore  # pragma: no cover
    except ImportError:  # pragma: no cover
        DDGS = None  # type: ignore


_SUPPORTED_PROVIDERS = {"brave", "duckduckgo"}


class RateLimiter:
    """Simple thread-safe rate limiter enforcing minimum spacing between calls."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._interval = max(0.0, float(min_interval_seconds))
        self._lock = threading.Lock()
        self._next_ready = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_ready - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_ready = now + self._interval


_rate_limiters: Dict[str, RateLimiter] = {}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_rate_limiter(provider: str) -> RateLimiter:
    if provider not in _rate_limiters:
        if provider == "duckduckgo":
            interval = _env_float("DUCKDUCKGO_SEARCH_MIN_INTERVAL", 1.5)
        else:
            interval = _env_float("BRAVE_SEARCH_MIN_INTERVAL", 1.0)
        _rate_limiters[provider] = RateLimiter(interval)
    return _rate_limiters[provider]


def get_active_search_provider(explicit: Optional[str] = None) -> str:
    provider = (explicit or os.getenv("SEARCH_PROVIDER", "brave")).strip().lower()
    return provider if provider in _SUPPORTED_PROVIDERS else "brave"


def web_search(query: str, *, provider: Optional[str] = None, **params: Any) -> Dict[str, Any]:
    resolved = get_active_search_provider(provider)
    limiter = _get_rate_limiter(resolved)
    limiter.acquire()

    if resolved == "duckduckgo":
        return _duckduckgo_web_search(query, **params)

    timeout = params.pop("timeout", 20)
    payload = brave_web_search(query, timeout=timeout, **params)
    normalized = _normalize_brave_results(payload)
    return {
        "provider": "brave",
        "query": query,
        "results": normalized,
        "raw": payload,
    }


def _normalize_brave_results(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in extract_web_results(payload):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        meta_url = item.get("meta_url") if isinstance(item.get("meta_url"), dict) else {}
        host = ""
        if isinstance(meta_url, dict):
            host = str(meta_url.get("host") or "").strip()
        source = str(item.get("source") or host or "").strip()
        normalized.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "description": str(item.get("description") or item.get("snippet") or "").strip(),
                "source": source,
            }
        )
    return normalized


def _duckduckgo_web_search(query: str, **params: Any) -> Dict[str, Any]:
    if DDGS is None:
        raise ImportError(
            "duckduckgo-search dependency is not installed. Add 'duckduckgo-search' to requirements.txt "
            "and install it to use the DuckDuckGo provider."
        )

    ddg_kwargs: Dict[str, Any] = {}
    for key in ("region", "safesearch", "timelimit", "max_results", "page", "backend"):
        if key in params and params[key] is not None:
            ddg_kwargs[key] = params[key]
    proxy = params.get("proxy", os.getenv("DDGS_PROXY"))
    timeout = params.get("timeout", 10)

    with DDGS(proxy=proxy, timeout=timeout) as ddgs:  # type: ignore
        raw_results = list(ddgs.text(query, **ddg_kwargs))

    normalized: List[Dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("href") or item.get("url") or "").strip()
        if not url:
            continue
        normalized.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "description": str(item.get("body") or item.get("description") or item.get("snippet") or "").strip(),
                "source": str(item.get("source") or "").strip(),
            }
        )

    return {
        "provider": "duckduckgo",
        "query": query,
        "results": normalized,
        "raw": raw_results,
        "params": {
            "proxy": proxy,
            **ddg_kwargs,
        },
    }


__all__ = ["web_search", "get_active_search_provider"]
