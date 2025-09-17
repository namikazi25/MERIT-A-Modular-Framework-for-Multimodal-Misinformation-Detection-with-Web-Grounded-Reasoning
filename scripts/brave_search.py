from __future__ import annotations

"""
Thin wrapper around Brave Search API (Web Search).

Env var:
  - BRAVE_API_KEY (aka x-subscription-token)

Usage:
    from scripts.brave_search import brave_web_search, extract_web_results
    res = brave_web_search("When did Westworld first season premiere?")
    results = extract_web_results(res)
"""

import os
from typing import Any, Dict, List

import requests


BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def brave_web_search(query: str, *, timeout: int = 20, **params: Any) -> Dict[str, Any]:
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise ValueError("Missing BRAVE_API_KEY in environment (.env)")

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "x-subscription-token": api_key,
    }
    req_params = {"q": query}
    # Allow light customization via params (e.g., country, safesearch)
    req_params.update({k: v for k, v in (params or {}).items() if v is not None})

    resp = requests.get(BRAVE_ENDPOINT, headers=headers, params=req_params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def extract_web_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract top-level web.results list if present; else empty list."""
    try:
        web = payload.get("web") or {}
        results = web.get("results") or []
        return [r for r in results if isinstance(r, dict)]
    except Exception:
        return []


__all__ = ["brave_web_search", "extract_web_results"]

