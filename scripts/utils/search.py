from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


_DEFAULT_URL_KEYS: Sequence[str] = ("url", "href")
_DEFAULT_TITLE_KEYS: Sequence[str] = ("title",)
_DEFAULT_DESC_KEYS: Sequence[str] = ("description", "snippet", "body")
_DEFAULT_SOURCE_KEYS: Sequence[str] = ("source",)


def _first_non_empty(item: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def normalize_search_results(
    items: Iterable[Dict[str, Any]],
    *,
    url_keys: Sequence[str] = _DEFAULT_URL_KEYS,
    title_keys: Sequence[str] = _DEFAULT_TITLE_KEYS,
    desc_keys: Sequence[str] = _DEFAULT_DESC_KEYS,
    source_keys: Sequence[str] = _DEFAULT_SOURCE_KEYS,
) -> List[Dict[str, str]]:
    """Normalize heterogeneous search result dictionaries into a standard schema."""
    normalized: List[Dict[str, str]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        url = _first_non_empty(raw, url_keys)
        if not url:
            continue
        normalized.append(
            {
                "title": _first_non_empty(raw, title_keys),
                "url": url,
                "description": _first_non_empty(raw, desc_keys),
                "source": _first_non_empty(raw, source_keys),
            }
        )
    return normalized


def normalize_search_payload(
    payload: Dict[str, Any],
    *,
    legacy_extractor: Optional[Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]]] = None,
    result_key: str = "results",
) -> List[Dict[str, str]]:
    """Normalize search payloads that use a `results` list or a legacy extractor."""
    results = payload.get(result_key)
    if isinstance(results, Iterable):
        normalized = normalize_search_results(results)
        if normalized:
            return normalized

    if legacy_extractor is not None:
        try:
            legacy_items = legacy_extractor(payload)
        except Exception:
            legacy_items = []
        return normalize_search_results(legacy_items)

    return []


__all__ = ["normalize_search_results", "normalize_search_payload"]
