from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

JsonObjectFallback = Union[Dict[str, Any], Callable[[str], Dict[str, Any]]]
JsonArrayFallback = Union[List[Any], Callable[[str], List[Any]]]


def _apply_object_fallback(fallback: Optional[JsonObjectFallback], text: str) -> Optional[Dict[str, Any]]:
    if fallback is None:
        return None
    if callable(fallback):
        return fallback(text)
    return dict(fallback)


def _apply_array_fallback(fallback: Optional[JsonArrayFallback], text: str) -> Optional[List[Any]]:
    if fallback is None:
        return None
    if callable(fallback):
        return fallback(text)
    return list(fallback)


def extract_json_object(text: str, *, fallback: Optional[JsonObjectFallback] = None) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from text.

    Returns the parsed dict or the provided fallback. If parsing fails and no
    fallback is supplied, returns None.
    """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return _apply_object_fallback(fallback, text)


def extract_json_array(text: str, *, fallback: Optional[JsonArrayFallback] = None) -> Optional[List[Any]]:
    """Extract the first JSON array from text, or return the fallback."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return _apply_array_fallback(fallback, text)


__all__ = ["extract_json_object", "extract_json_array"]
