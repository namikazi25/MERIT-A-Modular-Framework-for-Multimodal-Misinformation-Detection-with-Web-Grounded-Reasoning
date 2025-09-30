from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_dataset_records(path: Path) -> List[Dict[str, Any]]:
    """Load dataset records from a JSON file with flexible top-level shapes."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [rec for rec in data if isinstance(rec, dict)]
    if isinstance(data, dict):
        for key in ("data", "items", "records", "samples"):
            value = data.get(key)
            if isinstance(value, list):
                return [rec for rec in value if isinstance(rec, dict)]
        for value in data.values():
            if isinstance(value, list):
                return [rec for rec in value if isinstance(rec, dict)]
    raise ValueError(f"Unsupported JSON structure in dataset file: {path}")


def normalize_record_path(path: Optional[str], image_root: Optional[Path]) -> Optional[str]:
    """Resolve dataset paths relative to an optional image root."""
    if not path:
        return None
    p = Path(str(path))
    if not p.is_absolute() and image_root:
        p = Path(image_root) / str(path).lstrip("/\\")
    try:
        return str(p.resolve())
    except Exception:
        return str(p.expanduser().absolute())


def coerce_index(value: Any) -> Optional[int]:
    """Coerce dataset index values to integers, ignoring booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def resolve_image_candidates(image_root: Path, img_path: str) -> List[str]:
    """Return possible normalized keys for matching dataset image paths."""
    p = Path(img_path)
    candidates: List[str] = []
    if p.is_absolute():
        candidates.append(str(p.resolve()))
        candidates.append(str((image_root / str(p).lstrip(os.sep)).resolve()))
    else:
        candidates.append(str((image_root / p).resolve()))
    rel = _to_rel_under_root(image_root, img_path)
    candidates.append(rel)
    candidates.append(_to_rel_under_root(image_root, "/" + rel))

    unique: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _to_rel_under_root(image_root: Path, path: str) -> str:
    normalized = path.replace("\\", "/").lstrip("/")
    try:
        rel = Path(path).relative_to(image_root)
        return rel.as_posix()
    except Exception:
        return normalized


__all__ = [
    "load_dataset_records",
    "normalize_record_path",
    "coerce_index",
    "resolve_image_candidates",
]
