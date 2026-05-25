from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Dict


_IMAGE_MIME_MAP: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def guess_mime_type(path: str, *, default: str = "image/png") -> str:
    """Guess an image MIME type based on file extension."""
    ext = Path(path or "").suffix.lower()
    return _IMAGE_MIME_MAP.get(ext, default)


def image_to_data_url(path: str, *, default_mime: str = "image/png") -> str:
    """Encode an image file as a base64 data URL, caching repeated reads."""
    resolved = str(Path(path).resolve())
    return _image_to_data_url_cached(resolved, default_mime)


@lru_cache(maxsize=256)
def _image_to_data_url_cached(path: str, default_mime: str) -> str:
    with open(path, "rb") as handle:
        data = handle.read()
    b64 = base64.b64encode(data).decode("ascii")
    mime = guess_mime_type(path, default=default_mime)
    return f"data:{mime};base64,{b64}"


__all__ = ["guess_mime_type", "image_to_data_url"]
