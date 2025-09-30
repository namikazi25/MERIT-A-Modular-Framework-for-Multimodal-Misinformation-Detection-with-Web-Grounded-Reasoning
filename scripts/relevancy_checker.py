from __future__ import annotations

"""
LLM-based relevancy check between a news headline and an image.

This module provides a function that uses the switchable loader from
scripts.llm_loader to query an LLM (OpenAI, Google Gemini, DeepInfra, or OpenRouter)
about whether a given image is relevant to a headline.

Example
    from scripts.llm_loader import LLMModelLoader
    from scripts.relevancy_checker import assess_image_headline_relevancy

    loader = LLMModelLoader({"provider": "openai", "model": "gpt-4o"})
    res = assess_image_headline_relevancy("path/to/image.jpg", "Breaking: ...", loader)
    print(res)  # {"aligned": true/false, "confidence": 0.0-1.0, "explanation": "..."}

Note: For backward compatibility, this module also exports
`assess_image_headline_alignment` as an alias of `assess_image_headline_relevancy`.
"""

import base64
import json
import os
from typing import Any, Dict, Optional

from scripts.llm_loader import LLMModelLoader


_DEFAULT_SYSTEM_PROMPT = (
    "You are a careful fact-checking assistant. Given a news headline and a "
    "reference image, judge if the image is relevant to the headline (i.e., "
    "the image plausibly depicts the claim in the headline). Respond with strict JSON only, no extra text."
)


def _guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext in (".png",):
        return "image/png"
    if ext in (".webp",):
        return "image/webp"
    if ext in (".gif",):
        return "image/gif"
    if ext in (".bmp",):
        return "image/bmp"
    # Fallback
    return "image/png"


def _image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    mime = _guess_mime(image_path)
    return f"data:{mime};base64,{b64}"


def _build_user_instruction(headline: str) -> str:
    return (
        "Task: Determine whether the image is relevant to the headline.\n"
        f"Headline: {headline}\n\n"
        "Output JSON strictly with keys: \n"
        "  aligned: boolean (true if relevant/aligned, false if not)\n"
        "  confidence: number in [0,1]\n"
        "  explanation: short textual rationale\n"
        "Example: {\"aligned\": true, \"confidence\": 0.78, \"explanation\": \"...\"}"
    )


def _parse_json_response(text: str) -> Dict[str, Any]:
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Extract first top-level JSON object heuristically
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    # Fallback minimal schema
    return {"aligned": None, "confidence": None, "explanation": text.strip()}


def assess_image_headline_relevancy(
    image_path: str,
    headline: str,
    loader: LLMModelLoader,
    *,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask the selected LLM whether an image is relevant to a headline.

    Returns a dict with keys: aligned (bool), confidence (float 0..1), explanation (str).
    """
    data_url = _image_to_data_url(image_path)
    model = loader.get_model()

    sys_msg = system_prompt or _DEFAULT_SYSTEM_PROMPT
    user_text = _build_user_instruction(headline)

    messages = [
        {"role": "system", "content": sys_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": data_url},
            ],
        },
    ]

    # Invoke the model directly with structured content (multimodal)
    resp = model.invoke(messages)
    text = getattr(resp, "content", resp)
    parsed = _parse_json_response(text if isinstance(text, str) else str(text))

    # Normalize fields
    aligned = parsed.get("aligned")
    if isinstance(aligned, str):
        aligned_l = aligned.strip().lower()
        aligned = True if aligned_l in ("true", "yes", "aligned", "relevant") else False if aligned_l in ("false", "no", "misaligned", "irrelevant") else None
        parsed["aligned"] = aligned

    conf = parsed.get("confidence")
    try:
        if conf is not None:
            parsed["confidence"] = float(conf)
    except Exception:
        parsed["confidence"] = None

    expl = parsed.get("explanation")
    if not isinstance(expl, str):
        parsed["explanation"] = str(expl)

    return {
        "aligned": parsed.get("aligned"),
        "confidence": parsed.get("confidence"),
        "explanation": parsed.get("explanation"),
        "raw": text,
    }


# Backward-compatibility alias
assess_image_headline_alignment = assess_image_headline_relevancy

__all__ = [
    "assess_image_headline_relevancy",
    "assess_image_headline_alignment",  # alias for compatibility
]
