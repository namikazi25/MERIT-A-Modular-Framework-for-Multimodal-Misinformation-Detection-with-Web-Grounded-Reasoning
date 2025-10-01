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
    print(res)  # {"aligned": true/"partial"/false, "confidence": 0.0-1.0, "explanation": "..."}

Note: For backward compatibility, this module also exports
`assess_image_headline_alignment` as an alias of `assess_image_headline_relevancy`.
"""

from typing import Any, Dict, Optional

from scripts.llm_loader import LLMModelLoader
from scripts.utils.json_utils import extract_json_object
from scripts.utils.media import image_to_data_url


_DEFAULT_SYSTEM_PROMPT = (
    "You are a careful fact-checking assistant. Given a news headline and a "
    "reference image, judge how well the image supports the headline (i.e., "
    "whether it clearly depicts the claim, partially fits the context, or is "
    "unrelated). Respond with strict JSON only, no extra text."
)

def _build_user_instruction(headline: str) -> str:
    return (
        "Task: Determine whether the image is relevant to the headline.\n"
        "Consider three levels:\n"
        "1. Clear match: Image visibly depicts the main claim\n"
        "2. Partial match: Image shows the right context/setting, but you can't verify all specific details (they might exist but aren't prominent)\n"
        "3. Mismatch: Image shows different content or contradicts the claim\n\n"
        f"Headline: {headline}\n\n"
        "Output JSON strictly with keys:\n"
        "  aligned: boolean or \"partial\"\n"
        "    - true: image clearly shows what headline claims\n"
        "    - \"partial\": image shows the general context, but specific details mentioned in headline are not clearly visible (could exist but can't confirm from image alone)\n"
        "    - false: image contradicts or is irrelevant to headline\n"
        "  confidence: number in [0,1]\n"
        "  explanation: short textual rationale (for partial matches, explain what you CAN see and what you CANNOT verify)\n"
        "Example: {\"aligned\": \"partial\", \"confidence\": 0.62, \"explanation\": \"...\"}"
    )
def assess_image_headline_relevancy(
    image_path: str,
    headline: str,
    loader: LLMModelLoader,
    *,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask the selected LLM whether an image is relevant to a headline.

    Returns a dict with keys: aligned (bool or "partial"), confidence (float 0..1), explanation (str).
    """
    data_url = image_to_data_url(image_path)
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
    raw_text = text if isinstance(text, str) else str(text)
    parsed = extract_json_object(
        raw_text,
        fallback=lambda payload: {
            "aligned": None,
            "confidence": None,
            "explanation": payload.strip(),
        },
    )
    if parsed is None:
        parsed = {}

    # Normalize fields
    aligned = parsed.get("aligned")
    if isinstance(aligned, str):
        aligned_l = aligned.strip().lower()
        if aligned_l in ("true", "yes", "aligned", "relevant", "clear"):
            parsed["aligned"] = True
        elif aligned_l in ("partial", "partially", "context", "uncertain"):
            parsed["aligned"] = "partial"
        elif aligned_l in ("false", "no", "misaligned", "irrelevant", "mismatch"):
            parsed["aligned"] = False
        else:
            parsed["aligned"] = None

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
        "raw": raw_text,
    }


# Backward-compatibility alias
assess_image_headline_alignment = assess_image_headline_relevancy

__all__ = [
    "assess_image_headline_relevancy",
    "assess_image_headline_alignment",  # alias for compatibility
]
