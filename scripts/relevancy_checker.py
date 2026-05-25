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
    "You are an image-headline relevancy assessor. Evaluate if the image depicts "
    "the specific subject and context described in the headline.\n"
    "Respond with strict JSON only."
)

def _build_user_instruction(headline: str) -> str:
    return (
        f"Headline: {headline}\n\n"
        
        "Task: Does the image show the specific subject/event from the headline?\n\n"
        
        "Classification:\n"
        "- aligned=true: Image clearly depicts the specific subject/event\n"
        "- aligned=partial: Image shows related content but lacks confirmation of specifics\n"
        "- aligned=false: Image shows different subject/event\n\n"
        
        "Confidence calibration (CRITICAL):\n"
        "- 0.9-1.0: Can identify specific people/places/events mentioned in headline\n"
        "- 0.7-0.9: Shows correct general context but cannot confirm specific details\n"
        "- 0.5-0.7: Shows related content but connection is weak or ambiguous\n"
        "- 0.3-0.5: Superficial similarity only, likely wrong subject\n"
        "- 0.0-0.3: No meaningful connection\n\n"
        
        "For partial alignment:\n"
        "- High confidence (0.7+): Right subject, details not fully visible\n"
        "- Low confidence (<0.7): Possibly wrong subject, superficial match\n\n"
        
        "Output JSON:\n"
        "{\n"
        "  \"aligned\": \"true\" | \"partial\" | \"false\",\n"
        "  \"confidence\": 0.0-1.0,\n"
        "  \"explanation\": \"What you observed and why this confidence\"\n"
        "}"
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
