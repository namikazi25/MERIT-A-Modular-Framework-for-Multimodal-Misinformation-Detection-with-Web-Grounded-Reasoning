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
    "You are a fact-checking assistant. Judge how well an image supports a headline. "
    "Focus on whether the image authentically represents what the headline claims. "
    "Respond with strict JSON only."
)

def _build_user_instruction(headline: str) -> str:
    return (
        "Task: Does this image reasonably fit with the headline?\n\n"
        
        "Guidelines:\n\n"
        
        "ALIGNED (true):\n"
        "- Image clearly shows the subject or event mentioned\n"
        "- Image provides appropriate context for the headline\n"
        "- Reasonable connection exists even if not all details visible\n"
        "- Example: Headline about protest → image shows crowd with signs\n"
        "- Example: Headline about person → image shows that person\n\n"
        
        "PARTIAL (\"partial\"):\n"
        "- Image shows related context but key specifics unclear\n"
        "- Right general setting but cannot confirm exact details\n"
        "- Example: Headline names specific person → image shows someone but unclear who\n"
        "- Example: Headline claims specific location → image shows a location but unclear which\n"
        "- Loose connection that could fit but lacks confirmation\n\n"
        
        "MISALIGNED (false):\n"
        "- Image shows clearly different subject or event\n"
        "- No reasonable connection to headline\n"
        "- Image contradicts the headline claim\n"
        "- Example: Headline about football → image shows basketball\n"
        "- Example: Headline about Person A → image clearly shows Person B\n\n"
        
        f"Headline: {headline}\n\n"
        
        "Be reasonable: Accept loose connections for aligned. "
        "Only use false when clearly wrong or unrelated.\n\n"
        
        "Output JSON:\n"
        "{\n"
        "  \"aligned\": true | \"partial\" | false,\n"
        "  \"confidence\": 0.0-1.0,\n"
        "  \"explanation\": \"What you see and why you chose this rating\"\n"
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
