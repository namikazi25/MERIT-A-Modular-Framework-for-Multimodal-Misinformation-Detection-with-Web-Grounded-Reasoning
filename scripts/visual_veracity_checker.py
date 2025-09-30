from __future__ import annotations

"""
LLM-based visual veracity checker for a single image.

Asks a provider model (OpenAI, Google Gemini, DeepInfra, or OpenRouter) whether the image
appears AI-generated, manipulated, or contains elements that seem out of place.

Output format (strict JSON requested from the model):
  - ai_generated: boolean (true if likely AI-generated/manipulated or suspicious)
  - confidence: number in [0,1]
  - explanation: short textual rationale
  - anomalies: optional list of short strings describing issues

Usage:
    from scripts.llm_loader import LLMModelLoader
    from scripts.visual_veracity_checker import assess_image_visual_veracity

    loader = LLMModelLoader({"provider": "openai", "model": "gpt-4o"})
    res = assess_image_visual_veracity("path/to/image.jpg", loader)
    print(res)
"""

from typing import Any, Dict, Optional

from scripts.llm_loader import LLMModelLoader
from scripts.utils.json_utils import extract_json_object
from scripts.utils.media import image_to_data_url


_DEFAULT_SYSTEM_PROMPT = (
    "You are a careful image forensics assistant. Given a single image, "
    "assess whether it appears AI-generated or manipulated, and whether any "
    "visual elements seem out of place (e.g., inconsistent lighting, warped hands, "
    "nonsensical text, reflections, impossible shadows). Respond with strict JSON only."
)
def _build_user_instruction() -> str:
    return (
        "Task: Examine the image and decide whether it appears AI-generated or manipulated, "
        "or contains elements that seem out of place.\n\n"
        "Output JSON strictly with keys: \n"
        "  ai_generated: boolean (true if likely AI-generated/manipulated or suspicious)\n"
        "  confidence: number in [0,1]\n"
        "  explanation: short textual rationale\n"
        "  anomalies: optional array of short strings (e.g., ['extra finger', 'warped text'])\n"
        "Example: {\"ai_generated\": false, \"confidence\": 0.22, \"explanation\": \"...\", \"anomalies\": []}"
    )
def assess_image_visual_veracity(
    image_path: str,
    loader: LLMModelLoader,
    *,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask the selected LLM whether the image seems AI-generated or manipulated.

    Returns a dict with keys: ai_generated (bool), confidence (float 0..1), explanation (str),
    optionally anomalies (list[str]), and raw (original text response).
    """
    data_url = image_to_data_url(image_path)
    model = loader.get_model()

    sys_msg = system_prompt or _DEFAULT_SYSTEM_PROMPT
    user_text = _build_user_instruction()

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

    resp = model.invoke(messages)
    text = getattr(resp, "content", resp)
    raw_text = text if isinstance(text, str) else str(text)
    parsed = extract_json_object(
        raw_text,
        fallback=lambda payload: {
            "ai_generated": None,
            "confidence": None,
            "explanation": payload.strip(),
        },
    )
    if parsed is None:
        parsed = {}

    # Normalize fields
    ai_gen = parsed.get("ai_generated")
    if isinstance(ai_gen, str):
        v = ai_gen.strip().lower()
        ai_gen = True if v in ("true", "yes", "ai", "likely", "manipulated") else False if v in ("false", "no", "authentic", "genuine") else None
        parsed["ai_generated"] = ai_gen

    conf = parsed.get("confidence")
    try:
        if conf is not None:
            parsed["confidence"] = float(conf)
    except Exception:
        parsed["confidence"] = None

    expl = parsed.get("explanation")
    if not isinstance(expl, str):
        parsed["explanation"] = str(expl)

    # anomalies is optional; ensure list if present
    anomalies = parsed.get("anomalies")
    if anomalies is not None and not isinstance(anomalies, list):
        parsed["anomalies"] = [str(anomalies)]

    return {
        "ai_generated": parsed.get("ai_generated"),
        "confidence": parsed.get("confidence"),
        "explanation": parsed.get("explanation"),
        "anomalies": parsed.get("anomalies", []),
        "raw": raw_text,
    }


__all__ = ["assess_image_visual_veracity"]
