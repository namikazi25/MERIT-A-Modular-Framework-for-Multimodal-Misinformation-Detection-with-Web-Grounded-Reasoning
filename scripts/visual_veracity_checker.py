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
    "You are an AI image detection expert. Analyze images for signs of AI generation or manipulation.\n"
    "Respond with strict JSON only."
)

def _build_user_instruction() -> str:
    return (
        "Task: Examine this image for signs of AI generation or manipulation.\n\n"
        
        "Detection criteria:\n\n"
        
        "Technical artifacts:\n"
        "- Warped hands, extra fingers, impossible anatomy\n"
        "- Nonsensical text, garbled signs\n"
        "- Inconsistent lighting, impossible shadows\n"
        "- Unnatural textures, blending errors\n\n"
        
        "Contextual anomalies:\n"
        "- Surreal object combinations (e.g., clown in bathroom mirror)\n"
        "- Impossible scenarios or physics violations\n"
        "- Dreamlike or fantastical elements in otherwise normal scenes\n"
        "- Objects that don't belong in the context\n\n"
        
        "Be suspicious when:\n"
        "- Technically perfect BUT contextually bizarre\n"
        "- Minor details don't make sense\n"
        "- Scene feels 'off' even without obvious artifacts\n\n"
        
        "Confidence calibration:\n"
        "- 0.8-1.0: Clear technical artifacts OR impossible context\n"
        "- 0.6-0.8: Strong evidence, multiple anomalies\n"
        "- 0.4-0.6: Moderate suspicion, some anomalies\n"
        "- 0.2-0.4: Minor concerns, could be unusual real photo\n"
        "- 0.0-0.2: Appears genuine\n\n"
        
        "Prioritize contextual impossibility as much as technical quality.\n\n"
        
        "Output JSON:\n"
        "{\n"
        "  \"ai_generated\": boolean,\n"
        "  \"confidence\": 0.0-1.0,\n"
        "  \"explanation\": \"What you observed\",\n"
        "  \"anomalies\": [\"specific issues\"]\n"
        "}"
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
