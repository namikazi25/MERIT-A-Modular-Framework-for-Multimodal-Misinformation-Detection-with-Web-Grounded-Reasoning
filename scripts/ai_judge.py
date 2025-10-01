from __future__ import annotations

"""
AI Judge: Decide if the headline+image pair constitutes misinformation.

Input is the Final Structured Output produced by main.py, e.g.:
{
  "image_path": str,
  "headline": str,
  "relevancy": {"aligned": bool|"partial"|None, "confidence": float|None, "explanation": str},
  "visual_veracity": {"ai_generated": bool|None, "confidence": float|None, "explanation": str, "anomalies": []},
  "best_qa_per_chain": [ {"question": str, "answer": str, "confidence": float|None, "citations": [...]}, ... ]
  ...
}

This module exposes `judge_from_structured(final_obj, loader)` that returns a dict:
{
  "label": "Misinformation" | "Not Misinformation" | "Uncertain",
  "confidence": float in [0,1],
  "logprob_confidence": float in [0,1] | None,
  "rationale": str,
  "key_factors": [str, ...],
  "raw": str | None,  # raw LLM response if any
  "logprob_stats": Dict[str, Any] | None,
}

It attempts an LLM judgment first (OpenAI/Google/DeepInfra/OpenRouter via scripts.llm_loader). If that
fails (e.g., missing API keys), it falls back to a deterministic heuristic using
relevancy + visual veracity signals.
"""

import json
from typing import Any, Dict, List, Optional

from scripts.llm_loader import LLMModelLoader
from scripts.utils.json_utils import extract_json_object


_SYSTEM_PROMPT = (
    "You are a misinformation detector evaluating image-headline pairings.\n\n"
    
    "Decision rules:\n\n"
    
    "NOT MISINFORMATION when ALL true:\n"
    "- Headline is factually accurate (verified by Q/A)\n"
    "- Image reasonably relates to headline (aligned=true OR partial)\n"
    "- Image is genuine (ai_generated=false)\n\n"
    
    "Partial alignment is normal for news photos. Context images are acceptable.\n\n"
    
    "MISINFORMATION when ANY true:\n"
    "- Headline makes verifiably false claims (Q/A contradicts)\n"
    "- Image shows wrong subject/event (aligned=false)\n"
    "- Image is AI-generated (ai_generated=true)\n\n"
    
    "Edge cases:\n"
    "- Partial + true headline + real image = Not Misinformation\n"
    "- Partial + false headline = Misinformation\n"
    "- False alignment (regardless of other factors) = Misinformation\n\n"
    
    "Component reliability note:\n"
    "Visual veracity and relevancy checkers may occasionally err.\n"
    "When components disagree, weigh the strength of each signal:\n"
    "- Strong signals: False headline (high confidence Q/A), completely wrong image\n"
    "- Weak signals: Low confidence assessments, borderline cases\n\n"
    
    "Return JSON:\n"
    "{\n"
    "  \"label\": \"Misinformation\" | \"Not Misinformation\",\n"
    "  \"confidence\": 0.0-1.0,\n"
    "  \"rationale\": \"Brief explanation\",\n"
    "  \"key_factors\": [\"factor1\", \"factor2\"]\n"
    "}"
)


def _build_user_message(final_obj: Dict[str, Any]) -> str:
    # Keep payload compact but informative
    compact = {
        "headline": final_obj.get("headline"),
        "image_path": final_obj.get("image_path"),
        "relevancy": final_obj.get("relevancy", {}),
        "visual_veracity": final_obj.get("visual_veracity", {}),
        # include only minimal Q/A details to keep prompt small
        "best_qa_per_chain": [
            {
                "question": it.get("question"),
                "answer": it.get("answer"),
                "confidence": it.get("confidence"),
                "citations_count": len(it.get("citations") or []),
            }
            for it in (final_obj.get("best_qa_per_chain") or [])
            if isinstance(it, dict)
        ],
    }
    return (
        "You are given the following analysis JSON for a headline+image pair.\n"
        "Make a final misinformation judgment.\n\n"
        f"Analysis JSON (compact):\n{json.dumps(compact, ensure_ascii=False)}\n\n"
        "Respond ONLY with JSON: {\"label\":..., \"confidence\":..., \"rationale\":..., \"key_factors\":[...]}"
    )


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _heuristic_judge(final_obj: Dict[str, Any]) -> Dict[str, Any]:
    rel = final_obj.get("relevancy", {}) or {}
    ver = final_obj.get("visual_veracity", {}) or {}

    rel_aligned = rel.get("aligned")
    rel_conf = _safe_float(rel.get("confidence")) or 0.0
    ver_ai = ver.get("ai_generated")
    ver_conf = _safe_float(ver.get("confidence")) or 0.0

    # Map to scores: positive => misinformation, negative => not
    rel_score = 0.0
    if rel_aligned is True:
        rel_score = -rel_conf
    elif rel_aligned is False:
        rel_score = +rel_conf
    elif isinstance(rel_aligned, str) and str(rel_aligned).lower() == "partial":
        rel_score = -0.4 * rel_conf

    ver_score = 0.0
    if ver_ai is True:
        ver_score = +0.5 * ver_conf  # risk signal; weaker than direct misalignment
    elif ver_ai is False:
        ver_score = -0.25 * ver_conf

    score = 0.7 * rel_score + 0.3 * ver_score

    if score >= 0.3:
        label = "Misinformation"
    elif score <= -0.3:
        label = "Not Misinformation"
    else:
        label = "Uncertain"

    # Confidence: map |score| to [0.3, 0.9]
    base = 0.3 + min(0.6, abs(score))
    confidence = round(min(1.0, max(0.0, base)), 2)

    key_factors: List[str] = []
    if rel_aligned is True:
        key_factors.append(f"Relevancy aligned (conf ~{rel_conf:.2f})")
    elif rel_aligned is False:
        key_factors.append(f"Relevancy misaligned (conf ~{rel_conf:.2f})")
    elif isinstance(rel_aligned, str) and str(rel_aligned).lower() == "partial":
        key_factors.append(f"Relevancy partial alignment (conf ~{rel_conf:.2f})")
    if ver_ai is True:
        key_factors.append(f"Image likely AI/manipulated (conf ~{ver_conf:.2f})")
    elif ver_ai is False:
        key_factors.append(f"Image likely authentic (conf ~{ver_conf:.2f})")

    rationale = (
        "Heuristic decision combining relevancy and visual veracity signals. "
        "High misalignment pushes toward misinformation; partial alignment tilts toward the claim when other evidence agrees; AI/manipulation raises risk."
    )

    return {
        "label": label,
        "confidence": confidence,
        "logprob_confidence": None,
        "rationale": rationale,
        "key_factors": key_factors,
        "raw": None,
        "logprob_stats": None,
    }


def judge_from_structured(final_obj: Dict[str, Any], loader: Optional[LLMModelLoader]) -> Dict[str, Any]:
    """
    Make a misinformation judgment from the final structured output.

    Attempts LLM-based decision first; if that fails or no loader is provided,
    falls back to a deterministic heuristic.
    """
    if loader is not None:
        try:
            model = loader.get_model()
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(final_obj)},
            ]
            resp = model.invoke(messages)
            text = getattr(resp, "content", resp)
            raw_text = text if isinstance(text, str) else str(text)
            parsed = extract_json_object(
                raw_text,
                fallback=lambda payload: {
                    "label": "Uncertain",
                    "confidence": 0.5,
                    "rationale": payload.strip(),
                    "key_factors": [],
                },
            ) or {}
            logprob_summary = getattr(resp, "logprobs", None)
            logprob_confidence: Optional[float] = None
            if isinstance(logprob_summary, dict):
                geo_prob = logprob_summary.get("geo_mean_prob")
                try:
                    if geo_prob is not None:
                        logprob_confidence = max(0.0, min(1.0, float(geo_prob)))
                except Exception:
                    logprob_confidence = None

            label = str(parsed.get("label", "Uncertain")).strip()
            # Normalize label
            low = label.lower()
            if "not" in low and "mis" in low:
                norm = "Not Misinformation"
            elif low.startswith("mis"):
                norm = "Misinformation"
            elif low.startswith("uncertain") or low.startswith("unknown"):
                norm = "Uncertain"
            else:
                norm = "Uncertain"

            conf = parsed.get("confidence")
            try:
                confidence = float(conf)
            except Exception:
                confidence = 0.5

            rationale = parsed.get("rationale")
            if not isinstance(rationale, str):
                rationale = str(rationale)

            key_factors = parsed.get("key_factors")
            if not isinstance(key_factors, list):
                key_factors = [str(key_factors)] if key_factors is not None else []

            return {
                "label": norm,
                "confidence": max(0.0, min(1.0, confidence)),
                "logprob_confidence": logprob_confidence,
                "rationale": rationale,
                "key_factors": key_factors,
                "raw": raw_text,
                "logprob_stats": logprob_summary if isinstance(logprob_summary, dict) else None,
            }
        except Exception:
            # Fall back to heuristic if LLM call fails
            pass

    return _heuristic_judge(final_obj)


__all__ = ["judge_from_structured"]
