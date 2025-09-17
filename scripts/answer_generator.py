from __future__ import annotations

"""
Answer generation from web search results using an LLM.

Given a question and a normalized web search payload (see scripts.search_provider),
summarize the likely answer using the descriptions/snippets and cite sources via
their URLs. If sources disagree, summarize differing viewpoints and cite each
source.

Returns a dict with:
  - answer: str
  - citations: list[{url, title}]
  - confidence: float|None
  - rationale: str
  - raw: str (raw LLM text)
  - search_provider: str|None
"""

import json
from typing import Any, Dict, List, Optional

from scripts.llm_loader import LLMModelLoader


_DEFAULT_SYSTEM_PROMPT = (
    "You are a careful fact-checking assistant. Using the provided web snippets, "
    "answer the user's question concisely and cite sources. If sources disagree, "
    "summarize the differing views and cite each. Respond with strict JSON only."
)


def _build_user_text(question: str, sources: List[Dict[str, str]]) -> str:
    lines = [f"Question: {question}", "\nSources:"]
    for i, s in enumerate(sources, start=1):
        title = s.get("title") or "(no title)"
        url = s.get("url") or ""
        desc = s.get("description") or ""
        lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {desc}")
    lines.append(
        "\nInstructions: Produce strict JSON with keys:\n"
        "  answer: short textual answer (2-5 sentences)\n"
        "  citations: array of objects {url, title} for the sources you used\n"
        "  confidence: number in [0,1]\n"
        "  rationale: one or two sentences on how you arrived at the answer\n"
    )
    return "\n".join(lines)


def _parse_json_response(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except Exception:
            pass
    return {"answer": text.strip(), "citations": []}


def _normalize_results(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    results = payload.get("results")
    if isinstance(results, list) and results:
        normalized: List[Dict[str, str]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            normalized.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": url,
                    "description": str(item.get("description") or "").strip(),
                }
            )
        if normalized:
            return normalized

    try:
        from scripts.brave_search import extract_web_results as _legacy_extract  # late import to avoid cycle

        legacy_results = _legacy_extract(payload)
    except Exception:
        legacy_results = []

    normalized: List[Dict[str, str]] = []
    for item in legacy_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        normalized.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "description": str(item.get("description") or "").strip(),
            }
        )
    return normalized


def generate_answer_from_search(
    question: str,
    search_payload: Dict[str, Any],
    loader: LLMModelLoader,
    *,
    max_sources: int = 5,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_results = _normalize_results(search_payload)
    sources: List[Dict[str, str]] = []
    for r in normalized_results[: max(1, int(max_sources))]:
        sources.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("description", ""),
            }
        )

    model = loader.get_model()
    sys_msg = system_prompt or _DEFAULT_SYSTEM_PROMPT
    user_text = _build_user_text(question, sources)

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_text},
    ]
    resp = model.invoke(messages)
    text = getattr(resp, "content", resp)
    parsed = _parse_json_response(text if isinstance(text, str) else str(text))

    # Normalize fields
    if not isinstance(parsed.get("answer"), str):
        parsed["answer"] = str(parsed.get("answer"))
    cits = parsed.get("citations")
    if not isinstance(cits, list):
        parsed["citations"] = []
    for i, c in enumerate(parsed.get("citations", [])):
        if not isinstance(c, dict):
            parsed["citations"][i] = {"url": str(c)}
        else:
            # ensure keys exist
            c.setdefault("url", "")
            c.setdefault("title", "")
    try:
        if parsed.get("confidence") is not None:
            parsed["confidence"] = float(parsed.get("confidence"))
    except Exception:
        parsed["confidence"] = None
    if not isinstance(parsed.get("rationale"), str):
        parsed["rationale"] = str(parsed.get("rationale"))

    return {
        "answer": parsed.get("answer"),
        "citations": parsed.get("citations", []),
        "confidence": parsed.get("confidence"),
        "rationale": parsed.get("rationale"),
        "raw": text,
        "sources_used": sources,
        "search_provider": search_payload.get("provider"),
    }


__all__ = ["generate_answer_from_search"]
