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

from typing import Any, Dict, List, Optional

from scripts.llm_loader import LLMModelLoader
from scripts.utils.json_utils import extract_json_object
from scripts.utils.search import normalize_search_payload


_DEFAULT_SYSTEM_PROMPT = (
    "You are a careful fact-checking assistant. Using the provided web snippets, "
    "answer the user's question concisely and cite sources. If sources disagree, "
    "summarize the differing views and cite each. Respond with strict JSON only."
)


def _build_user_text(question: str, sources: List[Dict[str, str]]) -> str:
    lines = [f"Question: {question}", "\nSources:"]
    for i, source in enumerate(sources, start=1):
        title = source.get("title") or "(no title)"
        url = source.get("url") or ""
        desc = source.get("description") or ""
        lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {desc}")
    lines.append(
        "\nInstructions: Produce strict JSON with keys:\n"
        "  answer: short textual answer (2-5 sentences)\n"
        "  citations: array of objects {url, title} for the sources you used\n"
        "  confidence: number in [0,1]\n"
        "  rationale: one or two sentences on how you arrived at the answer\n"
    )
    return "\n".join(lines)


def generate_answer_from_search(
    question: str,
    search_payload: Dict[str, Any],
    loader: LLMModelLoader,
    *,
    max_sources: int = 5,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        from scripts.brave_search import extract_web_results as _legacy_extract  # late import to avoid cycle
    except Exception:
        _legacy_extract = None

    normalized_results = normalize_search_payload(
        search_payload,
        legacy_extractor=_legacy_extract,
    )

    sources: List[Dict[str, str]] = []
    for result in normalized_results[: max(1, int(max_sources))]:
        sources.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("description", ""),
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
    raw_text = text if isinstance(text, str) else str(text)
    parsed = extract_json_object(
        raw_text,
        fallback=lambda payload: {"answer": payload.strip(), "citations": []},
    ) or {}

    if not isinstance(parsed.get("answer"), str):
        parsed["answer"] = str(parsed.get("answer"))

    citations = parsed.get("citations")
    if not isinstance(citations, list):
        parsed["citations"] = []
    for index, citation in enumerate(parsed.get("citations", [])):
        if not isinstance(citation, dict):
            parsed["citations"][index] = {"url": str(citation)}
        else:
            citation.setdefault("url", "")
            citation.setdefault("title", "")

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
        "raw": raw_text,
        "sources_used": sources,
        "search_provider": search_payload.get("provider"),
    }


__all__ = ["generate_answer_from_search"]
