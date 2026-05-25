from __future__ import annotations

"""
LLM-based selection of best Q/A per chain and follow-up question proposal.

Inputs:
  - chains: list[list[str]] questions per chain
  - answers_by_question: mapping question -> answer dict from answer_generator

Outputs:
  - selected: list[dict] one per chain with keys {question, answer, confidence, citations}
  - followups: list[list[str]] follow-up questions per chain (default 3 each)
"""

import json
from typing import Any, Dict, List, Optional

from scripts.llm_loader import LLMModelLoader
from scripts.question_generator import generate_investigative_questions


_DEFAULT_SYSTEM_PROMPT = (
    "You are a verification strategist. Given multiple question+answer pairs for a chain, "
    "choose the single pair that most advances verification of the headline and image. "
    "Prefer specific, well-cited, high-confidence answers that reduce uncertainty. Respond with strict JSON only."
)


def _build_selection_user_text(headline: str, chain_items: List[Dict[str, Any]]) -> str:
    lines = [f"Headline: {headline}", "Candidates:"]
    for i, item in enumerate(chain_items, start=1):
        q = item.get("question", "")
        a = item.get("answer", "")
        conf = item.get("confidence")
        cits = item.get("citations", [])
        cit_str = "; ".join([str(c.get("url", "")) for c in cits if isinstance(c, dict)])
        lines.append(f"[{i}] Q: {q}\nA: {a}\nConfidence: {conf}\nSources: {cit_str}")
    lines.append(
        "\nInstructions: Return JSON with keys:\n"
        "  index: 1-based index of the best candidate\n"
        "  reason: short textual justification\n"
    )
    return "\n".join(lines)


def _parse_selection(text: str) -> Dict[str, Any]:
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
    return {"index": 1, "reason": text.strip()}


def select_best_qa_and_propose_followups(
    image_path: str,
    headline: str,
    chains: List[List[str]],
    answers_by_question: Dict[str, Dict[str, Any]],
    loader: LLMModelLoader,
    *,
    followups_per_chain: int = 3,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    model = loader.get_model()
    sys_msg = system_prompt or _DEFAULT_SYSTEM_PROMPT

    selected: List[Dict[str, Any]] = []
    followups: List[List[str]] = [[] for _ in chains]

    # Prepare prior list to avoid duplicates in follow-ups
    prior_questions: List[str] = [q for chain in chains for q in chain]

    for chain in chains:
        # Build items with available answers
        items: List[Dict[str, Any]] = []
        for q in chain:
            ans = answers_by_question.get(q) or {}
            items.append(
                {
                    "question": q,
                    "answer": ans.get("answer"),
                    "confidence": ans.get("confidence"),
                    "citations": ans.get("citations", []),
                }
            )

        # LLM selection, with confidence fallback
        user_text = _build_selection_user_text(headline, items)
        try:
            resp = model.invoke([
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_text},
            ])
            text = getattr(resp, "content", resp)
            parsed = _parse_selection(text if isinstance(text, str) else str(text))
            idx = int(parsed.get("index", 1))
        except Exception:
            # Fallback: pick highest confidence
            best_i, best_c = 0, -1.0
            for i, it in enumerate(items):
                try:
                    c = float(it.get("confidence") or 0.0)
                except Exception:
                    c = 0.0
                if c > best_c:
                    best_c, best_i = c, i
            idx = best_i + 1

        idx = max(1, min(idx, len(items)))
        pick = items[idx - 1] if items else {}
        selected.append(pick)

    generated_followups: List[List[str]] = []
    if followups_per_chain > 0 and chains:
        try:
            qres = generate_investigative_questions(
                image_path,
                headline,
                loader,
                chains=len(chains),
                questions_per_chain=followups_per_chain,
                prior_questions=prior_questions,
            )
            generated_followups = qres.get("chains") or []
        except Exception:
            generated_followups = []

    for chain_index in range(len(chains)):
        if chain_index < len(generated_followups):
            followups[chain_index] = generated_followups[chain_index] or []

    return {"selected": selected, "followups": followups}


__all__ = ["select_best_qa_and_propose_followups"]
