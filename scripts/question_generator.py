from __future__ import annotations

"""
Generate investigative, Google-like search queries given a headline and image.

The function queries the selected provider model (OpenAI, Google Gemini, DeepInfra,
or OpenRouter) to produce a small chain of question lists. Defaults to 3 chains with 3 questions
each. It keeps a running set of previously generated questions for a sample to
avoid duplicates across the chains and can optionally condition on previously
answered questions to encourage follow-up exploration.

Returns a dict with:
  - chains: list[list[str]] (questions per chain)
  - unique_questions: list[str] (flattened, de-duplicated)
  - raws: list[str] (raw LLM outputs per chain)
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from scripts.llm_loader import LLMModelLoader
from scripts.utils.json_utils import extract_json_array
from scripts.utils.media import image_to_data_url


_DEFAULT_SYSTEM_PROMPT = (
    "You are an investigative assistant skilled at verifying news claims. "
    "Given a headline and an accompanying image, generate concise, Google-style "
    "search queries that a fact-checker would use to assess misinformation. "
    "Prefer neutral, specific terms (entities, locations, dates). Respond with strict JSON only."
)

_MAX_CONTEXT_QA = 5
def _make_instruction(headline: str, k: int, prior: List[str], answered: List[Tuple[str, str]]) -> str:
    prior_block = "\n".join(f"- {q}" for q in prior) if prior else "(none)"
    answered_block = (
        "\n".join(f"- Q: {q}\n  A: {a}" for q, a in answered)
        if answered
        else "(none yet)"
    )
    return (
        "Task: Propose Google-style search queries to verify whether the headline is misinformation, "
        "taking the image into account.\n"
        f"Headline: {headline}\n\n"
        f"Previously generated (avoid duplicates):\n{prior_block}\n\n"
        f"Context from earlier Q/A (use to refine or deepen investigation):\n{answered_block}\n\n"
        f"Generate exactly {k} distinct queries, concise (max 12 words), \n"
        "avoiding punctuation unless needed. Prefer entity names, places, dates.\n\n"
        "Output JSON strictly as an array of strings.\n"
        "Example: [\"Event name location date verification\", \"Feature in image anomaly term\"]"
    )
def generate_investigative_questions(
    image_path: str,
    headline: str,
    loader: LLMModelLoader,
    *,
    chains: int = 3,
    questions_per_chain: int = 3,
    prior_questions: Optional[List[str]] = None,
    prior_answers: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    data_url = image_to_data_url(image_path)
    model = loader.get_model()

    sys_msg = system_prompt or _DEFAULT_SYSTEM_PROMPT
    prior: List[str] = list(prior_questions or [])
    seen: Set[str] = {q.strip().lower() for q in prior}

    answered_pairs: List[Tuple[str, str]] = []
    if prior_answers:
        items = list(prior_answers.items())[-_MAX_CONTEXT_QA:]
        for q, meta in items:
            question_text = str(q).strip()
            answer_text = ""
            if isinstance(meta, dict):
                answer_text = str(meta.get("answer") or meta.get("rationale") or meta.get("raw") or "").strip()
            else:
                answer_text = str(meta).strip()
            if not answer_text:
                continue
            # Collapse whitespace and trim to keep prompt concise
            answer_text = " ".join(answer_text.split())
            if len(answer_text) > 220:
                answer_text = answer_text[:217] + "..."
            answered_pairs.append((question_text, answer_text))

    out_chains: List[List[str]] = []
    raws: List[str] = []

    for _ in range(max(0, int(chains))):
        instr = _make_instruction(headline, int(questions_per_chain), prior, answered_pairs)
        messages = [
            {"role": "system", "content": sys_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instr},
                    {"type": "image_url", "image_url": data_url},
                ],
            },
        ]
        resp = model.invoke(messages)
        text = getattr(resp, "content", resp)
        raw_text = text if isinstance(text, str) else str(text)
        raws.append(raw_text)
        candidates = extract_json_array(
            raw_text,
            fallback=lambda payload: [
                stripped
                for stripped in (
                    ln.strip(" -•\t") for ln in payload.splitlines() if ln.strip()
                )
                if len(stripped) > 2
            ],
        ) or []

        chain_list: List[str] = []
        for q in candidates:
            qn = str(q).strip()
            key = qn.lower()
            if not qn or key in seen:
                continue
            chain_list.append(qn)
            seen.add(key)
            prior.append(qn)
            if len(chain_list) >= int(questions_per_chain):
                break

        out_chains.append(chain_list)

    unique_questions = list({q for chain in out_chains for q in chain})
    return {"chains": out_chains, "unique_questions": unique_questions, "raws": raws}


__all__ = ["generate_investigative_questions"]
