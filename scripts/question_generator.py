from __future__ import annotations

"""
Generate investigative, Google-like search queries given a headline and image.

The function queries the selected provider model (OpenAI GPT-4o or Google Gemini)
to produce a small chain of question lists. Defaults to 3 chains with 3 questions
each. It keeps a running set of previously generated questions for a sample to
avoid duplicates across the chains.

Returns a dict with:
  - chains: list[list[str]] (questions per chain)
  - unique_questions: list[str] (flattened, de-duplicated)
  - raws: list[str] (raw LLM outputs per chain)
"""

import base64
import json
import os
from typing import Any, Dict, List, Optional, Set

from scripts.llm_loader import LLMModelLoader


_DEFAULT_SYSTEM_PROMPT = (
    "You are an investigative assistant skilled at verifying news claims. "
    "Given a headline and an accompanying image, generate concise, Google-style "
    "search queries that a fact-checker would use to assess misinformation. "
    "Prefer neutral, specific terms (entities, locations, dates). Respond with strict JSON only."
)


def _guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext in (".png",):
        return "image/png"
    if ext in (".webp",):
        return "image/webp"
    if ext in (".gif",):
        return "image/gif"
    if ext in (".bmp",):
        return "image/bmp"
    return "image/png"


def _image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    mime = _guess_mime(image_path)
    return f"data:{mime};base64,{b64}"


def _make_instruction(headline: str, k: int, prior: List[str]) -> str:
    prior_block = "\n".join(f"- {q}" for q in prior) if prior else "(none)"
    return (
        "Task: Propose Google-style search queries to verify whether the headline is misinformation, "
        "taking the image into account.\n"
        f"Headline: {headline}\n\n"
        f"Previously generated (avoid duplicates):\n{prior_block}\n\n"
        f"Generate exactly {k} distinct queries, concise (max 12 words), \n"
        "avoiding punctuation unless needed. Prefer entity names, places, dates.\n\n"
        "Output JSON strictly as an array of strings.\n"
        "Example: [\"Event name location date verification\", \"Feature in image anomaly term\"]"
    )


def _parse_json_array(text: str) -> List[str]:
    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if isinstance(x, (str, int, float)) or x]
    except Exception:
        pass
    # Heuristic extract
    s, e = text.find("["), text.rfind("]")
    if s != -1 and e != -1 and e > s:
        snippet = text[s : e + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if isinstance(x, (str, int, float)) or x]
        except Exception:
            pass
    # Fallback: split lines
    lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    return [ln for ln in lines if len(ln) > 2]


def generate_investigative_questions(
    image_path: str,
    headline: str,
    loader: LLMModelLoader,
    *,
    chains: int = 3,
    questions_per_chain: int = 3,
    prior_questions: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    data_url = _image_to_data_url(image_path)
    model = loader.get_model()

    sys_msg = system_prompt or _DEFAULT_SYSTEM_PROMPT
    prior: List[str] = list(prior_questions or [])
    seen: Set[str] = {q.strip().lower() for q in prior}

    out_chains: List[List[str]] = []
    raws: List[str] = []

    for _ in range(max(0, int(chains))):
        instr = _make_instruction(headline, int(questions_per_chain), prior)
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
        raws.append(text if isinstance(text, str) else str(text))
        candidates = _parse_json_array(raws[-1])

        chain_list: List[str] = []
        for q in candidates:
            qn = q.strip()
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

