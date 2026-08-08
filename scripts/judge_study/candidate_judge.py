"""Task B.2: parameterized candidate judge — renders the pinned prompt + evidence
per the candidate config and calls the model.

Confidence elicitation:
  raw   -> model outputs numeric confidence
  verbal-> model outputs high/medium/low/none, mapped to 0.9/0.6/0.3/0.1
Abstain variants return abstain=True when the model says so.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from scripts.utils.json_utils import extract_json_object

VERBAL_MAP = {"high": 0.9, "medium": 0.6, "low": 0.3, "none": 0.1}
EVIDENCE_OPTS = {
    "none": (0, 0),
    "top3_300": (3, 1200),
    "top5_500": (5, 2000),
    "full": (99, 2000),
}


def truncate(text: Any, max_chars: int) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= max_chars else s[:max_chars] + "...[truncated]"


def render_evidence(bundle: dict, evidence: str) -> str:
    max_src, max_chars = EVIDENCE_OPTS.get(evidence, (5, 2000))
    docs = bundle.get("documents") or {}
    lines = []
    for q, items in docs.items():
        lines.append(f"Q: {q}")
        if not items:
            lines.append("  (no documents retrieved)")
        for i, d in enumerate(items[:max_src], start=1):
            lines.append(f"  [{i}] {d.get('title')} | {d.get('url')} | {truncate(d.get('description'), max_chars)}")
    return "\n".join(lines) if lines else "(no evidence)"


def render_modules(bundle: dict) -> str:
    rel = bundle.get("relevancy") or {}
    vis = bundle.get("visual_veracity") or {}
    ans = []
    for a in (bundle.get("answers") or []):
        if a.get("question") is None:
            continue
        ans.append(f"- Q: {a.get('question')}\n  A: {truncate(a.get('answer'), 600)}\n  conf: {a.get('confidence')}")
    return json.dumps({
        "relevancy": {"aligned": rel.get("aligned"), "confidence": rel.get("confidence"),
                      "explanation": truncate(rel.get("explanation"), 300)},
        "visual_veracity": {"ai_generated": vis.get("ai_generated"), "confidence": vis.get("confidence"),
                            "explanation": truncate(vis.get("explanation"), 300)},
        "answers": ans,
    }, ensure_ascii=False, indent=1)


def judge_candidate(bundle: dict, loader, cand: dict) -> dict:
    prompt = open(cand["prompt_path"]).read()
    ev = cand["evidence"]
    user = (f"CLAIM: {bundle.get('claim')}\n\n"
            f"MODULE SIGNALS:\n{render_modules(bundle)}")
    if ev != "none":
        user += f"\n\nRETRIEVED EVIDENCE:\n{render_evidence(bundle, ev)}"
    user += "\n\nReturn your verdict JSON."

    resp = loader.get_model().invoke([{"role": "system", "content": prompt},
                                      {"role": "user", "content": user}])
    raw = str(getattr(resp, "content", resp))
    parsed = extract_json_object(raw, fallback=lambda p: {})
    label = parsed.get("label")
    low = str(label).strip().lower() if label else ""
    if "not" in low and "mis" in low:
        label = "Not Misinformation"
    elif low.startswith("mis"):
        label = "Misinformation"
    else:
        label = None
    if label is None:
        return {"label": None, "confidence": None, "rationale": "PARSE_FAILURE",
                "abstain": False, "extra": {"parse_ok": False, "calls": 1}}

    abstain = bool(parsed.get("abstain")) and cand["abstain"]
    if abstain:
        return {"label": None, "confidence": 0.0, "rationale": parsed.get("rationale"),
                "abstain": True, "extra": {"parse_ok": True, "calls": 1,
                                           "abstain_reason": parsed.get("abstain_reason")}}

    if cand["confidence"] == "verbal":
        cl = str(parsed.get("confidence_label") or "").strip().lower()
        conf = VERBAL_MAP.get(cl, 0.5)
    else:
        try:
            conf = float(parsed.get("confidence"))
        except Exception:
            conf = None
    return {"label": label, "confidence": conf, "rationale": parsed.get("rationale"),
            "abstain": False, "extra": {"parse_ok": True, "calls": 1,
                                        "evidence_items": parsed.get("evidence_items") or []}}
