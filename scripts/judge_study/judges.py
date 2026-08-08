"""Phase 2 judge implementations (J1-J5) on evidence bundles.

Common output schema: {label, confidence, rationale, abstain, extra}
label: "Misinformation" | "Not Misinformation" | None (J5 abstain or parse failure)
abstain: bool (only J5)
extra: per-judge (evidence_items, arguments, parse_ok, calls)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from scripts.ai_judge import judge_from_structured
from scripts.judge_study.judge_only import reconstruct_judge_input
from scripts.utils.json_utils import extract_json_object

POS = "Misinformation"
NEG = "Not Misinformation"
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prompts", "judge_study")


def load_prompt(name: str) -> str:
    with open(os.path.join(PROMPT_DIR, name)) as fh:
        return fh.read()


def truncate(text: Any, max_chars: int = 2000) -> str:
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= max_chars else s[:max_chars] + "...[truncated]"


def _norm_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    low = str(label).strip().lower()
    if "not" in low and "mis" in low:
        return NEG
    if low.startswith("mis"):
        return POS
    return None  # includes Uncertain / parse failures


def _parse_fail() -> dict:
    return {"label": None, "confidence": None, "rationale": "PARSE_FAILURE",
            "abstain": False, "extra": {"parse_ok": False, "calls": 1}}


def evidence_blocks(bundle: dict, max_chars: int = 2000) -> str:
    """Render retrieved evidence per question, truncated."""
    lines = []
    docs = bundle.get("documents") or {}
    for q, items in docs.items():
        lines.append(f"Q: {q}")
        if not items:
            lines.append("  (no documents retrieved)")
        for i, d in enumerate(items[:5], start=1):
            t = d.get("title") or ""
            desc = truncate(d.get("description"), max_chars)
            lines.append(f"  [{i}] {t} | {d.get('url')} | {desc}")
    return "\n".join(lines) if lines else "(no evidence)"


def module_block(bundle: dict) -> str:
    rel = bundle.get("relevancy") or {}
    vis = bundle.get("visual_veracity") or {}
    ans = bundle.get("answers") or []
    ans_lines = []
    for a in ans:
        if a.get("question") is None:
            continue
        conf = a.get("confidence")
        ans_lines.append(f"- Q: {a.get('question')}\n  A: {truncate(a.get('answer'), 600)}\n  conf: {conf}")
    return json.dumps({
        "relevancy": {"aligned": rel.get("aligned"), "confidence": rel.get("confidence"),
                      "explanation": truncate(rel.get("explanation"), 300)},
        "visual_veracity": {"ai_generated": vis.get("ai_generated"), "confidence": vis.get("confidence"),
                            "explanation": truncate(vis.get("explanation"), 300)},
        "answers": ans_lines,
    }, ensure_ascii=False, indent=1)


# ---------------- J1: current rule (unchanged logic) ----------------
def judge_j1(bundle: dict, loader, cfg: Optional[dict] = None) -> dict:
    final_obj = reconstruct_judge_input(bundle)
    j = judge_from_structured(final_obj, loader)
    label = _norm_label(j.get("label"))
    parse_ok = j.get("raw") is not None and label is not None
    return {"label": label, "confidence": j.get("confidence"),
            "rationale": j.get("rationale"), "abstain": False,
            "extra": {"parse_ok": parse_ok, "calls": 1, "raw": j.get("raw")}}


# ---------------- J2: evidence-grounded ----------------
def judge_j2(bundle: dict, loader, cfg: Optional[dict] = None) -> dict:
    sys_p = load_prompt("j2_evidence_grounded.txt")
    user = (f"CLAIM: {bundle.get('claim')}\n\n"
            f"MODULE SIGNALS:\n{module_block(bundle)}\n\n"
            f"RETRIEVED EVIDENCE:\n{evidence_blocks(bundle)}\n\n"
            "Return your verdict JSON.")
    resp = loader.get_model().invoke([{"role": "system", "content": sys_p},
                          {"role": "user", "content": user}])
    raw = getattr(resp, "content", resp)
    parsed = extract_json_object(str(raw), fallback=lambda p: {})
    label = _norm_label(parsed.get("label"))
    if label is None:
        return _parse_fail()
    conf = parsed.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        conf = None
    return {"label": label, "confidence": conf,
            "rationale": parsed.get("rationale"), "abstain": False,
            "extra": {"parse_ok": True, "calls": 1,
                      "evidence_items": parsed.get("evidence_items") or []}}


# ---------------- J3: structured debate (3 calls) ----------------
def judge_j3(bundle: dict, loader, cfg: Optional[dict] = None) -> dict:
    base = (f"CLAIM: {bundle.get('claim')}\n\n"
            f"MODULE SIGNALS:\n{module_block(bundle)}\n\n"
            f"RETRIEVED EVIDENCE:\n{evidence_blocks(bundle)}")
    # call 1: prosecution
    r1 = loader.get_model().invoke([{"role": "system", "content": load_prompt("j3_debate_fake.txt")},
                        {"role": "user", "content": base}])
    p1 = extract_json_object(str(getattr(r1, "content", r1)), fallback=lambda p: {})
    # call 2: defense
    r2 = loader.get_model().invoke([{"role": "system", "content": load_prompt("j3_debate_authentic.txt")},
                        {"role": "user", "content": base}])
    p2 = extract_json_object(str(getattr(r2, "content", r2)), fallback=lambda p: {})
    # call 3: adjudicate
    adj = (f"{base}\n\nPROSECUTION (fake) ARGUMENTS:\n{json.dumps(p1, ensure_ascii=False)[:4000]}\n\n"
           f"DEFENSE (authentic) ARGUMENTS:\n{json.dumps(p2, ensure_ascii=False)[:4000]}\n\n"
           "Adjudicate now.")
    r3 = loader.get_model().invoke([{"role": "system", "content": load_prompt("j3_adjudicate.txt")},
                        {"role": "user", "content": adj}])
    p3 = extract_json_object(str(getattr(r3, "content", r3)), fallback=lambda p: {})
    label = _norm_label(p3.get("label"))
    if label is None:
        return _parse_fail()
    try:
        conf = float(p3.get("confidence"))
    except Exception:
        conf = None
    return {"label": label, "confidence": conf, "rationale": p3.get("rationale"),
            "abstain": False,
            "extra": {"parse_ok": True, "calls": 3,
                      "prosecution": p1, "defense": p2}}


# ---------------- J5: abstaining judge (J2 setup + abstain) ----------------
def judge_j5(bundle: dict, loader, cfg: Optional[dict] = None) -> dict:
    sys_p = load_prompt("j5_abstaining.txt")
    user = (f"CLAIM: {bundle.get('claim')}\n\n"
            f"MODULE SIGNALS:\n{module_block(bundle)}\n\n"
            f"RETRIEVED EVIDENCE:\n{evidence_blocks(bundle)}\n\n"
            "Return your verdict JSON.")
    resp = loader.get_model().invoke([{"role": "system", "content": sys_p},
                          {"role": "user", "content": user}])
    parsed = extract_json_object(str(getattr(resp, "content", resp)), fallback=lambda p: {})
    abstain = bool(parsed.get("abstain")) or parsed.get("label") is None
    if abstain:
        return {"label": None, "confidence": 0.0,
                "rationale": parsed.get("rationale") or parsed.get("abstain_reason"),
                "abstain": True,
                "extra": {"parse_ok": True, "calls": 1,
                          "abstain_reason": parsed.get("abstain_reason"),
                          "evidence_items": parsed.get("evidence_items") or []}}
    label = _norm_label(parsed.get("label"))
    if label is None:
        return _parse_fail()
    try:
        conf = float(parsed.get("confidence"))
    except Exception:
        conf = None
    return {"label": label, "confidence": conf, "rationale": parsed.get("rationale"),
            "abstain": False,
            "extra": {"parse_ok": True, "calls": 1,
                      "evidence_items": parsed.get("evidence_items") or []}}


# ---------------- J2 variants (Task D) ----------------
def evidence_blocks_top3(bundle: dict, max_sources: int = 3, max_chars: int = 1200) -> str:
    """Evidence rendering: top-N sources per question, tighter truncation (~300 tokens)."""
    lines = []
    docs = bundle.get("documents") or {}
    for q, items in docs.items():
        lines.append(f"Q: {q}")
        if not items:
            lines.append("  (no documents retrieved)")
        for i, d in enumerate(items[:max_sources], start=1):
            lines.append(f"  [{i}] {d.get('title')} | {d.get('url')} | {truncate(d.get('description'), max_chars)}")
    return "\n".join(lines) if lines else "(no evidence)"


def _judge_j2_variant(bundle: dict, loader, prompt_name: str, renderer) -> dict:
    sys_p = load_prompt(prompt_name)
    user = (f"CLAIM: {bundle.get('claim')}\n\n"
            f"MODULE SIGNALS:\n{module_block(bundle)}\n\n"
            f"RETRIEVED EVIDENCE:\n{renderer(bundle)}\n\n"
            "Return your verdict JSON.")
    resp = loader.get_model().invoke([{"role": "system", "content": sys_p},
                                      {"role": "user", "content": user}])
    parsed = extract_json_object(str(getattr(resp, "content", resp)), fallback=lambda p: {})
    label = _norm_label(parsed.get("label"))
    if label is None:
        return _parse_fail()
    try:
        conf = float(parsed.get("confidence"))
    except Exception:
        conf = None
    return {"label": label, "confidence": conf, "rationale": parsed.get("rationale"),
            "abstain": False,
            "extra": {"parse_ok": True, "calls": 1,
                      "evidence_items": parsed.get("evidence_items") or []}}


def judge_j2a(bundle: dict, loader, cfg=None) -> dict:
    return _judge_j2_variant(bundle, loader, "j2a_top3_300.txt", evidence_blocks_top3)


def judge_j2b(bundle: dict, loader, cfg=None) -> dict:
    return _judge_j2_variant(bundle, loader, "j2b_citation_required.txt", evidence_blocks)


def judge_j2c(bundle: dict, loader, cfg=None) -> dict:
    return _judge_j2_variant(bundle, loader, "j2c_reasoning_first.txt", evidence_blocks)


JUDGES = {"J1": judge_j1, "J2": judge_j2, "J3": judge_j3, "J5": judge_j5,
          "J2A": judge_j2a, "J2B": judge_j2b, "J2C": judge_j2c}
PROMPT_FILES = {
    "J1": ["prompts/judge_study/j1_rule_prompt.txt"],
    "J2": ["prompts/judge_study/j2_evidence_grounded.txt"],
    "J3": ["prompts/judge_study/j3_debate_fake.txt", "prompts/judge_study/j3_debate_authentic.txt",
           "prompts/judge_study/j3_adjudicate.txt"],
    "J5": ["prompts/judge_study/j5_abstaining.txt"],
    "J2A": ["prompts/judge_study/j2a_top3_300.txt"],
    "J2B": ["prompts/judge_study/j2b_citation_required.txt"],
    "J2C": ["prompts/judge_study/j2c_reasoning_first.txt"],
}
