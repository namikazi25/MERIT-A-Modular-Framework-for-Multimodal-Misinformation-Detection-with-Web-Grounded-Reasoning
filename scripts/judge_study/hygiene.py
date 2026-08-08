"""Task C: retrieval-hygiene interventions, implemented as bundle transforms.

Each transform returns (new_bundle, stats) and is A/B-tested by hygiene_ab.py
(run J1 on dev-core with/without the intervention; adopt iff no significant
degradation vs the unmodified pipeline).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional, Tuple

BOILERPLATE_PATTERNS = [
    re.compile(r"cookie|privacy policy|accept all|manage preferences", re.I),
    re.compile(r"sign ?in|log ?in|subscribe|newsletter", re.I),
    re.compile(r"^\s*(home|about|contact|advertise|terms|faq)\s*$", re.I),
    re.compile(r"share on (facebook|twitter|linkedin)|follow us", re.I),
    re.compile(r"javascript|window\.|function\(|getElementById|addEventListener", re.I),
    re.compile(r"^\s*(load more|see more|related stories|you may also like)\s*$", re.I),
]


def dedup_documents(bundle: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """C.1: remove near-duplicate documents across questions (same URL or normalized text)."""
    docs = bundle.get("documents") or {}
    seen_text, seen_url = set(), set()
    removed = 0
    kept = 0
    new_docs = {}
    for q, items in docs.items():
        out = []
        for d in items:
            url = (d.get("url") or "").strip().rstrip("/")
            desc = re.sub(r"\s+", " ", (d.get("description") or "")).strip()[:500]
            key_text = desc.lower() if len(desc) > 40 else ""
            key_url = url.lower()
            dup = (key_url and key_url in seen_url) or (key_text and key_text in seen_text)
            if dup:
                removed += 1
                continue
            seen_url.add(key_url)
            if key_text:
                seen_text.add(key_text)
            out.append(d)
            kept += 1
        new_docs[q] = out
    nb = dict(bundle)
    nb["documents"] = new_docs
    total = removed + kept
    return nb, {"removed": removed, "kept": kept,
                "dedup_rate": round(removed / total, 4) if total else 0.0}


def strip_boilerplate(bundle: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """C.2: strip boilerplate lines from extracted text (nav, cookies, footers)."""
    docs = bundle.get("documents") or {}
    removed_chars = 0
    new_docs = {}
    for q, items in docs.items():
        out = []
        for d in items:
            desc = d.get("description") or ""
            lines = desc.split("\n")
            kept_lines = []
            for ln in lines:
                s = ln.strip()
                if not s:
                    continue
                if any(p.search(s) for p in BOILERPLATE_PATTERNS):
                    removed_chars += len(s)
                    continue
                kept_lines.append(ln)
            nd = dict(d)
            nd["description"] = "\n".join(kept_lines)
            out.append(nd)
        new_docs[q] = out
    nb = dict(bundle)
    nb["documents"] = new_docs
    return nb, {"removed_chars": removed_chars, "removed_tokens_est": removed_chars // 4}


def rewrite_queries(bundle: Dict[str, Any], cache, provider: str = "duckduckgo") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """C.3: claim-focused query rewriting. Rewritten queries go through the cache.
    Uses headline key-phrases instead of question-verbatim search."""
    claim = bundle.get("claim") or ""
    # extract key phrases: drop stopwords, keep top 8 tokens
    stop = set("the a an is are was were of in on at to for with and or not do does did how what when where why which who".split())
    toks = [t for t in re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", claim.lower()) if t not in stop]
    keyphrases = " ".join(list(dict.fromkeys(toks))[:8])  # ordered unique, capped
    docs = bundle.get("documents") or {}
    new_docs = {}
    hits = misses = 0
    for q, items in docs.items():
        if not items:
            new_docs[q] = []
            continue
        rewritten = f"{keyphrases} {q}" if keyphrases else q
        payload, meta = cache.search(rewritten, provider=provider)
        results = _payload_results(payload)
        new_docs[q] = [
            {"url": r.get("url"), "title": r.get("title"),
             "description": r.get("description") or r.get("snippet")}
            for r in results[:5]
        ]
        if meta.get("cache_hit"):
            hits += 1
        else:
            misses += 1
    nb = dict(bundle)
    nb["documents"] = new_docs
    nb["query_rewritten"] = True
    return nb, {"cache_hits": hits, "cache_misses": misses,
                "cache_hit_rate": round(hits / (hits + misses), 4) if (hits + misses) else 0.0}


def _payload_results(payload: dict) -> list:
    if not isinstance(payload, dict):
        return []
    for key in ("results", "web_results", "organic"):
        if isinstance(payload.get(key), list):
            return payload[key]
    web = payload.get("web")
    if isinstance(web, dict) and isinstance(web.get("results"), list):
        return web["results"]
    return []


def rerank_evidence(bundle: Dict[str, Any], loader) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """C.4: re-rank documents by claim relevance, scored by the local Qwen model."""
    claim = bundle.get("claim") or ""
    docs = bundle.get("documents") or {}
    new_docs = {}
    total_rank_changes = 0
    n_questions = 0
    for q, items in docs.items():
        if not items:
            new_docs[q] = []
            continue
        n_questions += 1
        prompt = (
            "Rate relevance 0-10 of each snippet to this claim. Claim: " + claim + "\n\n"
            + "\n".join(f"[{i}] {d.get('title')} | {(d.get('description') or '')[:300]}" for i, d in enumerate(items))
            + "\n\nReturn ONLY JSON: {\"scores\": [0-10,...]}"
        )
        try:
            resp = loader.get_model().invoke([
                {"role": "system", "content": "You are a relevance scorer. Be strict."},
                {"role": "user", "content": prompt},
            ])
            text = str(getattr(resp, "content", resp))
            m = re.search(r"\[([0-9.,\s]*)\]", text)
            scores = []
            if m:
                scores = [float(x) for x in m.group(1).split(",") if x.strip() != ""]
            if len(scores) != len(items):
                scores = list(range(len(items), 0, -1))  # fallback: keep order
            order = sorted(range(len(items)), key=lambda i: -scores[i])
            reranked = [items[i] for i in order]
            total_rank_changes += sum(1 for a, b in zip(range(len(order)), order) if a != b)
            new_docs[q] = reranked
        except Exception:
            new_docs[q] = items  # keep original order on failure
    nb = dict(bundle)
    nb["documents"] = new_docs
    nb["reranked"] = True
    return nb, {"rank_changes": total_rank_changes, "n_questions": n_questions,
                "avg_rank_changes": round(total_rank_changes / n_questions, 2) if n_questions else 0}
