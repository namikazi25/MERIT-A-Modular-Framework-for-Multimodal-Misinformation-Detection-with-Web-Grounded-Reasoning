
"""Stage A (Gather): produce evidence bundles (Task 0.1).

Modes:
  frozen : backfill bundles from an existing run's JSONL (no new LLM/search calls).
  live   : run relevancy, visual veracity, question generation, cached web search,
           and answer extraction; write bundles (mirrors main.py's per-sample flow).

Bundle schema (v1):
  sample_id, image_path, claim, relevancy, visual_veracity,
  questions (list), questions_complete (bool),
  documents: {question: [{url, title, description}]},
  answers: [{question, answer, confidence, citations, selected}],
  upstream: {provider, model}, source, schema_version

Usage:
  python -m scripts.judge_study.gather frozen --jsonl results/X.jsonl --out evidence/<run_id>
  python -m scripts.judge_study.gather live --samples samples.jsonl --provider lmstudio \
      --model qwen/qwen3.6-35b-a3b --out evidence/<run_id> [--no-cache] [--search duckduckgo]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.getcwd())  # allow `python -m scripts.judge_study.gather` from repo root

# Load .env (mirrors main.py)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from scripts.llm_loader import LLMModelLoader, ModelConfig
from scripts.judge_study.manifest import make_manifest, write_manifest
from scripts.judge_study.search_cache import SearchCache

SCHEMA_VERSION = 1


def _payload_results(payload: dict) -> list:
    """Best-effort extraction of result list from a web_search payload."""
    if not isinstance(payload, dict):
        return []
    for key in ("results", "web_results", "organic"):
        if isinstance(payload.get(key), list):
            return payload[key]
    web = payload.get("web")
    if isinstance(web, dict) and isinstance(web.get("results"), list):
        return web["results"]
    return []


def bundle_from_frozen(obj: dict) -> dict:
    """Build an evidence bundle from one JSONL record of an existing run."""
    best = obj.get("best_qa_per_chain") or []
    answers = []
    documents: dict = {}
    for it in best:
        if not isinstance(it, dict):
            continue
        q = it.get("question")
        cits = it.get("citations") or []
        answers.append({
            "question": q,
            "answer": it.get("answer"),
            "confidence": it.get("confidence"),
            "citations": cits,
            "selected": True,
        })
        if q is not None:
            documents[str(q)] = [
                {"url": c.get("url") if isinstance(c, dict) else str(c),
                 "title": c.get("title") if isinstance(c, dict) else "",
                 "description": None}  # frozen runs did not persist extracted text
                for c in cits
            ]
    sample_id = str(obj.get("dataset_order_index") if obj.get("dataset_order_index") is not None
                     else obj.get("sample_index") or obj.get("image_path"))
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "image_path": obj.get("image_path"),
        "claim": obj.get("headline"),
        "relevancy": obj.get("relevancy") or {},
        "visual_veracity": obj.get("visual_veracity") or {},
        "questions": [a["question"] for a in answers if a.get("question") is not None],
        "questions_complete": False,  # old runs recorded only the selected per-chain Q/A
        "documents": documents,
        "answers": answers,
        "upstream": {"provider": obj.get("provider"), "model": obj.get("model")},
        "source": "frozen",
    }


def gather_frozen(jsonl_path: str, out_dir: str, run_id: str, repo_root: str, notes: str = "") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    bundles = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            bundles.append(bundle_from_frozen(json.loads(line)))
    for b in bundles:
        with open(os.path.join(out_dir, f"{b['sample_id']}.json"), "w") as fh:
            json.dump(b, fh, ensure_ascii=False, indent=1)
    manifest = make_manifest(
        phase="phase-0", stage="A", run_id=run_id,
        model={"provider": "frozen", "id": "n/a", "revision": "n/a", "quant": "n/a"},
        prompt_files=[], evidence_version=run_id, seed=None,
        sample_ids=[b["sample_id"] for b in bundles],
        temperature=0.0, repo_root=repo_root,
        notes=f"frozen backfill from {jsonl_path}. " + notes,
        extra={"mode": "frozen", "source_jsonl": jsonl_path, "schema_version": SCHEMA_VERSION},
    )
    write_manifest(manifest, os.path.join(out_dir, "manifest.json"))
    return manifest


def gather_live(samples_jsonl: str, out_dir: str, run_id: str, repo_root: str,
                provider: str, model: str, temperature: float,
                search_provider: str, cache_enabled: bool,
                q_chains: int, q_per_chain: int, answer_max_sources: int,
                notes: str = "") -> dict:
    from scripts.relevancy_checker import assess_image_headline_relevancy
    from scripts.visual_veracity_checker import assess_image_visual_veracity
    from scripts.question_generator import generate_investigative_questions
    from scripts.answer_generator import generate_answer_from_search
    from scripts.qa_selector import select_best_qa_and_propose_followups

    samples = [json.loads(l) for l in open(samples_jsonl) if l.strip()]
    manifest = make_manifest(
        phase="phase-0", stage="A", run_id=run_id,
        model={"provider": provider, "id": model, "revision": "n/a", "quant": "n/a"},
        prompt_files=[], evidence_version=run_id, seed=None,
        sample_ids=[str(s.get("sample_id", s.get("image_path"))) for s in samples],
        temperature=temperature, repo_root=repo_root,
        notes=f"live gather from {samples_jsonl}. search={search_provider} cache={'on' if cache_enabled else 'OFF'}. " + notes,
        extra={"mode": "live", "samples_jsonl": samples_jsonl, "schema_version": SCHEMA_VERSION,
               "q_chains": q_chains, "q_per_chain": q_per_chain, "answer_max_sources": answer_max_sources},
    )
    write_manifest(manifest, os.path.join(out_dir, "manifest.json"))

    loader = LLMModelLoader(ModelConfig(provider=provider, model=model or None, temperature=temperature))
    cache = SearchCache(enabled=cache_enabled, verbose=True)

    for si, s in enumerate(samples, start=1):
        img = str(s["image_path"])
        headline = str(s.get("headline", ""))
        sample_id = str(s.get("sample_id", img))
        print(f"[{si}/{len(samples)}] {sample_id}")

        rel = assess_image_headline_relevancy(img, headline, loader)
        ver = assess_image_visual_veracity(img, loader)

        all_questions: list = []
        documents: dict = {}
        answers: list = []
        prior_questions: list = []
        answers_global: dict = {}

        for chain_idx in range(1, q_chains + 1):
            qres = generate_investigative_questions(
                img, headline, loader, chains=1,
                questions_per_chain=q_per_chain,
                prior_questions=prior_questions,
                prior_answers=answers_global,
            )
            chain = (qres.get("chains", [[]]) or [[]])[0]
            prior_questions.extend(chain)
            all_questions.extend(chain)

            answers_by_q: dict = {}
            for q in chain:
                payload, meta = cache.search(q, provider=search_provider)
                documents[str(q)] = [
                    {"url": r.get("url"), "title": r.get("title"), "description": r.get("description") or r.get("snippet")}
                    for r in _payload_results(payload)
                ]
                ans = generate_answer_from_search(q, payload, loader, max_sources=answer_max_sources)
                ans["question"] = q
                ans["cache_hit"] = meta["cache_hit"]
                answers_by_q[q] = ans
                answers.append(ans)
            answers_global.update(answers_by_q)

            try:
                sel = select_best_qa_and_propose_followups(
                    img, headline, [chain], answers_by_q, loader, followups_per_chain=3,
                )
                best = (sel.get("selected", [{}]) or [{}])[0]
                fqs = (sel.get("followups", [[]]) or [[]])[0]
                if best and best.get("question") in answers_by_q:
                    answers_by_q[best["question"]]["selected"] = True
                prior_questions.extend(fqs or [])
            except Exception as e:
                print(f"  selection error: {e}")

        bundle = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "image_path": img,
            "claim": headline,
            "relevancy": rel,
            "visual_veracity": ver,
            "questions": all_questions,
            "questions_complete": True,
            "documents": documents,
            "answers": [a for a in answers if a.get("question") is not None],
            "upstream": {"provider": provider, "model": loader.config.model},
            "source": "live",
        }
        with open(os.path.join(out_dir, f"{sample_id}.json"), "w") as fh:
            json.dump(bundle, fh, ensure_ascii=False, indent=1)

    print(f"wrote {len(samples)} bundles to {out_dir}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A (Gather): evidence bundles")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("frozen")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--repo-root", default=os.getcwd())

    p = sub.add_parser("live")
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--repo-root", default=os.getcwd())
    p.add_argument("--provider", default=os.getenv("ALIGN_PROVIDER", "lmstudio"))
    p.add_argument("--model", default=os.getenv("ALIGN_MODEL", "qwen/qwen3.6-35b-a3b"))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--search", default=os.getenv("SEARCH_PROVIDER", "duckduckgo"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--q-chains", type=int, default=3)
    p.add_argument("--q-per-chain", type=int, default=3)
    p.add_argument("--answer-max-sources", type=int, default=5)

    args = ap.parse_args()
    if args.mode == "frozen":
        gather_frozen(args.jsonl, args.out, args.run_id, args.repo_root)
    else:
        gather_live(
            args.samples, args.out, args.run_id, args.repo_root,
            provider=args.provider, model=args.model, temperature=args.temperature,
            search_provider=args.search, cache_enabled=not args.no_cache,
            q_chains=args.q_chains, q_per_chain=args.q_per_chain,
            answer_max_sources=args.answer_max_sources,
        )


if __name__ == "__main__":
    main()
