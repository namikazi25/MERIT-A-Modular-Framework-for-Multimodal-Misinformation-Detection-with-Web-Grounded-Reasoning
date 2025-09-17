#!/usr/bin/env python3
"""
Quick sanity run: load 10 samples using the MMFakeBench PyTorch dataset.

This script imports the dataset from `scripts.dataloader` and prints a few
fields to verify integration with the local data layout.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List
import argparse
from datetime import datetime

# Align checker integration
from scripts.relevancy_checker import assess_image_headline_relevancy
from scripts.visual_veracity_checker import assess_image_visual_veracity
from scripts.question_generator import generate_investigative_questions
from scripts.brave_search import brave_web_search
from scripts.answer_generator import generate_answer_from_search
from scripts.qa_selector import select_best_qa_and_propose_followups
from scripts.llm_loader import LLMModelLoader
from scripts.ai_judge import judge_from_structured

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()  # Loads variables from a local .env file if present
except Exception:
    pass


def _resolve_json_path() -> Path:
    # Prefer the source JSON if present; otherwise, fallback to the root JSON.
    p1 = Path("data/MMFakeBench_test/source/MMFakeBench_test.json")
    p2 = Path("data/MMFakeBench_test/MMFakeBench_test.json")
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError("Could not find MMFakeBench_test.json under data/MMFakeBench_test.")


def _print_sample(i: int, sample: Dict[str, Any]) -> None:
    text = sample.get("text")
    text_disp = (text[:60] + "...") if isinstance(text, str) and len(text) > 60 else text
    img = sample.get("image")
    if hasattr(img, "shape"):
        img_info = f"tensor{tuple(img.shape)}"
    elif hasattr(img, "size"):
        try:
            img_info = f"PIL{img.size}"
        except Exception:
            img_info = "image"
    else:
        img_info = "<none>"
    ip = Path(sample.get("image_path", "")).name
    gt = sample.get("gt_answers")
    fake = sample.get("fake_cls")
    print(f"[{i:02d}] img={img_info} file={ip} gt={gt} fake={fake} text={text_disp}")


def main() -> None:
    from scripts.dataloader import MMFakeBenchDataset, build_torch_dataloader

    parser = argparse.ArgumentParser(description="MMFakeBench quick sanity run + align check")
    parser.add_argument("--image", type=str, default=None, help="Path to an image for align check")
    parser.add_argument("--headline", type=str, default=None, help="Headline text for align check")
    # Provider is controlled solely via env var ALIGN_PROVIDER to keep a single switch.
    parser.add_argument(
        "--model", type=str, default=os.getenv("ALIGN_MODEL", None),
        help="LLM model name (default depends on provider; can also set ALIGN_MODEL)",
    )
    parser.add_argument(
        "--temperature", type=float, default=float(os.getenv("ALIGN_TEMPERATURE", "0.2")),
        help="LLM temperature for align check",
    )
    parser.add_argument(
        "--relevancy-limit",
        type=int,
        default=int(os.getenv("RELEVANCY_MAX_CHECKS", "1")),
        help=(
            "Deprecated: use --max-samples. Max number of dataset samples to pass through the relevancy checker when --image/--headline are not provided."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.getenv("PIPELINE_MAX_SAMPLES", os.getenv("RELEVANCY_MAX_CHECKS", "1"))),
        help=(
            "How many dataset samples to run through the full pipeline when --image/--headline are not provided. "
            "Controlled by PIPELINE_MAX_SAMPLES in .env."
        ),
    )
    parser.add_argument("--q-chains", type=int, default=int(os.getenv("Q_CHAINS", "3")), help="How many question chains to generate per sample (default 3)")
    parser.add_argument("--q-per-chain", type=int, default=int(os.getenv("Q_PER_CHAIN", "3")), help="How many questions per chain (default 3)")
    parser.add_argument("--answer-questions", action="store_true", default=os.getenv("ANSWER_ENABLE", "0") == "1", help="Use Brave Search + LLM to answer generated questions")
    parser.add_argument("--answer-max-sources", type=int, default=int(os.getenv("ANSWER_MAX_SOURCES", "5")), help="Max sources to pass to LLM per question")
    parser.add_argument("--emit-json", action="store_true", default=os.getenv("EMIT_FINAL_JSON", "1") == "1", help="Print final structured JSON per sample for downstream use")
    parser.add_argument("--judge", action="store_true", default=os.getenv("JUDGE_ENABLE", "1") == "1", help="Run AI judge on final structured output and print/append decision")
    parser.add_argument("--save-jsonl", type=str, default=os.getenv("PIPELINE_OUTPUT_JSONL", ""), help="If set, append each final structured object to this JSONL file")
    parser.add_argument("--html-report", type=str, default=os.getenv("PIPELINE_HTML_REPORT", ""), help="If set, write an HTML report for this run's outputs")
    parser.add_argument("--html-title", type=str, default=os.getenv("PIPELINE_HTML_TITLE", "Pipeline Results"), help="Title for the HTML report")
    parser.add_argument("--html-inline-images", action="store_true", default=os.getenv("PIPELINE_HTML_INLINE_IMAGES", "0") == "1", help="Embed images into HTML as base64 data URLs for portability")
    args = parser.parse_args()

    # Default results folder outputs when not explicitly provided
    if not args.save_jsonl:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.save_jsonl = f"results/run-{run_id}.jsonl"
    if not args.html_report:
        # Pair HTML path with the same run id as jsonl if possible
        try:
            base = os.path.splitext(os.path.basename(args.save_jsonl))[0]
            run_id = base.replace("run-", "")
            args.html_report = f"results/run-{run_id}.html"
        except Exception:
            args.html_report = "results/run.html"

    json_path = _resolve_json_path()
    image_root = Path("data/MMFakeBench_test")

    # Try to use torchvision transforms if available; otherwise fall back to PIL images.
    transform = None
    try:
        from torchvision import transforms as T  # type: ignore

        transform = T.ToTensor()
    except Exception:
        transform = None

    # Build dataset
    ds = MMFakeBenchDataset(
        json_path=str(json_path),
        image_root=str(image_root),
        balanced=False,
        seed=42,
        image_transform=None,
        skip_missing=True,
        verbose=True,
    )

    # If torch is present, exercise the DataLoader; else iterate directly.
    used_loader = False
    try:
        import torch  # noqa: F401

        # If transform is None (PIL), provide a safe collate that keeps images as lists.
        def safe_collate(batch: List[Dict[str, Any]]):
            out: Dict[str, List[Any]] = {}
            for item in batch:
                for k, v in item.items():
                    out.setdefault(k, []).append(v)
            return out

        collate_fn = None if transform is not None else safe_collate
        loader = build_torch_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False, collate_fn=collate_fn)

        printed = 0
        preview_cap = min(10, max(0, int(args.max_samples)))
        for batch in loader:
            # Iterate items within the batch to print up to 10 samples
            bs = len(batch["image"]) if "image" in batch else len(next(iter(batch.values())))
            for j in range(bs):
                sample = {k: (v[j] if isinstance(v, list) or hasattr(v, "__getitem__") else v) for k, v in batch.items()}
                _print_sample(printed, sample)
                printed += 1
                if printed >= preview_cap:
                    used_loader = True
                    break
            if printed >= preview_cap:
                break
    except Exception:
        used_loader = False

    if not used_loader:
        # Fallback: iterate the dataset directly
        for i in range(min(max(0, int(args.max_samples)), len(ds))):
            _print_sample(i, ds[i])

    # Relevancy check: use provided args if present; else take first N dataset samples
    pairs = []
    if args.image and args.headline:
        pairs = [(args.image, args.headline, None)]
    else:
        # Use --max-samples as the primary control, fallback to --relevancy-limit
        limit = max(0, int(args.max_samples if args.max_samples is not None else args.relevancy_limit))
        for i in range(min(limit, len(ds))):
            sample = ds[i]
            pairs.append((sample.get("image_path"), str(sample.get("text")), sample))

    if not pairs:
        print("\n--- Relevancy Check ---")
        print("No items to check (relevancy-limit is 0).")
        return

    print("\n--- Relevancy + Visual Veracity Checks ---")
    try:
        loader = LLMModelLoader({
            "provider": os.getenv("ALIGN_PROVIDER", "openai"),
            "model": args.model,  # None → provider default inside loader
            "temperature": args.temperature,
        })
    except Exception as e:
        print("Checker setup failed:", e)
        print("Hints: set OPENAI_API_KEY or GOOGLE_API_KEY or DEEPINFRA_API_KEY; set ALIGN_PROVIDER=openai|google|deepinfra; optionally pass --model/--image/--headline/--relevancy-limit.")
        return

    run_outputs = []
    for idx, (img_path, headline, sample_meta) in enumerate(pairs, start=1):
        print(f"\n[Sample {idx}/{len(pairs)}]")
        print(f"Using image: {img_path}")
        print(f"Headline: {headline[:120]}{'...' if len(headline) > 120 else ''}")

        # Snapshot usage before this sample
        usage_before = dict(getattr(loader, "usage_total", {"prompt": 0, "completion": 0, "total": 0}))

        # Relevancy check first
        rel = None
        try:
            rel = assess_image_headline_relevancy(str(img_path), headline, loader)
            print("Relevancy:")
            print(f"  aligned: {rel.get('aligned')}")
            print(f"  confidence: {rel.get('confidence')}")
            print(f"  explanation: {rel.get('explanation')}")
        except Exception as e:
            print("  Relevancy check error:", e)

        # Visual veracity check next
        ver = None
        try:
            ver = assess_image_visual_veracity(str(img_path), loader)
            print("Visual Veracity:")
            print(f"  ai_generated: {ver.get('ai_generated')}")
            print(f"  confidence: {ver.get('confidence')}")
            print(f"  explanation: {ver.get('explanation')}")
            anomalies = ver.get('anomalies') or []
            if anomalies:
                print(f"  anomalies: {', '.join(map(str, anomalies))}")
        except Exception as e:
            print("  Visual veracity check error:", e)

        # Investigative question generation (sequential chains)
        prior_questions: List[str] = []
        final_selected: List[Dict[str, Any]] = []
        question_chains: List[List[str]] = []
        answers_by_question_global: Dict[str, Dict[str, Any]] = {}

        for chain_idx in range(1, int(args.q_chains) + 1):
            # Generate one chain at a time, avoiding duplicates with prior_questions
            try:
                qres = generate_investigative_questions(
                    str(img_path),
                    headline,
                    loader,
                    chains=1,
                    questions_per_chain=args.q_per_chain,
                    prior_questions=prior_questions,
                )
                chain = (qres.get("chains", [[]]) or [[]])[0]
                print(f"Questions (Chain {chain_idx}):")
                for qi, q in enumerate(chain, start=1):
                    print(f"  {qi}. {q}")
            except Exception as e:
                print(f"  Question generation error (chain {chain_idx}):", e)
                chain = []

            prior_questions.extend(chain)
            question_chains.append(chain)

            # Optionally answer and select best for this chain
            if args.answer_questions and chain:
                answers_by_question: Dict[str, Dict[str, Any]] = {}
                print("  Answers:")
                for q in chain:
                    print(f"    Q: {q}")
                    try:
                        search_payload = brave_web_search(q)
                        ans = generate_answer_from_search(q, search_payload, loader, max_sources=args.answer_max_sources)
                        answers_by_question[q] = ans
                        print(f"      A: {ans.get('answer')}")
                        cits = ans.get("citations") or []
                        if cits:
                            print("      Sources:")
                            for c in cits:
                                url = c.get("url") if isinstance(c, dict) else str(c)
                                title = c.get("title") if isinstance(c, dict) else ""
                                print(f"        - {title} {url}")
                    except Exception as e:
                        print("      Answer error:", e)
                # merge per-chain answers into global mapping
                answers_by_question_global.update(answers_by_question)

                try:
                    sel = select_best_qa_and_propose_followups(
                        str(img_path),
                        headline,
                        [chain],
                        answers_by_question,
                        loader,
                        followups_per_chain=3,
                    )
                    best = (sel.get("selected", [{}]) or [{}])[0]
                    fqs = (sel.get("followups", [[]]) or [[]])[0]

                    if best:
                        final_selected.append(best)
                        print("  Best Q/A for this chain:")
                        print(f"    Q: {best.get('question')}")
                        print(f"    A: {best.get('answer')}")
                        print(f"    confidence: {best.get('confidence')}")
                        cits = best.get('citations') or []
                        if cits:
                            print("    Sources:")
                            for c in cits:
                                url = c.get("url") if isinstance(c, dict) else str(c)
                                title = c.get("title") if isinstance(c, dict) else ""
                                print(f"      - {title} {url}")
                    # Use follow-ups to enrich the prior pool (no printing)
                    prior_questions.extend(fqs or [])
                except Exception as e:
                    print("  Selection/follow-up error:", e)

        # Final aggregated best Q/A list across chains (for downstream use)
        best_qa_list = final_selected
        if best_qa_list:
            print("\n  Final Best Q/A list (one per chain):")
            for ci, it in enumerate(best_qa_list, start=1):
                print(f"    Chain {ci} best:")
                print(f"      Q: {it.get('question')}")
                print(f"      A: {it.get('answer')}")
                print(f"      confidence: {it.get('confidence')}")
                cits = it.get('citations') or []
                if cits:
                    print("      Sources:")
                    for c in cits:
                        url = c.get("url") if isinstance(c, dict) else str(c)
                        title = c.get("title") if isinstance(c, dict) else ""
                        print(f"        - {title} {url}")

        # Emit final structured JSON object per sample for downstream steps (slim schema)
        if args.emit_json:
            # keep only allowed fields in best_qa_per_chain
            best_slim = []
            for it in best_qa_list:
                if not isinstance(it, dict):
                    continue
                best_slim.append({
                    "question": it.get("question"),
                    "answer": it.get("answer"),
                    "confidence": it.get("confidence"),
                    "citations": it.get("citations") or [],
                })
            run_params = {
                "provider": os.getenv("ALIGN_PROVIDER", "openai"),
                "model": args.model,
                "temperature": float(args.temperature),
                "q_chains": int(args.q_chains),
                "q_per_chain": int(args.q_per_chain),
                "answer_questions": bool(args.answer_questions),
                "answer_max_sources": int(args.answer_max_sources),
            }

            sample_details = {}
            if isinstance(sample_meta, dict):
                sample_details = {
                    "text": sample_meta.get("text"),
                    "image_path": sample_meta.get("image_path"),
                    "text_source": sample_meta.get("text_source"),
                    "image_source": sample_meta.get("image_source"),
                    "gt_answers": sample_meta.get("gt_answers"),
                    "fake_cls": sample_meta.get("fake_cls"),
                }

            # Compute per-sample usage delta
            usage_after = getattr(loader, "usage_total", {"prompt": 0, "completion": 0, "total": 0})
            sample_usage = {
                "prompt": max(0, int(usage_after.get("prompt", 0)) - int(usage_before.get("prompt", 0))),
                "completion": max(0, int(usage_after.get("completion", 0)) - int(usage_before.get("completion", 0))),
                "total": max(0, int(usage_after.get("total", 0)) - int(usage_before.get("total", 0))),
            }

            final_obj = {
                "image_path": str(img_path),
                "headline": headline,
                "provider": run_params["provider"],
                "model": run_params["model"],
                "settings": {
                    "q_chains": run_params["q_chains"],
                    "q_per_chain": run_params["q_per_chain"],
                    "answer_questions": run_params["answer_questions"],
                    "answer_max_sources": run_params["answer_max_sources"],
                    "temperature": run_params["temperature"],
                },
                "run_params": run_params,
                "relevancy": rel or {},
                "visual_veracity": ver or {},
                "best_qa_per_chain": best_slim,
                "token_usage": sample_usage,
                "sample_details": sample_details,
            }
            # Run AI judge if requested
            judgement = None
            if args.judge:
                try:
                    judgement = judge_from_structured(final_obj, loader)
                    final_obj["judgement"] = judgement
                except Exception as e:
                    print("  AI judge error:", e)

            print("\n== Final Structured Output (JSON) ==")
            try:
                print(json.dumps(final_obj, ensure_ascii=False, indent=2))
            except Exception:
                # Fallback: best-effort string conversion
                def _stringify(o):
                    if isinstance(o, dict):
                        return {k: _stringify(v) for k, v in o.items()}
                    if isinstance(o, list):
                        return [_stringify(x) for x in o]
                    try:
                        json.dumps(o)
                        return o
                    except Exception:
                        return str(o)
                print(json.dumps(_stringify(final_obj), ensure_ascii=False, indent=2))

            # Also print judge summary for quick scan
            if args.judge and judgement:
                print("\n== AI Judge Decision ==")
                print(f"  label: {judgement.get('label')}")
                print(f"  confidence: {judgement.get('confidence')}")
                rat = judgement.get('rationale')
                if rat:
                    print(f"  rationale: {rat}")
                factors = judgement.get('key_factors') or []
                if factors:
                    print("  key_factors:")
                    for f in factors:
                        print(f"    - {f}")

            # Optionally append to a JSONL sink for later evaluation
            out_path = (args.save_jsonl or "").strip()
            if out_path:
                try:
                    out_dir = os.path.dirname(out_path)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(final_obj, ensure_ascii=False) + "\n")
                except Exception as e:
                    print("  Warning: failed to append to --save-jsonl:", e)

            # Collect for HTML per-run report
            run_outputs.append(final_obj)
            # Print token usage summary for this sample
            print("  Token usage (sample): prompt={} completion={} total={}".format(
                sample_usage.get("prompt", 0), sample_usage.get("completion", 0), sample_usage.get("total", 0)
            ))

    # Print total token usage across all processed samples
    try:
        grand = getattr(loader, "usage_total", {"prompt": 0, "completion": 0, "total": 0})
        print("\n=== Token Usage (Run Total) ===")
        print(f"prompt={grand.get('prompt', 0)} completion={grand.get('completion', 0)} total={grand.get('total', 0)}")
    except Exception:
        pass

    # Generate HTML report for this run if requested
    html_path = (args.html_report or "").strip()
    if html_path and run_outputs:
        try:
            from scripts.report_html import render_html_report
            # Try to compute metrics over the full JSONL if provided and dataset exists
            metrics = None
            ds_json = _resolve_json_path()
            if args.save_jsonl:
                try:
                    from scripts.evaluate import evaluate as _eval
                    metrics = _eval(Path(args.save_jsonl), ds_json, Path("data/MMFakeBench_test"), save_report=None)
                except Exception:
                    metrics = None
            render_html_report(run_outputs, metrics, html_path, title=args.html_title, inline_images=bool(args.html_inline_images))
            base, _ = os.path.splitext(html_path)
            csv_path = base + ".tokens.csv"
            print(f"\nWrote HTML report to {html_path}")
            if os.path.exists(csv_path):
                print(f"Token usage CSV: {csv_path}")
        except Exception as e:
            print("\nWarning: failed to write HTML report:", e)


if __name__ == "__main__":
    main()
