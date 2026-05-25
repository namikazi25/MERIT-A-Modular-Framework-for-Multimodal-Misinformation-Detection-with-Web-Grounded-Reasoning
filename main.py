#!/usr/bin/env python3
"""
Quick sanity run: load 10 samples using the MMFakeBench PyTorch dataset.

This script imports the dataset from `scripts.dataloader` and prints a few
fields to verify integration with the local data layout.
"""

from __future__ import annotations

import concurrent.futures
import os
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
from datetime import datetime, timezone

# Align checker integration
from scripts.relevancy_checker import assess_image_headline_relevancy
from scripts.visual_veracity_checker import assess_image_visual_veracity
from scripts.question_generator import generate_investigative_questions
from scripts.answer_generator import generate_answer_from_search
from scripts.qa_selector import select_best_qa_and_propose_followups
from scripts.llm_loader import LLMModelLoader
from scripts.ai_judge import judge_from_structured
from scripts.search_provider import get_active_search_provider, web_search
from scripts.checkpoint import CheckpointManager, CheckpointState, build_args_snapshot

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()  # Loads variables from a local .env file if present
except Exception:
    pass


def _resolve_json_path(dataset_root: Path, dataset_json: Optional[str] = None) -> Path:
    """Resolve the dataset JSON under ``dataset_root``."""

    dataset_root = dataset_root.expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    candidates: List[Path] = []

    if dataset_json:
        override = Path(dataset_json).expanduser()
        if not override.is_absolute():
            override = dataset_root / override
        candidates.append(override)

    dataset_name = dataset_root.name
    candidates.append(dataset_root / "source" / f"{dataset_name}.json")
    candidates.append(dataset_root / f"{dataset_name}.json")

    if not dataset_json:
        for parent in (dataset_root / "source", dataset_root):
            if parent.exists():
                for extra in sorted(parent.glob("*.json")):
                    if extra not in candidates:
                        candidates.append(extra)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates) or "(no candidates)"
    raise FileNotFoundError(
        f"Could not find dataset JSON under {dataset_root}. Checked: {searched}"
    )


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
    print(f"[{i:02d}] img={img_info} file={ip} text={text_disp}")


RUN_METADATA_ENV_KEYS = [
    "ALIGN_PROVIDER",
    "ALIGN_MODEL",
    "ALIGN_TEMPERATURE",
    "PIPELINE_MAX_SAMPLES",
    "PIPELINE_HTML_REPORT",
    "PIPELINE_HTML_TITLE",
    "PIPELINE_HTML_INLINE_IMAGES",
    "PIPELINE_CHECKPOINT_DIR",
    "PIPELINE_CHECKPOINT_SIZE",
    "PIPELINE_RESUME",
    "PIPELINE_OUTPUT_JSONL",
    "PIPELINE_DISABLE_RELEVANCY",
    "PIPELINE_DISABLE_VISUAL",
    "PIPELINE_DISABLE_QUESTIONS",
    "PIPELINE_DISABLE_JUDGE",
    "SEARCH_PROVIDER",
    "ANSWER_ENABLE",
    "ANSWER_MAX_SOURCES",
    "Q_CHAINS",
    "Q_PER_CHAIN",
    "PIPELINE_DATASET_ROOT",
    "PIPELINE_DATASET_JSON",
    "PIPELINE_WORKERS",
]


def _collect_env_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for key in RUN_METADATA_ENV_KEYS:
        val = os.getenv(key)
        if val is not None:
            snap[key] = val
    return snap


def _collect_git_metadata() -> Dict[str, Any]:
    info: Dict[str, Any] = {"commit": "unknown", "dirty": None}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        if commit:
            info["commit"] = commit
    except Exception:
        pass
    try:
        status = subprocess.check_output(["git", "status", "--short"], stderr=subprocess.DEVNULL, text=True)
        info["dirty"] = bool(status.strip())
    except Exception:
        info.setdefault("dirty", None)
    return info


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_existing_outputs(jsonl_path: str) -> List[Dict[str, Any]]:
    path = Path(jsonl_path)
    if not path.exists():
        return []

    outputs: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    outputs.append(obj)
    except Exception:
        return []
    return outputs


def _process_one_sample(
    sample_meta: Dict[str, Any],
    order_idx: int,
    total_samples: int,
    loader: "LLMModelLoader",
    args: "argparse.Namespace",
    run_relevancy: bool,
    run_visual: bool,
    run_questions: bool,
    run_judge: bool,
    answer_questions_enabled: bool,
    search_provider: str,
    module_config: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Process one sample through the full pipeline.

    Returns ``(final_obj_or_none, ddg_metrics)``.  The caller (main thread) is
    responsible for JSONL writing, checkpointing, and appending to run_outputs.
    """
    img_path = sample_meta.get("image_path")
    headline = sample_meta.get("headline", "")
    img_path = str(img_path) if img_path is not None else ""
    headline = str(headline)

    human_index = order_idx + 1
    print(f"\n[Sample {human_index}/{total_samples}]")
    print(f"Using image: {img_path}")
    print(f"Headline: {headline[:120]}{'...' if len(headline) > 120 else ''}")

    # Snapshot usage before this sample
    usage_before = dict(getattr(loader, "usage_total", {"prompt": 0, "completion": 0, "total": 0}))

    # Steps 1 & 2: run relevancy and visual veracity concurrently (they are independent)
    fut_rel = None
    fut_ver = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _step12_pool:
        if run_relevancy:
            fut_rel = _step12_pool.submit(assess_image_headline_relevancy, str(img_path), headline, loader)
        if run_visual:
            fut_ver = _step12_pool.submit(assess_image_visual_veracity, str(img_path), loader)
    # Both futures complete when the executor context exits

    rel: Optional[Dict[str, Any]] = None
    if run_relevancy:
        if fut_rel is not None:
            try:
                rel = fut_rel.result()
                print("Relevancy:")
                print(f"  aligned: {rel.get('aligned')}")
                print(f"  confidence: {rel.get('confidence')}")
                print(f"  explanation: {rel.get('explanation')}")
            except Exception as e:
                print("  Relevancy check error:", e)
    else:
        print("Relevancy: skipped (--disable-relevancy)")

    ver: Optional[Dict[str, Any]] = None
    if run_visual:
        if fut_ver is not None:
            try:
                ver = fut_ver.result()
                print("Visual Veracity:")
                print(f"  ai_generated: {ver.get('ai_generated')}")
                print(f"  confidence: {ver.get('confidence')}")
                print(f"  explanation: {ver.get('explanation')}")
                anomalies = ver.get('anomalies') or []
                if anomalies:
                    print(f"  anomalies: {', '.join(map(str, anomalies))}")
            except Exception as e:
                print("  Visual veracity check error:", e)
    else:
        print("Visual veracity: skipped (--disable-visual)")

    # Create a per-sample DuckDuckGo batcher to avoid cross-sample queue interference
    local_batcher = None
    if answer_questions_enabled and search_provider == "duckduckgo":
        try:
            from scripts.duckduckgo_batcher import DuckDuckGoBatcher
            local_batcher = DuckDuckGoBatcher.from_env()
        except Exception as e:
            print("DuckDuckGo batch search unavailable; using sequential requests:", e)

    # Investigative question generation (sequential chains)
    best_qa_list: List[Dict[str, Any]] = []
    answers_by_question_global: Dict[str, Dict[str, Any]] = {}

    if run_questions:
        prior_questions: List[str] = []
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
                    prior_answers=answers_by_question_global,
                )
                chain = (qres.get("chains", [[]]) or [[]])[0]
                print(f"Questions (Chain {chain_idx}):")
                for qi, q in enumerate(chain, start=1):
                    print(f"  {qi}. {q}")
            except Exception as e:
                print(f"  Question generation error (chain {chain_idx}):", e)
                chain = []

            prior_questions.extend(chain)

            # Optionally answer and select best for this chain
            if answer_questions_enabled and chain:
                answers_by_question: Dict[str, Dict[str, Any]] = {}
                print("  Answers:")
                batch_results: Dict[str, Dict[str, Any]] = {}
                batch_job_ids: Dict[str, str] = {}

                if local_batcher is not None:
                    for q in chain:
                        try:
                            batch_job_ids[q] = local_batcher.enqueue(q)
                        except Exception as enqueue_err:
                            print(f"    Q: {q}")
                            print("      Answer error:", enqueue_err)
                    if batch_job_ids:
                        batch_results = local_batcher.execute()
                        batch_metrics = local_batcher.last_batch_metrics()
                        if batch_metrics:
                            unique = int(batch_metrics.get("unique", 0))
                            executed = int(batch_metrics.get("unique_executed", 0))
                            cache_hits = int(batch_metrics.get("cache_hits", 0))
                            retries = int(batch_metrics.get("retry_count", 0))
                            errors = int(batch_metrics.get("unique_error", 0))
                            avg_ms = float(batch_metrics.get("avg_duration_ms", 0.0))
                            if unique or cache_hits:
                                print(
                                    "      Batch stats: "
                                    f"unique={unique} executed={executed} cache_hits={cache_hits} "
                                    f"errors={errors} retries={retries} avg_ms={avg_ms:.0f}"
                                )

                for q in chain:
                    print(f"    Q: {q}")
                    try:
                        search_payload: Optional[Dict[str, Any]] = None
                        if local_batcher is not None and q in batch_job_ids:
                            result = batch_results.get(batch_job_ids[q])
                            if result:
                                err = result.get("error")
                                if err is not None:
                                    print("      Batch search error:", err)
                                else:
                                    search_payload = result.get("payload")
                        if search_payload is None:
                            search_payload = web_search(q, provider=search_provider)

                        ans = generate_answer_from_search(
                            q,
                            search_payload,
                            loader,
                            max_sources=args.answer_max_sources,
                        )
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
                        best_qa_list.append(best)
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
    else:
        print("Questions: skipped (--disable-questions)")
        if bool(args.answer_questions):
            print("  Note: --disable-questions overrides --answer-questions")

    # Final aggregated best Q/A list across chains (for downstream use)
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

    ddg_metrics = local_batcher.get_metrics() if local_batcher is not None else {}

    # Emit final structured JSON object per sample for downstream steps (slim schema)
    if not args.emit_json:
        return None, ddg_metrics

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
        "answer_questions": bool(answer_questions_enabled),
        "answer_max_sources": int(args.answer_max_sources),
        "search_provider": search_provider,
        "questions_enabled": bool(run_questions),
    }

    # Compute per-sample usage delta
    usage_after = getattr(loader, "usage_total", {"prompt": 0, "completion": 0, "total": 0})
    sample_usage = {
        "prompt": max(0, int(usage_after.get("prompt", 0)) - int(usage_before.get("prompt", 0))),
        "completion": max(0, int(usage_after.get("completion", 0)) - int(usage_before.get("completion", 0))),
        "total": max(0, int(usage_after.get("total", 0)) - int(usage_before.get("total", 0))),
    }

    final_obj: Dict[str, Any] = {
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
            "search_provider": run_params["search_provider"],
        },
        "run_params": run_params,
        "relevancy": rel or {},
        "visual_veracity": ver or {},
        "best_qa_per_chain": best_slim,
        "token_usage": sample_usage,
        "modules": module_config.copy(),
    }
    final_obj["sample_index"] = human_index
    dataset_index = sample_meta.get("dataset_index")
    if dataset_index is not None:
        try:
            final_obj["dataset_order_index"] = int(dataset_index)
        except Exception:
            final_obj["dataset_order_index"] = dataset_index
    else:
        final_obj["dataset_order_index"] = sample_meta.get("order_index")
    final_obj["iteration_index"] = sample_meta.get("order_index")

    # Attach dataset metadata for downstream rendering/analysis
    details_src = sample_meta.get("sample_details")
    details: Dict[str, Any] = {}
    if isinstance(details_src, dict):
        details.update({k: v for k, v in details_src.items() if v is not None})

    if dataset_index is not None and "dataset_index" not in details:
        try:
            details["dataset_index"] = int(dataset_index)
        except Exception:
            details["dataset_index"] = dataset_index

    if details:
        final_obj["sample_details"] = details

    # Run AI judge if requested
    judgement = None
    if run_judge:
        try:
            judgement = judge_from_structured(final_obj, loader)
            final_obj["judgement"] = judgement
        except Exception as e:
            print("  AI judge error:", e)
    else:
        print("AI judge: skipped (--disable-judge)")

    print("\n== Final Structured Output (JSON) ==")
    try:
        print(json.dumps(final_obj, ensure_ascii=False, indent=2))
    except Exception:
        # Fallback: best-effort string conversion
        def _stringify(o: Any) -> Any:
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
    if run_judge and judgement:
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

    print("  Token usage (sample): prompt={} completion={} total={}".format(
        sample_usage.get("prompt", 0), sample_usage.get("completion", 0), sample_usage.get("total", 0)
    ))

    return final_obj, ddg_metrics


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
    parser.add_argument(
        "--answer-questions",
        action="store_true",
        default=os.getenv("ANSWER_ENABLE", "0") == "1",
        help="Use web search (controlled by SEARCH_PROVIDER) + LLM to answer generated questions",
    )
    parser.add_argument("--answer-max-sources", type=int, default=int(os.getenv("ANSWER_MAX_SOURCES", "5")), help="Max sources to pass to LLM per question")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.getenv("PIPELINE_DATASET_ROOT", "data/MMFakeBench_test"),
        help="Root directory containing the MMFakeBench split to process (default: data/MMFakeBench_test)",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=os.getenv("PIPELINE_DATASET_JSON", ""),
        help="Optional dataset JSON path (absolute or relative to --dataset-root)",
    )
    parser.add_argument(
        "--dataset-stratify",
        type=str,
        default=os.getenv("PIPELINE_DATASET_STRATIFY", "fake_cls"),
        help="Comma-separated dataset fields to stratify the sampling order (set to '' to disable).",
    )
    parser.add_argument("--emit-json", action="store_true", default=os.getenv("EMIT_FINAL_JSON", "1") == "1", help="Print final structured JSON per sample for downstream use")
    parser.add_argument("--judge", action="store_true", default=os.getenv("JUDGE_ENABLE", "1") == "1", help="Run AI judge on final structured output and print/append decision")
    parser.add_argument("--save-jsonl", type=str, default=os.getenv("PIPELINE_OUTPUT_JSONL", ""), help="If set, append each final structured object to this JSONL file")
    parser.add_argument("--html-report", type=str, default=os.getenv("PIPELINE_HTML_REPORT", ""), help="If set, write an HTML report for this run's outputs")
    parser.add_argument("--html-title", type=str, default=os.getenv("PIPELINE_HTML_TITLE", "Pipeline Results"), help="Title for the HTML report")
    parser.add_argument("--html-inline-images", action="store_true", default=os.getenv("PIPELINE_HTML_INLINE_IMAGES", "0") == "1", help="Embed images into HTML as base64 data URLs for portability")
    parser.add_argument("--disable-relevancy", action="store_true", default=os.getenv("PIPELINE_DISABLE_RELEVANCY", "0") == "1", help="Skip the relevancy checker step")
    parser.add_argument("--disable-visual", action="store_true", default=os.getenv("PIPELINE_DISABLE_VISUAL", "0") == "1", help="Skip the visual veracity checker step")
    parser.add_argument("--disable-questions", action="store_true", default=os.getenv("PIPELINE_DISABLE_QUESTIONS", "0") == "1", help="Skip investigative question generation and answering")
    parser.add_argument("--disable-judge", action="store_true", default=os.getenv("PIPELINE_DISABLE_JUDGE", "0") == "1", help="Skip the AI judge step")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.getenv("PIPELINE_CHECKPOINT_DIR", ""),
        help="Directory for checkpoint files (defaults to results/checkpoints/<run-id>)",
    )
    parser.add_argument(
        "--checkpoint-size",
        type=int,
        default=int(os.getenv("PIPELINE_CHECKPOINT_SIZE", "100") or "100"),
        help="Write a checkpoint after this many samples",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=os.getenv("PIPELINE_RESUME", ""),
        help="Path to a checkpoint file to resume from",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("PIPELINE_WORKERS", "1")),
        help=(
            "Number of samples to process in parallel (default 1 = sequential). "
            "Each worker uses a separate LLM loader instance. "
            "Controlled by PIPELINE_WORKERS env var."
        ),
    )
    args = parser.parse_args()

    run_relevancy = not args.disable_relevancy
    run_visual = not args.disable_visual
    run_questions = not args.disable_questions
    run_judge = not args.disable_judge
    answer_questions_enabled = bool(args.answer_questions) and run_questions
    module_config = {
        "relevancy": run_relevancy,
        "visual_veracity": run_visual,
        "questions": run_questions,
        "question_answering": answer_questions_enabled,
        "judge": run_judge,
    }

    resume_state: Optional[CheckpointState] = None
    resume_path = Path(args.resume).expanduser() if args.resume else None
    if resume_path:
        if not resume_path.exists():
            print(f"\nResume checkpoint not found: {resume_path}")
            return
        try:
            resume_state = CheckpointState.from_file(resume_path)
        except Exception as exc:
            print(f"\nFailed to load checkpoint {resume_path}: {exc}")
            return

    run_id: Optional[str] = resume_state.run_id if resume_state else None

    if resume_state:
        if resume_state.jsonl_path and not args.save_jsonl:
            args.save_jsonl = resume_state.jsonl_path
        if resume_state.html_report_path and not args.html_report:
            args.html_report = resume_state.html_report_path
        if not args.checkpoint_dir:
            args.checkpoint_dir = str(resume_path.parent)

    # Default results folder outputs when not explicitly provided
    if not args.save_jsonl:
        fresh_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.save_jsonl = f"results/run-{fresh_id}.jsonl"

    base_name = os.path.splitext(os.path.basename(args.save_jsonl))[0]
    if not run_id:
        run_id = base_name

    if resume_state and resume_state.jsonl_path:
        try:
            if Path(args.save_jsonl).resolve() != Path(resume_state.jsonl_path).resolve():
                print("\nResume checkpoint was created for a different --save-jsonl path. Aborting to avoid mixing runs.")
                return
        except Exception:
            pass
        # Prefer checkpoint's canonical run id when it differs from derived base name
        if resume_state.run_id and resume_state.run_id != run_id:
            run_id = resume_state.run_id

    if not args.html_report:
        try:
            args.html_report = f"results/{run_id}.html"
        except Exception:
            args.html_report = "results/run.html"

    if not args.checkpoint_dir:
        args.checkpoint_dir = str(Path("results") / "checkpoints" / run_id)

    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.exists():
        print(f"\nDataset root not found: {dataset_root}")
        return

    dataset_json_override = (args.dataset_json or "").strip() or None
    try:
        json_path = _resolve_json_path(dataset_root, dataset_json_override)
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return

    stratify_fields = [field.strip() for field in (args.dataset_stratify or "").split(",") if field.strip()]

    args.checkpoint_size = max(1, int(args.checkpoint_size))

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    args_snapshot = build_args_snapshot(dict(vars(args)), drop_keys={"resume"})
    manager = CheckpointManager(
        run_id=run_id,
        checkpoint_dir=checkpoint_dir,
        chunk_size=args.checkpoint_size,
        args_snapshot=args_snapshot,
        jsonl_path=args.save_jsonl,
        html_report_path=args.html_report or None,
        resume_state=resume_state,
    )

    start_wall = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    metadata_path = Path(args.save_jsonl).with_suffix(".metadata.json")
    run_metadata: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "paths": {
            "jsonl": args.save_jsonl,
            "html": args.html_report,
            "checkpoint_dir": str(checkpoint_dir),
        },
        "dataset": {
            "root": str(dataset_root),
            "json": str(json_path),
            "stratify": stratify_fields or None,
        },
        "modules": module_config,
        "args": args_snapshot,
        "environment": _collect_env_snapshot(),
        "git": _collect_git_metadata(),
        "resume_from": str(resume_path) if resume_state else None,
        "status": "initializing",
    }

    def _update_run_metadata(extra: Dict[str, Any]) -> None:
        run_metadata.update(extra)
        _write_json(metadata_path, run_metadata)

    _update_run_metadata({})

    existing_outputs: List[Dict[str, Any]] = []
    if resume_state:
        existing_outputs = _load_existing_outputs(args.save_jsonl)
        if existing_outputs:
            if len(existing_outputs) > manager.processed_count:
                manager.sync_to(len(existing_outputs))
            elif len(existing_outputs) < manager.processed_count:
                print(
                    "\nWarning: checkpoint indicates more samples than present in the JSONL output."
                )

        resume_start = manager.next_index
        print(
            f"\nResuming run '{run_id}' from checkpoint {resume_path} starting at dataset index {resume_start + 1}."
        )
        _update_run_metadata({
            "status": "resuming",
            "resume_from": str(resume_path),
            "progress": {
                "processed_samples": manager.processed_count,
            },
        })

    search_provider = get_active_search_provider()

    image_root = dataset_root

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
        return_image=False,
        stratify_by=stratify_fields,
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
    samples: List[Dict[str, Any]] = []
    if args.image and args.headline:
        samples.append(
            {
                "order_index": 0,
                "image_path": args.image,
                "headline": args.headline,
                "sample_details": None,
            }
        )
    else:
        # Use --max-samples as the primary control, fallback to --relevancy-limit
        limit = max(0, int(args.max_samples if args.max_samples is not None else args.relevancy_limit))
        total = min(limit, len(ds))
        for order_index in range(total):
            sample = ds[order_index]
            sample_details = {
                "dataset_index": sample.get("dataset_index"),
                "gt_answers": sample.get("gt_answers"),
                "fake_cls": sample.get("fake_cls"),
                "text_source": sample.get("text_source"),
                "image_source": sample.get("image_source"),
            }
            samples.append(
                {
                    "order_index": order_index,
                    "dataset_index": sample.get("dataset_index"),
                    "image_path": sample.get("image_path"),
                    "headline": str(sample.get("text", "")),
                    "sample_details": sample_details,
                }
            )

    sample_target = len(samples)
    _update_run_metadata({
        "status": "ready",
        "target_samples": sample_target,
    })

    if not samples:
        print("\n--- Relevancy Check ---")
        print("No items to check (relevancy-limit is 0).")
        _update_run_metadata({
            "status": "no_samples",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - start_wall, 3),
        })
        return

    print("\n--- Relevancy + Visual Veracity Checks ---")
    n_workers = max(1, int(getattr(args, "workers", 1)))
    loader_cfg = {
        "provider": os.getenv("ALIGN_PROVIDER", "openai"),
        "model": args.model,  # None → provider default inside loader
        "temperature": args.temperature,
    }
    try:
        loader = LLMModelLoader(loader_cfg)
    except Exception as e:
        print("Checker setup failed:", e)
        print("Hints: set OPENAI_API_KEY or GOOGLE_API_KEY or DEEPINFRA_API_KEY or OPENROUTER_API_KEY; set ALIGN_PROVIDER=openai|google|deepinfra|openrouter; optionally pass --model/--image/--headline/--relevancy-limit.")
        return

    # Create one LLMModelLoader per worker slot to avoid usage_total contention.
    # For n_workers == 1 we reuse the single loader (unchanged behaviour).
    if n_workers == 1:
        worker_loaders: List[LLMModelLoader] = [loader]
    else:
        try:
            worker_loaders = [LLMModelLoader(loader_cfg) for _ in range(n_workers)]
        except Exception as e:
            print("Worker loader setup failed:", e)
            return

    if answer_questions_enabled:
        print(f"\nSearch answers will use provider: {search_provider}")

    run_outputs = list(existing_outputs)
    total_samples = sample_target
    start_index = min(manager.next_index, total_samples)

    if start_index >= total_samples:
        print("\nNo remaining samples to process."
              f" Already completed {manager.processed_count} of {total_samples} target samples.")
        _update_run_metadata({
            "status": "up_to_date",
            "progress": {
                "processed_samples": manager.processed_count,
                "target_samples": total_samples,
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - start_wall, 3),
        })
        return

    _update_run_metadata({
        "status": "running",
        "progress": {
            "processed_samples": manager.processed_count,
            "target_samples": total_samples,
            "resume_index": start_index,
        },
    })

    processed_any = False
    ddg_totals_list: List[Dict[str, Any]] = []
    run_summary: Dict[str, Any] = {}

    # Fixed args tuple forwarded to every _process_one_sample call
    _sample_call_args = (
        args, run_relevancy, run_visual, run_questions, run_judge,
        answer_questions_enabled, search_provider, module_config,
    )

    def _handle_one_result(
        final_obj: Optional[Dict[str, Any]],
        worker_ddg: Dict[str, Any],
    ) -> None:
        """Write JSONL, update run_outputs, tick checkpoint — must run in main thread."""
        nonlocal processed_any
        if final_obj is not None:
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
            run_outputs.append(final_obj)
        if worker_ddg:
            ddg_totals_list.append(worker_ddg)
        processed_any = True
        manager.record_sample()
        checkpoint_path = manager.maybe_save()
        if checkpoint_path:
            print(f"  Saved checkpoint: {checkpoint_path}")
            resume_hint = manager.resume_hint(checkpoint_path)
            if resume_hint:
                print(f"  Resume with: python main.py {resume_hint} [other flags]")
            _update_run_metadata({
                "progress": {
                    "processed_samples": manager.processed_count,
                    "target_samples": total_samples,
                },
                "last_checkpoint": str(checkpoint_path),
            })

    if n_workers == 1:
        # Sequential path — identical to the original behaviour, no futures overhead
        for order_idx in range(start_index, total_samples):
            final_obj, worker_ddg = _process_one_sample(
                samples[order_idx], order_idx, total_samples, worker_loaders[0],
                *_sample_call_args,
            )
            _handle_one_result(final_obj, worker_ddg)
    else:
        # Parallel path: submit all samples, then drain futures in **submission order**
        # so that processed_count increments sequentially and --resume works correctly.
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures_ordered = [
                pool.submit(
                    _process_one_sample,
                    samples[order_idx], order_idx, total_samples,
                    worker_loaders[order_idx % n_workers],
                    *_sample_call_args,
                )
                for order_idx in range(start_index, total_samples)
            ]
            for fut in futures_ordered:
                try:
                    final_obj, worker_ddg = fut.result()
                except Exception as e:
                    print(f"  Sample worker error: {e}")
                    final_obj, worker_ddg = None, {}
                _handle_one_result(final_obj, worker_ddg)

    # Aggregate DDG metrics from per-sample batchers
    ddg_totals: Dict[str, Any] = {}
    for wm in ddg_totals_list:
        for k, v in wm.items():
            if isinstance(v, (int, float)):
                ddg_totals[k] = ddg_totals.get(k, 0) + v

    if processed_any and manager.processed_count % manager.chunk_size != 0:
        _update_run_metadata({
            "progress": {
                "processed_samples": manager.processed_count,
                "target_samples": total_samples,
            }
        })
        final_checkpoint = manager.maybe_save(force=True)
        if final_checkpoint:
            print(f"\nSaved final checkpoint: {final_checkpoint}")
            resume_hint = manager.resume_hint(final_checkpoint)
            if resume_hint:
                print(f"Resume with: python main.py {resume_hint} [other flags]")
            _update_run_metadata({
                "last_checkpoint": str(final_checkpoint),
            })

    if ddg_totals:
        total_batches = int(ddg_totals.get("total_batches", 0))
        total_unique = int(ddg_totals.get("total_unique", 0))
        total_exec = int(ddg_totals.get("total_unique_executed", 0))
        total_cache_hits = int(ddg_totals.get("total_cache_hits", 0))
        total_errors = int(ddg_totals.get("total_errors", 0))
        total_retries = int(ddg_totals.get("total_retries", 0))
        total_duration_ms = float(ddg_totals.get("total_duration_ms", 0.0))
        avg_ms = total_duration_ms / total_exec if total_exec else 0.0
        print(
            "\nDuckDuckGo batch summary: batches={} unique={} executed={} cache_hits={} errors={} retries={} avg_ms={:.0f}".format(
                total_batches,
                total_unique,
                total_exec,
                total_cache_hits,
                total_errors,
                total_retries,
                avg_ms,
            )
        )
        run_summary["duckduckgo_batch"] = {
            "batches": total_batches,
            "unique": total_unique,
            "executed": total_exec,
            "cache_hits": total_cache_hits,
            "errors": total_errors,
            "retries": total_retries,
            "avg_ms": round(avg_ms, 0),
        }

    # Print total token usage across all processed samples
    # Merge usage from all per-worker loaders into a single grand total.
    token_totals: Dict[str, Any] = {}
    try:
        grand: Dict[str, Any] = {"prompt": 0, "completion": 0, "total": 0}
        for wl in worker_loaders:
            for k in grand:
                grand[k] = grand.get(k, 0) + int(wl.usage_total.get(k, 0) or 0)
        print("\n=== Token Usage (Run Total) ===")
        print(f"prompt={grand.get('prompt', 0)} completion={grand.get('completion', 0)} total={grand.get('total', 0)}")
        token_totals = {
            "prompt": int(grand.get("prompt", 0) or 0),
            "completion": int(grand.get("completion", 0) or 0),
            "total": int(grand.get("total", 0) or 0),
        }
    except Exception:
        pass

    if token_totals:
        run_summary["token_usage"] = token_totals

    # Generate HTML report for this run if requested
    html_path = (args.html_report or "").strip()
    if html_path and run_outputs:
        try:
            from scripts.report_html import render_html_report
            # Ensure outputs are in sample order (matters when workers > 1)
            run_outputs.sort(key=lambda o: o.get("sample_index", 0))
            # Try to compute metrics over the full JSONL if provided and dataset exists
            metrics = None
            ds_json = _resolve_json_path(dataset_root, dataset_json_override)
            if args.save_jsonl:
                try:
                    from scripts.evaluate import evaluate as _eval
                    metrics = _eval(Path(args.save_jsonl), ds_json, dataset_root, save_report=None)
                except Exception:
                    metrics = None
            render_html_report(
                run_outputs,
                metrics,
                html_path,
                title=args.html_title,
                inline_images=bool(args.html_inline_images),
                dataset_json_path=str(ds_json),
                dataset_image_root=str(image_root),
                run_summary=run_summary or None,
            )
            base, _ = os.path.splitext(html_path)
            csv_path = base + ".tokens.csv"
            print(f"\nWrote HTML report to {html_path}")
            if os.path.exists(csv_path):
                print(f"Token usage CSV: {csv_path}")
        except Exception as e:
            print("\nWarning: failed to write HTML report:", e)

    final_status = "completed" if processed_any else "no_samples_processed"
    _update_run_metadata({
        "status": final_status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - start_wall, 3),
        "progress": {
            "processed_samples": manager.processed_count,
            "target_samples": total_samples,
        },
        "outputs_count": len(run_outputs),
        "last_checkpoint": str(manager.last_checkpoint_path) if manager.last_checkpoint_path else run_metadata.get("last_checkpoint"),
    })


if __name__ == "__main__":
    main()
