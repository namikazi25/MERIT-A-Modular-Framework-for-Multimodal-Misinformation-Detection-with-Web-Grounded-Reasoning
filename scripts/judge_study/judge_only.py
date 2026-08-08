"""Stage B (Judge) entrypoint (Task 0.1).

Reads evidence bundles produced by Stage A and produces labels using the
*current* judge logic (scripts.ai_judge.judge_from_structured) — unchanged.

No network calls except the single judge LLM invocation per sample; no vision.

Rules honoured here:
  - manifest written BEFORE the run starts (rule 3)
  - spend cap enforced (rule 5): paid runs refused when MAX_API_SPEND_USD undefined
  - retries: exponential backoff, logged, retried tokens counted (rule 6)
  - temperature 0.0 (determinism, rule 7)

Usage:
  python -m scripts.judge_study.judge_only \
      --evidence-dir evidence/<run_id> \
      --provider openai --model gpt-4o-mini-2024-07-18 \
      --out results/judge_study/<run_id> [--max-attempts 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from scripts.ai_judge import judge_from_structured, _SYSTEM_PROMPT
from scripts.llm_loader import LLMModelLoader, ModelConfig
from scripts.judge_study.manifest import make_manifest, write_manifest, sha256_file
from scripts.judge_study.spend import SpendTracker, LOCAL_PROVIDERS

PROMPT_FILE = "prompts/judge_study/j1_rule_prompt.txt"


def _write_prompt_file() -> str:
    os.makedirs(os.path.dirname(PROMPT_FILE), exist_ok=True)
    if not os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "w") as fh:
            fh.write(_SYSTEM_PROMPT)
    else:
        # guard against drift between the file and the live module constant
        with open(PROMPT_FILE) as fh:
            if fh.read() != _SYSTEM_PROMPT:
                raise RuntimeError(
                    PROMPT_FILE + " does not match the current _SYSTEM_PROMPT in scripts/ai_judge.py; "
                    "refresh the prompt file (it is frozen for traceability)."
                )
    return PROMPT_FILE


def load_bundles(evidence_dir: str) -> list:
    bundles = []
    for fn in sorted(os.listdir(evidence_dir)):
        if not fn.endswith(".json") or fn == "manifest.json":
            continue
        with open(os.path.join(evidence_dir, fn)) as fh:
            bundles.append(json.load(fh))
    return bundles


def reconstruct_judge_input(bundle: dict) -> dict:
    """Rebuild the exact final_obj fields the judge consumes (mirrors main.py)."""
    answers = bundle.get("answers") or []
    return {
        "headline": bundle.get("claim"),
        "image_path": bundle.get("image_path"),
        "relevancy": bundle.get("relevancy") or {},
        "visual_veracity": bundle.get("visual_veracity") or {},
        "best_qa_per_chain": [
            {
                "question": a.get("question"),
                "answer": a.get("answer"),
                "confidence": a.get("confidence"),
                "citations": a.get("citations") or [],
            }
            for a in answers if isinstance(a, dict)
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B (Judge): labels from evidence bundles")
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--provider", default=os.getenv("ALIGN_PROVIDER", "openai"))
    ap.add_argument("--model", default=os.getenv("ALIGN_MODEL", "gpt-4o-mini-2024-07-18"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True, help="output dir for labels/manifest/log")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--repo-root", default=os.getcwd())
    ap.add_argument("--env", default=".env")
    ap.add_argument("--phase", default="phase-0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_id = os.path.basename(args.out.rstrip("/"))

    bundles = load_bundles(args.evidence_dir)
    sample_ids = [b["sample_id"] for b in bundles]
    prompt_file = _write_prompt_file()

    tracker = SpendTracker(env_path=args.env)
    if args.provider not in LOCAL_PROVIDERS and tracker.cap_undefined:
        print("[HALT] MAX_API_SPEND_USD is not defined in .env; paid judge runs are refused (hard rule 5).")
        sys.exit(2)

    manifest = make_manifest(
        phase=args.phase, stage="B", run_id=run_id,
        model={"provider": args.provider, "id": args.model, "revision": "n/a", "quant": "n/a"},
        prompt_files=[prompt_file],
        evidence_version=os.path.basename(args.evidence_dir.rstrip("/")),
        seed=None, sample_ids=sample_ids,
        temperature=args.temperature, repo_root=args.repo_root,
        notes="current judge logic (judge_from_structured) unchanged; no vision; single LLM call per sample.",
        extra={"evidence_dir": args.evidence_dir, "max_attempts": args.max_attempts,
               "prompt_sha256": sha256_file(prompt_file)},
    )
    write_manifest(manifest, os.path.join(args.out, "manifest.json"))

    log_path = os.path.join(args.out, "run.log")
    log = open(log_path, "w")

    loader = LLMModelLoader(ModelConfig(provider=args.provider, model=args.model or None, temperature=args.temperature))
    model = loader.get_model()

    labels = []
    n_error = 0
    for i, b in enumerate(bundles, start=1):
        final_obj = reconstruct_judge_input(b)
        usage_before = dict(getattr(loader, "usage_total", {"prompt": 0, "completion": 0, "total": 0}))
        judgement = None
        attempts = 0
        while attempts < args.max_attempts:
            attempts += 1
            try:
                judgement = judge_from_structured(final_obj, loader)
                if judgement.get("raw") is None and args.provider not in LOCAL_PROVIDERS:
                    # heuristic fallback inside judge_from_structured -> treat as failure
                    log.write(f"{b['sample_id']} attempt {attempts}: heuristic fallback (raw=None); retrying\n")
                    log.flush()
                    judgement = None
                else:
                    break
            except Exception as e:
                log.write(f"{b['sample_id']} attempt {attempts}: exception {e!r}\n")
                log.flush()
                judgement = None
            if attempts < args.max_attempts:
                delay = 2 ** (attempts - 1)
                log.write(f"{b['sample_id']} backing off {delay}s\n")
                log.flush()
                time.sleep(delay)

        usage_after = dict(getattr(loader, "usage_total", {"prompt": 0, "completion": 0, "total": 0}))
        p_tok = max(0, int(usage_after.get("prompt", 0)) - int(usage_before.get("prompt", 0)))
        c_tok = max(0, int(usage_after.get("completion", 0)) - int(usage_before.get("completion", 0)))
        retried = max(0, attempts - 1)  # failed attempts had no usage; estimate conservatively
        tracker.add(args.model, p_tok, c_tok, retried_prompt=retried * 64, retried_completion=0)

        if judgement is None:
            n_error += 1
            labels.append({"sample_id": b["sample_id"], "label": "JUDGE_ERROR", "confidence": None,
                           "raw_present": False, "attempts": attempts})
            log.write(f"{b['sample_id']} FAILED after {attempts} attempts -> JUDGE_ERROR\n")
        else:
            labels.append({"sample_id": b["sample_id"], "label": judgement.get("label"),
                           "confidence": judgement.get("confidence"),
                           "raw_present": judgement.get("raw") is not None, "attempts": attempts})
        log.write(f"{b['sample_id']} -> {labels[-1]['label']} (attempts={attempts}) {tracker.summary()}\n")
        log.flush()
        if (i % 20) == 0:
            print(f"[{i}/{len(bundles)}] done; cost so far: ${tracker.total_cost():.4f}")

    log.close()

    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "label", "confidence", "raw_present", "attempts"])
        w.writeheader()
        w.writerows(labels)

    with open(os.path.join(args.out, "labels.jsonl"), "w") as fh:
        for row in labels:
            fh.write(json.dumps(row) + "\n")

    with open(os.path.join(args.out, "spend.json"), "w") as fh:
        json.dump({"tokens": tracker.tokens, "cost_usd": tracker.total_cost(),
                   "cap": tracker.cap, "cap_undefined": tracker.cap_undefined,
                   "judge_errors": n_error}, fh, indent=2)

    print(f"done: {len(labels)} labels, {n_error} judge errors, cost ${tracker.total_cost():.4f}, "
          f"cap_ok={tracker.check_cap()['ok']}")
    print(f"outputs in {args.out}")


if __name__ == "__main__":
    main()
