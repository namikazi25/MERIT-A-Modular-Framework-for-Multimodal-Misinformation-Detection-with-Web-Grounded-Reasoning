"""Phase 2 bake-off runner: J1/J2/J3/J5 over an evidence dir with a judge model.

Manifest written before the run (rule 3); retries with backoff, logged (rule 6);
spend tracked against MAX_API_SPEND_USD (rule 5); temperature 0.0 (rule 7).
Parse failures are logged per judge and never silently averaged (brief).

J4 (learned aggregator) is LLM-free and handled by j4_aggregator.py.

Usage:
  python -m scripts.judge_study.judge_bakeoff --evidence-dir evidence/live200-v2 \
      --judge J2 --provider openai --model gpt-4o-mini-2024-07-18 \
      --out results/judge_study/phase2/J2-gpt --phase phase-2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from scripts.llm_loader import LLMModelLoader, ModelConfig
from scripts.judge_study.manifest import make_manifest, write_manifest, sha256_file
from scripts.judge_study.spend import SpendTracker, LOCAL_PROVIDERS
from scripts.judge_study.judges import JUDGES, PROMPT_FILES


def load_bundles(evidence_dir: str) -> list:
    bundles = []
    for fn in sorted(os.listdir(evidence_dir)):
        if not fn.endswith(".json") or fn in ("manifest.json", "refetch_manifest.json"):
            continue
        with open(os.path.join(evidence_dir, fn)) as fh:
            bundles.append(json.load(fh))
    return bundles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--judge", required=True, choices=sorted(JUDGES.keys()))
    ap.add_argument("--provider", default=os.getenv("ALIGN_PROVIDER", "openai"))
    ap.add_argument("--model", default=os.getenv("ALIGN_MODEL", "gpt-4o-mini-2024-07-18"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--repo-root", default=os.getcwd())
    ap.add_argument("--env", default=".env")
    ap.add_argument("--phase", default="phase-2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_id = os.path.basename(args.out.rstrip("/"))
    bundles = load_bundles(args.evidence_dir)
    sample_ids = [b["sample_id"] for b in bundles]

    tracker = SpendTracker(env_path=args.env)
    if args.provider not in LOCAL_PROVIDERS and tracker.cap_undefined:
        print("[HALT] MAX_API_SPEND_USD undefined; paid runs refused (rule 5).")
        sys.exit(2)

    prompt_files = PROMPT_FILES[args.judge]
    manifest = make_manifest(
        phase=args.phase, stage="B", run_id=run_id,
        model={"provider": args.provider, "id": args.model, "revision": "n/a", "quant": "n/a"},
        prompt_files=prompt_files,
        evidence_version=os.path.basename(args.evidence_dir.rstrip("/")),
        seed=None, sample_ids=sample_ids,
        temperature=args.temperature, repo_root=args.repo_root,
        notes=f"Phase 2 judge {args.judge}. Parse failures logged, never averaged silently.",
        extra={"judge": args.judge, "evidence_dir": args.evidence_dir,
               "max_attempts": args.max_attempts,
               "prompt_shas": {p: sha256_file(p) for p in prompt_files}},
    )
    write_manifest(manifest, os.path.join(args.out, "manifest.json"))

    judge_fn = JUDGES[args.judge]
    loader = LLMModelLoader(ModelConfig(provider=args.provider, model=args.model or None, temperature=args.temperature))
    log = open(os.path.join(args.out, "run.log"), "w")
    log_path = os.path.join(args.out, "run.log")

    rows, parse_failures = [], []
    n_abstain = 0
    for i, b in enumerate(bundles, start=1):
        usage_before = dict(getattr(loader, "usage_total", {"prompt": 0, "completion": 0, "total": 0}))
        result = None
        attempts = 0
        while attempts < args.max_attempts:
            attempts += 1
            try:
                result = judge_fn(b, loader)
                # retry only on parse failure / exception, NOT on J5 abstention
                if result.get("label") is None and not result.get("abstain"):
                    log.write(f"{b['sample_id']} attempt {attempts}: parse failure; retrying\n")
                    log.flush()
                    result = None
                else:
                    break
            except Exception as e:
                log.write(f"{b['sample_id']} attempt {attempts}: exception {e!r}\n")
                log.flush()
                result = None
            if attempts < args.max_attempts:
                delay = 2 ** (attempts - 1)
                log.write(f"{b['sample_id']} backing off {delay}s\n")
                log.flush()
                time.sleep(delay)

        usage_after = dict(getattr(loader, "usage_total", {"prompt": 0, "completion": 0, "total": 0}))
        p_tok = max(0, int(usage_after.get("prompt", 0)) - int(usage_before.get("prompt", 0)))
        c_tok = max(0, int(usage_after.get("completion", 0)) - int(usage_before.get("completion", 0)))
        retried = max(0, attempts - 1)
        tracker.add(args.model, p_tok, c_tok, retried_prompt=retried * 64, retried_completion=0)

        if result is None:
            row = {"sample_id": b["sample_id"], "label": "JUDGE_ERROR", "confidence": None,
                   "abstain": False, "parse_ok": False, "attempts": attempts, "calls": None}
            parse_failures.append(row)
            log.write(f"{b['sample_id']} FAILED after {attempts} attempts\n")
        else:
            row = {"sample_id": b["sample_id"], "label": result.get("label"),
                   "confidence": result.get("confidence"), "abstain": bool(result.get("abstain")),
                   "parse_ok": result.get("extra", {}).get("parse_ok", True),
                   "attempts": attempts, "calls": result.get("extra", {}).get("calls", 1)}
            if result.get("abstain"):
                n_abstain += 1
            if not row["parse_ok"]:
                parse_failures.append(row)
        rows.append(row)
        log.write(f"{b['sample_id']} -> {row['label']} abstain={row['abstain']} attempts={attempts} "
                  f"{tracker.summary()}\n")
        log.flush()
        if i % 25 == 0:
            print(f"[{i}/{len(bundles)}] {args.judge} done; cost ${tracker.total_cost():.4f}")

    log.close()

    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "label", "confidence", "abstain", "parse_ok", "attempts", "calls"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "labels.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "parse_failures.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(parse_failures)
    with open(os.path.join(args.out, "spend.json"), "w") as fh:
        json.dump({"judge": args.judge, "tokens": tracker.tokens, "cost_usd": tracker.total_cost(),
                   "cap": tracker.cap, "cap_undefined": tracker.cap_undefined,
                   "n": len(rows), "parse_failures": len(parse_failures), "abstentions": n_abstain},
                  fh, indent=2)

    print(f"{args.judge}: {len(rows)} labels, parse_failures={len(parse_failures)}, "
          f"abstentions={n_abstain}, cost ${tracker.total_cost():.4f} -> {args.out}")


if __name__ == "__main__":
    main()
