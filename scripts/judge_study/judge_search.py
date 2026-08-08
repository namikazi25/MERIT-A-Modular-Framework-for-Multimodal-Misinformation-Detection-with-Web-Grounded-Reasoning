"""Task B: judge auto-search runner (successive halving).

Runs a candidate list over a sample subset with a given judge model (temp 0,
concurrency 1 per B.1 for scored runs), writes per-candidate labels + manifests,
and scores macro-F1 / acc / F1 with bootstrap CIs vs the J1 baseline.

Usage:
  python -m scripts.judge_study.judge_search --candidates manifests/candidates-50-v1.json \
      --evidence-dir evidence/devcore-100-v1 --split manifests/split-500-v1.json \
      --subset dev-core --provider lmstudio --model qwen/qwen3.6-35b-a3b \
      --baseline results/judge_study/taskC/labels_none.csv --out results/judge_study/taskB/r1 \
      --ids C01,C02,...   (or --all)
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
from scripts.judge_study.manifest import make_manifest, write_manifest, sha256_file, git_state
from scripts.judge_study.spend import SpendTracker, LOCAL_PROVIDERS
from scripts.judge_study.candidate_judge import judge_candidate
from scripts.judge_study.stats import accuracy, binary_f1, bootstrap_ci, mcnemar
from scripts.judge_study.hygiene import dedup_documents, strip_boilerplate


def load_bundles(evidence_dir: str) -> list:
    out = []
    for fn in sorted(os.listdir(evidence_dir)):
        if not fn.endswith(".json") or fn in ("manifest.json", "refetch_manifest.json"):
            continue
        out.append(json.load(open(os.path.join(evidence_dir, fn))))
    return out


def interventions(bundle):
    """Adopted Task C interventions (C1 dedup, C2 boilerplate) as defaults."""
    nb, _ = dedup_documents(bundle)
    nb, _ = strip_boilerplate(nb)
    return nb


def run_candidate(cand, bundles, loader, tracker, model_id, out_dir):
    rows = []
    pf = 0
    for b in bundles:
        nb = interventions(b)
        try:
            j = judge_candidate(nb, loader, cand)
            if j.get("label") is None and not j.get("abstain"):
                pf += 1
            rows.append({"sample_id": b.get("sample_id"), "label": j.get("label"),
                         "confidence": j.get("confidence"), "abstain": bool(j.get("abstain")),
                         "parse_ok": j.get("extra", {}).get("parse_ok", True),
                         "rationale": (j.get("rationale") or "")[:800]})
        except Exception as e:
            pf += 1
            rows.append({"sample_id": b.get("sample_id"), "label": "JUDGE_ERROR",
                         "confidence": None, "abstain": False, "parse_ok": False, "rationale": str(e)[:200]})
    with open(os.path.join(out_dir, f"{cand['id']}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "label", "confidence", "abstain", "parse_ok", "rationale"])
        w.writeheader()
        w.writerows(rows)
    return pf


def score(cand_id, out_dir, sid2gt, baseline=None):
    lab = {}
    with open(os.path.join(out_dir, f"{cand_id}.csv")) as f:
        for r in csv.DictReader(f):
            lab[r["sample_id"]] = r
    common = [s for s in sid2gt if s in lab and lab[s]["label"] in ("Misinformation", "Not Misinformation")]
    y = [sid2gt[s] for s in common]
    p = [lab[s]["label"] for s in common]
    if not y:
        return None
    def macro(y, p):
        cls = set(y)
        return sum(binary_f1(y, p, c) for c in cls) / len(cls)
    mac, mlo, mhi = bootstrap_ci(y, p, macro)
    acc, alo, ahi = bootstrap_ci(y, p, accuracy)
    f1, flo, fhi = bootstrap_ci(y, p, binary_f1)
    out = {"id": cand_id, "n": len(common), "decided": len(common),
           "macro_f1": round(mac, 4), "macro_ci": [round(mlo, 4), round(mhi, 4)],
           "acc": round(acc, 4), "acc_ci": [round(alo, 4), round(ahi, 4)],
           "f1": round(f1, 4), "f1_ci": [round(flo, 4), round(fhi, 4)],
           "abstentions": sum(1 for s in common if lab[s]["abstain"] == "True")}
    if baseline is not None:
        pb = [baseline[s]["label"] for s in common]
        mn = mcnemar(y, pb, p)
        out["vs_baseline_p"] = mn["p"]
        out["vs_baseline_sig"] = mn["significant"]
        # B.3 criterion: candidate beats J1 iff CI excludes J1's point estimate
        b_acc = accuracy(y, pb)
        out["beats_baseline_ci"] = bool(alo > b_acc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--subset", default="dev-core", choices=["dev-core", "dev", "holdout"])
    ap.add_argument("--provider", default="lmstudio")
    ap.add_argument("--model", default="qwen/qwen3.6-35b-a3b")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--baseline", default=None, help="J1 baseline labels csv for the same samples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=None, help="comma-separated candidate ids, or --all")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--repo-root", default=os.getcwd())
    args = ap.parse_args()

    cands = json.load(open(args.candidates))["candidates"]
    if args.ids and args.ids != "--all":
        want = set(args.ids.split(","))
        cands = [c for c in cands if c["id"] in want]
    print(f"candidates to run: {len(cands)}")

    # sample set + GT from split manifest
    sp = json.load(open(args.split))
    if args.subset == "dev-core":
        dc = json.load(open("manifests/devcore-100-v1.json"))
        core = {c["image_path"] for c in dc["dev_core"]}
        items = [d for d in sp["dev"] if d["image_path"] in core]
    elif args.subset == "dev":
        items = sp["dev"]
    else:
        items = sp["holdout"]
    def norm(p): return p.replace("data/MMFakeBench_test", "").replace("\\", "/")
    sid2gt = {}
    bundle_by_path = {d["image_path"]: d["bundle"] for d in items}
    for d in items:
        sid2gt.setdefault(str(d.get("sample_id") or d["image_path"]), None)
    bundles = []
    for d in items:
        b = json.load(open(d["bundle"]))
        if b.get("sample_id") is None:
            b["sample_id"] = d["image_path"]
        bundles.append(b)
    # gt lookup by bundle image_path
    gt_by_ip = {norm(d["image_path"]): ("Misinformation" if str(d.get("gt")) == "Fake" else "Not Misinformation") for d in items}
    sid2gt = {str(b.get("sample_id")): gt_by_ip[norm(b.get("image_path"))] for b in bundles if norm(b.get("image_path")) in gt_by_ip}
    print(f"samples: {len(bundles)}; gt mapped: {len(sid2gt)}")

    tracker = SpendTracker(env_path=args.env)
    if args.provider not in LOCAL_PROVIDERS and tracker.cap_undefined:
        print("[HALT] cap undefined; paid runs refused"); sys.exit(2)

    os.makedirs(args.out, exist_ok=True)
    loader = LLMModelLoader(ModelConfig(provider=args.provider, model=args.model or None, temperature=args.temperature))

    baseline = None
    if args.baseline:
        baseline = {}
        with open(args.baseline) as f:
            for r in csv.DictReader(f):
                baseline[r["sample_id"]] = r

    results = []
    for c in cands:
        t0 = time.time()
        pf = run_candidate(c, bundles, loader, tracker, args.model, args.out)
        s = score(c["id"], args.out, sid2gt, baseline)
        if s is None:
            print(f"{c['id']}: no scored samples!", flush=True)
            continue
        s["parse_failures"] = pf
        s["seconds"] = round(time.time() - t0, 1)
        results.append(s)
        print(f"{c['id']}: macroF1={s['macro_f1']} acc={s['acc']} f1={s['f1']} "
              f"pf={pf} {s['seconds']}s" + (f" beatsJ1={s.get('beats_baseline_ci')}" if baseline else ""), flush=True)
        # per-candidate manifest
        m = make_manifest(
            phase="task-B", stage="B", run_id=f"{args.out.split('/')[-1]}/{c['id']}",
            model={"provider": args.provider, "id": args.model, "revision": "n/a", "quant": "n/a"},
            prompt_files=[c["prompt_path"]],
            evidence_version=args.subset, seed=None,
            sample_ids=list(sid2gt.keys()), temperature=args.temperature,
            repo_root=args.repo_root,
            notes="Task B candidate scored run; concurrency 1 (B.1); interventions C1+C2 active; temp 0.",
            extra={"candidate": c["id"], "axes": {k: c[k] for k in ("evidence", "citation", "reasoning", "confidence", "abstain")},
                   "prompt_sha256": c["prompt_sha256"], "concurrency": 1,
                   "interventions_active": ["C1", "C2"]})
        write_manifest(m, os.path.join(args.out, f"{c['id']}.manifest.json"))

    results.sort(key=lambda r: -r["macro_f1"])
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"n_candidates": len(results), "results": results}, f, indent=1)
    print("\n=== ranking (macro F1) ===")
    for r in results[:15]:
        print(f"  {r['id']:12s} macroF1={r['macro_f1']:.4f} acc={r['acc']:.4f} beatsJ1={r.get('beats_baseline_ci')}")


if __name__ == "__main__":
    main()
