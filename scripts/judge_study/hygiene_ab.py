"""Task C: retrieval-hygiene A/B tests on the dev core (all local, $0).

Runs J1 (local qwen3.6-35b-a3b) on dev-core bundles with and without each
intervention; outputs labels per intervention; a comparison step then reports
acc/F1 with CIs + McNemar vs the unmodified baseline.

Usage:
  python -m scripts.judge_study.hygiene_ab run --devcore manifests/devcore-100-v1.json \
      --split manifests/split-500-v1.json --out results/judge_study/taskC --interventions C1,C2,C3,C4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.getcwd())

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from scripts.llm_loader import LLMModelLoader, ModelConfig
from scripts.judge_study.judges import judge_j1
from scripts.judge_study.hygiene import dedup_documents, strip_boilerplate, rewrite_queries, rerank_evidence
from scripts.judge_study.search_cache import SearchCache


def load_devcore_bundles(devcore_path: str, split_path: str) -> list:
    dc = json.load(open(devcore_path))
    sp = json.load(open(split_path))
    bundle_by_path = {d["image_path"]: d["bundle"] for d in sp["dev"] + sp["holdout"]}
    out = []
    for c in dc["dev_core"]:
        ip = c["image_path"]
        assert ip in bundle_by_path, ip
        b = json.load(open(bundle_by_path[ip]))
        b["sample_id"] = c.get("sample_id") or b.get("sample_id")
        out.append(b)
    return out


def apply_intervention(bundle, name, cache=None, provider="duckduckgo"):
    if name == "C1":
        nb, stats = dedup_documents(bundle)
    elif name == "C2":
        nb, stats = strip_boilerplate(bundle)
    elif name == "C3":
        nb, stats = rewrite_queries(bundle, cache, provider)
    elif name == "C4":
        nb, stats = rerank_evidence(bundle, None)  # loader injected below
    else:
        return bundle, {}
    nb["intervention"] = name
    nb["intervention_stats"] = stats
    return nb, stats


def run_intervention(bundles, name, loader, cache=None):
    rows = []
    for b in bundles:
        nb, stats = apply_intervention(b, name, cache)
        if name == "C4":
            nb, stats = rerank_evidence(nb, loader)
        try:
            j = judge_j1(nb, loader)
            label = j["label"] if j["label"] is not None else "JUDGE_ERROR"
            conf = j["confidence"]
        except Exception as e:
            label, conf = "JUDGE_ERROR", None
        rows.append({"sample_id": b["sample_id"], "label": label, "confidence": conf,
                     "intervention": name, "stats": json.dumps(stats)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--devcore", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interventions", default="C1,C2,C3,C4")
    p.add_argument("--provider", default="lmstudio")
    p.add_argument("--model", default="qwen/qwen3.6-35b-a3b")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bundles = load_devcore_bundles(args.devcore, args.split)
    print(f"dev-core bundles: {len(bundles)}")
    loader = LLMModelLoader(ModelConfig(provider=args.provider, model=args.model, temperature=0.0))
    cache = SearchCache(enabled=True, verbose=False)

    interventions = ["none"] + [x.strip() for x in args.interventions.split(",")]
    all_rows = []
    for name in interventions:
        print(f"running J1-local + intervention {name} ...", flush=True)
        rows = run_intervention(bundles, name, loader, cache)
        with open(os.path.join(args.out, f"labels_{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "label", "confidence", "intervention", "stats"])
            w.writeheader()
            w.writerows(rows)
        all_rows.extend(rows)
        print(f"  {name}: {len(rows)} labels", flush=True)

    with open(os.path.join(args.out, "all_labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "label", "confidence", "intervention", "stats"])
        w.writeheader()
        w.writerows(all_rows)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
