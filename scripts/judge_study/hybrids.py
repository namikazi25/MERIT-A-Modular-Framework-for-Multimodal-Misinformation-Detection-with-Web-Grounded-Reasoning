"""Step 3: hybrids H1-H3 on gpt dev-core under the same protocol as candidates.

H1: sufficiency-gated RULE — deterministic heuristic judge (judge_from_structured with
    loader=None); if evidence_sufficient=false -> abstain.
H2: J4 + evidence features — logistic regression (5-fold CV, supervised, NOT training-free).
H3: two-stage sufficiency -> LLM rule / abstain — if sufficient, current LLM judge (J1,
    gpt-4o-mini); if insufficient, abstain.

Sufficiency ratings: results/judge_study/sufficiency_dev.jsonl (gpt-rated, one call/sample).
Protocol: CI criterion vs J1-gpt baseline on the same samples; manifests per hybrid.

Usage:
  python -m scripts.judge_study.hybrids --out results/judge_study/taskB/hybrids \
      --provider openai --model gpt-4o-mini-2024-07-18
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

import numpy as np

from scripts.llm_loader import LLMModelLoader, ModelConfig
from scripts.ai_judge import judge_from_structured
from scripts.judge_study.judge_only import reconstruct_judge_input
from scripts.judge_study.manifest import make_manifest, write_manifest
from scripts.judge_study.stats import accuracy, binary_f1, bootstrap_ci, mcnemar
from scripts.judge_study.j4_aggregator import LogisticRegression, features as j4_features

POS = "Misinformation"


def load_sufficiency():
    out = {}
    if os.path.exists("results/judge_study/sufficiency_dev.jsonl"):
        for l in open("results/judge_study/sufficiency_dev.jsonl"):
            r = json.loads(l)
            out[r["sample_id"]] = r
    return out


def evidence_features(bundle):
    docs = bundle.get("documents") or {}
    n_docs = sum(len(v) for v in docs.values())
    ev_chars = sum(len((d.get("description") or "")) for v in docs.values() for d in v)
    return [float(len(docs)), float(n_docs), float(ev_chars / 4.0)]


def h1_rule_gated(bundle, suff, loader):
    """Deterministic heuristic judge gated on sufficiency -> abstain when insufficient."""
    suff_ok = bool(suff.get("evidence_sufficient"))
    final_obj = reconstruct_judge_input(bundle)
    j = judge_from_structured(final_obj, None)  # None -> heuristic (deterministic)
    if not suff_ok:
        return {"label": None, "confidence": 0.0, "abstain": True,
                "abstain_reason": "insufficient_evidence", "rationale": j.get("rationale")}
    return {"label": j.get("label"), "confidence": j.get("confidence"),
            "abstain": False, "rationale": j.get("rationale")}


def h3_two_stage(bundle, suff, loader):
    """Sufficient -> current LLM judge (J1, gpt); insufficient -> abstain."""
    suff_ok = bool(suff.get("evidence_sufficient"))
    final_obj = reconstruct_judge_input(bundle)
    j = judge_from_structured(final_obj, loader)
    if not suff_ok:
        return {"label": None, "confidence": 0.0, "abstain": True,
                "abstain_reason": "insufficient_evidence", "rationale": j.get("rationale")}
    return {"label": j.get("label"), "confidence": j.get("confidence"),
            "abstain": False, "rationale": j.get("rationale")}


def h2_lr(X, y, seed=42):
    """5-fold CV logistic regression (J4 + evidence features). Returns (oof_labels, oof_proba, fold_metrics)."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(y))
    folds = np.array_split(order, 5)
    oof = np.full(len(y), np.nan)
    oof_p = np.full(len(y), np.nan)
    fm = []
    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(5) if j != fi])
        mu, sd = X[train_idx].mean(0), X[train_idx].std(0) + 1e-9
        lr = LogisticRegression()
        lr.fit((X[train_idx] - mu) / sd, y[train_idx])
        proba = lr.predict_proba((X[test_idx] - mu) / sd)
        oof[test_idx] = (proba >= 0.5).astype(int)
        oof_p[test_idx] = proba
        tp = sum(1 for t, q in zip(y[test_idx], oof[test_idx]) if t == 1 and q == 1)
        acc = (sum(1 for t, q in zip(y[test_idx], oof[test_idx]) if t == q)) / len(test_idx)
        fm.append({"fold": fi + 1, "acc": round(acc, 4)})
    return oof, oof_p, fm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--repo-root", default=os.getcwd())
    args = ap.parse_args()

    # dev-core samples + GT
    sp = json.load(open("manifests/split-500-v1.json"))
    dc = json.load(open("manifests/devcore-100-v1.json"))
    core = {c["image_path"] for c in dc["dev_core"]}
    items = [d for d in sp["dev"] if d["image_path"] in core]
    suff = load_sufficiency()
    print("dev-core items:", len(items), "| sufficiency ratings available:", sum(1 for d in items if str(d.get("sample_id") or d['image_path']) in suff))

    bundles = []
    for d in items:
        b = json.load(open(d["bundle"]))
        if b.get("sample_id") is None:
            b["sample_id"] = d["image_path"]
        bundles.append(b)
    sid2gt = {}
    for d, b in zip(items, bundles):
        sid = str(b.get("sample_id"))
        g = d.get("gt")
        sid2gt[sid] = "Misinformation" if str(g) == "Fake" else "Not Misinformation"

    loader = LLMModelLoader(ModelConfig(provider=args.provider, model=args.model or None, temperature=0.0))

    os.makedirs(args.out, exist_ok=True)
    rows = {h: [] for h in ["H1", "H2", "H3"]}

    for b in bundles:
        sid = str(b.get("sample_id"))
        s = suff.get(sid, {"evidence_sufficient": True})
        # H1
        r1 = h1_rule_gated(b, s, loader)
        rows["H1"].append({"sample_id": sid, "label": r1["label"], "confidence": r1["confidence"],
                           "abstain": r1["abstain"]})
        # H3
        r3 = h3_two_stage(b, s, loader)
        rows["H3"].append({"sample_id": sid, "label": r3["label"], "confidence": r3["confidence"],
                           "abstain": r3["abstain"]})

    # H2: LR features (J4 + evidence), 5-fold CV on dev-core GT
    X = np.array([j4_features(b) + evidence_features(b) for b in bundles], dtype=float)
    y = np.array([1 if sid2gt[str(b.get("sample_id"))] == POS else 0 for b in bundles], dtype=float)
    oof, oof_p, fm = h2_lr(X, y)
    for i, b in enumerate(bundles):
        sid = str(b.get("sample_id"))
        rows["H2"].append({"sample_id": sid,
                           "label": "Misinformation" if oof[i] == 1 else "Not Misinformation",
                           "confidence": round(float(oof_p[i]), 4), "abstain": False})

    # score each hybrid vs J1-gpt baseline on dev-core
    j1dev = {}
    with open("results/judge_study/j1-gpt-dev.csv") as f:
        for r in csv.DictReader(f):
            j1dev[r["sample_id"]] = r
    j1_core = {s: j1dev[s] for s in sid2gt if s in j1dev}

    print(f"{'hybrid':5s} {'acc':>6s} {'acc_ci':>15s} {'f1':>6s} {'abst':>4s} {'vsJ1':>18s}")
    summary = {}
    for h in ["H1", "H2", "H3"]:
        lab = {r["sample_id"]: r for r in rows[h]}
        common = [s for s in sid2gt if s in lab and lab[s]["label"] in (POS, "Not Misinformation")]
        y = [sid2gt[s] for s in common]
        p = [lab[s]["label"] for s in common]
        acc, lo, hi = bootstrap_ci(y, p, accuracy)
        f1, flo, fhi = bootstrap_ci(y, p, binary_f1)
        abst = sum(1 for s in sid2gt if s in lab and lab[s]["abstain"])
        mn = mcnemar(y, [j1_core[s]["label"] for s in common], p)
        summary[h] = {"acc": acc, "acc_ci": [lo, hi], "f1": f1, "abstentions": abst,
                      "vs_J1_p": mn["p"], "vs_J1_sig": mn["significant"],
                      "beats_J1_ci": bool(lo > accuracy(y, [j1_core[s]["label"] for s in common]))}
        print(f"{h:5s} {acc:6.3f} [{lo:.3f},{hi:.3f}] {f1:6.3f} {abst:4d} p={mn['p']:.4f} {'SIG' if mn['significant'] else 'n.s.'} beatsJ1CI={summary[h]['beats_J1_ci']}")
        with open(os.path.join(args.out, f"{h}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "label", "confidence", "abstain"])
            w.writeheader()
            w.writerows(rows[h])
    summary["h2_folds"] = fm
    json.dump(summary, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    for h in ["H1", "H2", "H3"]:
        m = make_manifest(phase="task-B", stage="B", run_id=f"hybrids/{h}",
                          model={"provider": args.provider, "id": args.model, "revision": "n/a", "quant": "n/a"},
                          prompt_files=[], evidence_version="dev-core",
                          seed=None, sample_ids=list(sid2gt.keys()), temperature=0.0,
                          repo_root=args.repo_root,
                          notes=f"Hybrid {h}; sufficiency ratings from gpt pass; H2 supervised 5-fold CV (NOT training-free).",
                          extra={"hybrid": h, "concurrency": 1, "interventions_active": ["C1", "C2"]})
        write_manifest(m, os.path.join(args.out, f"{h}.manifest.json"))
    print("done ->", args.out)


if __name__ == "__main__":
    main()
