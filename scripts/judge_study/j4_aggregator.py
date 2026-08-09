"""J4: learned aggregator (logistic regression on the module confidence vector).

LLM-free. 5-fold cross-validation on the 200 samples; mean +/- std across folds.
NOT training-free (flagged per brief): each fold trains on the other folds' GT.

Features per bundle:
  rel_aligned(+1/-1/0), rel_conf, vis_ai(+1/-1/0), vis_conf,
  n_answers, mean_answer_conf, max_answer_conf
Standardization fit on train folds only.

Usage:
  python -m scripts.judge_study.j4_aggregator --evidence-dir evidence/live200-v2 \
      --gt-jsonl results/gpt4omini-200.jsonl --out results/judge_study/phase2/J4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np

from scripts.judge_study.manifest import make_manifest, write_manifest

POS = "Misinformation"


class LogisticRegression:
    def __init__(self, l2: float = 0.01, steps: int = 2000, lr: float = 0.5):
        self.l2, self.steps, self.lr = l2, steps, lr
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        w = np.zeros(Xb.shape[1])
        for _ in range(self.steps):
            p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
            grad = Xb.T @ (p - y) + self.l2 * np.r_[w[:-1], 0.0]
            w -= self.lr * grad / Xb.shape[0]
        self.w = w

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        return 1.0 / (1.0 + np.exp(-np.clip(Xb @ self.w, -30, 30)))


def features(bundle: dict) -> list:
    rel = bundle.get("relevancy") or {}
    vis = bundle.get("visual_veracity") or {}
    al = rel.get("aligned")
    rel_al = 1.0 if al is True else (-1.0 if al is False else 0.0)
    va = vis.get("ai_generated")
    vis_ai = 1.0 if va is True else (-1.0 if va is False else 0.0)
    def f(x):
        try:
            return float(x)
        except Exception:
            return 0.0
    ans = [a for a in (bundle.get("answers") or []) if a.get("question") is not None]
    confs = [f(a.get("confidence")) for a in ans if a.get("confidence") is not None]
    return [rel_al, f(rel.get("confidence")), vis_ai, f(vis.get("confidence")),
            float(len(ans)), float(np.mean(confs)) if confs else 0.0,
            float(np.max(confs)) if confs else 0.0]


def load_gt(jsonl_path: str) -> dict:
    gt = {}
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            sid = str(d.get("dataset_order_index") if d.get("dataset_order_index") is not None else d.get("sample_index"))
            sd = d.get("sample_details") or {}
            gt[sid] = 1 if str(sd.get("gt_answers")) == "Fake" else 0
    return gt


def load_bundles(evidence_dir: str) -> dict:
    out = {}
    for fn in sorted(os.listdir(evidence_dir)):
        if not fn.endswith(".json") or fn in ("manifest.json", "refetch_manifest.json"):
            continue
        b = json.load(open(os.path.join(evidence_dir, fn)))
        out[b["sample_id"]] = b
    return out


def metrics(y, p):
    tp = sum(1 for t, q in zip(y, p) if t == 1 and q == 1)
    fp = sum(1 for t, q in zip(y, p) if t == 0 and q == 1)
    fn = sum(1 for t, q in zip(y, p) if t == 1 and q == 0)
    tn = sum(1 for t, q in zip(y, p) if t == 0 and q == 0)
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, f1, prec, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--gt-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=os.getcwd())
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bundles = load_bundles(args.evidence_dir)
    gt = load_gt(args.gt_jsonl)
    common = [sid for sid in bundles if sid in gt]
    X = np.array([features(bundles[s]) for s in common], dtype=float)
    y = np.array([gt[s] for s in common], dtype=float)

    manifest = make_manifest(
        phase="phase-2", stage="B", run_id=os.path.basename(args.out.rstrip("/")),
        model={"provider": "numpy", "id": "LogisticRegression(L2=0.01)", "revision": "n/a", "quant": "n/a"},
        prompt_files=[], evidence_version=os.path.basename(args.evidence_dir.rstrip("/")),
        seed=args.seed, sample_ids=common, temperature=0.0, repo_root=args.repo_root,
        notes="J4 learned aggregator: logistic regression on module confidence vector; 5-fold CV; "
              "NOT training-free (each fold trained on other folds' GT).",
        extra={"model_class": "LogisticRegression", "l2": 0.01, "steps": 2000, "features": 7, "k": 5},
    )
    write_manifest(manifest, os.path.join(args.out, "manifest.json"))

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(common))
    folds = np.array_split(order, 5)
    oof = np.full(len(common), np.nan)
    oof_proba = np.full(len(common), np.nan)
    fold_metrics = []
    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(5) if j != fi])
        mu, sd = X[train_idx].mean(0), X[train_idx].std(0) + 1e-9
        Xtr = (X[train_idx] - mu) / sd
        Xte = (X[test_idx] - mu) / sd
        lr = LogisticRegression()
        lr.fit(Xtr, y[train_idx])
        proba = lr.predict_proba(Xte)
        pred = (proba >= 0.5).astype(int)
        oof[test_idx] = pred
        oof_proba[test_idx] = proba
        acc, f1, prec, rec = metrics(y[test_idx], pred)
        fold_metrics.append({"fold": fi + 1, "n": len(test_idx), "acc": round(acc, 4),
                             "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4)})

    accs = [m["acc"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]
    acc, f1, prec, rec = metrics(y, oof)
    with open(os.path.join(args.out, "fold_metrics.json"), "w") as f:
        json.dump({"folds": fold_metrics,
                   "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
                   "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
                   "oof_acc": round(acc, 4), "oof_f1": round(f1, 4),
                   "oof_precision": round(prec, 4), "oof_recall": round(rec, 4)}, f, indent=2)
    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "label", "confidence", "abstain", "parse_ok", "attempts", "calls"])
        for i, sid in enumerate(common):
            w.writerow([sid, "Misinformation" if oof[i] == 1 else "Not Misinformation",
                        round(float(oof_proba[i]), 4), False, True, 1, 0])
    print(f"J4 fold acc: mean={np.mean(accs):.4f} +/- {np.std(accs):.4f} | fold f1: mean={np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"J4 out-of-fold: acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f}")


if __name__ == "__main__":
    main()
