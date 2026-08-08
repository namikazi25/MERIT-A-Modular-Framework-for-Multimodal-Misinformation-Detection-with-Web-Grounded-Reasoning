"""Phase 1 analysis: judge comparison on frozen v1 evidence.

Inputs:
  --gt-jsonl   results/gpt4omini-200.jsonl  (sample_details: gt_answers, fake_cls)
  --judge-a    labels csv from judge_only run (e.g. local)
  --judge-b    labels csv from judge_only run (e.g. gpt-4o-mini, today's re-judged baseline)
  --out-dir    where to write tables + PR plot

Outputs: per-judge metrics with 95% bootstrap CIs, McNemar, threshold-sweep PR
curves (CSV + overlaid plot), per-fake_cls breakdown.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.judge_study.stats import accuracy, binary_f1, bootstrap_ci, mcnemar

POS = "Misinformation"
NEG = "Not Misinformation"


def load_gt(jsonl_path: str) -> dict:
    gt = {}
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            sid = str(d.get("dataset_order_index") if d.get("dataset_order_index") is not None else d.get("sample_index"))
            sd = d.get("sample_details") or {}
            gt[sid] = {
                "gt": POS if str(sd.get("gt_answers")) == "Fake" else NEG,
                "fake_cls": sd.get("fake_cls"),
                "image_path": d.get("image_path"),
            }
    return gt


def load_labels(csv_path: str) -> dict:
    labels = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            labels[row["sample_id"]] = row
    return labels


def signed_score(label: str, conf: float) -> float:
    """Higher = more likely misinformation. Judge operating point = 0."""
    conf = 0.0 if conf is None else float(conf)
    return conf if label == POS else -conf


def metrics_block(y_true, y_pred, name: str) -> dict:
    acc_p, acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, accuracy)
    f1_p, f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, binary_f1)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == POS and p == POS)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == NEG and p == POS)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == POS and p == NEG)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == NEG and p == NEG)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "judge": name, "n": len(y_true),
        "accuracy": round(acc_p, 4), "acc_ci": [round(acc_lo, 4), round(acc_hi, 4)],
        "f1": round(f1_p, 4), "f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
        "precision": round(prec, 4), "recall": round(rec, 4),
        "cm": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
    }


def pr_curve(y_true, scores) -> list:
    """Sort by descending score, compute precision/recall at each threshold."""
    pairs = sorted(zip(scores, y_true), key=lambda x: -x[0])
    n_pos = sum(1 for t in y_true if t == POS)
    tp = fp = 0
    pts = []
    seen = set()
    for i, (s, t) in enumerate(pairs):
        if t == POS:
            tp += 1
        else:
            fp += 1
        # record unique thresholds
        if s not in seen:
            seen.add(s)
            prec = tp / (tp + fp) if (tp + fp) else 1.0
            rec = tp / n_pos if n_pos else 0.0
            pts.append({"threshold": s, "precision": round(prec, 4), "recall": round(rec, 4),
                        "tp": tp, "fp": fp})
    # add terminal point (all pos)
    pts.append({"threshold": -1.01, "precision": 1.0, "recall": 1.0, "tp": n_pos, "fp": 0})
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-jsonl", required=True)
    ap.add_argument("--judge-a", required=True, help="csv: local judge")
    ap.add_argument("--judge-b", required=True, help="csv: gpt-4o-mini judge (today)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--a-name", default="local qwen3.6-35b-a3b")
    ap.add_argument("--b-name", default="gpt-4o-mini (today)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gt = load_gt(args.gt_jsonl)
    la = load_labels(args.judge_a)
    lb = load_labels(args.judge_b)

    common = [sid for sid in gt if sid in la and sid in lb]
    y_true = [gt[s]["gt"] for s in common]
    y_a = [la[s]["label"] for s in common]
    y_b = [lb[s]["label"] for s in common]
    s_a = [signed_score(la[s]["label"], la[s]["confidence"]) for s in common]
    s_b = [signed_score(lb[s]["label"], lb[s]["confidence"]) for s in common]

    rows = [
        metrics_block(y_true, y_a, args.a_name),
        metrics_block(y_true, y_b, args.b_name),
    ]
    mn = mcnemar(y_true, y_a, y_b)

    # per fake_cls
    cls_rows = []
    for cls in sorted({gt[s]["fake_cls"] for s in common}):
        idx = [i for i, s in enumerate(common) if gt[s]["fake_cls"] == cls]
        m_a = metrics_block([y_true[i] for i in idx], [y_a[i] for i in idx], args.a_name)
        m_b = metrics_block([y_true[i] for i in idx], [y_b[i] for i in idx], args.b_name)
        m_a["fake_cls"] = cls; m_b["fake_cls"] = cls
        cls_rows.append(m_a); cls_rows.append(m_b)

    out = {
        "common_n": len(common),
        "overall": rows,
        "mcnemar_a_vs_b": mn,
        "per_fake_cls": cls_rows,
        "pr_a": pr_curve(y_true, s_a),
        "pr_b": pr_curve(y_true, s_b),
        "a_name": args.a_name, "b_name": args.b_name,
    }
    with open(os.path.join(args.out_dir, "phase1_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    # CSV tables
    with open(os.path.join(args.out_dir, "overall.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out_dir, "per_fake_cls.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cls_rows[0].keys()))
        w.writeheader(); w.writerows(cls_rows)
    with open(os.path.join(args.out_dir, "pr_a.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "tp", "fp"])
        w.writeheader(); w.writerows(out["pr_a"])
    with open(os.path.join(args.out_dir, "pr_b.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "tp", "fp"])
        w.writeheader(); w.writerows(out["pr_b"])

    # PR plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for tag, pr in [("a", out["pr_a"]), ("b", out["pr_b"])]:
            pass
        pr_a, pr_b = out["pr_a"], out["pr_b"]
        plt.figure(figsize=(7, 6))
        plt.plot([p["recall"] for p in pr_a], [p["precision"] for p in pr_a], marker=".", label=args.a_name)
        plt.plot([p["recall"] for p in pr_b], [p["precision"] for p in pr_b], marker=".", label=args.b_name)
        # mark the judges' operating points (threshold >= 0)
        for pr, name, c in [(pr_a, args.a_name, "C0"), (pr_b, args.b_name, "C1")]:
            op = [p for p in pr if p["threshold"] >= 0]
            if op:
                p0 = op[0]
                plt.scatter([p0["recall"]], [p0["precision"]], marker="X", s=120, color=c, zorder=5)
                plt.annotate(f"{name} op", (p0["recall"], p0["precision"]),
                             textcoords="offset points", xytext=(6, 6), fontsize=8, color=c)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Phase 1: judge PR curves (frozen v1 evidence, 200 samples)")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "pr_overlay.png"), dpi=150)
        print("plot saved")
    except Exception as e:
        print("plot failed:", e)

    print(json.dumps({"overall": rows, "mcnemar": mn}, indent=1))


if __name__ == "__main__":
    main()
