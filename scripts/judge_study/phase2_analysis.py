"""Phase 2 master analysis: per-judge metrics + CIs, McNemar matrix,
J5 risk-coverage, per-class deltas, cost accounting.

Usage:
  python -m scripts.judge_study.phase2_analysis \
      --gt-jsonl results/gpt4omini-200.jsonl \
      --judges J1:results/judge_study/phase2/J1-gpt/labels.csv,J2:... \
      --out results/judge_study/phase2/report
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


def load_gt(jsonl_path: str) -> dict:
    gt = {}
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            sid = str(d.get("dataset_order_index") if d.get("dataset_order_index") is not None else d.get("sample_index"))
            sd = d.get("sample_details") or {}
            gt[sid] = {"gt": POS if str(sd.get("gt_answers")) == "Fake" else "Not Misinformation",
                       "fake_cls": sd.get("fake_cls")}
    return gt


def load_labels(csv_path: str) -> dict:
    out = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            out[r["sample_id"]] = r
    return out


def ece_brier(y_true, y_pred, confs):
    """10-bin equal-width calibration. Returns (ece, brier)."""
    n = len(y_true)
    if n == 0:
        return None, None
    correct = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]
    brier = sum((c - f) ** 2 for c, f in zip(correct, confs)) / n
    ece = 0.0
    for k in range(10):
        lo, hi = k / 10, (k + 1) / 10
        idx = [i for i, c in enumerate(confs) if lo <= c < hi or (k == 9 and c == 1.0)]
        if not idx:
            continue
        bin_acc = sum(correct[i] for i in idx) / len(idx)
        bin_conf = sum(confs[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(bin_acc - bin_conf)
    return round(ece, 4), round(brier, 4)


def macro_f1(y_true, y_pred):
    classes = sorted(set(y_true))
    f1s = []
    for c in classes:
        f1s.append(binary_f1(y_true, y_pred, c))
    return sum(f1s) / len(f1s) if f1s else 0.0


def balanced_acc(y_true, y_pred):
    classes = sorted(set(y_true))
    recs = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        n_c = sum(1 for t in y_true if t == c)
        recs.append(tp / n_c if n_c else 0.0)
    return sum(recs) / len(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-jsonl", required=True)
    ap.add_argument("--judges", required=True, help="comma-separated Name:path.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--binning", default="10-bin equal-width")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gt = load_gt(args.gt_jsonl)
    judges = {}
    for spec in args.judges.split(","):
        name, path = spec.split(":")
        judges[name] = load_labels(path)

    common = [s for s in gt if all(s in judges[j] for j in judges)]
    y_true = {s: gt[s]["gt"] for s in common}
    y_true_list = [y_true[s] for s in common]

    rows = []
    for name, lab in judges.items():
        y_pred = [lab[s]["label"] for s in common]
        confs = []
        valid = []
        for s in common:
            c = lab[s].get("confidence")
            try:
                cf = float(c)
            except Exception:
                cf = None
            if lab[s]["label"] in (POS, "Not Misinformation") and cf is not None:
                valid.append(s)
                confs.append(cf)
        decided = [s for s in common if lab[s]["label"] in (POS, "Not Misinformation")]
        y_d = [y_true[s] for s in decided]
        p_d = [lab[s]["label"] for s in decided]
        c_d = [float(lab[s]["confidence"]) for s in decided if lab[s].get("confidence") not in (None, "")]

        acc, lo, hi = bootstrap_ci(y_d, p_d, accuracy)
        f1, flo, fhi = bootstrap_ci(y_d, p_d, binary_f1)
        ece, brier = ece_brier(y_d, p_d, c_d)
        pf = sum(1 for s in common if not (lab[s].get("parse_ok", "True") in ("True", True, "1")))
        rows.append({
            "judge": name,
            "n": len(common), "decided": len(decided), "abstained": len(common) - len(decided),
            "accuracy": round(acc, 4), "acc_ci": [round(lo, 4), round(hi, 4)],
            "f1": round(f1, 4), "f1_ci": [round(flo, 4), round(fhi, 4)],
            "macro_f1": round(macro_f1(y_d, p_d), 4),
            "balanced_acc": round(balanced_acc(y_d, p_d), 4),
            "ece": ece, "brier": brier, "parse_failures": pf,
        })

    # McNemar matrix (on decided-only common samples per pair? use all labeled samples)
    mn_matrix = {}
    names = list(judges.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa = [judges[a][s]["label"] for s in common]
            pb = [judges[b][s]["label"] for s in common]
            mn = mcnemar(y_true_list, pa, pb)
            mn_matrix[f"{a}_vs_{b}"] = mn

    # J5 risk-coverage
    j5 = judges.get("J5-gpt") or next((v for k, v in judges.items() if k.startswith("J5")), None)
    rc = None
    if j5 is not None:
        decided_s = [s for s in common if j5[s]["label"] in (POS, "Not Misinformation")]
        scored = sorted(
            [(float(j5[s]["confidence"]), 1 if y_true[s] == j5[s]["label"] else 0) for s in decided_s],
            key=lambda x: -x[0])
        pts = []
        cum_correct = 0
        total = len(common)
        for i, (conf, ok) in enumerate(scored):
            cum_correct += ok
            coverage = (i + 1) / total
            risk = 1 - cum_correct / (i + 1)
            pts.append({"threshold": round(conf, 4), "coverage": round(coverage, 4), "risk": round(risk, 4)})
        rc = pts
        with open(os.path.join(args.out, "j5_risk_coverage.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["threshold", "coverage", "risk"])
            w.writeheader()
            w.writerows(pts)

    with open(os.path.join(args.out, "master_table.json"), "w") as f:
        json.dump({"n": len(common), "binning": args.binning, "rows": rows,
                   "mcnemar": mn_matrix, "j5_risk_coverage": rc}, f, indent=2)
    with open(os.path.join(args.out, "master_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'judge':10s} {'acc':>7s} {'acc_ci':>16s} {'f1':>6s} {'f1_ci':>16s} {'macF1':>6s} {'bacc':>6s} {'ECE':>6s} {'Brier':>7s} {'dec':>5s} {'abs':>4s} {'pf':>3s}")
    for r in rows:
        print(f"{r['judge']:10s} {r['accuracy']:7.3f} [{r['acc_ci'][0]:.3f},{r['acc_ci'][1]:.3f}] "
              f"{r['f1']:6.3f} [{r['f1_ci'][0]:.3f},{r['f1_ci'][1]:.3f}] {r['macro_f1']:6.3f} "
              f"{r['balanced_acc']:6.3f} {str(r['ece']):>6s} {str(r['brier']):>7s} {r['decided']:5d} {r['abstained']:4d} {r['parse_failures']:3d}")
    print("\nMcNemar (p < 0.05 => significant):")
    for k, v in mn_matrix.items():
        print(f"  {k:16s} p={v['p']:.4f} {'SIG' if v['significant'] else 'n.s.'} (b={v['b']},c={v['c']})")


if __name__ == "__main__":
    main()
