from __future__ import annotations

"""
Evaluate pipeline outputs against MMFakeBench ground truth.

Usage examples:
  - Evaluate a JSONL of pipeline outputs:
      python -m scripts.evaluate \
        --outputs outputs.jsonl \
        --dataset-json data/MMFakeBench_test/MMFakeBench_test.json \
        --image-root data/MMFakeBench_test

The evaluator computes:
  - Confusion matrix and metrics (Accuracy, Precision, Recall, F1) for the AI judge
    where Positive = "Misinformation" and GT positive = gt_answers == "Fake".
    Two views are reported:
      • Filtered: excludes samples where the judge predicted 'Uncertain'.
      • Penalized (all-in): includes all samples and treats 'Uncertain' as incorrect
        by assigning the opposite of ground-truth (degrades accuracy and F1 accordingly).
  - Visual veracity detection vs. image_source: Positive GT if image_source == "AI-generated Image".

Additionally, uncertainty statistics are reported (count and rate among all outputs).

Outputs a human-readable summary and a JSON metrics object if --save-report is provided.
"""

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _norm_path(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def _to_rel_under_root(image_root: Path, path: str) -> str:
    # Normalize to a relative posix-like key under image_root, stripping any leading '/'
    s = path.replace("\\", "/")
    s = s.lstrip("/")
    try:
        # If it's absolute under the workspace, try to relativize to root
        p = Path(path)
        rel = p.relative_to(image_root)
        return rel.as_posix()
    except Exception:
        return s


def _resolve_candidates(image_root: Path, img_path: str) -> List[str]:
    p = Path(img_path)
    cands: List[str] = []
    # Absolute normalized
    if p.is_absolute():
        cands.append(_norm_path(p))
        # Rebased absolute-like to image_root
        cands.append(_norm_path(image_root / str(p).lstrip(os.sep)))
    else:
        cands.append(_norm_path((image_root / p)))
    # Relative under root
    cands.append(_to_rel_under_root(image_root, img_path))
    # Also store version without any leading '/'
    cands.append(_to_rel_under_root(image_root, "/" + _to_rel_under_root(image_root, img_path)))
    # De-dup
    seen = set()
    out: List[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _load_gt_map(dataset_json: Path, image_root: Path) -> Dict[str, Dict[str, Any]]:
    with open(dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Build mapping from multiple keys (abs + rel) -> {gt, image_source}
    m: Dict[str, Dict[str, Any]] = {}
    for rec in data:
        img = rec.get("image_path")
        if img is None:
            continue
        for key in _resolve_candidates(image_root, img):
            m[key] = {
                "gt_answers": rec.get("gt_answers"),
                "image_source": rec.get("image_source"),
                "text": rec.get("text"),
            }
    return m


def _metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    # Per-class recall: positive=misinformation, negative=not misinformation
    rec_pos = rec
    rec_neg = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "recall_pos": round(rec_pos, 4),
        "recall_neg": round(rec_neg, 4),
        "support": total,
    }


def _confusion_and_metrics(y_true: List[int], y_pred: List[int]) -> Tuple[Dict[str, int], Dict[str, float]]:
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        elif t == 0 and p == 0:
            tn += 1
    cm = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return cm, _metrics_from_counts(tp, fp, fn, tn)


def _label_to_pred(label: str) -> Optional[int]:
    low = (label or "").strip().lower()
    if low.startswith("mis"):  # misinformation
        return 1
    if "not" in low and "mis" in low:  # not misinformation
        return 0
    if low.startswith("uncertain") or low.startswith("unknown"):
        return None
    return None


def _ai_truth_from_source(src: Any) -> Optional[int]:
    if not isinstance(src, str):
        return None
    return 1 if src.strip().lower() == "ai-generated image" else 0


def evaluate(outputs_path: Path, dataset_json: Path, image_root: Path, save_report: Optional[Path] = None) -> Dict[str, Any]:
    gt_map = _load_gt_map(dataset_json, image_root)

    preds: List[Dict[str, Any]] = []
    with open(outputs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            preds.append(obj)

    # Aggregate for judge
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_pred_all_keep_flags: List[bool] = []  # exclude when uncertain for filtered metrics

    # Visual veracity AI detection
    y_true_ai_all: List[int] = []              # filtered: only where prediction exists
    y_pred_ai_all: List[int] = []
    y_true_ai_all_all: List[int] = []          # all-in: include all with known GT, penalize missing preds
    y_pred_ai_all_all: List[int] = []
    ai_truth_total = 0                         # count of samples with known GT (image_source)

    missing = 0
    label_counter = Counter()
    for obj in preds:
        img_path = str(obj.get("image_path", ""))
        # Try flexible matching
        gt = None
        for key in _resolve_candidates(image_root, img_path):
            gt = gt_map.get(key)
            if gt is not None:
                break
        if gt is None:
            missing += 1
            continue

        # Ground truth for misinformation
        y_true = 1 if str(gt.get("gt_answers")) == "Fake" else 0

        # Predicted label from judge
        j = obj.get("judgement") or {}
        label = j.get("label") or ""
        label_counter[label] += 1
        pred = _label_to_pred(label)
        y_true_all.append(y_true)
        if pred is None:
            # Treat uncertain as incorrect for the all-in metric by assigning the opposite class
            # so it degrades both precision and recall conservatively.
            y_pred_all.append(1 - y_true)
            y_pred_all_keep_flags.append(False)
        else:
            y_pred_all.append(pred)
            y_pred_all_keep_flags.append(True)

        # Visual veracity AI detection
        ai_true = _ai_truth_from_source(gt.get("image_source"))
        ai_pred = obj.get("visual_veracity", {}).get("ai_generated")
        if ai_true is not None:
            ai_truth_total += 1
            # filtered view: only include when model predicted
            if isinstance(ai_pred, bool):
                y_true_ai_all.append(ai_true)
                y_pred_ai_all.append(1 if ai_pred else 0)
                # all-in view: same prediction
                y_true_ai_all_all.append(ai_true)
                y_pred_ai_all_all.append(1 if ai_pred else 0)
            else:
                # all-in view: penalize missing as incorrect by assigning opposite of GT
                y_true_ai_all_all.append(ai_true)
                y_pred_ai_all_all.append(1 - ai_true)

    # Metrics for judge (exclude uncertain; clean, no-penalty view)
    y_true_f = [t for t, keep in zip(y_true_all, y_pred_all_keep_flags) if keep]
    y_pred_f = [p for p, keep in zip(y_pred_all, y_pred_all_keep_flags) if keep]
    cm_filtered, metrics_filtered = _confusion_and_metrics(y_true_f, y_pred_f)

    # Metrics for judge (all predictions; 'Uncertain' penalized as incorrect)
    cm_all, metrics_all = _confusion_and_metrics(y_true_all, y_pred_all)

    # Metrics for AI detection via visual veracity
    cm_ai, metrics_ai = _confusion_and_metrics(y_true_ai_all, y_pred_ai_all)
    cm_ai_all, metrics_ai_all = _confusion_and_metrics(y_true_ai_all_all, y_pred_ai_all_all)

    total_preds = len(y_true_all)
    uncertain_count = total_preds - len(y_true_f)
    uncertain_rate = (uncertain_count / total_preds) if total_preds else 0.0

    report = {
        "counts": {
            "total_outputs": len(preds),
            "missing_in_gt_map": missing,
            "judge_label_distribution": dict(label_counter),
            "uncertain_count": uncertain_count,
            "uncertain_rate": round(uncertain_rate, 4),
            "visual_veracity_gt_known": ai_truth_total,
            "visual_veracity_pred_covered": len(y_true_ai_all),
            "visual_veracity_coverage_rate": round((len(y_true_ai_all) / ai_truth_total) if ai_truth_total else 0.0, 4),
        },
        # Primary, clean metrics for the judge (Uncertain excluded)
        "judge_metrics": {  # legacy key for backward-compatibility
            "confusion_matrix": cm_filtered,
            "metrics": metrics_filtered,
        },
        # All predictions; Uncertain penalized as incorrect
        "judge_metrics_all": {
            "confusion_matrix": cm_all,
            "metrics": metrics_all,
        },
        "visual_veracity_ai_detection": {
            "confusion_matrix": cm_ai,
            "metrics": metrics_ai,
        },
        "visual_veracity_ai_detection_all": {
            "confusion_matrix": cm_ai_all,
            "metrics": metrics_ai_all,
        },
    }

    # Pretty print summary
    def _pp(title: str, cm: Dict[str, int], m: Dict[str, float]):
        print(f"\n== {title} ==")
        print(f"  Accuracy: {m['accuracy']}")
        print(f"  Recall (Misinformation): {m['recall_pos']}")
        print(f"  Recall (Not Misinformation): {m['recall_neg']}")
        print(f"  F1: {m['f1']}")
        print(f"  Support: {m['support']}")
        print(f"  Confusion Matrix: TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}")

    print("Evaluation summary")
    print(f"  Outputs read: {len(preds)}; missing GT matches: {missing}")
    print(f"  Judge labels: {dict(label_counter)}")
    print(f"  Uncertain: {uncertain_count} ({round(uncertain_rate*100,2)}%)")
    _pp("Judge (filtered; excludes Uncertain)", report["judge_metrics"]["confusion_matrix"], report["judge_metrics"]["metrics"])
    _pp("Judge (all predictions; Uncertain penalized)", report["judge_metrics_all"]["confusion_matrix"], report["judge_metrics_all"]["metrics"])
    cm_ai_d = report["visual_veracity_ai_detection"]["confusion_matrix"]
    m_ai_d = report["visual_veracity_ai_detection"]["metrics"]
    _pp("Visual veracity: AI-generated detection (filtered)", cm_ai_d, m_ai_d)
    cm_ai_all_d = report["visual_veracity_ai_detection_all"]["confusion_matrix"]
    m_ai_all_d = report["visual_veracity_ai_detection_all"]["metrics"]
    _pp("Visual veracity: AI-generated detection (all; missing penalized)", cm_ai_all_d, m_ai_all_d)
    ai_total = cm_ai_all_d["TP"] + cm_ai_all_d["FN"]
    covered = len(y_true_ai_all)
    print(f"  AI-generated images in GT: {ai_total}; correctly flagged (filtered): {cm_ai_d['TP']} of {covered} covered")

    if save_report is not None:
        try:
            with open(save_report, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\nReport saved to {save_report}")
        except Exception as e:
            print("\nWarning: failed to save report:", e)

    return report


def main():
    p = argparse.ArgumentParser(description="Evaluate pipeline outputs vs MMFakeBench ground truth")
    p.add_argument("--outputs", type=str, required=True, help="Path to JSONL of final pipeline outputs")
    p.add_argument("--dataset-json", type=str, default=str(Path("data/MMFakeBench_test/MMFakeBench_test.json")), help="Path to dataset JSON")
    p.add_argument("--image-root", type=str, default=str(Path("data/MMFakeBench_test")), help="Image root to resolve dataset image paths")
    p.add_argument("--save-report", type=str, default="", help="Optional path to save metrics JSON report")
    args = p.parse_args()

    outputs = Path(args.outputs)
    dataset_json = Path(args.dataset_json)
    image_root = Path(args.image_root)
    save_report = Path(args.save_report) if args.save_report else None

    evaluate(outputs, dataset_json, image_root, save_report)


if __name__ == "__main__":
    main()
