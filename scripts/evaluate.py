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
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.utils.dataset import load_dataset_records, resolve_image_candidates


def _load_gt_map(dataset_json: Path, image_root: Path) -> Dict[str, Dict[str, Any]]:
    records = load_dataset_records(dataset_json)
    mapping: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        img = rec.get("image_path")
        if img is None:
            continue
        for key in resolve_image_candidates(image_root, str(img)):
            mapping[key] = {
                "gt_answers": rec.get("gt_answers"),
                "image_source": rec.get("image_source"),
                "text": rec.get("text"),
            }
    return mapping


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


def _calibration_from_records(records: List[Tuple[float, int]], bins: int = 10) -> Dict[str, Any]:
    if not records:
        return {
            "total": 0,
            "bins": [],
            "brier_score": None,
            "expected_calibration_error": None,
            "bin_count": bins,
        }

    total = len(records)
    bin_buckets: List[List[Tuple[float, int]]] = [[] for _ in range(bins)]
    for conf, correct in records:
        idx = min(int(conf * bins), bins - 1)
        bin_buckets[idx].append((conf, correct))

    bin_infos: List[Dict[str, Any]] = []
    ece = 0.0
    for i, bucket in enumerate(bin_buckets):
        low = i / bins
        high = 1.0 if i == bins - 1 else (i + 1) / bins
        if bucket:
            count = len(bucket)
            sum_conf = sum(c for c, _ in bucket)
            sum_correct = sum(corr for _, corr in bucket)
            avg_conf = sum_conf / count
            accuracy = sum_correct / count
            ece += abs(accuracy - avg_conf) * (count / total)
        else:
            count = 0
            avg_conf = None
            accuracy = None
        bin_infos.append(
            {
                "bin_lower": round(low, 3),
                "bin_upper": round(high, 3),
                "count": count,
                "avg_confidence": round(avg_conf, 4) if avg_conf is not None else None,
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
            }
        )

    brier = sum((conf - correct) ** 2 for conf, correct in records) / total
    return {
        "total": total,
        "bins": bin_infos,
        "brier_score": round(brier, 6),
        "expected_calibration_error": round(ece, 6),
        "bin_count": bins,
    }


def _write_metrics_csv(path: Path, report: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []

    counts = report.get("counts", {})
    if counts:
        count_row = {"category": "summary", "name": "counts"}
        count_row.update(counts)
        rows.append(count_row)

    def add_metric_row(name: str, key: str) -> None:
        data = report.get(key) or {}
        cm = data.get("confusion_matrix") or {}
        metrics = data.get("metrics") or {}
        if not metrics:
            return
        row = {
            "category": "metric",
            "name": name,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "recall_pos": metrics.get("recall_pos"),
            "recall_neg": metrics.get("recall_neg"),
            "support": metrics.get("support"),
            "TP": cm.get("TP"),
            "FP": cm.get("FP"),
            "FN": cm.get("FN"),
            "TN": cm.get("TN"),
        }
        rows.append(row)

    add_metric_row("judge_filtered", "judge_metrics")
    add_metric_row("judge_all", "judge_metrics_all")
    add_metric_row("visual_veracity_filtered", "visual_veracity_ai_detection")
    add_metric_row("visual_veracity_all", "visual_veracity_ai_detection_all")

    calibration = report.get("judge_confidence_calibration") or {}
    if calibration:
        rows.append(
            {
                "category": "calibration_summary",
                "name": "judge_confidence",
                "brier_score": calibration.get("brier_score"),
                "expected_calibration_error": calibration.get("expected_calibration_error"),
                "samples": calibration.get("total"),
            }
        )
        for idx, bin_info in enumerate(calibration.get("bins", [])):
            row = {
                "category": "calibration_bin",
                "name": f"bin_{idx}",
                "bin_lower": bin_info.get("bin_lower"),
                "bin_upper": bin_info.get("bin_upper"),
                "count": bin_info.get("count"),
                "avg_confidence": bin_info.get("avg_confidence"),
                "accuracy": bin_info.get("accuracy"),
            }
            rows.append(row)

    logprob_cal = report.get("judge_logprob_calibration") or {}
    if logprob_cal:
        rows.append(
            {
                "category": "calibration_summary",
                "name": "judge_logprob_confidence",
                "brier_score": logprob_cal.get("brier_score"),
                "expected_calibration_error": logprob_cal.get("expected_calibration_error"),
                "samples": logprob_cal.get("total"),
            }
        )
        for idx, bin_info in enumerate(logprob_cal.get("bins", [])):
            row = {
                "category": "calibration_bin",
                "name": f"logprob_bin_{idx}",
                "bin_lower": bin_info.get("bin_lower"),
                "bin_upper": bin_info.get("bin_upper"),
                "count": bin_info.get("count"),
                "avg_confidence": bin_info.get("avg_confidence"),
                "accuracy": bin_info.get("accuracy"),
            }
            rows.append(row)

    if not rows:
        raise ValueError("No data available to write CSV metrics")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_reliability_plot(calibration: Dict[str, Any], title: str, path: Path) -> None:
    if not calibration or not calibration.get("total"):
        return

    bins = calibration.get("bins") or []
    if not bins:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Warning: unable to import matplotlib for '{title}' plot: {exc}")
        return

    centers: List[float] = []
    accuracies: List[float] = []
    confidences: List[float] = []
    for bin_info in bins:
        low = bin_info.get("bin_lower")
        high = bin_info.get("bin_upper")
        if low is None or high is None:
            continue
        center = (float(low) + float(high)) / 2.0
        centers.append(center)
        accuracies.append(float(bin_info.get("accuracy") or 0.0))
        confidences.append(float(bin_info.get("avg_confidence") or 0.0))

    if not centers:
        return

    width = 1.0 / max(1, len(centers))
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.bar(centers, confidences, width=width, alpha=0.6, label="Avg confidence")
    plt.bar(centers, accuracies, width=width, alpha=0.6, label="Accuracy")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence bin")
    plt.ylabel("Rate")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate(
    outputs_path: Path,
    dataset_json: Path,
    image_root: Path,
    save_report: Optional[Path] = None,
    save_csv: Optional[Path] = None,
    save_calibration_dir: Optional[Path] = None,
) -> Dict[str, Any]:
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
    confidence_records: List[Tuple[float, int]] = []
    logprob_records: List[Tuple[float, int]] = []
    for obj in preds:
        img_path = str(obj.get("image_path", ""))
        # Try flexible matching
        gt = None
        for key in resolve_image_candidates(image_root, img_path):
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
            conf_val = j.get("confidence")
            if isinstance(conf_val, (int, float)):
                conf = max(0.0, min(1.0, float(conf_val)))
                confidence_records.append((conf, 1 if pred == y_true else 0))
            lp_val = j.get("logprob_confidence")
            if isinstance(lp_val, (int, float)):
                lp_conf = max(0.0, min(1.0, float(lp_val)))
                logprob_records.append((lp_conf, 1 if pred == y_true else 0))

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

    calibration = _calibration_from_records(confidence_records)
    report["judge_confidence_calibration"] = calibration
    report["counts"]["judge_confidence_samples"] = calibration.get("total", 0)

    logprob_calibration = _calibration_from_records(logprob_records)
    report["judge_logprob_calibration"] = logprob_calibration
    report["counts"]["judge_logprob_samples"] = logprob_calibration.get("total", 0)

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

    if calibration.get("total"):
        print(
            "\nJudge confidence calibration: "
            f"samples={calibration['total']} "
            f"brier={calibration['brier_score']} "
            f"ece={calibration['expected_calibration_error']}"
        )
    else:
        print("\nJudge confidence calibration: no confident predictions available")

    if logprob_calibration.get("total"):
        print(
            "  Logprob-derived confidence calibration: "
            f"samples={logprob_calibration['total']} "
            f"brier={logprob_calibration['brier_score']} "
            f"ece={logprob_calibration['expected_calibration_error']}"
        )

    if save_report is not None:
        try:
            with open(save_report, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\nReport saved to {save_report}")
        except Exception as e:
            print("\nWarning: failed to save report:", e)

    if save_csv is not None:
        try:
            _write_metrics_csv(save_csv, report)
            print(f"Metrics CSV saved to {save_csv}")
        except Exception as e:
            print("Warning: failed to save CSV metrics:", e)

    if save_calibration_dir is not None:
        try:
            save_calibration_dir.mkdir(parents=True, exist_ok=True)
            _save_reliability_plot(
                calibration,
                "Self-reported confidence reliability",
                save_calibration_dir / "judge_confidence.png",
            )
            _save_reliability_plot(
                logprob_calibration,
                "Logprob-derived confidence reliability",
                save_calibration_dir / "judge_logprob.png",
            )
            print(f"Calibration plots saved to {save_calibration_dir}")
        except Exception as e:
            print("Warning: failed to save calibration plots:", e)

    return report


def main():
    p = argparse.ArgumentParser(description="Evaluate pipeline outputs vs MMFakeBench ground truth")
    p.add_argument("--outputs", type=str, required=True, help="Path to JSONL of final pipeline outputs")
    p.add_argument("--dataset-json", type=str, default=str(Path("data/MMFakeBench_test/MMFakeBench_test.json")), help="Path to dataset JSON")
    p.add_argument("--image-root", type=str, default=str(Path("data/MMFakeBench_test")), help="Image root to resolve dataset image paths")
    p.add_argument("--save-report", type=str, default="", help="Optional path to save metrics JSON report")
    p.add_argument("--save-csv", type=str, default="", help="Optional path to save metrics summary CSV")
    p.add_argument("--save-calibration-dir", type=str, default="", help="Optional directory to save reliability diagrams")
    args = p.parse_args()

    outputs = Path(args.outputs)
    dataset_json = Path(args.dataset_json)
    image_root = Path(args.image_root)
    save_report = Path(args.save_report) if args.save_report else None
    save_csv = Path(args.save_csv) if args.save_csv else None
    save_calibration_dir = Path(args.save_calibration_dir) if args.save_calibration_dir else None

    evaluate(outputs, dataset_json, image_root, save_report, save_csv, save_calibration_dir)


if __name__ == "__main__":
    main()
