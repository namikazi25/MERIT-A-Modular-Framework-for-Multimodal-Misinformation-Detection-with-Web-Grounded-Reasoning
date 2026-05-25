#!/usr/bin/env python3
"""Recompute evaluation metrics for existing result runs.

For each JSONL file under a results directory, this script runs the
standard evaluator to emit fresh ``.metrics.json``/``.metrics.csv`` files
and optionally refreshes the HTML report so the precision metric (and other
updates) are visible.

Example usage:

    python -m scripts.backfill_metrics \
        --results-dir results \
        --dataset-json data/MMFakeBench_test/MMFakeBench_test.json \
        --image-root data/MMFakeBench_test

Use ``--skip-existing`` to leave runs with a precision value untouched and
``--skip-html`` when only the metrics artifacts need regeneration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.evaluate import evaluate
from scripts.report_html import render_html_report


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    objs.append(obj)
    except FileNotFoundError:
        pass
    return objs


def _metrics_have_precision(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    candidates: List[Dict[str, Any]] = []
    for key in (
        "judge_metrics_overall",
        "judge_metrics",
        "visual_veracity_ai_detection",
        "visual_veracity_ai_detection_all",
    ):
        payload = data.get(key)
        if isinstance(payload, dict):
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                candidates.append(metrics)

    for metrics in candidates:
        if "precision" in metrics and metrics.get("precision") is not None:
            return True
    return False


def _load_run_summary(metadata_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    summary = data.get("run_summary")
    if isinstance(summary, dict):
        return summary
    return None


def _calibration_dir(base: Path) -> Path:
    stem = base.with_suffix("")
    return stem.with_suffix(".calibration")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute evaluation metrics for a set of runs")
    parser.add_argument("--results-dir", default="results", help="Directory containing run JSONL files")
    parser.add_argument(
        "--dataset-json",
        default=str(Path("data/MMFakeBench_test/MMFakeBench_test.json")),
        help="Dataset JSON file used for evaluation",
    )
    parser.add_argument(
        "--image-root",
        default=str(Path("data/MMFakeBench_test")),
        help="Root directory for dataset images",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if precision already exists")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose metrics already contain precision")
    parser.add_argument("--skip-html", action="store_true", help="Do not rewrite HTML reports")
    parser.add_argument("--calibration", action="store_true", help="Persist calibration plots alongside metrics")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    dataset_json = Path(args.dataset_json)
    image_root = Path(args.image_root)

    if not results_dir.exists():
        sys.exit(f"Results directory not found: {results_dir}")

    if not dataset_json.exists():
        sys.exit(f"Dataset JSON not found: {dataset_json}")

    if not image_root.exists():
        sys.exit(f"Image root not found: {image_root}")

    jsonl_paths = sorted(results_dir.glob("*.jsonl"))
    if not jsonl_paths:
        print(f"No JSONL files found under {results_dir}")
        return

    processed = 0
    for jsonl_path in jsonl_paths:
        metrics_json_path = jsonl_path.with_suffix(".metrics.json")
        metrics_csv_path = jsonl_path.with_suffix(".metrics.csv")
        html_path = jsonl_path.with_suffix(".html")
        metadata_path = jsonl_path.with_suffix(".metadata.json")

        if args.skip_existing and _metrics_have_precision(metrics_json_path):
            print(f"[skip] {jsonl_path.name} already has precision")
            continue

        if not args.force and not args.skip_existing and _metrics_have_precision(metrics_json_path):
            print(f"[info] {jsonl_path.name} already contains precision – use --force to recompute")
            continue

        print(f"[run] Recomputing metrics for {jsonl_path.name}")
        calibration_dir = _calibration_dir(jsonl_path) if args.calibration else None

        metrics = evaluate(
            jsonl_path,
            dataset_json,
            image_root,
            save_report=metrics_json_path,
            save_csv=metrics_csv_path,
            save_calibration_dir=calibration_dir,
        )

        if not args.skip_html and html_path.exists():
            objs = _load_jsonl(jsonl_path)
            run_summary = _load_run_summary(metadata_path)
            render_html_report(
                objs,
                metrics,
                str(html_path),
                dataset_json_path=str(dataset_json),
                dataset_image_root=str(image_root),
                run_summary=run_summary,
            )
            print(f"[ok] Updated {html_path.name}")

        processed += 1

    print(f"Processed {processed} run(s)")


if __name__ == "__main__":
    main()
