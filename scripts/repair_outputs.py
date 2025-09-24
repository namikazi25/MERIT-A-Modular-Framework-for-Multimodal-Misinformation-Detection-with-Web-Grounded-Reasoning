#!/usr/bin/env python3
"""Repair pipeline JSONL outputs by reattaching MMFakeBench ground truth metadata.

This utility aligns each JSON line object with the source dataset based on
dataset index, image path, or headline text and ensures the following fields
are present and accurate:
  - dataset_order_index (true dataset index)
  - sample_details.{dataset_index, gt_answers, fake_cls, text_source, image_source}

It rewrites the JSONL file in-place while keeping a timestamped backup unless
`--dry-run` is specified.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_dataset_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "records", "samples"):
            val = data.get(key)
            if isinstance(val, list):
                return val
        for val in data.values():
            if isinstance(val, list):
                return val
    raise ValueError(f"Unsupported JSON structure in dataset file: {path}")


def _normalize(path: Optional[str], image_root: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    p = Path(str(path))
    if not p.is_absolute() and image_root:
        p = image_root / str(path).lstrip("/\\")
    try:
        return str(p.resolve())
    except Exception:
        return str(p.expanduser().absolute())


def _parse_index(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _record_matches(
    obj: Dict[str, Any],
    rec: Dict[str, Any],
    image_root: Optional[Path],
) -> bool:
    headline_obj = obj.get("headline")
    headline_rec = rec.get("text")
    if isinstance(headline_obj, str) and isinstance(headline_rec, str):
        if headline_obj.strip() == headline_rec.strip():
            return True

    img_obj = _normalize(obj.get("image_path"), image_root)
    img_rec = _normalize(rec.get("image_path"), image_root)
    if img_obj and img_rec and img_obj == img_rec:
        return True
    return False


def _pick_dataset_record(
    obj: Dict[str, Any],
    idx_map: Dict[int, Dict[str, Any]],
    img_map: Dict[str, Tuple[int, Dict[str, Any]]],
    text_map: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    image_root: Optional[Path],
) -> Optional[Tuple[int, Dict[str, Any], str]]:
    sources: List[str] = []

    idx = _parse_index(obj.get("dataset_order_index"))
    if idx is not None and idx in idx_map:
        rec = idx_map[idx]
        if _record_matches(obj, rec, image_root):
            return idx, rec, "dataset_order_index"
        sources.append("dataset_order_index_mismatch")
    else:
        sources.append("dataset_order_index")

    sample_details = obj.get("sample_details")
    if isinstance(sample_details, dict):
        idx = _parse_index(sample_details.get("dataset_index"))
        if idx is not None and idx in idx_map:
            rec = idx_map[idx]
            if _record_matches(obj, rec, image_root):
                return idx, rec, "sample_details.dataset_index"
        sources.append("sample_details.dataset_index")

    img_norm = _normalize(obj.get("image_path"), image_root)
    if img_norm and img_norm in img_map:
        ds_idx, rec = img_map[img_norm]
        return ds_idx, rec, "image_path"
    sources.append("image_path")

    headline = obj.get("headline")
    if isinstance(headline, str):
        candidates = text_map.get(headline.strip()) or []
        if len(candidates) == 1:
            ds_idx, rec = candidates[0]
            return ds_idx, rec, "headline"
        if len(candidates) > 1:
            # Ambiguous match; return first but flag via source label.
            ds_idx, rec = candidates[0]
            return ds_idx, rec, "headline*"
    sources.append("headline")

    return None


def _update_sample(obj: Dict[str, Any], idx: int, rec: Dict[str, Any]) -> None:
    obj["dataset_order_index"] = idx

    sd = obj.get("sample_details")
    if not isinstance(sd, dict):
        sd = {}
    sd.setdefault("dataset_index", idx)

    def _maybe_set(key: str) -> None:
        if key in rec and rec.get(key) is not None:
            sd[key] = rec.get(key)

    _maybe_set("gt_answers")
    _maybe_set("fake_cls")
    _maybe_set("text_source")
    _maybe_set("image_source")

    obj["sample_details"] = sd


def _rewrite_jsonl(path: Path, objects: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def repair_outputs(
    outputs_path: Path,
    dataset_json: Path,
    image_root: Optional[Path],
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    records = _load_dataset_records(dataset_json)
    idx_map = {idx: rec for idx, rec in enumerate(records)}

    img_map: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    text_map: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}

    img_root = image_root.resolve() if image_root else None

    for idx, rec in enumerate(records):
        img_norm = _normalize(rec.get("image_path"), img_root)
        if img_norm and img_norm not in img_map:
            img_map[img_norm] = (idx, rec)
        headline = rec.get("text")
        if isinstance(headline, str):
            text_map.setdefault(headline.strip(), []).append((idx, rec))

    repaired: List[Dict[str, Any]] = []
    unmatched: List[Tuple[int, str]] = []
    source_counts: Dict[str, int] = {}

    with outputs_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                unmatched.append((line_no, "json_decode_error"))
                continue

            result = _pick_dataset_record(obj, idx_map, img_map, text_map, img_root)
            if result is None:
                unmatched.append((line_no, "no_match"))
            else:
                ds_idx, rec, source = result
                _update_sample(obj, ds_idx, rec)
                source_counts[source] = source_counts.get(source, 0) + 1

            repaired.append(obj)

    stats = {
        "total": len(repaired),
        "matched": len(repaired) - len(unmatched),
        "unmatched": unmatched,
        "source_counts": source_counts,
    }

    if dry_run:
        return stats

    backup = outputs_path.with_suffix(outputs_path.suffix + f".{int(time.time())}.bak")
    outputs_path.replace(backup)
    _rewrite_jsonl(outputs_path, repaired)
    stats["backup"] = str(backup)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair pipeline outputs JSONL using MMFakeBench ground truth")
    parser.add_argument("--outputs", required=True, help="Path to the JSONL file produced by the pipeline")
    parser.add_argument("--dataset-json", required=True, help="Path to MMFakeBench JSON file")
    parser.add_argument("--image-root", default="data/MMFakeBench_test", help="Dataset image root (default: data/MMFakeBench_test)")
    parser.add_argument("--dry-run", action="store_true", help="Only report mismatches without rewriting the file")
    args = parser.parse_args()

    outputs_path = Path(args.outputs)
    dataset_json = Path(args.dataset_json)
    image_root = Path(args.image_root) if args.image_root else None

    stats = repair_outputs(outputs_path, dataset_json, image_root, dry_run=args.dry_run)

    print(f"Processed {stats['total']} objects; matched {stats['matched']}")
    if stats.get("backup"):
        print(f"Backup written to: {stats['backup']}")
    if stats.get("source_counts"):
        print("Match sources:")
        for source, count in sorted(stats["source_counts"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  {source}: {count}")
    if stats.get("unmatched"):
        print("Unmatched entries:")
        for line_no, reason in stats["unmatched"][:20]:
            print(f"  line {line_no}: {reason}")
        if len(stats["unmatched"]) > 20:
            remaining = len(stats["unmatched"]) - 20
            print(f"  ... and {remaining} more")


if __name__ == "__main__":
    main()
