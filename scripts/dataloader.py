"""
PyTorch dataset and dataloader utilities for MMFakeBench.

Features
- Loads samples from JSON + images under a root folder.
- Returns dict with keys: text, image_path, text_source, image_source, gt_answers, fake_cls, image.
- Optional balancing by `gt_answers` via undersampling. Falls back to random if not possible.
- Optional balancing by `fake_cls` (original/textual/visual/mismatch) with configurable totals.
- Deterministic shuffling via local RNG and seed.
- Works with torchvision-style image transforms that return tensors.

Usage
    from scripts.dataloader import MMFakeBenchDataset, build_torch_dataloader

    ds = MMFakeBenchDataset(
        json_path="data/MMFakeBench_test/source/MMFakeBench_test.json",
        image_root="data/MMFakeBench_test",
        balanced=True,
        seed=42,
        image_transform=None,  # e.g., torchvision transforms.ToTensor()
    )

    # For batching and parallel loading, ensure transform returns tensors
    # and then rely on PyTorch's default collation.
    loader = build_torch_dataloader(ds, batch_size=8, num_workers=2, pin_memory=True)

    for batch in loader:
        # batch is a dict of tensors/lists
        ...
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from PIL import Image

from scripts.utils.dataset import load_dataset_records


def _normalize_class_key(value: Any) -> str:
    # Convert gt_answers into a stable string key for balancing buckets.
    if isinstance(value, (list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


@dataclass
class _KeyMap:
    text: str = "text"
    image: str = "image_path"
    text_source: str = "text_source"
    image_source: str = "image_source"
    gt: str = "gt_answers"
    fake: str = "fake_cls"


class MMFakeBenchDataset:
    """PyTorch-friendly dataset for MMFakeBench.

    Parameters
    - json_path: Path to JSON file with records.
    - image_root: Root folder where image paths in JSON are resolved.
    - balanced: If True, undersample to balance classes by `gt_answers`.
    - balance_mode: Currently supports 'undersample'. If not feasible, falls back to unbalanced.
    - balance_fake_cls: If True, undersample to balance classes equally by `fake_cls`.
    - fake_cls_balance_total: Optional total sample budget (must be divisible by number of fake_cls labels).
      When omitted, the dataset uses the minimum available count across classes.
    - fake_cls_labels: Override the ordered list of fake classes to balance (defaults to
      ["original", "textual_veracity_distortion", "visual_veracity_distortion", "mismatch"]).
    - limit: Optional cap on number of samples after balancing/shuffling.
    - seed: RNG seed for deterministic shuffling and balancing.
    - skip_missing: If True, drops records with missing images. Else raises.
    - image_transform: Optional callable to transform PIL image (e.g., ToTensor()).
    - return_image: If False, do not load/return image (keeps metadata only).
    - keys: Optional mapping to override expected JSON field names.
    - verbose: If True, prints basic dataset stats.
    """

    def __init__(
        self,
        json_path: str | os.PathLike,
        image_root: str | os.PathLike,
        *,
        balanced: bool = False,
        balance_mode: str = "undersample",
        balance_fake_cls: bool = False,
        fake_cls_balance_total: Optional[int] = None,
        fake_cls_labels: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        seed: int = 42,
        skip_missing: bool = True,
        image_transform: Optional[Callable[[Image.Image], Any]] = None,
        return_image: bool = True,
        keys: Optional[Dict[str, str]] = None,
        verbose: bool = False,
    ) -> None:
        self.json_path = Path(json_path)
        self.image_root = Path(image_root)
        self.balanced = balanced
        self.balance_mode = balance_mode
        self.balance_fake_cls = balance_fake_cls
        self.fake_cls_balance_total = fake_cls_balance_total
        default_labels = [
            "original",
            "textual_veracity_distortion",
            "visual_veracity_distortion",
            "mismatch",
        ]
        if fake_cls_labels is None:
            self.fake_cls_labels = default_labels
        else:
            labels_list = [str(label) for label in fake_cls_labels]
            self.fake_cls_labels = labels_list if labels_list else default_labels
        self.limit = limit
        self.seed = seed
        self.skip_missing = skip_missing
        self.image_transform = image_transform
        self.return_image = return_image
        self.verbose = verbose

        km_dict = keys or {}
        self.km = _KeyMap(
            text=km_dict.get("text", _KeyMap.text),
            image=km_dict.get("image", _KeyMap.image),
            text_source=km_dict.get("text_source", _KeyMap.text_source),
            image_source=km_dict.get("image_source", _KeyMap.image_source),
            gt=km_dict.get("gt", _KeyMap.gt),
            fake=km_dict.get("fake", _KeyMap.fake),
        )

        self._rng = random.Random(self.seed)

        # Load and index base records
        records = load_dataset_records(self.json_path)
        self._records: List[Dict[str, Any]] = []
        self._missing_count = 0

        for original_idx, rec in enumerate(records):
            try:
                img_rel = rec[self.km.image]
                # Resolve image path relative to image_root. Some datasets include a leading
                # slash (e.g., "/real/...") even though paths are intended to be relative.
                raw_path = Path(str(img_rel))
                if raw_path.is_absolute():
                    # Try rebasing absolute-looking paths under image_root by stripping leading '/'
                    rebased = self.image_root / str(raw_path).lstrip(os.sep)
                    img_path = rebased if rebased.exists() else raw_path
                else:
                    img_path = self.image_root / raw_path

                if not img_path.exists():
                    if self.skip_missing:
                        self._missing_count += 1
                        continue
                    else:
                        raise FileNotFoundError(f"Image not found: {img_path}")

                # Build normalized record used at runtime
                self._records.append(
                    {
                        "text": rec.get(self.km.text, None),
                        "image_path": str(img_path),
                        "text_source": rec.get(self.km.text_source, None),
                        "image_source": rec.get(self.km.image_source, None),
                        "gt_answers": rec.get(self.km.gt, None),
                        "fake_cls": rec.get(self.km.fake, None),
                        "dataset_index": original_idx,
                    }
                )
            except KeyError as e:
                if self.skip_missing:
                    continue
                else:
                    raise KeyError(f"Missing required key in record: {e}") from e

        if not self._records:
            raise RuntimeError("No valid records found after loading/validation.")

        # Build sampling order (indices into _records)
        self._indices: List[int] = list(range(len(self._records)))
        self._limit_consumed_by_fake_balance = False

        if self.balance_fake_cls:
            self._apply_fake_cls_balance()
        elif self.balanced and self.balance_mode == "undersample":
            buckets: Dict[str, List[int]] = {}
            for idx, rec in enumerate(self._records):
                key = _normalize_class_key(rec["gt_answers"])  # can be str/list/etc.
                buckets.setdefault(key, []).append(idx)

            if len(buckets) >= 2:
                min_count = min(len(v) for v in buckets.values())
                if min_count > 0:
                    selected: List[int] = []
                    for v in buckets.values():
                        picked = list(v)
                        self._rng.shuffle(picked)
                        selected.extend(picked[:min_count])
                    self._rng.shuffle(selected)
                    self._indices = selected
                else:
                    # Fall back to random unbalanced
                    self._rng.shuffle(self._indices)
            else:
                # Not enough classes to balance; fall back to random
                self._rng.shuffle(self._indices)
        else:
            # Unbalanced order; shuffle deterministically
            self._rng.shuffle(self._indices)

        if self.limit is not None and not self._limit_consumed_by_fake_balance:
            self._indices = self._indices[: int(self.limit)]

        if self.verbose:
            total = len(self._records)
            kept = len(self._indices)
            msg = (
                f"Loaded records: {total}, kept (post-balance/limit): {kept}, "
                f"missing images skipped: {self._missing_count}"
            )
            print(msg)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = self._indices[idx]
        rec = self._records[base_idx]

        out: Dict[str, Any] = {
            "text": rec["text"],
            "image_path": rec["image_path"],
            "text_source": rec["text_source"],
            "image_source": rec.get("image_source"),
            "gt_answers": rec["gt_answers"],
            "fake_cls": rec["fake_cls"],
            "dataset_index": rec.get("dataset_index"),
        }

        if self.return_image:
            img = Image.open(rec["image_path"]).convert("RGB")
            if self.image_transform is not None:
                img = self.image_transform(img)
            out["image"] = img

        return out

    def _apply_fake_cls_balance(self) -> None:
        buckets: Dict[str, List[int]] = {label: [] for label in self.fake_cls_labels}
        for idx, rec in enumerate(self._records):
            cls = rec.get("fake_cls")
            if cls in buckets:
                buckets[cls].append(idx)

        missing = [label for label, values in buckets.items() if not values]
        if missing:
            raise ValueError(
                "Fake class balancing requested but the dataset lacks samples for: "
                + ", ".join(missing)
            )

        target_total: Optional[int] = None
        if self.fake_cls_balance_total is not None:
            target_total = int(self.fake_cls_balance_total)
        elif self.limit is not None:
            target_total = int(self.limit)

        class_count = len(self.fake_cls_labels)
        per_class_target: int

        if target_total is not None:
            if target_total <= 0:
                raise ValueError("fake_cls_balance_total (or limit) must be > 0 when balancing fake_cls.")
            if target_total % class_count != 0:
                raise ValueError(
                    f"Requested total {target_total} is not divisible by the number of fake_cls buckets ({class_count})."
                )
            requested_per_class = target_total // class_count
            available_per_class = min(len(bucket) for bucket in buckets.values())
            per_class_target = min(requested_per_class, available_per_class)
            if per_class_target < requested_per_class and self.verbose:
                print(
                    "fake_cls balance: capped per-class samples at "
                    f"{per_class_target} due to limited data (requested {requested_per_class})."
                )
        else:
            per_class_target = min(len(bucket) for bucket in buckets.values())
            if per_class_target == 0:
                raise ValueError("Not enough samples per fake_cls bucket to perform balancing.")

        selected: List[int] = []
        for label in self.fake_cls_labels:
            indices = list(buckets[label])
            self._rng.shuffle(indices)
            selected.extend(indices[:per_class_target])

        self._rng.shuffle(selected)
        self._indices = selected
        self._limit_consumed_by_fake_balance = target_total is not None


def _default_worker_init_fn(worker_id: int) -> None:
    # Ensure per-worker deterministic behavior for Python, NumPy, and Torch
    try:
        import torch
        base_seed = torch.initial_seed() % 2**32
    except Exception:
        base_seed = random.randrange(1, 2**32 - 1)

    random.seed(base_seed + worker_id)
    try:
        import numpy as np  # type: ignore

        np.random.seed((base_seed + worker_id) % 2**32)
    except Exception:
        pass


def build_torch_dataloader(
    dataset: MMFakeBenchDataset,
    *,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
):
    """Construct a PyTorch DataLoader for the given dataset.

    Notes
    - The dataset already encapsulates shuffling/balancing order; we set shuffle=False.
    - If batching images, ensure `dataset.image_transform` returns tensors; otherwise
      default collation will fail on PIL Images.
    - Provides a default worker_init_fn to seed random states per worker deterministically.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError("PyTorch is required to build a DataLoader.") from e

    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,  # order is determined inside the dataset
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=_default_worker_init_fn,
    )

    if persistent_workers is not None:
        kwargs["persistent_workers"] = persistent_workers
    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn

    return DataLoader(dataset, **kwargs)
