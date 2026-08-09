"""Task A.3: combine 200 + 300 -> 500; split 350 dev / 150 holdout, stratified on fake_cls; seal holdout.

Also pins the 100-sample dev core (Task B.3 / C / D / E).

Usage:
  python -m scripts.judge_study.split_500 --out manifests/split-500-v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.getcwd())

from scripts.judge_study.manifest import git_state

PROPS = {"mismatch": 0.30, "original": 0.30,
         "textual_veracity_distortion": 0.30, "visual_veracity_distortion": 0.10}


def norm(p: str) -> str:
    return p.replace("data/MMFakeBench_test", "").replace("\\", "/")


def load_bundle_ids(evidence_dir: str) -> dict:
    """image_path -> {sample_id, fake_cls, gt, bundle_file}"""
    out = {}
    for fn in sorted(os.listdir(evidence_dir)):
        if not fn.endswith(".json") or fn in ("manifest.json", "refetch_manifest.json"):
            continue
        b = json.load(open(os.path.join(evidence_dir, fn)))
        sd = b.get("sample_details") or {}
        ip = norm(b.get("image_path", ""))
        out[ip] = {"sample_id": b.get("sample_id"), "image_path": ip,
                   "fake_cls": sd.get("fake_cls"), "gt": sd.get("gt_answers"),
                   "bundle": os.path.join(evidence_dir, fn), "source": "unknown"}
    return out


def stratified_split(ids, n_dev, n_hold, seed):
    rng = random.Random(seed)
    dev, hold = [], []
    for cls in PROPS:
        pool = [i for i in ids if i["fake_cls"] == cls]
        rng.shuffle(pool)
        n = len(pool)
        n_hold_cls = round(n * n_hold / (n_dev + n_hold))
        hold.extend(pool[:n_hold_cls])
        dev.extend(pool[n_hold_cls:])
    rng.shuffle(dev); rng.shuffle(hold)
    return dev, hold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-evidence", default="evidence/live200-v2")
    ap.add_argument("--new-evidence", default="evidence/live300-v2")
    ap.add_argument("--out", default="manifests/split-500-v1.json")
    ap.add_argument("--dev-core-out", default="manifests/devcore-100-v1.json")
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--dev", type=int, default=350)
    ap.add_argument("--hold", type=int, default=150)
    ap.add_argument("--repo-root", default=os.getcwd())
    args = ap.parse_args()

    old = load_bundle_ids(args.old_evidence)
    new = load_bundle_ids(args.new_evidence)
    for v in old.values():
        v["source"] = "old200"
    for v in new.values():
        v["source"] = "new300"
    # sample_details may be absent in fresh gathers -> pull fake_cls/gt from the dataset
    if not all(x["fake_cls"] for x in old.values()) or not all(x["fake_cls"] for x in new.values()):
        recs = json.load(open("data/MMFakeBench_test/source/MMFakeBench_test.json"))
        for r in recs:
            ip = norm(r["image_path"])
            if ip in old and not old[ip]["fake_cls"]:
                old[ip]["fake_cls"] = r["fake_cls"]; old[ip]["gt"] = r["gt_answers"]
            if ip in new and not new[ip]["fake_cls"]:
                new[ip]["fake_cls"] = r["fake_cls"]; new[ip]["gt"] = r["gt_answers"]

    assert not (set(old) & set(new)), "overlap between old and new!"
    ids = list(old.values()) + list(new.values())
    print(f"total: {len(ids)} (old {len(old)} + new {len(new)})")

    dev, hold = stratified_split(ids, args.dev, args.hold, args.seed)
    dev_paths = {d["image_path"] for d in dev}
    hold_paths = {h["image_path"] for h in hold}
    assert not (dev_paths & hold_paths), "dev/holdout overlap!"
    print(f"dev {len(dev)} / holdout {len(hold)}; overlap: {len(dev_paths & hold_paths)} (verified 0)")
    from collections import Counter
    print("dev strata:", dict(Counter(d["fake_cls"] for d in dev)))
    print("hold strata:", dict(Counter(h["fake_cls"] for h in hold)))

    dev_ids = [d["image_path"] for d in dev]
    hold_ids = [h["image_path"] for h in hold]
    split = {
        "task": "A.3", "n_total": len(ids), "n_dev": len(dev), "n_hold": len(hold),
        "seed": args.seed, "strata": PROPS,
        "created": datetime.now(timezone.utc).isoformat(),
        "git": git_state(args.repo_root),
        "dev_list_hash": hashlib.sha256("\n".join(sorted(dev_ids)).encode()).hexdigest(),
        "holdout_list_hash": hashlib.sha256("\n".join(sorted(hold_ids)).encode()).hexdigest(),
        "holdout_sealed": True,
        "dev": [{"image_path": d["image_path"], "sample_id": d["sample_id"], "fake_cls": d["fake_cls"],
                 "gt": d["gt"], "source": d["source"], "bundle": d["bundle"]} for d in dev],
        "holdout": [{"image_path": h["image_path"], "sample_id": h["sample_id"], "fake_cls": h["fake_cls"],
                     "gt": h["gt"], "source": h["source"], "bundle": h["bundle"]} for h in hold],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(split, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("split sealed:", args.out)

    # dev core: 100 stratified from dev (pinned, used by B/C/D/E)
    rng = random.Random(args.seed + 1)
    core = []
    for cls in PROPS:
        pool = [d for d in dev if d["fake_cls"] == cls]
        rng.shuffle(pool)
        core.extend(pool[:round(100 * PROPS[cls])])
    core_ids = [c["image_path"] for c in core]
    core_manifest = {
        "task": "B.3-devcore", "n": len(core), "seed": args.seed + 1,
        "parent_split": args.out,
        "devcore_list_hash": hashlib.sha256("\n".join(sorted(core_ids)).encode()).hexdigest(),
        "created": datetime.now(timezone.utc).isoformat(),
        "git": git_state(args.repo_root),
        "dev_core": [{"image_path": c["image_path"], "sample_id": c["sample_id"],
                      "fake_cls": c["fake_cls"], "gt": c["gt"]} for c in core],
    }
    json.dump(core_manifest, open(args.dev_core_out, "w"), indent=1, ensure_ascii=False)
    print("dev core sealed:", args.dev_core_out, len(core))


if __name__ == "__main__":
    main()
