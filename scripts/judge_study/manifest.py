
"""Run manifests (hard rule 3): every run writes a manifest before it starts.

Schema: model id + revision/quant, prompt file + SHA256, evidence version id,
seed, sample list hash, timestamp, git commit.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def git_state(repo_root: str) -> Dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True
            ).stdout.strip()
        )
        return {"commit": commit or None, "dirty": dirty}
    except Exception as exc:  # pragma: no cover
        return {"commit": None, "dirty": None, "error": str(exc)}


def make_manifest(
    *,
    phase: str,
    stage: str,
    run_id: str,
    model: Dict[str, Any],
    prompt_files: Iterable[str],
    evidence_version: str,
    seed: Optional[int],
    sample_ids: Iterable[str],
    temperature: float,
    repo_root: str,
    notes: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sample_ids = list(sample_ids)
    manifest: Dict[str, Any] = {
        "phase": phase,
        "stage": stage,
        "run_id": run_id,
        "model": model,
        "prompt_files": [
            {"path": p, "sha256": sha256_file(p)} for p in prompt_files if os.path.exists(p)
        ],
        "evidence_version": evidence_version,
        "seed": seed,
        "sample_list_hash": sha256_text("\n".join(str(s) for s in sample_ids)),
        "sample_count": len(sample_ids),
        "temperature": temperature,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_state(repo_root),
    }
    if notes:
        manifest["notes"] = notes
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: Dict[str, Any], path: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
