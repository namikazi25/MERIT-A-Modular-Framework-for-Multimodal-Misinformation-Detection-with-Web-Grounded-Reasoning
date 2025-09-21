from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    return value


def build_args_snapshot(raw_args: Dict[str, Any], drop_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
    drop_keys = drop_keys or set()
    snapshot: Dict[str, Any] = {}
    for key, value in raw_args.items():
        if key in drop_keys:
            continue
        snapshot[key] = _normalize_value(value)
    return snapshot


def fingerprint_args(snapshot: Dict[str, Any]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CheckpointState:
    run_id: str
    processed_count: int
    chunk_size: int
    args_snapshot: Dict[str, Any]
    args_fingerprint: str
    jsonl_path: str
    html_report_path: Optional[str]
    created_at: str
    checkpoint_path: Path

    @property
    def next_index(self) -> int:
        return self.processed_count

    @classmethod
    def from_file(cls, path: Path) -> "CheckpointState":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            run_id=data["run_id"],
            processed_count=int(data.get("processed_count", 0)),
            chunk_size=int(data.get("chunk_size", 0) or 1),
            args_snapshot=data.get("args_snapshot", {}),
            args_fingerprint=data.get("args_fingerprint", ""),
            jsonl_path=data.get("jsonl_path", ""),
            html_report_path=data.get("html_report_path"),
            created_at=data.get("created_at", ""),
            checkpoint_path=path,
        )


class CheckpointManager:
    def __init__(
        self,
        *,
        run_id: str,
        checkpoint_dir: Path,
        chunk_size: int,
        args_snapshot: Dict[str, Any],
        jsonl_path: str,
        html_report_path: Optional[str] = None,
        resume_state: Optional[CheckpointState] = None,
    ) -> None:
        self.run_id = run_id
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.args_snapshot = args_snapshot
        self.args_fingerprint = fingerprint_args(args_snapshot)
        self.jsonl_path = jsonl_path
        self.html_report_path = html_report_path

        self._processed_count = 0
        self._last_path: Optional[Path] = None

        if resume_state:
            self._load_resume(resume_state)

        self.chunk_size = max(1, int(chunk_size))
        if resume_state and resume_state.chunk_size and resume_state.chunk_size != self.chunk_size:
            # Honour the resume state's chunk size to keep numbering consistent
            self.chunk_size = resume_state.chunk_size

    def _load_resume(self, state: CheckpointState) -> None:
        if state.run_id != self.run_id:
            raise ValueError(
                f"Checkpoint run_id '{state.run_id}' does not match expected '{self.run_id}'."
            )
        if state.args_fingerprint and state.args_fingerprint != self.args_fingerprint:
            raise ValueError(
                "Checkpoint arguments fingerprint does not match current run configuration."
            )
        self._processed_count = int(state.processed_count)
        if not state.args_snapshot:
            return
        # Preserve authoritative values from checkpoint for downstream consumers
        self.args_snapshot.update(state.args_snapshot)
        if state.jsonl_path:
            self.jsonl_path = state.jsonl_path
        if state.html_report_path is not None:
            self.html_report_path = state.html_report_path

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def next_index(self) -> int:
        return self._processed_count

    @property
    def last_checkpoint_path(self) -> Optional[Path]:
        return self._last_path

    def record_sample(self) -> None:
        self._processed_count += 1

    def sync_to(self, processed_count: int) -> None:
        if processed_count < self._processed_count:
            return
        self._processed_count = processed_count

    def maybe_save(self, *, force: bool = False) -> Optional[Path]:
        if not force and (self._processed_count % self.chunk_size != 0):
            return None
        path = self._write_checkpoint()
        self._last_path = path
        return path

    def _write_checkpoint(self) -> Path:
        filename = f"{self.run_id}.upto{self._processed_count:05d}.json"
        path = self.checkpoint_dir / filename
        payload = {
            "run_id": self.run_id,
            "processed_count": self._processed_count,
            "next_index": self.next_index,
            "chunk_size": self.chunk_size,
            "jsonl_path": self.jsonl_path,
            "html_report_path": self.html_report_path,
            "created_at": _now_iso(),
            "args_snapshot": self.args_snapshot,
            "args_fingerprint": self.args_fingerprint,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def resume_hint(self, checkpoint_path: Optional[Path] = None) -> str:
        path = checkpoint_path or self._last_path
        if not path:
            return ""
        return f"--resume {path}"
