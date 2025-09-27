from __future__ import annotations

"""Batch DuckDuckGo (DDGS) search helper with light rate limiting and retries."""

import hashlib
import os
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # Prefer the modern package name first
    from ddgs import DDGS  # type: ignore
except ImportError:  # pragma: no cover - fall back to the legacy module
    try:
        from duckduckgo_search import DDGS  # type: ignore  # pragma: no cover
    except ImportError:  # pragma: no cover
        DDGS = None  # type: ignore


_DDG_PARAM_KEYS = ("region", "safesearch", "timelimit", "max_results", "page", "backend")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class _Job:
    job_id: str
    query: str
    params: Dict[str, Any]
    key: str


class _RateLimiter:
    """Per-provider limiter enforcing minimum spacing between call starts."""

    def __init__(self, interval: float, jitter_pct: float, seed: Optional[int]) -> None:
        self._interval = max(0.0, float(interval))
        self._jitter = max(0.0, float(jitter_pct))
        self._lock = threading.Lock()
        self._next_ready = 0.0
        self._random = random.Random(seed) if seed is not None else random.Random()

    def acquire(self) -> None:
        if self._interval <= 0.0:
            return

        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._next_ready - now
                if wait <= 0:
                    # Randomize the spacing slightly to avoid periodic bursts
                    applied = self._interval
                    if self._jitter:
                        factor = 1.0 + self._random.uniform(-self._jitter, self._jitter)
                        applied = max(0.0, self._interval * factor)
                    self._next_ready = now + applied
                    return
            time.sleep(wait if wait > 0 else self._interval * 0.5)


def _normalize_results(raw_results: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("href") or item.get("url") or "").strip()
        if not url:
            continue
        normalized.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "description": str(
                    item.get("body") or item.get("description") or item.get("snippet") or ""
                ).strip(),
                "source": str(item.get("source") or "").strip(),
            }
        )
    return normalized


class DuckDuckGoBatcher:
    """Queue DuckDuckGo text queries and execute them with limited parallelism."""

    def __init__(
        self,
        *,
        max_workers: int = 2,
        min_interval: float = 1.8,
        jitter_pct: float = 0.25,
        retries: int = 2,
        backoff: float = 1.8,
        base_retry_delay: float = 1.0,
        job_deadline: float = 35.0,
        timeout: float = 12.0,
        proxy: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if DDGS is None:
            raise ImportError(
                "duckduckgo-search dependency is not installed. Add 'duckduckgo-search' to requirements.txt "
                "and install it to use the DuckDuckGo provider."
            )

        self._pool = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._rate_limiter = _RateLimiter(min_interval, jitter_pct, seed)
        self._retries = max(0, int(retries))
        self._backoff = max(1.0, float(backoff))
        self._base_retry_delay = max(0.0, float(base_retry_delay))
        self._job_deadline = max(1.0, float(job_deadline))
        self._timeout = max(1.0, float(timeout))
        self._proxy = proxy

        self._lock = threading.Lock()
        self._pending: Dict[str, _Job] = {}
        self._dedup: Dict[str, List[str]] = {}
        self._representatives: Dict[str, _Job] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._closed = False

        self._metrics_lock = threading.Lock()
        self._reset_metrics_unlocked()

    def _reset_metrics_unlocked(self) -> None:
        self._metrics: Dict[str, Any] = {
            "total_batches": 0,
            "total_enqueued": 0,
            "total_unique": 0,
            "total_unique_executed": 0,
            "total_duplicates": 0,
            "total_cache_hits": 0,
            "total_success": 0,
            "total_errors": 0,
            "total_retries": 0,
            "total_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "last_batch": {},
        }

    @classmethod
    def from_env(cls) -> "DuckDuckGoBatcher":
        return cls(
            max_workers=_env_int("DUCKDUCKGO_MAX_CONCURRENCY", 2),
            min_interval=_env_float("DUCKDUCKGO_MIN_INTERVAL", 1.8),
            jitter_pct=_env_float("DUCKDUCKGO_JITTER_PCT", 0.25),
            retries=_env_int("DUCKDUCKGO_RETRIES", 2),
            backoff=_env_float("DUCKDUCKGO_BACKOFF", 1.8),
            base_retry_delay=_env_float("DUCKDUCKGO_RETRY_DELAY", 1.0),
            job_deadline=_env_float("DUCKDUCKGO_JOB_DEADLINE", 35.0),
            timeout=_env_float("DUCKDUCKGO_TIMEOUT", 12.0),
            proxy=os.getenv("DDGS_PROXY"),
            seed=_env_int("DUCKDUCKGO_JITTER_SEED", 1234),
        )

    def enqueue(self, query: str, params: Optional[Dict[str, Any]] = None, *, job_id: Optional[str] = None) -> str:
        if self._closed:
            raise RuntimeError("Batcher is closed")

        q = str(query or "").strip()
        if not q:
            raise ValueError("Query must be a non-empty string")

        params = dict(params or {})
        job_id = job_id or hashlib.sha1(q.encode("utf-8")).hexdigest()[:16]
        key = self._make_key(q, params)

        job = _Job(job_id=job_id, query=q, params=params, key=key)

        with self._lock:
            self._pending[job_id] = job
            self._dedup.setdefault(key, []).append(job_id)
            self._representatives.setdefault(key, job)

        with self._metrics_lock:
            self._metrics["total_enqueued"] += 1

        return job_id

    def execute(self) -> Dict[str, Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Batcher is closed")

        with self._lock:
            pending = self._pending
            dedup = self._dedup
            representatives = self._representatives
            self._pending = {}
            self._dedup = {}
            self._representatives = {}

        if not pending:
            return {}

        batch_metrics: Dict[str, Any] = {
            "timestamp": time.time(),
            "queued": len(pending),
            "unique": len(dedup),
            "unique_executed": 0,
            "duplicates": max(0, len(pending) - len(dedup)),
            "cache_hits": 0,
            "unique_success": 0,
            "unique_error": 0,
            "retry_count": 0,
            "duration_ms_total": 0.0,
            "duration_ms_max": 0.0,
        }

        results: Dict[str, Dict[str, Any]] = {
            job_id: {"query": job.query, "params": dict(job.params), "payload": None, "error": None}
            for job_id, job in pending.items()
        }

        futures: Dict[str, Future] = {}
        for key, job_ids in dedup.items():
            job = representatives[key]
            cached = self._cache.get(key)
            if cached is not None:
                for job_id in job_ids:
                    results[job_id]["payload"] = cached
                batch_metrics["cache_hits"] += len(job_ids)
                continue

            futures[key] = self._pool.submit(self._run_job, job)

        for key, future in futures.items():
            job_ids = dedup[key]
            try:
                payload, attempts, duration = future.result()
                self._cache[key] = payload
                for job_id in job_ids:
                    results[job_id]["payload"] = payload

                batch_metrics["unique_executed"] += 1
                batch_metrics["unique_success"] += 1
                retries = max(0, int(attempts) - 1)
                batch_metrics["retry_count"] += retries
                duration_ms = float(duration) * 1000.0
                batch_metrics["duration_ms_total"] += duration_ms
                if duration_ms > batch_metrics["duration_ms_max"]:
                    batch_metrics["duration_ms_max"] = duration_ms
            except Exception as exc:  # propagate error string per job
                attempts = int(getattr(exc, "attempts", 0))
                duration = float(getattr(exc, "duration", 0.0))
                batch_metrics["unique_executed"] += 1
                batch_metrics["unique_error"] += 1
                if attempts:
                    batch_metrics["retry_count"] += max(0, attempts - 1)
                if duration:
                    duration_ms = duration * 1000.0
                    batch_metrics["duration_ms_total"] += duration_ms
                    if duration_ms > batch_metrics["duration_ms_max"]:
                        batch_metrics["duration_ms_max"] = duration_ms
                for job_id in job_ids:
                    results[job_id]["error"] = exc

        executed = batch_metrics["unique_executed"]
        batch_metrics["avg_duration_ms"] = (
            batch_metrics["duration_ms_total"] / executed if executed else 0.0
        )

        with self._metrics_lock:
            self._metrics["total_batches"] += 1
            self._metrics["total_unique"] += batch_metrics["unique"]
            self._metrics["total_unique_executed"] += batch_metrics["unique_executed"]
            self._metrics["total_duplicates"] += batch_metrics["duplicates"]
            self._metrics["total_cache_hits"] += batch_metrics["cache_hits"]
            self._metrics["total_success"] += batch_metrics["unique_success"]
            self._metrics["total_errors"] += batch_metrics["unique_error"]
            self._metrics["total_retries"] += batch_metrics["retry_count"]
            self._metrics["total_duration_ms"] += batch_metrics["duration_ms_total"]
            if batch_metrics["duration_ms_max"] > self._metrics["max_duration_ms"]:
                self._metrics["max_duration_ms"] = batch_metrics["duration_ms_max"]
            self._metrics["last_batch"] = dict(batch_metrics)

        return results

    def get_metrics(self, *, reset: bool = False) -> Dict[str, Any]:
        with self._metrics_lock:
            snapshot = {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in self._metrics.items()
            }
            if reset:
                self._reset_metrics_unlocked()
            return snapshot

    def last_batch_metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            last = self._metrics.get("last_batch", {})
            return dict(last)

    def close(self) -> None:
        if not self._closed:
            self._pool.shutdown(wait=True)
            self._closed = True

    def __enter__(self) -> "DuckDuckGoBatcher":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        self.close()

    # Internal helpers -------------------------------------------------

    def _make_key(self, query: str, params: Dict[str, Any]) -> str:
        norm_query = " ".join(query.lower().split())
        key_parts = [norm_query]
        for k in _DDG_PARAM_KEYS:
            if k in params and params[k] is not None:
                key_parts.append(f"{k}={params[k]}")
        return hashlib.sha256("|".join(key_parts).encode("utf-8")).hexdigest()

    def _run_job(self, job: _Job) -> Tuple[Dict[str, Any], int, float]:
        deadline = time.monotonic() + self._job_deadline
        attempt = 0
        delay = 0.0
        start_time = time.perf_counter()

        while True:
            if delay:
                time.sleep(delay)
            if time.monotonic() > deadline:
                err = TimeoutError(f"DuckDuckGo query deadline exceeded for '{job.query}'")
                err.attempts = attempt
                err.duration = time.perf_counter() - start_time
                raise err

            self._rate_limiter.acquire()
            attempt += 1
            try:
                payload = self._execute_once(job)
                duration = time.perf_counter() - start_time
                return payload, attempt, duration
            except Exception as exc:  # retry on transient failures
                exc.attempts = attempt
                exc.duration = time.perf_counter() - start_time
                if attempt > self._retries:
                    raise
                delay = self._compute_retry_delay(attempt)
                continue

    def _execute_once(self, job: _Job) -> Dict[str, Any]:
        proxy = job.params.get("proxy", self._proxy)
        timeout = job.params.get("timeout", self._timeout)
        ddg_kwargs: Dict[str, Any] = {}
        for key in _DDG_PARAM_KEYS:
            if key in job.params and job.params[key] is not None:
                ddg_kwargs[key] = job.params[key]

        with DDGS(proxy=proxy, timeout=timeout) as ddgs:  # type: ignore
            raw_results = list(ddgs.text(job.query, **ddg_kwargs))

        normalized = _normalize_results(raw_results)

        payload: Dict[str, Any] = {
            "provider": "duckduckgo",
            "query": job.query,
            "results": normalized,
            "raw": raw_results,
            "params": {"proxy": proxy, **ddg_kwargs},
        }

        return payload

    def _compute_retry_delay(self, attempt: int) -> float:
        if self._base_retry_delay <= 0:
            return 0.0
        factor = self._backoff ** max(0, attempt - 1)
        return min(30.0, self._base_retry_delay * factor)


__all__ = ["DuckDuckGoBatcher"]
