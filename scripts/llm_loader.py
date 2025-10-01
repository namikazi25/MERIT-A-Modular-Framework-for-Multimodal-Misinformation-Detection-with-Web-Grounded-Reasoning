from __future__ import annotations

"""
Provider-native LLM loader for OpenAI (GPT-4o), Google (Gemini), DeepInfra (Gemma 3 via OpenAI-compatible API), and OpenRouter (community models via OpenAI-compatible API).

No LangChain dependency. Exposes a minimal interface compatible with
scripts.align_checker: `get_model().invoke(messages)`.

Environment variables:
  - OPENAI_API_KEY (OpenAI)
  - GOOGLE_API_KEY (Google)
  - DEEPINFRA_API_KEY (DeepInfra)
  - OPENROUTER_API_KEY (OpenRouter)
  - OPENROUTER_MODEL (preferred OpenRouter model when ALIGN_MODEL is empty)
  - OPENROUTER_RATE_LIMIT_PER_MIN (optional override for rpm throttling; default 18)
  - OPENROUTER_MIN_INTERVAL_SECONDS (optional minimum spacing between requests)
  - OPENROUTER_DAILY_QUOTA (optional hard cap per UTC day; default 45)
"""

import base64
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Literal, Optional, Union, List

from httpx import HTTPStatusError
from openai import RateLimitError


Provider = Literal["openai", "google", "deepinfra", "openrouter"]


@dataclass
class ModelConfig:
    provider: Provider
    model: Optional[str] = None  # Provider default applied if None
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    timeout: Optional[int] = None  # seconds (best-effort)
    system_prompt: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _require_env(name: str, value: Optional[str]) -> str:
    v = value or os.getenv(name)
    if not v:
        raise ValueError(f"Missing required API key: set {name} or pass via config.api_key")
    return v


def _extract_system_prompt(messages: List[Dict[str, Any]], default: Optional[str]) -> str:
    if default:
        return default
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, list):
                # concatenate text parts
                parts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
                return "\n".join([p for p in parts if p])
            if isinstance(c, str):
                return c
    return "You are a helpful assistant."


def _to_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif p.get("type") == "image_url":
                    img = p.get("image_url")
                    url = img.get("url") if isinstance(img, dict) else img
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": str(content) if content is not None else ""})
    return out


def _parse_data_url(data_url: str) -> tuple[str, bytes]:
    # data:<mime>;base64,<data>
    if not data_url.startswith("data:"):
        raise ValueError("Expected data URL for image")
    header, b64 = data_url.split(",", 1)
    mime = header.split(";")[0][5:]
    return mime, base64.b64decode(b64)


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        iv = int(value)
        return iv if iv > 0 else None
    except Exception:
        return None


def _coerce_positive_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        fv = float(value)
        return fv if fv > 0 else None
    except Exception:
        return None


def _messages_are_text_only(messages: List[Dict[str, Any]]) -> bool:
    """Return True when every message contains only plain text segments."""

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    return False
                p_type = part.get("type")
                if p_type and p_type != "text":
                    return False
        elif content is None:
            continue
        else:
            # Any non-list content is treated as text once coerced to str downstream
            continue
    return True


def _truthy_env(var_name: str) -> Optional[bool]:
    val = os.getenv(var_name)
    if val is None:
        return None
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _logprob_request_settings(messages: List[Dict[str, Any]], extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Determine whether we should request logprobs for this invocation."""

    if not _messages_are_text_only(messages):
        return None

    extra_dict = extra if isinstance(extra, dict) else {}

    enabled: Optional[bool]
    if "capture_logprobs" in extra_dict:
        enabled = bool(extra_dict.get("capture_logprobs"))
    else:
        enabled = _truthy_env("ALIGN_CAPTURE_LOGPROBS")
        if enabled is None:
            enabled = _truthy_env("MIRAGE_CAPTURE_LOGPROBS")

    if not enabled:
        return None

    top_logprobs: Optional[int]
    if "logprob_top_k" in extra_dict:
        try:
            top_logprobs = int(extra_dict.get("logprob_top_k"))
        except Exception:
            top_logprobs = None
    else:
        env_top = os.getenv("ALIGN_LOGPROB_TOP_K") or os.getenv("MIRAGE_LOGPROB_TOP_K")
        try:
            top_logprobs = int(env_top) if env_top else None
        except Exception:
            top_logprobs = None

    settings: Dict[str, Any] = {"logprobs": True}
    if top_logprobs is not None and top_logprobs > 0:
        settings["top_logprobs"] = int(top_logprobs)
    return settings


def _extract_logprob_payload(choice: Any) -> Optional[Dict[str, Any]]:
    """Convert OpenAI ChatCompletionChoice.logprobs into a serializable summary."""

    logprobs = getattr(choice, "logprobs", None)
    if not logprobs and isinstance(choice, dict):
        logprobs = choice.get("logprobs")
    if not logprobs:
        return None

    content = getattr(logprobs, "content", None)
    if content is None and isinstance(logprobs, dict):
        content = logprobs.get("content")
    if not content:
        return None

    total_lp = 0.0
    count = 0

    def _as_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return None

    for item in content:
        logprob = None
        if isinstance(item, dict):
            logprob = _as_float(item.get("logprob"))
        else:
            logprob = _as_float(getattr(item, "logprob", None))
        if logprob is not None:
            total_lp += logprob
            count += 1

    if count == 0:
        return None

    avg_logprob = total_lp / count
    try:
        geo_mean_prob = math.exp(avg_logprob)
    except Exception:
        geo_mean_prob = None

    return {
        "token_count": count,
        "avg_logprob": avg_logprob,
        "geo_mean_prob": geo_mean_prob,
    }


class _OpenRouterRateLimiter:
    """Simple in-process rate limiter to stay under OpenRouter free-tier limits."""

    def __init__(
        self,
        *,
        requests_per_minute: Optional[int] = None,
        min_interval_seconds: Optional[float] = None,
        daily_quota: Optional[int] = None,
    ) -> None:
        self.requests_per_minute = _coerce_positive_int(requests_per_minute)
        self.min_interval_seconds = _coerce_positive_float(min_interval_seconds)
        if self.requests_per_minute:
            derived = 60.0 / float(self.requests_per_minute)
            if self.min_interval_seconds is None or self.min_interval_seconds < derived:
                self.min_interval_seconds = derived
        self.daily_quota = _coerce_positive_int(daily_quota)
        self._configured_daily_quota = self.daily_quota

        self._lock = threading.Lock()
        self._timestamps = deque()  # monotonic seen within the last minute
        self._last_call: Optional[float] = None
        self._current_day = datetime.now(timezone.utc).date()
        self._daily_count = 0
        self._blocked_until_wall: Optional[float] = None
        self._configured_requests_per_minute = self.requests_per_minute

    @classmethod
    def from_settings(cls, overrides: Optional[Dict[str, Any]] = None) -> "_OpenRouterRateLimiter":
        overrides = overrides or {}

        rpm = _coerce_positive_int(
            overrides.get("per_minute")
            if isinstance(overrides, dict)
            else None
        )
        if rpm is None:
            rpm = _coerce_positive_int(os.getenv("OPENROUTER_RATE_LIMIT_PER_MIN")) or 18

        interval = _coerce_positive_float(
            overrides.get("min_interval_seconds")
            if isinstance(overrides, dict)
            else None
        )
        if interval is None:
            interval = _coerce_positive_float(os.getenv("OPENROUTER_MIN_INTERVAL_SECONDS"))

        daily = _coerce_positive_int(
            overrides.get("daily_quota")
            if isinstance(overrides, dict)
            else None
        )
        if daily is None:
            daily = _coerce_positive_int(os.getenv("OPENROUTER_DAILY_QUOTA")) or 45

        return cls(
            requests_per_minute=rpm,
            min_interval_seconds=interval,
            daily_quota=daily,
        )

    def acquire(self) -> float:
        while True:
            wait = 0.0
            with self._lock:
                now = time.monotonic()
                # Reset daily counter if needed
                today = datetime.now(timezone.utc).date()
                if today != self._current_day:
                    self._current_day = today
                    self._daily_count = 0
                    self._blocked_until_wall = None
                    self.daily_quota = self._configured_daily_quota
                    if self._configured_requests_per_minute:
                        self.requests_per_minute = self._configured_requests_per_minute
                        derived = 60.0 / float(self.requests_per_minute)
                        if self.min_interval_seconds is None or self.min_interval_seconds < derived:
                            self.min_interval_seconds = derived

                if self.daily_quota is not None and self._daily_count >= self.daily_quota:
                    raise RuntimeError(
                        "OpenRouter daily quota reached. Reduce usage, purchase credits, or raise OPENROUTER_DAILY_QUOTA."
                    )

                if self._blocked_until_wall:
                    wall_now = time.time()
                    if wall_now < self._blocked_until_wall:
                        wait = max(wait, self._blocked_until_wall - wall_now)
                    else:
                        self._blocked_until_wall = None

                # Enforce min-interval spacing
                if self.min_interval_seconds and self._last_call is not None:
                    elapsed = now - self._last_call
                    if elapsed < self.min_interval_seconds:
                        wait = max(wait, self.min_interval_seconds - elapsed)

                # Enforce rolling per-minute limit
                if self.requests_per_minute:
                    cutoff = now - 60.0
                    while self._timestamps and self._timestamps[0] < cutoff:
                        self._timestamps.popleft()
                    if len(self._timestamps) >= self.requests_per_minute:
                        oldest = self._timestamps[0]
                        wait = max(wait, 60.0 - (now - oldest))

                if wait <= 0.0:
                    # Record the outbound request and proceed
                    self._timestamps.append(now)
                    self._last_call = now
                    self._daily_count += 1
                    return now

            time.sleep(min(wait, 60.0))

    def release_failure(self, token: Optional[float]) -> None:
        if token is None:
            return
        with self._lock:
            if self._timestamps and self._timestamps[-1] == token:
                self._timestamps.pop()
            self._daily_count = max(0, self._daily_count - 1)
            self._last_call = self._timestamps[-1] if self._timestamps else None

    def adjust_requests_per_minute(self, new_limit: Optional[int]) -> None:
        limit = _coerce_positive_int(new_limit)
        if not limit:
            return
        with self._lock:
            if self._configured_requests_per_minute:
                limit = min(limit, self._configured_requests_per_minute)
            self.requests_per_minute = limit
            derived = 60.0 / float(self.requests_per_minute)
            if self.min_interval_seconds is None or self.min_interval_seconds < derived:
                self.min_interval_seconds = derived

    def block_temporarily(self, reset_epoch_ms: Optional[int]) -> None:
        wall_reset: Optional[float] = None
        if reset_epoch_ms:
            try:
                wall_reset = max(time.time(), int(reset_epoch_ms) / 1000.0)
            except Exception:
                wall_reset = None
        if wall_reset is None:
            wall_reset = time.time() + (self.min_interval_seconds or 5.0)
        with self._lock:
            if self._blocked_until_wall is None or wall_reset > self._blocked_until_wall:
                self._blocked_until_wall = wall_reset

    def mark_exhausted(self, reset_epoch_ms: Optional[int]) -> None:
        wall_reset: Optional[float] = None
        if reset_epoch_ms:
            try:
                wall_reset = max(time.time(), int(reset_epoch_ms) / 1000.0)
            except Exception:
                wall_reset = None
        with self._lock:
            if self._daily_count > 0:
                if self.daily_quota is None or self._daily_count > self.daily_quota:
                    self.daily_quota = self._daily_count
            else:
                if self.daily_quota is None:
                    self.daily_quota = self._configured_daily_quota or 0
            if wall_reset is not None:
                if self._blocked_until_wall is None or wall_reset > self._blocked_until_wall:
                    self._blocked_until_wall = wall_reset

    def register_backoff(self, *, limit: Optional[int], reset_epoch_ms: Optional[int]) -> None:
        limit_int = _coerce_positive_int(limit)
        if limit_int is not None and limit_int <= 30:
            self.adjust_requests_per_minute(limit_int)
            self.block_temporarily(reset_epoch_ms)
        else:
            self.mark_exhausted(reset_epoch_ms)

class _OpenAIChatModel:
    def __init__(self, cfg: ModelConfig):
        from openai import OpenAI  # lazy import

        api_key = _require_env("OPENAI_API_KEY", cfg.api_key)
        self.client = OpenAI(api_key=api_key)
        self.cfg = cfg
        if not self.cfg.model:
            self.cfg.model = "gpt-5-nano"

    def invoke(self, messages: List[Dict[str, Any]]) -> Any:
        oai_messages = _to_openai_messages(messages)
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": oai_messages,
            "temperature": self.cfg.temperature,
        }
        if self.cfg.max_tokens is not None:
            kwargs["max_tokens"] = self.cfg.max_tokens
        if self.cfg.top_p is not None:
            kwargs["top_p"] = self.cfg.top_p
        logprob_settings = _logprob_request_settings(messages, self.cfg.extra)
        if logprob_settings:
            kwargs.update(logprob_settings)
        # timeout not directly supported per-call; rely on httpx default

        resp = self.client.chat.completions.create(**kwargs)
        # Extract content and usage
        content = resp.choices[0].message.content if resp.choices else ""
        usage = getattr(resp, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion = getattr(usage, "completion_tokens", None) if usage is not None else None
        total = getattr(usage, "total_tokens", None) if usage is not None else None
        usage_dict = {
            "prompt": int(prompt) if prompt is not None else None,
            "completion": int(completion) if completion is not None else None,
            "total": int(total) if total is not None else None,
        }
        payload = SimpleNamespace(content=content, usage=usage_dict)
        if resp.choices:
            logprob_summary = _extract_logprob_payload(resp.choices[0])
            if logprob_summary:
                payload.logprobs = logprob_summary
        return payload


class _GoogleChatModel:
    def __init__(self, cfg: ModelConfig):
        import google.generativeai as genai  # lazy import

        api_key = _require_env("GOOGLE_API_KEY", cfg.api_key)
        genai.configure(api_key=api_key)
        self.genai = genai
        self.cfg = cfg
        if not self.cfg.model:
            self.cfg.model = "gemini-2.5-flash"

    def invoke(self, messages: List[Dict[str, Any]]) -> Any:
        # Compose input from the last user message and system prompt
        sys_prompt = _extract_system_prompt(messages, self.cfg.system_prompt)
        user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)

        parts: List[Any] = []
        if user_msg is not None:
            content = user_msg.get("content")
            if isinstance(content, list):
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif p.get("type") == "image_url":
                        img = p.get("image_url")
                        url = img.get("url") if isinstance(img, dict) else img
                        try:
                            mime, data = _parse_data_url(url)
                            parts.append({"mime_type": mime, "data": data})
                        except Exception:
                            # If not data URL, attempt to treat as regular URL (not fetched here)
                            parts.append(p.get("text", ""))
            else:
                parts.append(str(content) if content is not None else "")

        model = self.genai.GenerativeModel(self.cfg.model, system_instruction=sys_prompt)
        gen_kwargs: Dict[str, Any] = {}
        if self.cfg.temperature is not None:
            gen_kwargs.setdefault("generation_config", {})["temperature"] = self.cfg.temperature
        if self.cfg.max_tokens is not None:
            gen_kwargs.setdefault("generation_config", {})["max_output_tokens"] = self.cfg.max_tokens
        if self.cfg.top_p is not None:
            gen_kwargs.setdefault("generation_config", {})["top_p"] = self.cfg.top_p

        resp = model.generate_content(parts, **gen_kwargs)
        text = getattr(resp, "text", None)
        if text is None and hasattr(resp, "candidates") and resp.candidates:
            # Fallback extraction
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                text = ""
        # Usage metadata (best-effort; may be None depending on SDK/version)
        usage_meta = getattr(resp, "usage_metadata", None)
        prompt = getattr(usage_meta, "input_token_count", None) if usage_meta is not None else None
        completion = getattr(usage_meta, "output_token_count", None) if usage_meta is not None else None
        total = getattr(usage_meta, "total_token_count", None) if usage_meta is not None else None
        usage_dict = {
            "prompt": int(prompt) if prompt is not None else None,
            "completion": int(completion) if completion is not None else None,
            "total": int(total) if total is not None else None,
        }
        return type("_Resp", (), {"content": text or "", "usage": usage_dict})()


class _DeepInfraChatModel:
    """OpenAI-compatible Chat Completions via DeepInfra endpoint.

    Uses OpenAI SDK with base_url set to DeepInfra's OpenAI-compatible API.
    Default model: "google/gemma-3-27b-it".
    """

    def __init__(self, cfg: ModelConfig):
        from openai import OpenAI  # lazy import

        api_key = _require_env("DEEPINFRA_API_KEY", cfg.api_key)
        # Point the OpenAI client at DeepInfra's OpenAI-compatible endpoint
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
        self.cfg = cfg
        if not self.cfg.model:
            self.cfg.model = "google/gemma-3-27b-it"

    def invoke(self, messages: List[Dict[str, Any]]) -> Any:
        oai_messages = _to_openai_messages(messages)
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": oai_messages,
            "temperature": self.cfg.temperature,
        }
        if self.cfg.max_tokens is not None:
            kwargs["max_tokens"] = self.cfg.max_tokens
        if self.cfg.top_p is not None:
            kwargs["top_p"] = self.cfg.top_p
        logprob_settings = _logprob_request_settings(messages, self.cfg.extra)
        if logprob_settings:
            kwargs.update(logprob_settings)

        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content if resp.choices else ""
        usage = getattr(resp, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion = getattr(usage, "completion_tokens", None) if usage is not None else None
        total = getattr(usage, "total_tokens", None) if usage is not None else None
        usage_dict = {
            "prompt": int(prompt) if prompt is not None else None,
            "completion": int(completion) if completion is not None else None,
            "total": int(total) if total is not None else None,
        }
        payload = SimpleNamespace(content=content, usage=usage_dict)
        if resp.choices:
            logprob_summary = _extract_logprob_payload(resp.choices[0])
            if logprob_summary:
                payload.logprobs = logprob_summary
        return payload


class _OpenRouterChatModel:
    """OpenAI-compatible Chat Completions via OpenRouter endpoint."""

    _rate_limiter: Optional[_OpenRouterRateLimiter] = None
    _rate_limiter_lock = threading.Lock()

    def __init__(self, cfg: ModelConfig):
        from openai import OpenAI  # lazy import

        api_key = _require_env("OPENROUTER_API_KEY", cfg.api_key)
        base_url = cfg.extra.get("base_url") if isinstance(cfg.extra, dict) else None
        self.client = OpenAI(api_key=api_key, base_url=base_url or "https://openrouter.ai/api/v1")
        self.cfg = cfg
        env_model = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENROUTER_DEFAULT_MODEL")
        if not self.cfg.model:
            self.cfg.model = env_model or "mistralai/mistral-small-3.2-24b-instruct:free"

        extra_dict = cfg.extra if isinstance(cfg.extra, dict) else {}

        env_headers = {}
        referer = os.getenv("OPENROUTER_SITE_URL") or os.getenv("OPENROUTER_HTTP_REFERER")
        if referer:
            env_headers["HTTP-Referer"] = referer
        title = os.getenv("OPENROUTER_SITE_NAME") or os.getenv("OPENROUTER_TITLE")
        if title:
            env_headers["X-Title"] = title

        cfg_headers = {}
        cfg_body: Dict[str, Any] = {}
        if extra_dict:
            maybe_headers = extra_dict.get("headers")
            if isinstance(maybe_headers, dict):
                cfg_headers = maybe_headers
            maybe_body = extra_dict.get("body")
            if isinstance(maybe_body, dict):
                cfg_body = maybe_body

        # Prefer config-supplied headers when keys collide.
        self.extra_headers = {**env_headers, **cfg_headers}
        self.extra_body = cfg_body

        rate_cfg = extra_dict.get("rate_limit") if isinstance(extra_dict, dict) else None
        if _OpenRouterChatModel._rate_limiter is None:
            with _OpenRouterChatModel._rate_limiter_lock:
                if _OpenRouterChatModel._rate_limiter is None:
                    _OpenRouterChatModel._rate_limiter = _OpenRouterRateLimiter.from_settings(rate_cfg if isinstance(rate_cfg, dict) else None)
        self._rate_limiter = _OpenRouterChatModel._rate_limiter
        # Allow per-instance override if explicitly provided
        if isinstance(rate_cfg, dict) and any(rate_cfg.get(k) for k in ("per_minute", "min_interval_seconds", "daily_quota")):
            self._rate_limiter = _OpenRouterRateLimiter.from_settings(rate_cfg)

    def _handle_rate_limit_headers(self, headers: Optional[Dict[str, Any]]) -> Optional[str]:
        if not headers:
            return None
        try:
            limit = headers.get("X-RateLimit-Limit")
            remaining = headers.get("X-RateLimit-Remaining")
            reset = headers.get("X-RateLimit-Reset")
        except AttributeError:
            limit = remaining = reset = None

        if self._rate_limiter is not None:
            self._rate_limiter.register_backoff(limit=limit, reset_epoch_ms=reset)

        limit_int = _coerce_positive_int(limit)
        remaining_int = _coerce_positive_int(remaining)
        if limit_int is not None and limit_int > 30 and (remaining_int is None or remaining_int <= 0):
            reset_hint = None
            if reset:
                try:
                    reset_ts = int(reset) / 1000.0
                    reset_dt = datetime.fromtimestamp(reset_ts, tz=timezone.utc)
                    reset_hint = reset_dt.strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    reset_hint = None
            if reset_hint:
                return (
                    f"OpenRouter daily free-model quota exhausted (limit {limit_int}). "
                    f"Next reset around {reset_hint}."
                )
            return (
                f"OpenRouter daily free-model quota exhausted (limit {limit_int}). "
                "Wait for the daily reset or add credits to increase limits."
            )
        return None

    def invoke(self, messages: List[Dict[str, Any]]) -> Any:
        attempt = 0
        oai_messages = _to_openai_messages(messages)
        while True:
            token = self._rate_limiter.acquire() if self._rate_limiter is not None else None
            kwargs: Dict[str, Any] = {
                "model": self.cfg.model,
                "messages": oai_messages,
                "temperature": self.cfg.temperature,
            }
            if self.cfg.max_tokens is not None:
                kwargs["max_tokens"] = self.cfg.max_tokens
            if self.cfg.top_p is not None:
                kwargs["top_p"] = self.cfg.top_p
            if self.extra_headers:
                kwargs["extra_headers"] = self.extra_headers
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            logprob_settings = _logprob_request_settings(messages, self.cfg.extra)
            if logprob_settings:
                kwargs.update(logprob_settings)

            try:
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content if resp.choices else ""
                usage = getattr(resp, "usage", None)
                prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
                completion = getattr(usage, "completion_tokens", None) if usage is not None else None
                total = getattr(usage, "total_tokens", None) if usage is not None else None
                usage_dict = {
                    "prompt": int(prompt) if prompt is not None else None,
                    "completion": int(completion) if completion is not None else None,
                    "total": int(total) if total is not None else None,
                }
                payload = SimpleNamespace(content=content, usage=usage_dict)
                if resp.choices:
                    logprob_summary = _extract_logprob_payload(resp.choices[0])
                    if logprob_summary:
                        payload.logprobs = logprob_summary
                return payload
            except HTTPStatusError as exc:
                if self._rate_limiter is not None:
                    self._rate_limiter.release_failure(token)
                if exc.response.status_code == 429:
                    message = self._handle_rate_limit_headers(getattr(exc.response, "headers", None))
                    if message:
                        raise RuntimeError(message) from exc
                    attempt += 1
                    if attempt >= 3:
                        raise
                    continue
                raise
            except RateLimitError as exc:
                if self._rate_limiter is not None:
                    self._rate_limiter.release_failure(token)
                response = getattr(exc, "response", None)
                headers = getattr(response, "headers", None) if response is not None else None
                message = self._handle_rate_limit_headers(headers)
                raise RuntimeError(message or str(exc)) from exc
            except Exception as exc:
                if self._rate_limiter is not None:
                    self._rate_limiter.release_failure(token)
                response = getattr(exc, "response", None)
                headers = getattr(response, "headers", None) if response is not None else None
                message = self._handle_rate_limit_headers(headers)
                if message:
                    raise RuntimeError(message) from exc
                raise


class LLMModelLoader:
    """Factory returning a minimal chat model with `.invoke(messages)` method."""

    def __init__(self, config: Union[ModelConfig, Dict[str, Any]]):
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        if isinstance(self.config.model, str) and not self.config.model.strip():
            self.config.model = None
        if self.config.provider not in ("openai", "google", "deepinfra", "openrouter"):
            raise ValueError("provider must be 'openai', 'google', 'deepinfra', or 'openrouter'")
        # Cumulative usage across all invocations via this loader
        self.usage_total = {"prompt": 0, "completion": 0, "total": 0}
        self._model_wrapper = None
        self._model_lock = threading.Lock()

    def get_model(self):
        if self._model_wrapper is not None:
            return self._model_wrapper

        with self._model_lock:
            if self._model_wrapper is None:
                if self.config.provider == "openai":
                    base_model = _OpenAIChatModel(self.config)
                elif self.config.provider == "openrouter":
                    base_model = _OpenRouterChatModel(self.config)
                elif self.config.provider == "deepinfra":
                    base_model = _DeepInfraChatModel(self.config)
                else:
                    base_model = _GoogleChatModel(self.config)

                loader_usage = self.usage_total

                class _UsageWrapped:
                    def __init__(self, inner):
                        self._inner = inner

                    def invoke(self, messages: List[Dict[str, Any]]) -> Any:
                        resp = self._inner.invoke(messages)
                        # Normalize and accumulate usage if present
                        usage = getattr(resp, "usage", None)
                        if isinstance(usage, dict):
                            p = usage.get("prompt") or 0
                            c = usage.get("completion") or 0
                            t = usage.get("total") or 0
                            try:
                                loader_usage["prompt"] += int(p)
                            except Exception:
                                pass
                            try:
                                loader_usage["completion"] += int(c)
                            except Exception:
                                pass
                            try:
                                loader_usage["total"] += int(t)
                            except Exception:
                                # Derive total if missing
                                try:
                                    loader_usage["total"] += int(p) + int(c)
                                except Exception:
                                    pass
                        return resp

                self._model_wrapper = _UsageWrapped(base_model)

        return self._model_wrapper

    def invoke_text(self, text: str, system_prompt: Optional[str] = None) -> str:
        model = self.get_model()
        sys = system_prompt or self.config.system_prompt or "You are a helpful assistant."
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": text},
        ]
        resp = model.invoke(messages)
        return getattr(resp, "content", str(resp))


__all__ = ["ModelConfig", "LLMModelLoader"]
