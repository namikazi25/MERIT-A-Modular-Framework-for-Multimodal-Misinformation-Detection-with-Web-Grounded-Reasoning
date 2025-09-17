from __future__ import annotations

"""
Provider-native LLM loader for OpenAI (GPT-4o), Google (Gemini), and DeepInfra (Gemma 3 via OpenAI-compatible API).

No LangChain dependency. Exposes a minimal interface compatible with
scripts.align_checker: `get_model().invoke(messages)`.

Environment variables:
  - OPENAI_API_KEY (OpenAI)
  - GOOGLE_API_KEY (Google)
  - DEEPINFRA_API_KEY (DeepInfra)
"""

import base64
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union, List


Provider = Literal["openai", "google", "deepinfra"]


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
        return type("_Resp", (), {"content": content, "usage": usage_dict})()


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
        return type("_Resp", (), {"content": content, "usage": usage_dict})()


class LLMModelLoader:
    """Factory returning a minimal chat model with `.invoke(messages)` method."""

    def __init__(self, config: Union[ModelConfig, Dict[str, Any]]):
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        if self.config.provider not in ("openai", "google", "deepinfra"):
            raise ValueError("provider must be 'openai', 'google', or 'deepinfra'")
        # Cumulative usage across all invocations via this loader
        self.usage_total = {"prompt": 0, "completion": 0, "total": 0}

    def get_model(self):
        # Instantiate provider-specific model
        if self.config.provider == "openai":
            base_model = _OpenAIChatModel(self.config)
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

        return _UsageWrapped(base_model)

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
