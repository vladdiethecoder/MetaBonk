"""Unified LLM/VLM client helpers.

This file provides a single place to configure real backends (Ollama/local,
OpenAI-compatible HTTP, etc.). Mock/fallback backends are intentionally not
supported: failures should be explicit to preserve troubleshooting signal.

Environment:
  - METABONK_LLM_BACKEND: "ollama" | "http" | "openai"
  - METABONK_LLM_MODEL: model name (default varies by caller)
  - METABONK_EMBED_MODEL: optional embedding model name (defaults to METABONK_LLM_MODEL)
  - METABONK_LLM_TEMPERATURE
  - METABONK_LLM_MAX_TOKENS
  - METABONK_LLM_BASE_URL: for http/openai backends (default http://127.0.0.1:8000)
  - METABONK_LLM_API_KEY: optional
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    HTTP = "http"          # OpenAI-compatible /v1/chat/completions
    OPENAI = "openai"      # Uses openai python package if installed


@dataclass
class LLMConfig:
    backend: LLMBackend = LLMBackend.OLLAMA
    model: str = "qwen2.5"
    temperature: float = 0.7
    max_tokens: int = 2048
    base_url: str = "http://127.0.0.1:8000"
    api_key: Optional[str] = None
    timeout_s: float = 30.0

    @classmethod
    def from_env(cls, default_model: str = "qwen2.5") -> "LLMConfig":
        backend = os.environ.get("METABONK_LLM_BACKEND", "ollama").lower()
        model = os.environ.get("METABONK_LLM_MODEL", default_model)
        try:
            temp = float(os.environ.get("METABONK_LLM_TEMPERATURE", "0.7"))
        except Exception:
            temp = 0.7
        try:
            max_tokens = int(os.environ.get("METABONK_LLM_MAX_TOKENS", "2048"))
        except Exception:
            max_tokens = 2048
        base_url = os.environ.get("METABONK_LLM_BASE_URL", "http://127.0.0.1:8000")
        api_key = os.environ.get("METABONK_LLM_API_KEY")
        try:
            timeout_s = float(os.environ.get("METABONK_LLM_TIMEOUT_S", "30"))
        except Exception:
            timeout_s = 30.0
        try:
            b = LLMBackend(backend)
        except Exception as e:
            raise ValueError(
                f"Unsupported METABONK_LLM_BACKEND={backend!r}. "
                f"Expected one of: {', '.join([x.value for x in LLMBackend])}"
            ) from e
        return cls(
            backend=b,
            model=model,
            temperature=temp,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
        )


def build_llm_fn(
    cfg: Optional[LLMConfig] = None,
    *,
    system_prompt: Optional[str] = None,
) -> Callable[[str], str]:
    """Return a callable(prompt)->text for the configured backend."""
    cfg = cfg or LLMConfig.from_env()

    if cfg.backend == LLMBackend.OLLAMA:
        try:
            import ollama  # type: ignore
        except Exception as e:
            raise ImportError(
                "METABONK_LLM_BACKEND=ollama requires the `ollama` python package."
            ) from e

        def _ollama_call(prompt: str) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            try:
                resp = ollama.chat(
                    model=cfg.model,
                    messages=messages,
                    options={
                        "temperature": cfg.temperature,
                        "num_predict": cfg.max_tokens,
                    },
                    stream=False,
                )
                # resp may be dict-like or pydantic object.
                if isinstance(resp, dict):
                    return (
                        resp.get("message", {}).get("content")
                        or resp.get("response")
                        or ""
                    )
                msg = getattr(resp, "message", None)
                if isinstance(msg, dict):
                    return msg.get("content", "")
                if msg is not None:
                    return getattr(msg, "content", "") or ""
                return getattr(resp, "response", "") or ""
            except Exception as e:
                raise RuntimeError(f"Ollama LLM call failed: {type(e).__name__}: {e}") from e

        return _ollama_call

    if cfg.backend == LLMBackend.OPENAI:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError(
                "METABONK_LLM_BACKEND=openai requires the `openai` python package."
            ) from e

        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

        def _openai_call(prompt: str) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            try:
                r = client.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                return r.choices[0].message.content or ""
            except Exception as e:
                raise RuntimeError(f"OpenAI LLM call failed: {type(e).__name__}: {e}") from e

        return _openai_call

    # HTTP OpenAI-compatible backend.
    try:
        import requests
    except Exception as e:
        raise ImportError(
            "METABONK_LLM_BACKEND=http requires the `requests` python package."
        ) from e

    def _http_call(prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        payload: Dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        try:
            url = cfg.base_url.rstrip("/")
            if not url.endswith("/v1/chat/completions"):
                url = url + "/v1/chat/completions"
            r = requests.post(url, json=payload, headers=headers, timeout=cfg.timeout_s)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"HTTP LLM call failed: {type(e).__name__}: {e}") from e

    return _http_call


def build_embed_fn(
    cfg: Optional[LLMConfig] = None,
) -> Callable[[str], List[float]]:
    """Return a callable(text)->embedding vector for the configured backend.

    This is used by System 2 components (e.g., MoR) for strategy selection,
    retrieval, and other numeric scoring. No mock fallback is provided.
    """

    cfg = cfg or LLMConfig.from_env(default_model=os.environ.get("METABONK_EMBED_MODEL", "qwen2.5"))
    embed_model = os.environ.get("METABONK_EMBED_MODEL")
    if embed_model:
        cfg.model = embed_model

    if cfg.backend == LLMBackend.OLLAMA:
        try:
            import ollama  # type: ignore
        except Exception as e:
            raise ImportError("Embedding backend ollama requires the `ollama` python package.") from e

        def _ollama_embed(text: str) -> List[float]:
            try:
                resp = ollama.embeddings(model=cfg.model, prompt=text, options=None, keep_alive=None)
                vec = resp.get("embedding") if isinstance(resp, dict) else getattr(resp, "embedding", None)
                if vec is None:
                    raise RuntimeError("Ollama embeddings response missing `embedding`.")
                return [float(x) for x in vec]
            except Exception as e:
                raise RuntimeError(f"Ollama embed failed: {type(e).__name__}: {e}") from e

        return _ollama_embed

    if cfg.backend == LLMBackend.OPENAI:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError("Embedding backend openai requires the `openai` python package.") from e

        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

        def _openai_embed(text: str) -> List[float]:
            try:
                r = client.embeddings.create(model=cfg.model, input=text)
                vec = r.data[0].embedding
                return [float(x) for x in vec]
            except Exception as e:
                raise RuntimeError(f"OpenAI embed failed: {type(e).__name__}: {e}") from e

        return _openai_embed

    # HTTP OpenAI-compatible embeddings endpoint.
    try:
        import requests
    except Exception as e:
        raise ImportError("Embedding backend http requires the `requests` python package.") from e

    def _http_embed(text: str) -> List[float]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        payload: Dict[str, Any] = {"model": cfg.model, "input": text}
        try:
            url = cfg.base_url.rstrip("/")
            if not url.endswith("/v1/embeddings"):
                url = url + "/v1/embeddings"
            r = requests.post(url, json=payload, headers=headers, timeout=cfg.timeout_s)
            r.raise_for_status()
            data = r.json()
            vec = data["data"][0]["embedding"]
            return [float(x) for x in vec]
        except Exception as e:
            raise RuntimeError(f"HTTP embed failed: {type(e).__name__}: {e}") from e

    return _http_embed


def build_embed_fn(cfg: Optional[LLMConfig] = None) -> Callable[[str], List[float]]:
    """Return a callable(text)->embedding[] for the configured backend.

    This intentionally does not provide mock/fallback embeddings: if a real
    backend is unavailable, it raises to preserve troubleshooting signal.
    """
    cfg = cfg or LLMConfig.from_env()

    if cfg.backend == LLMBackend.OLLAMA:
        try:
            import ollama  # type: ignore
        except Exception as e:
            raise ImportError(
                "METABONK_LLM_BACKEND=ollama requires the `ollama` python package."
            ) from e

        def _ollama_embed(text: str) -> List[float]:
            try:
                resp = ollama.embeddings(model=cfg.model, prompt=text)
                if isinstance(resp, dict):
                    emb = resp.get("embedding")
                else:
                    emb = getattr(resp, "embedding", None)
                if not isinstance(emb, list):
                    raise RuntimeError("ollama.embeddings returned no embedding list")
                return [float(x) for x in emb]
            except Exception as e:
                raise RuntimeError(f"Ollama embedding call failed: {type(e).__name__}: {e}") from e

        return _ollama_embed

    if cfg.backend == LLMBackend.OPENAI:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError(
                "METABONK_LLM_BACKEND=openai requires the `openai` python package."
            ) from e

        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

        def _openai_embed(text: str) -> List[float]:
            try:
                r = client.embeddings.create(model=cfg.model, input=text)
                emb = r.data[0].embedding
                return [float(x) for x in emb]
            except Exception as e:
                raise RuntimeError(f"OpenAI embedding call failed: {type(e).__name__}: {e}") from e

        return _openai_embed

    # HTTP OpenAI-compatible backend.
    try:
        import requests
    except Exception as e:
        raise ImportError(
            "METABONK_LLM_BACKEND=http requires the `requests` python package."
        ) from e

    def _http_embed(text: str) -> List[float]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        payload: Dict[str, Any] = {
            "model": cfg.model,
            "input": text,
        }
        try:
            url = cfg.base_url.rstrip("/")
            if not url.endswith("/v1/embeddings"):
                url = url + "/v1/embeddings"
            r = requests.post(url, json=payload, headers=headers, timeout=cfg.timeout_s)
            r.raise_for_status()
            data = r.json()
            emb = data["data"][0]["embedding"]
            if not isinstance(emb, list):
                raise RuntimeError("embeddings response missing embedding list")
            return [float(x) for x in emb]
        except Exception as e:
            raise RuntimeError(f"HTTP embedding call failed: {type(e).__name__}: {e}") from e

    return _http_embed
