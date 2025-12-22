"""Menu reasoning via VLM over SoM-tagged screenshots."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception as e:  # pragma: no cover
    raise RuntimeError("requests is required for VLM menu reasoning") from e
try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None  # type: ignore


@dataclass
class MenuAction:
    kind: str  # "click" | "key" | "noop"
    x: int = 0
    y: int = 0
    key: str = ""
    target_id: Optional[int] = None
    target_label: str = ""
    reason: str = ""


@dataclass
class MenuReasonerConfig:
    backend: str = os.environ.get("METABONK_VLM_BACKEND", "http")
    model: str = os.environ.get("METABONK_VLM_MODEL", "gpt-4o-mini")
    base_url: str = os.environ.get("METABONK_VLM_BASE_URL", "http://127.0.0.1:8000")
    api_key: str = os.environ.get("METABONK_VLM_API_KEY", "")
    temperature: float = float(os.environ.get("METABONK_VLM_TEMPERATURE", "0.2") or 0.2)
    max_tokens: int = int(os.environ.get("METABONK_VLM_MAX_TOKENS", "512") or 512)
    timeout_s: float = float(os.environ.get("METABONK_VLM_TIMEOUT_S", "30") or 30)
    ollama_host: str = os.environ.get("METABONK_VLM_OLLAMA_HOST", "")
    json_mode: bool = os.environ.get("METABONK_VLM_JSON_MODE", "0") in ("1", "true", "True")

    prompt_template: str = (
        "You are a menu navigation agent. The image has numbered UI elements.\n"
        "Goal: {goal}\n"
        "Hint: {hint}\n"
        "Elements (id, label, text):\n{elements}\n\n"
        "Choose the best element id to interact with.\n"
        "Respond ONLY as JSON: {{\"action\": \"click\"|\"key\"|\"noop\", "
        "\"id\": <int|null>, \"key\": <string|null>, \"reason\": <string>}}"
    )


class MenuReasoner:
    def __init__(self, cfg: Optional[MenuReasonerConfig] = None):
        self.cfg = cfg or MenuReasonerConfig()
        self._ollama_client = None
        if self.cfg.ollama_host and ollama is not None:
            try:
                self._ollama_client = ollama.Client(host=self.cfg.ollama_host)  # type: ignore[attr-defined]
            except Exception:
                self._ollama_client = None

    def _encode_image(self, image: Any) -> str:
        if hasattr(image, "save"):
            import io
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
        else:
            data = image  # assume bytes
        return base64.b64encode(data).decode("ascii")

    def _build_prompt(self, mapping: List[Dict[str, Any]], goal: str, hint: str) -> str:
        lines = []
        for item in mapping:
            label = str(item.get("label", ""))
            text = str(item.get("text", ""))
            lines.append(f"{item.get('id')}: {label} | {text}")
        return self.cfg.prompt_template.format(
            goal=goal,
            hint=hint or "",
            elements="\n".join(lines),
        )

    def _call_openai_vision(self, image: Any, prompt: str) -> str:
        b64 = self._encode_image(image)
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        url = self.cfg.base_url.rstrip("/")
        if not url.endswith("/v1/chat/completions"):
            url = url + "/v1/chat/completions"
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        if self.cfg.json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", 0)
            if self.cfg.json_mode and status in (400, 422):
                payload.pop("response_format", None)
                resp = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
                resp.raise_for_status()
            else:
                raise
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_ollama_vision(self, image: Any, prompt: str) -> str:
        if ollama is None:
            raise RuntimeError("ollama is not installed")
        b64 = self._encode_image(image)
        client = self._ollama_client or ollama
        resp = client.chat(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": "You are a menu navigation agent. Output JSON only."},
                {"role": "user", "content": prompt, "images": [b64]},
            ],
            options={
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        )
        return (resp.get("message") or {}).get("content", "")

    def _extract_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    def infer_action(
        self,
        image: Any,
        mapping: List[Dict[str, Any]],
        goal: str,
        hint: str = "",
    ) -> Optional[MenuAction]:
        if not mapping:
            return None
        prompt = self._build_prompt(mapping, goal, hint)
        try:
            if self.cfg.backend in ("ollama",):
                response = self._call_ollama_vision(image, prompt)
            elif self.cfg.backend in ("http", "openai"):
                response = self._call_openai_vision(image, prompt)
            else:
                return None
        except Exception:
            return None
        payload = self._extract_json(response)
        action = str(payload.get("action") or "").strip().lower()
        if action not in ("click", "key", "noop"):
            return None
        if action == "key":
            key = str(payload.get("key") or "").strip()
            return MenuAction(kind="key", key=key, reason=str(payload.get("reason") or ""))
        if action == "noop":
            return MenuAction(kind="noop", reason=str(payload.get("reason") or ""))
        try:
            tid = int(payload.get("id"))
        except Exception:
            tid = None
        target = None
        if tid is not None:
            for item in mapping:
                if int(item.get("id", -1)) == tid:
                    target = item
                    break
        if not target:
            target = mapping[0]
        cx, cy = target.get("center", [0, 0])
        return MenuAction(
            kind="click",
            x=int(cx),
            y=int(cy),
            target_id=int(target.get("id", 0)),
            target_label=str(target.get("label", "")),
            reason=str(payload.get("reason") or ""),
        )


__all__ = ["MenuAction", "MenuReasoner", "MenuReasonerConfig"]
