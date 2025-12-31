#!/usr/bin/env python3
"""
MetaBonk Centralized Cognitive Server.

Runs SGLang with Phi-3-Vision for multi-instance System 2 reasoning.
Communication:
- Receives JSON requests over ZeroMQ ROUTER from multiple game instances (DEALER).
- Returns JSON strategic directives.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import base64
import difflib
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

import torch
from PIL import Image

try:
    import sglang as sgl  # type: ignore
    from sglang import assistant, gen, image, system, user, user_begin, user_end  # type: ignore
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint  # type: ignore
except Exception:  # pragma: no cover
    sgl = None  # type: ignore
    assistant = None  # type: ignore
    gen = None  # type: ignore
    image = None  # type: ignore
    system = None  # type: ignore
    user = None  # type: ignore
    user_begin = None  # type: ignore
    user_end = None  # type: ignore
    RuntimeEndpoint = None  # type: ignore

from rl_integration import RLLogger
from temporal_processor import TemporalFrameProcessor
from zmq_bridge import ZeroMQBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("metabonk.cognitive_server")


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in ("1", "true", "yes", "on")


def _sanitize_agent_state_for_prompt(agent_state: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the prompt payload small and stable.

    Workers may attach OCR-derived UI text boxes. On splash screens the OCR output can contain
    long paragraphs, which bloats token counts and can exceed the model context length.

    This function is vision-only (it never injects game-specific heuristics); it only truncates
    and bounds the size of the structured context we pass to the VLM.
    """
    raw_state = dict(agent_state or {})

    # Keep only fields the prompt actually references. This materially reduces
    # token count (floats with long mantissas are especially expensive).
    state: Dict[str, Any] = {}
    for key in ("menu_hint", "health_ratio", "enemies_nearby", "frame_w", "frame_h", "ui_elements"):
        if key in raw_state:
            state[key] = raw_state.get(key)

    max_ui = 12
    try:
        max_ui = int(os.environ.get("METABONK_COGNITIVE_MAX_UI_ELEMENTS", str(max_ui)) or max_ui)
    except Exception:
        max_ui = 12
    max_ui = max(0, min(64, int(max_ui)))

    def _round_num(val: Any) -> Any:
        if isinstance(val, bool):
            return val
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return round(float(val), 3)
        if isinstance(val, str):
            try:
                return round(float(val), 3)
            except Exception:
                return val
        return val

    # Round top-level scalars for compactness.
    for key in ("health_ratio", "enemies_nearby", "frame_w", "frame_h"):
        if key in state:
            state[key] = _round_num(state.get(key))

    ui_elements = state.get("ui_elements")
    if isinstance(ui_elements, list) and max_ui >= 0:
        cleaned: List[Dict[str, Any]] = []
        for elem in ui_elements:
            if not isinstance(elem, dict):
                continue
            row: Dict[str, Any] = {}
            for key in ("name", "cx", "cy", "w", "h", "conf"):
                if key in elem:
                    row[key] = _round_num(elem.get(key))
            name = row.get("name")
            if isinstance(name, str):
                name_norm = " ".join(name.split())
                if len(name_norm) > 48:
                    name_norm = name_norm[:48]
                row["name"] = name_norm
            cleaned.append(row)
            if max_ui and len(cleaned) >= max_ui:
                break
        state["ui_elements"] = cleaned

    return state


if sgl is not None:  # pragma: no cover

    @sgl.function  # type: ignore[misc]
    def metabonk_strategist(
        s: Any,
        temporal_frames: List[str],
        agent_state_json: str,
        system_prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> None:
        """SGLang prompt for System2/3 directives."""

        s += system(system_prompt)  # type: ignore[misc]
        s += user_begin()  # type: ignore[misc]
        s += f"Agent State: {agent_state_json}\n\n"
        # `sglang.image()` expects a filesystem path, not a base64 string.
        for frame_path in list(temporal_frames or []):
            s += image(str(frame_path))  # type: ignore[misc]
        s += "\n\nTask: Provide a strategic directive. Output STRICT JSON only."
        s += user_end()  # type: ignore[misc]
        s += assistant(  # type: ignore[misc]
            gen(
                "response",
                max_tokens=int(max(1, min(512, int(max_tokens)))),
                temperature=float(max(0.0, float(temperature))),
            )
        )


_JSON_DECODER = json.JSONDecoder()


def _extract_json_obj(text: str) -> Dict[str, Any]:
    """Best-effort parse a JSON object from model output.

    The model sometimes emits code fences or trailing text; we accept the first JSON
    object we can decode and ignore any suffix.
    """
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Empty model response")

    def _normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _normalize(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return [_normalize(v) for v in obj]
        if isinstance(obj, list):
            return [_normalize(v) for v in obj]
        return obj

    def _first_braced_object(s: str) -> Optional[str]:
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        escape = False
        quote = ""
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\\\":
                    escape = True
                elif ch == quote:
                    in_str = False
                continue
            if ch in ("'", '"'):
                in_str = True
                quote = ch
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return None

    def _try_parse_obj(s: str) -> Optional[Dict[str, Any]]:
        cand = str(s or "").strip()
        if not cand:
            return None
        # Strict JSON.
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # JSON with prefix/suffix (take first decodable object).
        starts = [i for i in (cand.find("{"), cand.find("[")) if i >= 0]
        if starts:
            try:
                obj, _end = _JSON_DECODER.raw_decode(cand[min(starts) :])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        # Python-literal fallback (handles None/True/False, trailing commas, single quotes).
        braced = _first_braced_object(cand)
        if braced:
            try:
                obj = ast.literal_eval(braced)
                if isinstance(obj, dict):
                    return _normalize(obj)
            except Exception:
                pass
        return None

    # Try raw text, then fenced blocks (if any).
    candidates = [raw]
    candidates.extend(
        str(m.group(1) or "").strip()
        for m in re.finditer(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    )
    for cand in candidates:
        obj = _try_parse_obj(cand)
        if obj is not None:
            return obj

    # Truncated/malformed JSON fallback: extract a minimal subset of fields so we can
    # still return a non-error response (worker fills defaults for missing keys).
    fallback: Dict[str, Any] = {}

    def _extract_str_field(key: str) -> Optional[str]:
        m = re.search(rf"\"{re.escape(key)}\"\s*:\s*\"((?:\\.|[^\"])*)\"", raw, flags=re.DOTALL)
        if not m:
            return None
        val = m.group(1)
        try:
            return str(json.loads(f"\"{val}\""))
        except Exception:
            return str(val)

    for k in ("reasoning", "goal", "strategy"):
        v = _extract_str_field(k)
        if v is not None and v.strip():
            fallback[k] = v.strip()

    directive: Dict[str, Any] = {}
    act = _extract_str_field("action")
    if act:
        directive["action"] = act
    pr = _extract_str_field("priority")
    if pr:
        directive["priority"] = pr
    m = re.search(r"\"duration_seconds\"\s*:\s*([0-9]+(?:\.[0-9]+)?)", raw)
    if m:
        try:
            directive["duration_seconds"] = float(m.group(1))
        except Exception:
            pass
    m = re.search(
        r"\"target\"\s*:\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]",
        raw,
    )
    if m:
        try:
            directive["target"] = [float(m.group(1)), float(m.group(2))]
        except Exception:
            pass
    if directive:
        fallback["directive"] = directive

    m = re.search(r"\"confidence\"\s*:\s*([0-9]+(?:\.[0-9]+)?)", raw)
    if m:
        try:
            fallback["confidence"] = float(m.group(1))
        except Exception:
            pass

    if fallback:
        return fallback

    raise ValueError(f"Could not parse JSON object from response: {raw[:800]}")


def _detect_model_type(model_path: str) -> str:
    try:
        raw = (Path(model_path) / "config.json").read_text()
        cfg = json.loads(raw)
        return str(cfg.get("model_type") or "").strip().lower()
    except Exception:
        return ""


def _select_backend(*, model_path: str) -> str:
    env = str(os.environ.get("METABONK_COGNITIVE_BACKEND", "") or "").strip().lower()
    if env in ("hf", "transformers", "huggingface"):
        return "transformers"
    if env in ("sglang", "srt"):
        return "sglang"

    # AUTO: Default to Transformers for Phi-3-Vision unless explicitly overridden.
    # SGLang can be enabled via METABONK_COGNITIVE_BACKEND=sglang, and will spawn an
    # in-container SGLang server for faster inference.
    if _detect_model_type(model_path) == "phi3_v":
        return "transformers"

    return "sglang"


def _prepare_model_path_for_sglang(model_path: str) -> str:
    """Prepare a local overlay directory to satisfy SGLang's HF auto_map expectations.

    SGLang 0.5.6 expects custom models to define `auto_map["AutoModel"]` (not only
    `AutoModelForCausalLM`). Some HF repos (notably Phi-3-Vision) omit AutoModel,
    which causes runtime initialization to fail.

    When detected, we create a lightweight overlay dir under /tmp with:
    - symlinks to all files in the original model repo
    - a patched config.json that includes AutoModel
    - small compat classes that advertise attention-backend support

    This avoids mutating the mounted model directory (often read-only).
    """
    if _env_truthy("METABONK_COGNITIVE_DISABLE_MODEL_OVERLAY"):
        return model_path

    src_dir = Path(model_path)
    cfg_path = src_dir / "config.json"
    try:
        cfg_raw = cfg_path.read_text()
        cfg = json.loads(cfg_raw)
    except Exception:
        return model_path

    architectures = list(cfg.get("architectures") or [])
    model_type = str(cfg.get("model_type") or "").strip().lower()
    auto_map = dict(cfg.get("auto_map") or {})

    # Only patch known cases for now (Phi-3-Vision).
    is_phi3v = model_type == "phi3_v" or any(a.lower().startswith("phi3v") for a in architectures)
    if not is_phi3v:
        return model_path

    needs_auto_model = "AutoModel" not in auto_map and "AutoModelForCausalLM" in auto_map
    if not needs_auto_model:
        return model_path

    overlay_root = Path(os.environ.get("METABONK_COGNITIVE_MODEL_OVERLAY_DIR", "/tmp/metabonk_model_overlay"))
    try:
        overlay_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return model_path

    digest = hashlib.sha256((str(src_dir.resolve()) + ":" + cfg_raw).encode("utf-8")).hexdigest()[:12]
    dst_dir = overlay_root / digest
    cfg_dst = dst_dir / "config.json"
    if cfg_dst.exists():
        return str(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Symlink all repo files into the overlay, except config.json which we patch.
    for child in src_dir.iterdir():
        if child.name == "config.json":
            continue
        target = dst_dir / child.name
        if target.exists():
            continue
        try:
            target.symlink_to(child)
        except Exception as e:
            raise RuntimeError(f"Failed to create model overlay symlink: {target} -> {child}: {e}") from e

    # Add a tiny adapter module that marks the model backend compatible.
    compat_mod = "metabonk_sglang_compat"
    (dst_dir / f"{compat_mod}.py").write_text(
        "\n".join(
            [
                "# Auto-generated by MetaBonk cognitive_server for SGLang compatibility.",
                "# NOTE: transformers dynamic_module_utils does not recursively copy relative imports",
                "# for local directories. We list Phi-3-Vision's helper modules here (guarded) so they",
                "# are copied into the HF modules cache alongside this file.",
                "if False:",
                "    from .configuration_phi3_v import Phi3VConfig  # noqa: F401",
                "    from .image_embedding_phi3_v import Phi3ImageEmbedding  # noqa: F401",
                "    from .image_processing_phi3_v import Phi3VImageProcessor  # noqa: F401",
                "    from .processing_phi3_v import Phi3VProcessor  # noqa: F401",
                "",
                "from .modeling_phi3_v import (",
                "    PHI3_ATTENTION_CLASSES as _PHI3_ATTENTION_CLASSES,",
                "    Phi3SdpaAttention as _Phi3SdpaAttention,",
                "    Phi3VForCausalLM,",
                "    Phi3VModel,",
                ")",
                "import inspect",
                "# SGLang sets config._attn_implementation='sglang' for its Transformers wrapper.",
                "# Phi-3-Vision doesn't know this key, so we map it to SDPA as a safe default.",
                "_PHI3_ATTENTION_CLASSES.setdefault('sglang', _Phi3SdpaAttention)",
                "",
                "# SGLang's Transformers wrapper passes a `forward_batch` kwarg down to the HF",
                "# backbone forward(). Phi-3-Vision does not accept it, so we drop it.",
                "_orig_phi3v_forward = Phi3VModel.forward",
                "_phi3v_allowed = set(inspect.signature(_orig_phi3v_forward).parameters.keys())",
                "def _phi3v_forward_compat(self, *args, **kwargs):",
                "    # Drop any extra kwargs injected by SGLang's wrapper (e.g. forward_batch, attention_instances).",
                "    for k in list(kwargs.keys()):",
                "        if k not in _phi3v_allowed:",
                "            kwargs.pop(k, None)",
                "    return _orig_phi3v_forward(self, *args, **kwargs)",
                "Phi3VModel.forward = _phi3v_forward_compat",
                "try:",
                "    _orig_phi3vlm_forward = Phi3VForCausalLM.forward",
                "    _phi3vlm_allowed = set(inspect.signature(_orig_phi3vlm_forward).parameters.keys())",
                "    def _phi3vlm_forward_compat(self, *args, **kwargs):",
                "        for k in list(kwargs.keys()):",
                "            if k not in _phi3vlm_allowed:",
                "                kwargs.pop(k, None)",
                "        return _orig_phi3vlm_forward(self, *args, **kwargs)",
                "    Phi3VForCausalLM.forward = _phi3vlm_forward_compat",
                "except Exception:",
                "    pass",
                "",
                "class Phi3VModelCompat(Phi3VModel):",
                "    _supports_attention_backend = True",
                "",
                "class Phi3VForCausalLMCompat(Phi3VForCausalLM):",
                "    _supports_attention_backend = True",
                "",
            ]
        )
        + "\n"
    )

    # Patch config.json auto_map.
    patched_auto_map = dict(auto_map)
    patched_auto_map.setdefault("AutoModel", f"{compat_mod}.Phi3VModelCompat")
    patched_auto_map["AutoModelForCausalLM"] = f"{compat_mod}.Phi3VForCausalLMCompat"
    cfg["auto_map"] = patched_auto_map
    cfg_dst.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")

    logger.warning(
        "Patched model overlay created for SGLang (missing auto_map[AutoModel]): %s -> %s",
        str(src_dir),
        str(dst_dir),
    )
    return str(dst_dir)

def _prepare_model_path_for_transformers(model_path: str) -> str:
    """Prepare a local overlay directory for Transformers runtime patches.

    Today this is primarily used to enable SDPA for Phi-3-Vision by patching the
    model's `*_supports_sdpa` flag. This avoids editing the mounted model
    directory (often read-only) while allowing us to use faster attention
    backends when FlashAttention is unavailable.
    """
    if _env_truthy("METABONK_COGNITIVE_DISABLE_MODEL_OVERLAY"):
        return model_path
    if not _env_truthy("METABONK_COGNITIVE_ENABLE_SDPA_PATCH"):
        return model_path

    src_dir = Path(model_path)
    cfg_path = src_dir / "config.json"
    try:
        cfg_raw = cfg_path.read_text()
        cfg = json.loads(cfg_raw)
    except Exception:
        return model_path

    architectures = list(cfg.get("architectures") or [])
    model_type = str(cfg.get("model_type") or "").strip().lower()
    is_phi3v = model_type == "phi3_v" or any(str(a).lower().startswith("phi3v") for a in architectures)
    if not is_phi3v:
        return model_path

    src_modeling = src_dir / "modeling_phi3_v.py"
    try:
        src_text = src_modeling.read_text()
    except Exception:
        return model_path

    if re.search(r"_supports_sdpa\s*=\s*True", src_text):
        return model_path

    patched_text = re.sub(r"_supports_sdpa\s*=\s*False", "_supports_sdpa = True", src_text, count=1)
    if patched_text == src_text:
        return model_path

    overlay_root = Path(os.environ.get("METABONK_COGNITIVE_MODEL_OVERLAY_DIR", "/tmp/metabonk_model_overlay"))
    try:
        overlay_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return model_path

    digest = hashlib.sha256((str(src_dir.resolve()) + ":" + cfg_raw + ":sdpa").encode("utf-8")).hexdigest()[:12]
    dst_dir = overlay_root / f"{digest}-tf"
    cfg_dst = dst_dir / "config.json"
    if cfg_dst.exists():
        return str(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    for child in src_dir.iterdir():
        if child.name in ("config.json", "modeling_phi3_v.py"):
            continue
        target = dst_dir / child.name
        if target.exists():
            continue
        try:
            target.symlink_to(child)
        except Exception:
            return model_path

    (dst_dir / "modeling_phi3_v.py").write_text(patched_text)
    cfg_dst.write_text(cfg_raw)

    logger.warning(
        "Patched model overlay created for Transformers SDPA support: %s -> %s",
        str(src_dir),
        str(dst_dir),
    )
    return str(dst_dir)


METABONK_PURE_SYSTEM_PROMPT = """You are a strategic assistant for a vision-only game agent.

Return STRICT JSON only (one line), no extra keys, no prose:
{"directive":{"action":"approach|retreat|explore|interact","target":[x,y],"duration_seconds":n,"priority":"low|medium|high|critical"},"confidence":0..1}

If Agent State.menu_hint=true: action MUST be "interact" and target MUST be the primary advance button (PLAY/CONFIRM/CONTINUE/START/OK/NEXT/RESUME/RETRY).
- Prefer Agent State.ui_elements centers (cx,cy) when available.
- If unsure, target=[0.5,0.5] and confidence<=0.2.
"""

METABONK_GAME_SYSTEM_PROMPT = """You are a strategic AI agent playing MetaBonk, a fast-paced roguelike game.

**Your Role**: Provide high-level strategic directives based on visual analysis.

**Visual Input**: You will see 9 frames:
- 5 past frames (showing recent history)
- 1 current frame (now)
- 3 predicted future frames (what will likely happen)

Return ONLY a JSON object with keys: reasoning, goal, strategy, directive, confidence.
directive: {action, target:[x,y] 0..1, duration_seconds, priority}
"""


def _system_prompt() -> str:
    mode = str(os.environ.get("METABONK_COGNITIVE_PROMPT_MODE", "") or "").strip().lower()
    if mode in ("pure", "vision", "agnostic", "generic"):
        return METABONK_PURE_SYSTEM_PROMPT
    if mode in ("game", "metabonk", "classic"):
        return METABONK_GAME_SYSTEM_PROMPT
    if _env_truthy("METABONK_PURE_VISION_MODE"):
        return METABONK_PURE_SYSTEM_PROMPT
    return METABONK_GAME_SYSTEM_PROMPT

class CognitiveServer:
    def __init__(
        self,
        *,
        model_path: str,
        tp_size: int,
        quantization: str,
        zmq_port: int = 5555,
        max_running_requests: int = 64,
        sglang_port: int = 30000,
        rl_log_dir: Optional[str] = None,
        mock: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.zmq_port = int(zmq_port)
        self.sglang_port = int(sglang_port)
        self.tp_size = int(tp_size)
        self.quantization = str(quantization or "").strip()
        self.start_ts = float(time.time())
        # Requests sent before the backend is ready are treated as stale (common after restarts).
        self._ready_ts = float(self.start_ts)
        self.mock = bool(mock)

        self.backend = "mock" if self.mock else _select_backend(model_path=self.model_path)

        self._hf_model = None
        self._hf_processor = None
        self._sglang_proc: Optional[subprocess.Popen] = None
        self._sglang_backend = None
        if not self.mock:
            if self.backend == "sglang":
                self._start_sglang_server(max_running_requests=int(max_running_requests))
            elif self.backend == "transformers":
                self.model_path = _prepare_model_path_for_transformers(self.model_path)
                logger.info("Initializing Transformers backend...")
                from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor  # type: ignore

                processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

                # Prefer SDPA attention on modern PyTorch (fast, no extra deps). If anything
                # goes wrong, fall back to eager attention for compatibility.
                wanted_attn = str(
                    os.environ.get("METABONK_COGNITIVE_ATTN_IMPL", "flash_attention_2") or "flash_attention_2"
                ).strip().lower()
                if wanted_attn in ("", "auto"):
                    wanted_attn = "flash_attention_2"
                attn_candidates: List[str] = []
                for cand in (wanted_attn, "flash_attention_2", "sdpa", "eager"):
                    if cand and cand not in attn_candidates:
                        attn_candidates.append(cand)

                # Safe, best-effort runtime knobs (helps throughput on Blackwell/Hopper).
                try:
                    if hasattr(torch, "set_float32_matmul_precision"):
                        torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
                try:
                    if hasattr(torch.backends, "cudnn"):
                        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
                except Exception:
                    pass

                last_exc: Optional[BaseException] = None
                model = None
                applied_attn = "eager"
                for attn_impl in attn_candidates:
                    try:
                        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
                        try:
                            setattr(cfg, "_attn_implementation", attn_impl)
                        except Exception:
                            pass
                        try:
                            setattr(cfg, "_attn_implementation_internal", attn_impl)
                        except Exception:
                            pass
                        load_kwargs: Dict[str, Any] = {
                            "trust_remote_code": True,
                            "torch_dtype": "auto",
                            "config": cfg,
                        }
                        # Transformers may reject sdpa/flash via `attn_implementation=...` even when the
                        # underlying trust_remote_code model supports it via config._attn_implementation.
                        # Try the config-only path first, then fall back to the explicit arg.
                        try:
                            model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs).cuda()
                        except Exception:
                            model = AutoModelForCausalLM.from_pretrained(
                                self.model_path,
                                attn_implementation=attn_impl,
                                **load_kwargs,
                            ).cuda()
                        applied_attn = str(attn_impl)
                        break
                    except Exception as e:  # pragma: no cover
                        last_exc = e
                        logger.warning("Transformers init failed (attn_implementation=%s): %s", attn_impl, e)
                        model = None
                if model is None:  # pragma: no cover
                    raise RuntimeError(f"Failed to initialize Transformers backend: {last_exc}") from last_exc
                logger.info("Transformers backend initialized (attn_implementation=%s)", applied_attn)
                # Prefer fast KV caching; we keep prompts short and generate very few tokens.
                # Note: Phi-3-Vision uses the new `Cache` interface in its attention module.
                try:
                    setattr(model.config, "use_cache", True)
                except Exception:
                    pass
                try:
                    if hasattr(model, "generation_config") and model.generation_config is not None:
                        model.generation_config.use_cache = True
                        # Avoid pre-creating sliding-window cache layers for Phi-3-Vision.
                        # `dynamic_full` skips passing the model config to `DynamicCache`, which otherwise
                        # initializes a sliding-window cache based on `sliding_window`.
                        model.generation_config.cache_implementation = "dynamic_full"
                except Exception:
                    pass
                model.eval()
                self._hf_processor = processor
                self._hf_model = model
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

        self.temporal_processor = TemporalFrameProcessor()
        self.zmq_bridge = ZeroMQBridge(port=self.zmq_port)

        concurrency = int(max_running_requests)
        if self.backend == "transformers":
            try:
                concurrency = int(os.environ.get("METABONK_COGNITIVE_MAX_CONCURRENCY", "1") or 1)
            except Exception:
                concurrency = 1
        self._sema = asyncio.Semaphore(max(1, concurrency))
        self._tasks: set[asyncio.Task] = set()

        self.rl_logger: Optional[RLLogger] = RLLogger(rl_log_dir) if rl_log_dir else None

        self.request_count = 0
        self.metrics_request_count = 0
        self.total_latency_s = 0.0
        self.active_agents: set[str] = set()
        self._agent_counts: Dict[str, int] = {}
        self._agent_latency_s: Dict[str, float] = {}
        self._agent_last_ts: Dict[str, float] = {}

        if self.backend != "sglang":
            self._ready_ts = float(time.time())

        logger.info(
            "✅ Cognitive Server initialized (backend=%s, model=%s, zmq_port=%d)",
            self.backend,
            self.model_path,
            self.zmq_port,
        )

    def _wait_for_port(self, *, host: str, port: int, timeout_s: float = 60.0) -> None:
        deadline = float(time.time() + float(timeout_s))
        last_err: Optional[BaseException] = None
        while time.time() < deadline:
            try:
                with socket.create_connection((str(host), int(port)), timeout=0.25):
                    return
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(0.25)
        raise RuntimeError(f"SGLang server did not become ready on {host}:{port}: {last_err}")

    def _wait_for_sglang_http_ready(self, *, host: str, port: int, timeout_s: float = 180.0) -> None:
        """Wait until the SGLang HTTP API is ready.

        A TCP accept is not sufficient; the port can open before the model finishes loading.
        We poll the OpenAI-compatible `/v1/models` endpoint.
        """

        deadline = float(time.time() + float(timeout_s))
        url = f"http://{str(host)}:{int(port)}/v1/models"
        last_err: Optional[BaseException] = None
        while time.time() < deadline:
            try:
                req = Request(url, method="GET")
                with urlopen(req, timeout=1.0) as resp:
                    status = int(getattr(resp, "status", 200) or 200)
                    payload_raw = resp.read()
                if status != 200:
                    raise RuntimeError(f"status={status}")
                try:
                    payload = json.loads(payload_raw.decode("utf-8", errors="replace"))
                except Exception as e:
                    raise RuntimeError(f"invalid json: {e}") from e
                data = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(data, list) and data:
                    return
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(0.5)
        raise RuntimeError(f"SGLang HTTP readiness check failed for {url}: {last_err}")

    def _warmup_sglang(self, *, timeout_s: float = 180.0) -> None:
        """Warm up SGLang with a real generation request.

        `/v1/models` can become available before the multimodal generation path is ready.
        We loop until a tiny multimodal request succeeds so the first agent request doesn't
        pay the cold-start cost (and to avoid skewing latency metrics after restarts).
        """

        if self.backend != "sglang":
            return
        if self._sglang_backend is None or sgl is None or RuntimeEndpoint is None:
            return
        if "metabonk_strategist" not in globals():  # pragma: no cover
            return

        # Use a representative 16:9 frame so the first *real* request doesn't pay
        # extra compilation costs (e.g. different image shapes / multimodal kernels).
        try:
            img = Image.new("RGB", (640, 360), (0, 0, 0))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=70)
            warmup_frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return

        deadline = float(time.time() + float(timeout_s))
        last_err: Optional[BaseException] = None
        while time.time() < deadline:
            try:
                # Run through the same codepath we use for real traffic so the warmup
                # covers image resize, prompt building, and decode length.
                _ = self._run_inference_sglang(
                    frames=[warmup_frame_b64],
                    agent_state={
                        "menu_hint": True,
                        "ui_elements": [{"name": "CONFIRM", "cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.1, "conf": 0.9}],
                    },
                )
                return
            except Exception as e:
                last_err = e
                time.sleep(0.5)

        raise RuntimeError(f"SGLang warmup request did not succeed: {last_err}")

    def _start_sglang_server(self, *, max_running_requests: int) -> None:
        if RuntimeEndpoint is None or sgl is None:
            raise RuntimeError("SGLang is not installed; cannot enable METABONK_COGNITIVE_BACKEND=sglang")

        model_path = _prepare_model_path_for_sglang(self.model_path)
        port = int(self.sglang_port)
        host = "127.0.0.1"

        try:
            context_len = int(os.environ.get("METABONK_COGNITIVE_CONTEXT_LEN", "4096") or 4096)
        except Exception:
            context_len = 4096
        context_len = max(256, int(context_len))

        # NOTE: docker-compose often injects empty-string env vars (VAR="") when unset.
        # For these knobs, treat empty as "not set" and fall back to safe defaults.
        quant = str(self.quantization or os.environ.get("METABONK_COGNITIVE_SGLANG_QUANTIZATION") or "").strip()
        dtype = str(os.environ.get("METABONK_COGNITIVE_SGLANG_DTYPE") or "half").strip().lower()
        attn_backend = str(os.environ.get("METABONK_COGNITIVE_SGLANG_ATTENTION_BACKEND") or "flashinfer").strip()
        mm_attn_backend = str(os.environ.get("METABONK_COGNITIVE_SGLANG_MM_ATTENTION_BACKEND") or "sdpa").strip()
        kv_cache_dtype = str(os.environ.get("METABONK_COGNITIVE_SGLANG_KV_CACHE_DTYPE") or "").strip()
        mem_fraction_raw = str(os.environ.get("METABONK_COGNITIVE_SGLANG_MEM_FRACTION") or "").strip()
        disable_cuda_graph_env = os.environ.get("METABONK_COGNITIVE_SGLANG_DISABLE_CUDA_GRAPH")
        if disable_cuda_graph_env is not None and not str(disable_cuda_graph_env).strip():
            disable_cuda_graph_env = None
        disable_cuda_graph = _env_truthy("METABONK_COGNITIVE_SGLANG_DISABLE_CUDA_GRAPH")
        if disable_cuda_graph_env is None:
            try:
                disable_cuda_graph = _detect_model_type(str(model_path)) == "phi3_v"
            except Exception:
                disable_cuda_graph = False

        cmd: List[str] = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model_path),
            "--trust-remote-code",
            "--enable-multimodal",
            "--host",
            str(host),
            "--port",
            str(port),
            "--context-length",
            str(context_len),
            "--max-running-requests",
            str(max(1, int(max_running_requests))),
            "--dtype",
            str(dtype),
        ]
        if quant:
            cmd.extend(["--quantization", quant])
        if kv_cache_dtype:
            cmd.extend(["--kv-cache-dtype", kv_cache_dtype])
        if mem_fraction_raw:
            cmd.extend(["--mem-fraction-static", str(mem_fraction_raw)])
        if attn_backend:
            cmd.extend(["--attention-backend", attn_backend])
            cmd.extend(["--prefill-attention-backend", attn_backend])
            cmd.extend(["--decode-attention-backend", attn_backend])
        if mm_attn_backend:
            cmd.extend(["--mm-attention-backend", mm_attn_backend])
        if disable_cuda_graph:
            cmd.append("--disable-cuda-graph")

        logger.info("Starting SGLang server: %s", " ".join(cmd))
        self._sglang_proc = subprocess.Popen(cmd, start_new_session=True)
        self._wait_for_port(host=host, port=port, timeout_s=120.0)
        self._wait_for_sglang_http_ready(host=host, port=port, timeout_s=180.0)
        self._sglang_backend = RuntimeEndpoint(f"http://{host}:{port}")
        logger.info("✅ SGLang backend ready: %s", f"http://{host}:{port}")
        self._warmup_sglang()
        self._ready_ts = float(time.time())

    def _stop_sglang_server(self) -> None:
        proc = self._sglang_proc
        self._sglang_proc = None
        self._sglang_backend = None
        if proc is None:
            return
        try:
            proc.terminate()
        except Exception:
            return
        try:
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _normalize_frames(self, frames: List[str], *, agent_state: Optional[Dict[str, Any]] = None) -> List[str]:
        """Normalize incoming frames for the prompt image builder.

        By default we keep a 9-frame strip. When the server is configured to use
        single-frame mode (or when menu_hint + single-frame menu path is active),
        we keep only the most recent frame to reduce decode overhead.
        """
        frames = list(frames or [])
        if not frames:
            return frames

        if _env_truthy("METABONK_COGNITIVE_ALWAYS_SINGLE_FRAME"):
            return [frames[-1]]

        agent_state = dict(agent_state or {})
        menu_hint = agent_state.get("menu_hint")
        if isinstance(menu_hint, str):
            menu_hint = menu_hint.strip().lower() in ("1", "true", "yes", "on")
        if bool(menu_hint):
            menu_single_frame = str(
                os.environ.get("METABONK_COGNITIVE_MENU_SINGLE_FRAME", "1") or "1"
            ).strip().lower() in ("1", "true", "yes", "on")
            if menu_single_frame:
                return [frames[-1]]

        # Ensure exactly 9 frames for the temporal grid.
        if len(frames) < 9:
            frames = frames + [frames[-1]] * (9 - len(frames))
        elif len(frames) > 9:
            frames = frames[-9:]
        return frames

    def _make_temporal_image(
        self, frames_b64: List[str], *, agent_state: Optional[Dict[str, Any]] = None
    ) -> Image.Image:
        tile_edge = int(os.environ.get("METABONK_COGNITIVE_TILE_EDGE", "224") or 224)
        try:
            menu_tile_edge = int(os.environ.get("METABONK_COGNITIVE_TILE_EDGE_MENU") or tile_edge)
        except Exception:
            menu_tile_edge = tile_edge
        menu_single_frame = str(os.environ.get("METABONK_COGNITIVE_MENU_SINGLE_FRAME", "1") or "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        def _resize_max_edge(img: Image.Image, edge: int) -> Image.Image:
            edge = int(edge)
            if edge <= 0:
                return img
            w, h = img.size
            if max(w, h) <= edge:
                return img
            scale = float(edge) / float(max(1, max(w, h)))
            nw = max(1, int(round(float(w) * scale)))
            nh = max(1, int(round(float(h) * scale)))
            return img.resize((nw, nh), resample=Image.BILINEAR)

        agent_state = dict(agent_state or {})
        force_single = _env_truthy("METABONK_COGNITIVE_ALWAYS_SINGLE_FRAME")

        if force_single:
            try:
                raw = base64.b64decode(str(frames_b64[-1] if frames_b64 else ""), validate=False)
                img = Image.open(BytesIO(raw)).convert("RGB")
            except Exception:
                img = Image.new("RGB", (512, 512), (0, 0, 0))
            img = _resize_max_edge(img, int(tile_edge))
            return img

        menu_hint = agent_state.get("menu_hint")
        if isinstance(menu_hint, str):
            menu_hint = menu_hint.strip().lower() in ("1", "true", "yes", "on")
        if bool(menu_hint):
            if menu_single_frame:
                try:
                    raw = base64.b64decode(str(frames_b64[-1] if frames_b64 else ""), validate=False)
                    img = Image.open(BytesIO(raw)).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (512, 512), (0, 0, 0))
                img = _resize_max_edge(img, int(menu_tile_edge))
                return img
            tile_edge = int(menu_tile_edge)
        tile_edge = max(64, min(1024, tile_edge))

        tiles: List[Image.Image] = []
        for s in frames_b64:
            try:
                raw = base64.b64decode(str(s or ""), validate=False)
                img = Image.open(BytesIO(raw)).convert("RGB")
            except Exception:
                img = Image.new("RGB", (tile_edge, tile_edge), (0, 0, 0))
            w, h = img.size
            scale = min(float(tile_edge) / max(1.0, float(w)), float(tile_edge) / max(1.0, float(h)))
            nw = max(1, int(round(float(w) * scale)))
            nh = max(1, int(round(float(h) * scale)))
            if (nw, nh) != (w, h):
                img = img.resize((nw, nh), resample=Image.BILINEAR)
            tile = Image.new("RGB", (tile_edge, tile_edge), (0, 0, 0))
            ox = (tile_edge - nw) // 2
            oy = (tile_edge - nh) // 2
            tile.paste(img, (int(ox), int(oy)))
            tiles.append(tile)

        while len(tiles) < 9:
            tiles.append(Image.new("RGB", (tile_edge, tile_edge), (0, 0, 0)))
        if len(tiles) > 9:
            tiles = tiles[-9:]

        grid = Image.new("RGB", (tile_edge * 3, tile_edge * 3), (0, 0, 0))
        for idx, tile in enumerate(tiles[:9]):
            r = idx // 3
            c = idx % 3
            grid.paste(tile, (c * tile_edge, r * tile_edge))

        return grid

    def _run_inference_transformers(self, *, temporal_img: Image.Image, agent_state: Dict[str, Any]) -> str:
        if self.backend != "transformers":
            raise RuntimeError(f"_run_inference_transformers called with backend={self.backend}")
        if self._hf_model is None or self._hf_processor is None:
            raise RuntimeError("Transformers backend not initialized")
        img = temporal_img

        user_prompt = "<|user|>\n"
        assistant_prompt = "<|assistant|>\n"
        prompt_suffix = "<|end|>\n"
        prompt_state = _sanitize_agent_state_for_prompt(dict(agent_state or {}))
        state_json = json.dumps(prompt_state, ensure_ascii=False, separators=(",", ":"))
        prompt = (
            f"{user_prompt}{_system_prompt()}\n\n"
            f"<|image_1|>\n"
            f"Agent State: {state_json}\n\n"
            "Task: Analyze the temporal grid and provide a strategic directive. "
            "Respond with STRICT JSON only, no additional text.\n"
            f"{prompt_suffix}{assistant_prompt}"
        )

        inputs = self._hf_processor(prompt, [img], return_tensors="pt").to(self._hf_model.device)
        try:
            max_new = int(os.environ.get("METABONK_COGNITIVE_MAX_NEW_TOKENS", "64") or 64)
        except Exception:
            max_new = 64
        with torch.inference_mode():
            do_sample = _env_truthy("METABONK_COGNITIVE_DO_SAMPLE")
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max(1, min(1024, int(max_new))),
                "do_sample": bool(do_sample),
                "eos_token_id": self._hf_processor.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if do_sample:
                try:
                    temperature = float(os.environ.get("METABONK_COGNITIVE_TEMPERATURE", "0.7") or 0.7)
                except Exception:
                    temperature = 0.7
                gen_kwargs["temperature"] = float(max(0.0, temperature))
            generate_ids = self._hf_model.generate(**inputs, **gen_kwargs)
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        out = self._hf_processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return str(out)

    def _run_inference_sglang(self, *, frames: List[str], agent_state: Dict[str, Any]) -> str:
        if self.backend != "sglang":
            raise RuntimeError(f"_run_inference_sglang called with backend={self.backend}")
        if self._sglang_backend is None:
            raise RuntimeError("SGLang backend not initialized")
        if sgl is None or RuntimeEndpoint is None or "metabonk_strategist" not in globals():  # pragma: no cover
            raise RuntimeError("SGLang prompt function not available")

        # Reuse the same prompt image builder as the Transformers path (temporal grid / single-frame menu).
        # SGLang's `image()` helper expects a file path, so we write the prompt image to /tmp.
        tmp_path: Optional[Path] = None
        keep_tmp = _env_truthy("METABONK_COGNITIVE_SGLANG_KEEP_TMP_IMAGES")
        tmp_dir = Path(
            str(os.environ.get("METABONK_COGNITIVE_SGLANG_TMP_DIR") or "/tmp/metabonk_sglang_imgs")
        )
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            tmp_dir = Path("/tmp")
        frame_payloads = list(frames or [])
        temporal_img: Optional[Image.Image] = None

        try:
            max_new = int(os.environ.get("METABONK_COGNITIVE_MAX_NEW_TOKENS", "64") or 64)
        except Exception:
            max_new = 64
        try:
            temperature = float(os.environ.get("METABONK_COGNITIVE_TEMPERATURE", "0.0") or 0.0)
        except Exception:
            temperature = 0.0
        prompt_state = _sanitize_agent_state_for_prompt(dict(agent_state or {}))
        agent_state_json = json.dumps(prompt_state, ensure_ascii=False, separators=(",", ":"))
        try:
            with tempfile.NamedTemporaryFile(dir=tmp_dir, prefix="frame_", suffix=".jpg", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            # Workers already send JPEG frames. If we don't need to resize, write bytes directly.
            # If we *do* need to resize (common when METABONK_COGNITIVE_TILE_EDGE is small),
            # prefer decoding once and writing a smaller JPEG to reduce VLM compute.
            wrote_file = False
            if len(frame_payloads) == 1:
                try:
                    raw = base64.b64decode(str(frame_payloads[0] or ""), validate=False)
                except Exception:
                    raw = b""
                if raw:
                    try:
                        tile_edge = int(os.environ.get("METABONK_COGNITIVE_TILE_EDGE", "224") or 224)
                    except Exception:
                        tile_edge = 224
                    try:
                        menu_edge = int(os.environ.get("METABONK_COGNITIVE_TILE_EDGE_MENU") or tile_edge)
                    except Exception:
                        menu_edge = tile_edge
                    menu_hint = dict(agent_state or {}).get("menu_hint")
                    if isinstance(menu_hint, str):
                        menu_hint = menu_hint.strip().lower() in ("1", "true", "yes", "on")
                    if bool(menu_hint) and int(menu_edge) > 0:
                        tile_edge = int(menu_edge)
                    tile_edge = max(64, min(1024, int(tile_edge)))
                    force_resize = _env_truthy("METABONK_COGNITIVE_FORCE_RESIZE")

                    try:
                        with Image.open(BytesIO(raw)) as img_probe:
                            w, h = img_probe.size
                        need_resize = tile_edge > 0 and max(int(w), int(h)) > int(tile_edge)
                    except Exception:
                        need_resize = bool(force_resize)

                    if (not force_resize) and (not need_resize):
                        tmp_path.write_bytes(raw)
                        wrote_file = True
                    else:
                        with Image.open(BytesIO(raw)) as img0:
                            img = img0.convert("RGB")
                            w, h = img.size
                            if tile_edge > 0 and max(int(w), int(h)) > int(tile_edge):
                                scale = float(tile_edge) / float(max(1, max(int(w), int(h))))
                                nw = max(1, int(round(float(w) * scale)))
                                nh = max(1, int(round(float(h) * scale)))
                                img = img.resize((nw, nh), resample=Image.BILINEAR)
                            img.save(str(tmp_path), format="JPEG", quality=85)
                        wrote_file = True
                else:
                    wrote_file = False

            if not wrote_file:
                if temporal_img is None:
                    try:
                        temporal_img = self._make_temporal_image(frame_payloads, agent_state=agent_state)
                    except Exception:
                        temporal_img = Image.new("RGB", (512, 512), (0, 0, 0))
                temporal_img.save(str(tmp_path), format="JPEG", quality=85)

            result = metabonk_strategist.run(  # type: ignore[attr-defined]
                temporal_frames=[str(tmp_path)],
                agent_state_json=str(agent_state_json),
                system_prompt=_system_prompt(),
                max_tokens=int(max_new),
                temperature=float(temperature),
                backend=self._sglang_backend,
            )
            try:
                return str((result or {}).get("response") or "")
            except Exception:
                return str(result or {})
        finally:
            if tmp_path is not None and not keep_tmp:
                try:
                    tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
                except Exception:
                    try:
                        os.unlink(str(tmp_path))
                    except Exception:
                        pass

    def _normalize_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize slightly-off schema variants to what workers expect."""
        if not isinstance(strategy, dict):
            return {}

        # Some prompts/models emit the directive fields at the top-level instead of
        # nesting them under "directive".
        if "directive" not in strategy:
            if isinstance(strategy.get("action"), str) and any(
                k in strategy for k in ("target", "duration_seconds", "priority")
            ):
                strategy["directive"] = {
                    "action": strategy.get("action"),
                    "target": strategy.get("target"),
                    "duration_seconds": strategy.get("duration_seconds"),
                    "priority": strategy.get("priority"),
                }
            else:
                maybe_action = strategy.get("action")
                if isinstance(maybe_action, dict):
                    strategy["directive"] = maybe_action
        directive = strategy.get("directive")
        # Some models emit `directive: "interact"` with the remaining fields at the top-level.
        # Normalize this into the nested schema workers expect.
        if isinstance(directive, str) and directive.strip():
            strategy["directive"] = {
                "action": directive,
                "target": strategy.get("target"),
                "duration_seconds": strategy.get("duration_seconds"),
                "priority": strategy.get("priority"),
            }
            directive = strategy.get("directive")
        if not isinstance(directive, dict):
            directive = {}
            strategy["directive"] = directive

        reasoning = strategy.get("reasoning")
        if not isinstance(reasoning, str):
            strategy["reasoning"] = ""
        goal = strategy.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            strategy["goal"] = "survive"
        strat = strategy.get("strategy")
        if not isinstance(strat, str) or not strat.strip():
            strategy["strategy"] = "explore"

        action = str(directive.get("action") or "").strip().lower()
        if action not in ("approach", "retreat", "explore", "interact"):
            if action in ("move", "collect"):
                action = "explore"
            elif action in ("attack", "defend"):
                action = "approach"
            else:
                action = "explore"
        directive["action"] = action

        target = directive.get("target")
        if not (isinstance(target, (list, tuple)) and len(target) >= 2):
            target = [0.5, 0.5]
        try:
            tx = float(target[0])
            ty = float(target[1])
        except Exception:
            tx, ty = 0.5, 0.5
        directive["target"] = [max(0.0, min(1.0, tx)), max(0.0, min(1.0, ty))]

        try:
            dur = float(directive.get("duration_seconds") or 2.0)
        except Exception:
            dur = 2.0
        directive["duration_seconds"] = float(max(0.25, min(10.0, dur)))

        pr_raw = directive.get("priority")
        if isinstance(pr_raw, (int, float)):
            pr = "high" if float(pr_raw) >= 1.5 else "medium"
        else:
            pr = str(pr_raw or "medium").strip().lower()
        if pr not in ("critical", "high", "medium", "low"):
            pr = "medium"
        directive["priority"] = pr

        conf = strategy.get("confidence")
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        strategy["confidence"] = float(max(0.0, min(1.0, conf_f)))
        return strategy

    @staticmethod
    def _normalize_ui_text(text: str) -> str:
        cleaned = "".join(ch for ch in str(text or "").upper() if ch.isalnum())
        return cleaned[:64]

    @classmethod
    def _advance_token_similarity(cls, text: str) -> float:
        text_n = cls._normalize_ui_text(text)
        if not text_n:
            return 0.0
        tokens = (
            "CONFIRM",
            "CONTINUE",
            "PLAY",
            "START",
            "OK",
            "ACCEPT",
            "NEXT",
            "RESUME",
            "RETRY",
            "BEGIN",
            "GO",
            "YES",
        )
        best = 0.0
        for tok in tokens:
            if tok in text_n or text_n in tok:
                best = max(best, 0.98)
                continue
            best = max(best, difflib.SequenceMatcher(None, text_n, tok).ratio())
        return float(best)

    @classmethod
    def _pick_menu_advance_target(cls, ui_elements: Any) -> Optional[List[float]]:
        if not isinstance(ui_elements, list) or not ui_elements:
            return None

        best = None
        best_score = -1.0

        def _candidate(el: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(el, dict):
                return None
            try:
                cx = float(el.get("cx"))
                cy = float(el.get("cy"))
                ww = float(el.get("w") or 0.0)
                hh = float(el.get("h") or 0.0)
            except Exception:
                return None
            try:
                conf = float(el.get("conf") or 0.0)
            except Exception:
                conf = 0.0
            name = str(el.get("name") or "")
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                return None
            ww = max(0.0, min(1.0, ww))
            hh = max(0.0, min(1.0, hh))
            area = ww * hh
            return {"name": name, "cx": cx, "cy": cy, "w": ww, "h": hh, "conf": conf, "area": area}

        candidates: List[Dict[str, Any]] = []
        for el in ui_elements:
            c = _candidate(el)
            if c is None:
                continue
            # Ignore tiny tokens like version strings, currency, etc.
            if c["w"] < 0.03 or c["h"] < 0.015:
                continue
            candidates.append(c)

        if not candidates:
            return None

        for c in candidates:
            sim = cls._advance_token_similarity(c["name"])
            score = 3.0 * sim + 0.8 * float(c["area"]) + 0.4 * float(c["conf"]) + 0.2 * float(c["cy"])
            score -= 0.05 * abs(float(c["cx"]) - 0.5)
            if score > best_score:
                best_score = score
                best = c

        if best is not None:
            return [float(best["cx"]), float(best["cy"])]

        return None

    @classmethod
    def _snap_target_to_ui(cls, target: List[float], ui_elements: Any) -> Optional[List[float]]:
        if not (isinstance(target, (list, tuple)) and len(target) >= 2):
            return None
        if not isinstance(ui_elements, list) or not ui_elements:
            return None
        try:
            tx = float(target[0])
            ty = float(target[1])
        except Exception:
            return None
        if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
            return None

        best = None
        best_d2 = 1e9
        for el in ui_elements:
            if not isinstance(el, dict):
                continue
            try:
                cx = float(el.get("cx"))
                cy = float(el.get("cy"))
            except Exception:
                continue
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                continue
            d2 = (tx - cx) ** 2 + (ty - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (cx, cy)
        if best is None:
            return None
        if best_d2 <= (0.12 * 0.12):
            return [float(best[0]), float(best[1])]
        return None

    @classmethod
    def _apply_menu_target_postproc(cls, strategy: Dict[str, Any], agent_state: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(strategy, dict) or not isinstance(agent_state, dict):
            return strategy
        menu_hint = agent_state.get("menu_hint")
        if isinstance(menu_hint, str):
            menu_hint = menu_hint.strip().lower() in ("1", "true", "yes", "on")
        if not bool(menu_hint):
            return strategy

        ui_elements = agent_state.get("ui_elements")
        directive = strategy.get("directive")
        if not isinstance(directive, dict):
            return strategy
        action = str(directive.get("action") or "").strip().lower()
        if action != "interact":
            return strategy

        target = directive.get("target")
        if not (isinstance(target, (list, tuple)) and len(target) >= 2):
            target = [0.5, 0.5]
            directive["target"] = target

        try:
            tx = float(target[0])
            ty = float(target[1])
        except Exception:
            tx = ty = 0.5

        conf = strategy.get("confidence")
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        low_conf = conf_f <= 0.25
        looks_default = abs(tx - 0.5) <= 0.12 and abs(ty - 0.5) <= 0.12

        if low_conf and looks_default:
            picked = cls._pick_menu_advance_target(ui_elements)
            if picked is not None:
                directive["target"] = picked
                strategy["confidence"] = float(max(conf_f, 0.25))
                if not str(strategy.get("reasoning") or "").strip():
                    strategy["reasoning"] = "click advance"
                return strategy

        snapped = cls._snap_target_to_ui(list(target), ui_elements)
        if snapped is not None:
            directive["target"] = snapped
        return strategy

    def _mock_strategy(self, *, agent_id: str, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        health_ratio = None
        try:
            health_ratio = float(agent_state.get("health_ratio")) if agent_state.get("health_ratio") is not None else None
        except Exception:
            health_ratio = None
        if health_ratio is not None and health_ratio < 0.3:
            action = "retreat"
            goal = "recover"
            strategy = "defensive"
        else:
            action = "collect"
            goal = "survive"
            strategy = "explore"
        return {
            "agent_id": agent_id,
            "reasoning": "Mock mode: heuristic directive (no VLM loaded).",
            "goal": goal,
            "strategy": strategy,
            "directive": {"action": action, "target": [0.5, 0.5], "duration_seconds": 2.0, "priority": "low"},
            "confidence": 0.05,
            "mock": True,
        }

    def _metrics_snapshot(self) -> Dict[str, Any]:
        avg_ms = (self.total_latency_s / max(1, self.request_count)) * 1000.0 if self.request_count else 0.0
        per_agent: Dict[str, Dict[str, Any]] = {}
        for aid, cnt in self._agent_counts.items():
            lat_s = float(self._agent_latency_s.get(aid, 0.0))
            per_agent[aid] = {
                "requests": int(cnt),
                "avg_latency_ms": (lat_s / max(1, int(cnt))) * 1000.0,
                "last_ts": float(self._agent_last_ts.get(aid, 0.0)),
            }
        return {
            "type": "metrics",
            "timestamp": float(time.time()),
            "uptime_s": float(time.time() - self.start_ts),
            "request_count": int(self.request_count),
            "metrics_request_count": int(self.metrics_request_count),
            "active_agents": int(len(self.active_agents)),
            "avg_latency_ms": float(avg_ms),
            "per_agent": per_agent,
        }

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        req_type = str(request_data.get("type") or "").strip().lower()
        if req_type == "metrics":
            self.metrics_request_count += 1
            return self._metrics_snapshot()

        agent_id = str(request_data.get("agent_id") or "unknown")
        self.active_agents.add(agent_id)

        # Ignore stale requests (commonly delivered after server restart) so we don't skew metrics.
        sent_ts = request_data.get("timestamp")
        try:
            sent_ts_f = float(sent_ts) if sent_ts is not None else None
        except Exception:
            sent_ts_f = None
        if sent_ts_f is not None and float(sent_ts_f) < float(self._ready_ts or 0.0):
            return {
                "agent_id": agent_id,
                "reasoning": "Server warming/restarted; ignoring stale request.",
                "goal": "survive",
                "strategy": "explore",
                "directive": {"action": "explore", "target": [0.5, 0.5], "duration_seconds": 1.0, "priority": "low"},
                "confidence": 0.0,
                "stale": True,
                "inference_time_ms": (time.time() - start) * 1000.0,
                "timestamp": time.time(),
            }

        if self.mock:
            strategy = self._mock_strategy(agent_id=agent_id, agent_state=dict(request_data.get("state") or {}))
            strategy["inference_time_ms"] = (time.time() - start) * 1000.0
            strategy["timestamp"] = time.time()
            self.request_count += 1
            dt = float(time.time() - start)
            self.total_latency_s += dt
            self._agent_counts[agent_id] = int(self._agent_counts.get(agent_id, 0)) + 1
            self._agent_latency_s[agent_id] = float(self._agent_latency_s.get(agent_id, 0.0)) + dt
            self._agent_last_ts[agent_id] = float(time.time())
            if self.rl_logger is not None:
                try:
                    self.rl_logger.log_decision(agent_id=agent_id, request_data=request_data, response_data=strategy)
                except Exception:
                    pass
            return strategy

        try:
            agent_state = dict(request_data.get("state") or {})
            frames = self._normalize_frames(list(request_data.get("frames") or []), agent_state=agent_state)

            # Optional temporal preprocessing hook (unused by the prompt image).
            # This is CPU-expensive (JPEG decode + tensor ops). Keep it opt-in.
            if _env_truthy("METABONK_COGNITIVE_ENABLE_TEMPORAL_PROCESSOR"):
                try:
                    _ = self.temporal_processor.process(frames)
                except Exception:
                    pass

            # Run model inference in a worker thread so the asyncio loop can keep polling ZMQ.
            if self.backend == "sglang":
                response_text = await asyncio.to_thread(
                    self._run_inference_sglang,
                    frames=frames,
                    agent_state=agent_state,
                )
            else:
                temporal_img = self._make_temporal_image(frames, agent_state=agent_state)
                response_text = await asyncio.to_thread(
                    self._run_inference_transformers,
                    temporal_img=temporal_img,
                    agent_state=agent_state,
                )

            # Extract JSON object from response.
            strategy = self._normalize_strategy(_extract_json_obj(response_text))
            strategy = self._apply_menu_target_postproc(strategy, agent_state)

            strategy["agent_id"] = agent_id
            strategy["inference_time_ms"] = (time.time() - start) * 1000.0
            strategy["timestamp"] = time.time()

            self.request_count += 1
            dt = float(time.time() - start)
            self.total_latency_s += dt
            self._agent_counts[agent_id] = int(self._agent_counts.get(agent_id, 0)) + 1
            self._agent_latency_s[agent_id] = float(self._agent_latency_s.get(agent_id, 0.0)) + dt
            self._agent_last_ts[agent_id] = float(time.time())

            if self.rl_logger is not None:
                try:
                    self.rl_logger.log_decision(agent_id=agent_id, request_data=request_data, response_data=strategy)
                except Exception:
                    pass

            logger.info(
                "%s: directive generated (%.1fms) goal=%s",
                agent_id,
                float(strategy.get("inference_time_ms", 0.0)),
                strategy.get("goal", "unknown"),
            )

            return strategy

        except Exception as e:
            logger.error("%s: error processing request: %s", agent_id, e)
            try:
                dt = float(time.time() - start)
                self._agent_counts[agent_id] = int(self._agent_counts.get(agent_id, 0)) + 1
                self._agent_latency_s[agent_id] = float(self._agent_latency_s.get(agent_id, 0.0)) + dt
                self._agent_last_ts[agent_id] = float(time.time())
            except Exception:
                pass
            return {
                "agent_id": agent_id,
                "reasoning": f"Error: {e}",
                "goal": "survive",
                "strategy": "defensive",
                "directive": {
                    "action": "retreat",
                    "target": [0, 0],
                    "duration_seconds": 2.0,
                    "priority": "high",
                },
                "confidence": 0.0,
                "error": True,
                "inference_time_ms": (time.time() - start) * 1000.0,
                "timestamp": time.time(),
            }

    async def _handle_one(self, identity: bytes, request_data: Dict[str, Any]) -> None:
        req_type = str(request_data.get("type") or "").strip().lower()
        # Metrics/health checks must not be blocked behind long-running model inference.
        if req_type == "metrics":
            response = await self.process_request(request_data)
            await self.zmq_bridge.send(identity, response)
            return

        async with self._sema:
            response = await self.process_request(request_data)
            await self.zmq_bridge.send(identity, response)

    async def run(self) -> None:
        logger.info("🚀 Cognitive Server starting (ZMQ port %d)...", self.zmq_port)
        await self.zmq_bridge.start()

        try:
            while True:
                req = await self.zmq_bridge.receive(timeout_ms=100)
                if req is None:
                    await asyncio.sleep(0.001)
                    continue

                identity, request_data = req
                task = asyncio.create_task(self._handle_one(identity, request_data))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

                if self.request_count and (self.request_count % 10 == 0):
                    avg_ms = (self.total_latency_s / max(1, self.request_count)) * 1000.0
                    logger.info(
                        "📊 Metrics: %d requests, %d agents, avg latency %.1fms",
                        self.request_count,
                        len(self.active_agents),
                        avg_ms,
                    )
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            # Drain pending tasks.
            if self._tasks:
                await asyncio.gather(*list(self._tasks), return_exceptions=True)
            await self.zmq_bridge.stop()
            self._stop_sglang_server()
            logger.info("✅ Cognitive Server stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaBonk Cognitive Server")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--quantization", type=str, default="")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--rl-log-dir", type=str, default=os.environ.get("METABONK_RL_LOG_DIR"))
    parser.add_argument("--mock", action="store_true", help="Run without loading a VLM (for smoke tests)")

    args = parser.parse_args()

    mock = bool(args.mock) or _env_truthy("METABONK_COGNITIVE_MOCK")
    server = CognitiveServer(
        model_path=args.model_path,
        tp_size=args.tp_size,
        quantization=args.quantization,
        zmq_port=args.zmq_port,
        max_running_requests=args.max_running_requests,
        sglang_port=args.sglang_port,
        rl_log_dir=args.rl_log_dir,
        mock=mock,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
