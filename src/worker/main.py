"""Worker service entrypoint.

Recovery implementation:
 - launches (stubbed) game instance
 - starts (stubbed) capture
 - maintains a local trainer state
 - sends periodic heartbeats to the orchestrator
 - exposes /status and /config endpoints
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import math as _math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except Exception as e:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    Request = None  # type: ignore
    uvicorn = None  # type: ignore
    _import_error = e
else:
    _import_error = None

from src.common.schemas import CurriculumConfig, HEARTBEAT_SCHEMA_VERSION, Heartbeat, InstanceConfig, TrainerConfig
from .launcher import GameLauncher
from .stream import CaptureStream
try:
    from .synthetic_eye_stream import SyntheticEyeStream, SyntheticEyeFrame  # type: ignore
    from .synthetic_eye_cuda import SyntheticEyeCudaIngestor  # type: ignore
except Exception:  # pragma: no cover
    SyntheticEyeStream = None  # type: ignore
    SyntheticEyeFrame = None  # type: ignore
    SyntheticEyeCudaIngestor = None  # type: ignore
from .trainer import Trainer
from .perception import construct_observation
from .rollout import LearnerClient, RolloutBuffer, PixelRolloutBuffer
from .nvenc_streamer import NVENCConfig, NVENCStreamer, PixelFrame, CudaFrame
from .fifo_publisher import FifoH264Publisher, FifoPublishConfig
from .highlight_recorder import HighlightConfig, HighlightRecorder
from .stream_health import jpeg_luma_thumbnail, luma_mean_abs_diff
from .gameplay_detector import HybridGameplayDetector

try:
    from src.agent.cognitive_client import CognitiveClient  # type: ignore
except Exception:  # pragma: no cover
    CognitiveClient = None  # type: ignore
try:
    from src.agent.rl_integration import RLLogger  # type: ignore
except Exception:  # pragma: no cover
    RLLogger = None  # type: ignore

try:
    from src.agent.visual_exploration_reward import VisualExplorationReward  # type: ignore
except Exception:  # pragma: no cover
    VisualExplorationReward = None  # type: ignore

try:
    from src.bridge.unity_bridge import UnityBridge, BridgeConfig, GameFrame
except Exception:  # pragma: no cover
    UnityBridge = None  # type: ignore
    BridgeConfig = None  # type: ignore
    GameFrame = None  # type: ignore

try:
    from src.input.uinput_backend import UInputBackend, UInputError
except Exception:  # pragma: no cover
    UInputBackend = None  # type: ignore
    UInputError = Exception  # type: ignore
try:
    from src.input.xdotool_backend import XDoToolBackend, XDoToolError
except Exception:  # pragma: no cover
    XDoToolBackend = None  # type: ignore
    XDoToolError = Exception  # type: ignore
try:
    from src.input.libxdo_backend import LibXDoBackend, LibXDoError
except Exception:  # pragma: no cover
    LibXDoBackend = None  # type: ignore
    LibXDoError = Exception  # type: ignore


app: Optional["FastAPI"] = FastAPI(title="MetaBonk Worker") if FastAPI else None
if app:
    # Allow the dev UI (Vite) to fetch /stream.mp4 with MSE. Keep it permissive for LAN/dev.
    try:
        origins = os.environ.get("METABONK_CORS_ORIGINS", "").strip()
        # Default to permissive CORS so MSE `fetch()` can read worker streams from:
        # - Vite dev server (5173)
        # - Orchestrator / static hosting (any port / LAN host)
        # The stream endpoints are read-only and do not use credentials.
        allow = [o.strip() for o in origins.split(",") if o.strip()] if origins else ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow if allow else ["*"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    except Exception:
        pass


def _game_restart_possible() -> bool:
    return bool(
        os.environ.get("MEGABONK_CMD")
        or os.environ.get("MEGABONK_COMMAND")
        or os.environ.get("MEGABONK_CMD_TEMPLATE")
        or os.environ.get("MEGABONK_COMMAND_TEMPLATE")
    )


class WorkerService:
    def __init__(
        self,
        instance_id: str,
        policy_name: str,
        orch_url: str,
        learner_url: str,
        obs_dim: int = 100,
        frame_stack: Optional[int] = None,
        vision_url: str = "http://127.0.0.1:8050",
        display: Optional[str] = None,
        hparams: Optional[dict] = None,
        host: str = "0.0.0.0",
        port: int = 5000,
    ):
        self.instance_id = instance_id
        self.policy_name = policy_name
        self.orch_url = orch_url.rstrip("/")
        self.learner_url = learner_url.rstrip("/")
        self.display = display
        self.vision_url = vision_url.rstrip("/")
        self.hparams = hparams or TrainerConfig().model_dump()
        self.host = host
        self.port = port
        self.launcher = GameLauncher(instance_id=instance_id, display=display)

        # GStreamer plugin registry is global per-user by default. When launching many workers,
        # concurrent `Gst.init()` calls can contend on the shared registry lock and stall stream
        # startup. Use a per-worker registry file to keep initialization independent.
        if "GST_REGISTRY_1_0" not in os.environ:
            try:
                repo_root = Path(os.environ.get("METABONK_REPO_ROOT", "") or ".").expanduser()
                if not repo_root.is_absolute():
                    repo_root = (Path.cwd() / repo_root).resolve()
                reg_dir = repo_root / "temp" / "gst_registry"
                reg_dir.mkdir(parents=True, exist_ok=True)
                os.environ["GST_REGISTRY_1_0"] = str(reg_dir / f"registry.{self.instance_id}.bin")
            except Exception:
                pass
        try:
            # Best-effort warmup so the first HTTP stream request doesn't do a slow init from
            # a threadpool worker (which can hang on some stacks).
            from src.worker.nvenc_streamer import Gst as _Gst  # type: ignore

            if _Gst is not None:
                _Gst.init(None)
        except Exception:
            pass

        # Optional centralized VLM System 2/3 reasoning (ZeroMQ client).
        self._system2_enabled = str(os.environ.get("METABONK_SYSTEM2_ENABLED", "0") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._cognitive_client = None
        self._system2_last_request: Optional[dict] = None
        self._system2_last_response: Optional[dict] = None
        self._system2_last_response_ts: float = 0.0
        try:
            trace_max = int(os.environ.get("METABONK_SYSTEM2_TRACE_MAX", "20"))
        except Exception:
            trace_max = 20
        self._system2_reasoning_trace: Deque[str] = deque(maxlen=max(1, int(trace_max)))
        self._system2_rl_logger = None
        self._system2_active_decision_id: Optional[str] = None
        self._system2_active_started_ts: float = 0.0
        self._system2_active_reward_sum: float = 0.0
        self._system2_last_frame_add_ts: float = 0.0
        try:
            self._system2_blend = float(os.environ.get("METABONK_SYSTEM2_BLEND", "0.7"))
        except Exception:
            self._system2_blend = 0.7
        self._system2_blend = max(0.0, min(1.0, float(self._system2_blend)))
        try:
            cooldown_raw = str(os.environ.get("METABONK_SYSTEM2_UI_CLICK_COOLDOWN_S", "") or "").strip()
            if not cooldown_raw:
                # Back-compat (deprecated)
                cooldown_raw = str(os.environ.get("METABONK_SYSTEM2_MENU_CLICK_COOLDOWN_S", "1.0") or "1.0")
            self._system2_ui_click_cooldown_s = float(cooldown_raw or 1.0)
        except Exception:
            self._system2_ui_click_cooldown_s = 1.0
        self._system2_last_ui_click_ts = 0.0
        self._system2_last_ui_click_resp_ts = 0.0
        self._vlm_hints_used: int = 0
        self._vlm_hints_applied: int = 0
        self._system2_active_applied: bool = False
        self._system2_override_cont = str(os.environ.get("METABONK_SYSTEM2_OVERRIDE_CONT", "1") or "").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if self._system2_enabled and CognitiveClient is not None:
            try:
                server_url = os.environ.get("METABONK_COGNITIVE_SERVER_URL", "tcp://127.0.0.1:5555")
                freq_s = float(os.environ.get("METABONK_STRATEGY_FREQUENCY", "2.0"))
                jpeg_q = int(os.environ.get("METABONK_SYSTEM2_JPEG_QUALITY", "85"))
                max_edge = int(os.environ.get("METABONK_SYSTEM2_MAX_EDGE", "512"))
                self._cognitive_client = CognitiveClient(
                    agent_id=self.instance_id,
                    server_url=server_url,
                    request_frequency_s=freq_s,
                    jpeg_quality=jpeg_q,
                    max_edge=max_edge,
                )
                rl_on = str(os.environ.get("METABONK_RL_LOGGING", "0") or "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                if rl_on and RLLogger is not None:
                    self._system2_rl_logger = RLLogger(os.environ.get("METABONK_RL_LOG_DIR"))
                print(
                    f"[worker:{self.instance_id}] System2 enabled "
                    f"(server={server_url}, freq={freq_s:.2f}s)",
                    flush=True,
                )
            except Exception as e:
                self._cognitive_client = None
                self._system2_enabled = False
                print(f"[worker:{self.instance_id}] WARN: System2 disabled (init failed): {e}", flush=True)

        # Optional stream overlay text file (used by ffmpeg drawtext). We create it early so
        # the encoder can reference it even before the first thought packet arrives.
        try:
            overlay_on = str(os.environ.get("METABONK_STREAM_OVERLAY", "0") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            overlay_path = str(os.environ.get("METABONK_STREAM_OVERLAY_FILE", "") or "").strip()
            if overlay_on and not overlay_path:
                run_dir = os.environ.get("METABONK_RUN_DIR") or os.environ.get("MEGABONK_LOG_DIR") or ""
                if run_dir:
                    try:
                        worker_id = os.environ.get("METABONK_WORKER_ID", "0")
                    except Exception:
                        worker_id = "0"
                    overlay_path = str(Path(run_dir) / "logs" / f"worker_{worker_id}_overlay.txt")
                    os.environ["METABONK_STREAM_OVERLAY_FILE"] = overlay_path
            if overlay_path:
                from src.worker.stream_overlay import ensure_overlay_file  # type: ignore

                ensure_overlay_file(overlay_path)
        except Exception:
            pass
        try:
            wm_overlay = str(
                os.environ.get("METABONK_WORLD_OVERLAY_FILE")
                or os.environ.get("METABONK_WM_OVERLAY_FILE")
                or ""
            ).strip()
            if wm_overlay:
                from src.worker.stream_overlay import start_world_overlay_watcher  # type: ignore

                poll_s = float(os.environ.get("METABONK_WORLD_OVERLAY_POLL_S", "0.5"))
                max_kb = int(os.environ.get("METABONK_WORLD_OVERLAY_MAX_KB", "2048"))
                max_hz = float(os.environ.get("METABONK_WORLD_OVERLAY_MAX_HZ", "2.0"))
                dedupe = str(os.environ.get("METABONK_WORLD_OVERLAY_DEDUPE", "1") or "").strip().lower() not in (
                    "0",
                    "false",
                    "no",
                    "off",
                )
                start_world_overlay_watcher(
                    path=wm_overlay,
                    instance_id=self.instance_id,
                    poll_s=poll_s,
                    max_kb=max_kb,
                    max_hz=max_hz,
                    dedupe=dedupe,
                )
        except Exception:
            pass
        self._frame_source = str(os.environ.get("METABONK_FRAME_SOURCE", "pipewire") or "").strip().lower()
        if self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf"):
            if SyntheticEyeStream is None:
                raise RuntimeError("SyntheticEyeStream unavailable (missing src/worker/synthetic_eye_stream.py)")
            self.stream = SyntheticEyeStream(socket_path=os.environ.get("METABONK_FRAME_SOCK"))  # type: ignore[assignment]
            audit_log_path = os.environ.get("METABONK_DMABUF_AUDIT_LOG")
            if not audit_log_path:
                run_dir = os.environ.get("METABONK_RUN_DIR") or os.environ.get("MEGABONK_LOG_DIR") or ""
                if run_dir:
                    try:
                        worker_id = os.environ.get("METABONK_WORKER_ID", "0")
                    except Exception:
                        worker_id = "0"
                    audit_log_path = str(Path(run_dir) / "logs" / f"worker_{worker_id}_dmabuf.log")
            # CUDA ingest is required to signal release fences (avoids compositor deadlock).
            if SyntheticEyeCudaIngestor is None:
                raise RuntimeError("SyntheticEyeCudaIngestor unavailable (missing CUDA bindings)")
            try:
                audit_interval_s = float(os.environ.get("METABONK_DMABUF_AUDIT_INTERVAL_S", "2.0"))
            except Exception:
                audit_interval_s = 2.0
            self._synthetic_eye_ingestor = SyntheticEyeCudaIngestor(  # type: ignore[call-arg]
                audit_log_path=audit_log_path,
                audit_interval_s=audit_interval_s,
            )
            try:
                self._synthetic_eye_cu_stream_ptr = int(getattr(self._synthetic_eye_ingestor, "stream", 0) or 0)
            except Exception:
                self._synthetic_eye_cu_stream_ptr = 0
        else:
            use_dmabuf = True
            audit_log_path = os.environ.get("METABONK_DMABUF_AUDIT_LOG")
            if not audit_log_path:
                run_dir = os.environ.get("METABONK_RUN_DIR") or os.environ.get("MEGABONK_LOG_DIR") or ""
                if run_dir:
                    try:
                        worker_id = os.environ.get("METABONK_WORKER_ID", "0")
                    except Exception:
                        worker_id = "0"
                    audit_log_path = str(Path(run_dir) / "logs" / f"worker_{worker_id}_dmabuf.log")
            self.stream = CaptureStream(
                pipewire_node=os.environ.get("PIPEWIRE_NODE"),
                use_dmabuf=use_dmabuf,
                audit_log_path=audit_log_path,
            )
            self._synthetic_eye_ingestor = None
            self._synthetic_eye_cu_stream_ptr = 0
        # Starting a GStreamer PipeWire capture pipeline can be fragile on some systems
        # (driver/gi/gstreamer mismatches). For stream HUD purposes we only need NVENC,
        # so keep capture opt-in and default it off.
        self._gst_capture_enabled = os.environ.get("METABONK_GST_CAPTURE", "0") in ("1", "true", "True")

        # Synthetic Eye lock-step: advance one frame per step (PING -> next FRAME).
        self._synthetic_eye_lockstep = False
        self._synthetic_eye_lockstep_wait_s = 0.0
        if self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf"):
            self._synthetic_eye_lockstep = str(os.environ.get("METABONK_SYNTHETIC_EYE_LOCKSTEP", "0") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if not self._synthetic_eye_lockstep:
                self._synthetic_eye_lockstep = str(os.environ.get("METABONK_EYE_LOCKSTEP", "0") or "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
            try:
                self._synthetic_eye_lockstep_wait_s = float(
                    os.environ.get(
                        "METABONK_SYNTHETIC_EYE_LOCKSTEP_WAIT_S",
                        os.environ.get("METABONK_EYE_LOCKSTEP_WAIT_S", "0.5"),
                    )
                )
            except Exception:
                self._synthetic_eye_lockstep_wait_s = 0.5
            self._synthetic_eye_lockstep_wait_s = max(0.0, float(self._synthetic_eye_lockstep_wait_s))

        # Observation backend (policy input): detections (default) or pixels (end-to-end vision).
        self._obs_backend = str(os.environ.get("METABONK_OBS_BACKEND", "detections") or "").strip().lower()
        if not self._obs_backend:
            self._obs_backend = "detections"
        if self._obs_backend not in ("detections", "pixels", "hybrid", "cutile"):
            raise ValueError(f"invalid METABONK_OBS_BACKEND={self._obs_backend!r}")
        self._pixel_obs_enabled = self._obs_backend in ("pixels", "hybrid", "cutile")
        # Pixel observation refresh rate. When using Synthetic Eye, the drain loop can easily
        # become compute-bound if we normalize pixels for *every* compositor frame. Throttle
        # pixel obs refresh to a reasonable rate while still servicing fences at full speed.
        #
        # Note: this affects agent obs updates only (not the compositor/frame ingest rate).
        self._pixel_obs_last_emit_ts: float = 0.0
        self._pixel_obs_interval_s: float = 0.0
        self._pixel_obs_w = 0
        self._pixel_obs_h = 0
        self._pixel_ui_grid = str(os.environ.get("METABONK_PIXEL_UI_GRID", "") or "").strip().lower()
        if not self._pixel_ui_grid:
            # Keep the implicit default at 32 targets so it fully covers the screen without
            # truncating to the top-left.
            self._pixel_ui_grid = str(os.environ.get("METABONK_UI_GRID", "8x4") or "8x4").strip().lower()
        self._pixel_lock = threading.Lock()
        self._latest_pixel_obs = None
        self._latest_pixel_frame_id: Optional[int] = None
        self._latest_pixel_ts: float = 0.0
        self._latest_pixel_src_size: Optional[tuple[int, int]] = None
        self._cutile_obs = None
        # Spectator frames (higher-res, 16:9) are derived from Synthetic Eye frames, but are
        # intentionally decoupled from the agent observation tensor. This prevents "square
        # obs upscaled to 1080p with huge black bars" in the Stream UI.
        self._spectator_enabled = False
        self._spectator_w: int = 0
        self._spectator_h: int = 0
        self._spectator_interval_s: float = 0.0
        self._spectator_last_emit_ts: float = 0.0
        self._latest_spectator_obs = None
        self._latest_spectator_frame_id: Optional[int] = None
        self._latest_spectator_ts: float = 0.0
        self._latest_spectator_src_size: Optional[tuple[int, int]] = None
        self._pixel_stream_cache_frame_id: Optional[int] = None
        self._pixel_stream_cache_bytes: Optional[bytes] = None
        self._pixel_stream_cache_size: Optional[tuple[int, int]] = None
        self._pixel_stream_cache_ts: float = 0.0
        self._spectator_stream_cache_frame_id: Optional[int] = None
        self._spectator_stream_cache_bytes: Optional[bytes] = None
        self._spectator_stream_cache_size: Optional[tuple[int, int]] = None
        self._spectator_stream_cache_ts: float = 0.0
        if self._pixel_obs_enabled:
            if self._frame_source not in ("synthetic_eye", "smithay", "smithay_dmabuf"):
                raise RuntimeError(
                    f"METABONK_OBS_BACKEND={self._obs_backend} requires METABONK_FRAME_SOURCE=synthetic_eye (got {self._frame_source!r})"
                )
            try:
                import torch  # type: ignore

                if not torch.cuda.is_available():
                    raise RuntimeError("torch.cuda.is_available() is false")
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    f"METABONK_OBS_BACKEND={self._obs_backend} requires torch+CUDA available: {e}"
                ) from e
            try:
                self._pixel_obs_w = int(os.environ.get("METABONK_PIXEL_OBS_W", "128"))
                self._pixel_obs_h = int(os.environ.get("METABONK_PIXEL_OBS_H", "128"))
            except Exception:
                self._pixel_obs_w = 128
                self._pixel_obs_h = 128
            if self._pixel_obs_w <= 0 or self._pixel_obs_h <= 0:
                raise ValueError("METABONK_PIXEL_OBS_W/H must be positive")
            # CuTile obs mode: enforce prerequisites (no silent fallback).
            self._pixel_preprocess_backend = str(os.environ.get("METABONK_PIXEL_PREPROCESS_BACKEND", "") or "").strip().lower()
            if not self._pixel_preprocess_backend:
                self._pixel_preprocess_backend = "cutile" if self._obs_backend == "cutile" else "torch"
            if self._obs_backend == "cutile":
                try:
                    from src.worker.gpu_preprocess import HAS_CUTILE
                except Exception as e:  # pragma: no cover
                    raise RuntimeError(f"cuTile backend requested but gpu_preprocess import failed: {e}") from e
                if not HAS_CUTILE:
                    raise RuntimeError(
                        "METABONK_OBS_BACKEND=cutile requires cuTile+CuPy (install cuda-tile + cupy-cuda13x)."
                    )
                if (int(self._pixel_obs_h) % 32) != 0 or (int(self._pixel_obs_w) % 32) != 0:
                    raise ValueError(
                        f"METABONK_OBS_BACKEND=cutile requires METABONK_PIXEL_OBS_H/W to be multiples of 32 "
                        f"(got {int(self._pixel_obs_h)}x{int(self._pixel_obs_w)})"
                    )
                try:
                    from src.perception.cutile_observations import CuTileObsConfig, CuTileObservations

                    self._cutile_obs = CuTileObservations(
                        cfg=CuTileObsConfig(out_size=(int(self._pixel_obs_h), int(self._pixel_obs_w)))
                    )
                except Exception as e:  # pragma: no cover
                    raise RuntimeError(f"Failed to initialize CuTileObservations: {e}") from e
            # VisionActorCritic uses a 3x stride-2 CNN stem (downscale factor 8) before the
            # ViT-style patch embedding conv. If the patch size is too large relative to the
            # post-stem spatial size, Torch will raise:
            #   "Kernel size can't be greater than actual input size"
            #
            # Keep the default experience robust by choosing a safe patch size when the user
            # did not explicitly set one. For example, 96x96 obs -> 12x12 after stem, so a
            # patch size of 16 would crash; use 8 instead.
            if not str(os.environ.get("METABONK_VISION_PATCH", "") or "").strip():
                try:
                    min_hw = int(min(int(self._pixel_obs_w), int(self._pixel_obs_h)))
                    stem_hw = max(1, int(min_hw) // 8)
                    patch = 1
                    for cand in (16, 8, 4, 2, 1):
                        if int(cand) <= int(stem_hw):
                            patch = int(cand)
                            break
                    os.environ["METABONK_VISION_PATCH"] = str(int(patch))
                except Exception:
                    pass
            try:
                pix_fps = float(os.environ.get("METABONK_PIXEL_OBS_FPS", "20") or 0.0)
            except Exception:
                pix_fps = 20.0
            if pix_fps > 0:
                self._pixel_obs_interval_s = 1.0 / max(1.0, float(pix_fps))
        # Spectator frames are produced when Synthetic Eye is active and streaming is enabled.
        # Default the spectator source size to half of the target encode size (reduces pipe bandwidth)
        # while keeping enough detail for human viewing.
        if self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf"):
            self._spectator_enabled = str(os.environ.get("METABONK_STREAM", "1") or "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            raw_sw = os.environ.get("METABONK_SPECTATOR_W") or os.environ.get("METABONK_STREAM_SPECTATOR_W")
            raw_sh = os.environ.get("METABONK_SPECTATOR_H") or os.environ.get("METABONK_STREAM_SPECTATOR_H")
            raw_res = os.environ.get("METABONK_SPECTATOR_RES") or os.environ.get("METABONK_STREAM_SPECTATOR_RES")
            tw = None
            th = None
            try:
                raw_target = str(os.environ.get("METABONK_STREAM_NVENC_TARGET_SIZE", "") or "").strip().lower()
                if raw_target and "x" in raw_target:
                    a, b = [p.strip() for p in raw_target.split("x", 1)]
                    tw = int(a)
                    th = int(b)
            except Exception:
                tw, th = None, None
            try:
                if (not raw_sw or not raw_sh) and raw_res and "x" in str(raw_res).lower():
                    try:
                        a, b = [p.strip() for p in str(raw_res).lower().split("x", 1)]
                        raw_sw = a
                        raw_sh = b
                    except Exception:
                        pass
                if raw_sw and raw_sh:
                    self._spectator_w = int(raw_sw)
                    self._spectator_h = int(raw_sh)
                elif tw and th:
                    self._spectator_w = max(320, int(tw) // 2)
                    self._spectator_h = max(180, int(th) // 2)
                else:
                    self._spectator_w = 960
                    self._spectator_h = 540
            except Exception:
                self._spectator_w = 960
                self._spectator_h = 540
            if self._spectator_w % 2:
                self._spectator_w += 1
            if self._spectator_h % 2:
                self._spectator_h += 1
            if self._spectator_w <= 0 or self._spectator_h <= 0:
                self._spectator_w = 960
                self._spectator_h = 540
            try:
                sfps = int(
                    os.environ.get(
                        "METABONK_SPECTATOR_FPS",
                        os.environ.get("METABONK_STREAM_FPS", "30"),
                    )
                    or "30"
                )
            except Exception:
                sfps = 30
            sfps = max(1, int(sfps))
            self._spectator_interval_s = 1.0 / float(sfps)

        self._obs_dim_raw = int(obs_dim)
        env_stack = os.environ.get("METABONK_FRAME_STACK")
        try:
            stack = int(frame_stack) if frame_stack is not None else int(env_stack or 1)
        except Exception:
            stack = 1
        # Pixel observations are already "high bandwidth"; prefer temporal memory (LSTM) to channel stacking.
        if self._pixel_obs_enabled:
            self._frame_stack = 1
        else:
            self._frame_stack = max(1, stack)
        self._obs_stack = deque(maxlen=self._frame_stack)
        stacked_dim = self._obs_dim_raw * self._frame_stack
        trainer_obs_dim = 0 if self._pixel_obs_enabled else stacked_dim
        self.trainer = Trainer(policy_name=policy_name, hparams=self.hparams, obs_dim=int(trainer_obs_dim))
        self.learner = LearnerClient(self.learner_url)
        if self._pixel_obs_enabled:
            try:
                max_size = int(os.environ.get("METABONK_PIXEL_ROLLOUT_MAX_SIZE", "256"))
            except Exception:
                max_size = 256
            self.rollout = PixelRolloutBuffer(
                instance_id=instance_id,
                policy_name=policy_name,
                hparams=self.hparams,
                obs_width=int(self._pixel_obs_w),
                obs_height=int(self._pixel_obs_h),
                obs_channels=3,
                max_size=max(1, int(max_size)),
            )
        else:
            self.rollout = RolloutBuffer(
                instance_id=instance_id,
                policy_name=policy_name,
                hparams=self.hparams,
                max_size=int(self.hparams.get("batch_size", 2048)),
            )
        self.rollout.eval_mode = os.environ.get("METABONK_EVAL_MODE", "0") in ("1", "true", "True")
        try:
            self.rollout.eval_seed = int(os.environ.get("METABONK_EVAL_SEED", "0")) or None
        except Exception:
            self.rollout.eval_seed = None
        self.curriculum = CurriculumConfig()
        self._stop = threading.Event()
        self._boot_ts = time.time()
        self._config_poll_s = float(os.environ.get("METABONK_CONFIG_POLL_S", "30.0"))
        self._warned_no_reward_frame = False
        self._eval_clip_on_done = os.environ.get("METABONK_EVAL_CLIP_ON_DONE", "0") in ("1", "true", "True")
        self._policy_version: Optional[int] = None
        self._last_policy_update_ts: float = 0.0
        self._last_policy_fetch_ts: float = 0.0
        self._last_policy_warn_ts: float = 0.0
        self._last_stream_ok_ts: float = 0.0
        self._last_stream_heal_ts: float = 0.0
        self._stream_health_checks: int = 0
        self._stream_heals: int = 0
        self._last_frame_var_ts: float = 0.0
        self._last_frame_var: Optional[float] = None
        self._black_frame_since: float = 0.0
        self._frozen_frame_since: float = 0.0
        self._last_frame_luma = None
        self._last_frame_luma_diff: Optional[float] = None
        self._obs_frame_times: Deque[float] = deque(maxlen=120)
        self._obs_fps: Optional[float] = None
        # Action cadence telemetry (independent of "meaningful" step_count).
        self._act_times: Deque[float] = deque(maxlen=240)
        self._act_hz: Optional[float] = None
        self._actions_total: int = 0
        self._pipewire_health_ts: float = 0.0
        self._pipewire_daemon_ok: Optional[bool] = None
        self._pipewire_session_ok: Optional[bool] = None
        # Synthetic Eye resilience: restart worker on compositor resets/stalls to avoid training on frozen frames.
        self._synthetic_eye_reset_restart = str(
            os.environ.get("METABONK_SYNTHETIC_EYE_RESET_RESTART", "1")
        ).strip().lower() in ("1", "true", "yes", "on")
        # Synthetic Eye orientation: some capture stacks produce bottom-up buffers.
        self._synthetic_eye_vflip = str(
            os.environ.get("METABONK_SYNTHETIC_EYE_VFLIP", "")
            or os.environ.get("METABONK_EYE_VFLIP", "")
            or "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        try:
            self._synthetic_eye_stall_restart_s = float(
                os.environ.get("METABONK_SYNTHETIC_EYE_STALL_RESTART_S", "60.0")
            )
        except Exception:
            self._synthetic_eye_stall_restart_s = 60.0
        self._synthetic_eye_drain_enabled: bool = False
        self._synthetic_eye_drain_last_error_ts: float = 0.0
        self._game_restart_enabled = os.environ.get("METABONK_GAME_RESTART", "1") in ("1", "true", "True")
        self._game_restart_possible = _game_restart_possible()
        self._game_restart_count = 0
        self._last_game_restart_ts = 0.0
        self._game_restart_failed = False
        self._last_step_seen: int = 0
        self._last_step_ts: float = 0.0
        self._last_state_ts: float = 0.0
        self._reward_log = os.environ.get("METABONK_REWARD_LOG", "0") in ("1", "true", "True")
        try:
            self._action_log_freq = int(os.environ.get("METABONK_ACTION_LOG_FREQ", "0"))
        except Exception:
            self._action_log_freq = 0
        self._last_action_log_step = -1
        self._action_clip_enabled = os.environ.get("METABONK_ACTION_CLIP", "1") in ("1", "true", "True")
        try:
            self._action_clip_min = float(os.environ.get("METABONK_ACTION_CLIP_MIN", "-1.0"))
        except Exception:
            self._action_clip_min = -1.0
        try:
            self._action_clip_max = float(os.environ.get("METABONK_ACTION_CLIP_MAX", "1.0"))
        except Exception:
            self._action_clip_max = 1.0
        self._action_clip_logged = False
        self._reward_hit_saved = False
        self._reward_hit_frame_path = os.environ.get(
            "METABONK_REWARD_HIT_FRAME_PATH", str(Path("temp") / "reward_hits")
        )
        self._reward_hit_once = os.environ.get("METABONK_REWARD_HIT_ONCE", "1") in ("1", "true", "True")
        self._gameplay_hit_saved = False
        # Synthetic Eye streaming: optionally cache a full-res CUDA frame for the NVENC path.
        # This must remain GPU-resident (no CPU fallbacks) and may be consumed by the
        # Synthetic Eye GStreamer/NVENC streamer backend.
        self._latest_spectator_cuda = None
        self._latest_spectator_cuda_frame_id = None
        self._latest_spectator_cuda_ts = 0.0
        self._latest_spectator_cuda_src_size = None
        self._gameplay_hit_frame_path = os.environ.get(
            "METABONK_GAMEPLAY_HIT_FRAME_PATH", str(Path("temp") / "gameplay_hits")
        )
        self._gameplay_hit_once = os.environ.get("METABONK_GAMEPLAY_HIT_ONCE", "1") in ("1", "true", "True")
        try:
            self._gameplay_strip_n = int(os.environ.get("METABONK_GAMEPLAY_STRIP_N", "16"))
        except Exception:
            self._gameplay_strip_n = 16
        self._frame_ring_enabled = os.environ.get("METABONK_FRAME_RING", "1") in ("1", "true", "True")
        try:
            ring_size = int(os.environ.get("METABONK_FRAME_RING_SIZE", "120"))
        except Exception:
            ring_size = 120
        self._frame_ring: Deque[dict] = deque(maxlen=max(1, ring_size))
        try:
            self._frame_black_mean = float(os.environ.get("METABONK_FRAME_BLACK_MEAN", "8.0"))
        except Exception:
            self._frame_black_mean = 8.0
        try:
            self._frame_black_p99 = float(os.environ.get("METABONK_FRAME_BLACK_P99", "20.0"))
        except Exception:
            self._frame_black_p99 = 20.0
        try:
            self._frame_black_sat = float(os.environ.get("METABONK_FRAME_BLACK_SAT", "5.0"))
        except Exception:
            self._frame_black_sat = 5.0
        self._hud_enabled = os.environ.get("METABONK_HUD_DETECT", "1") in ("1", "true", "True")
        try:
            self._hud_sat_min = float(os.environ.get("METABONK_HUD_SAT_MIN", "8.0"))
        except Exception:
            self._hud_sat_min = 8.0
        try:
            self._hud_on_frames = int(os.environ.get("METABONK_HUD_ON_FRAMES", "6"))
        except Exception:
            self._hud_on_frames = 6
        try:
            self._hud_off_frames = int(os.environ.get("METABONK_HUD_OFF_FRAMES", "10"))
        except Exception:
            self._hud_off_frames = 10
        try:
            self._hud_roi_x0 = float(os.environ.get("METABONK_HUD_MINIMAP_X0", "0.70"))
            self._hud_roi_x1 = float(os.environ.get("METABONK_HUD_MINIMAP_X1", "0.98"))
            self._hud_roi_y0 = float(os.environ.get("METABONK_HUD_MINIMAP_Y0", "0.02"))
            self._hud_roi_y1 = float(os.environ.get("METABONK_HUD_MINIMAP_Y1", "0.35"))
        except Exception:
            self._hud_roi_x0 = 0.70
            self._hud_roi_x1 = 0.98
            self._hud_roi_y0 = 0.02
            self._hud_roi_y1 = 0.35
        try:
            self._hud_hough_dp = float(os.environ.get("METABONK_HUD_HOUGH_DP", "1.2"))
        except Exception:
            self._hud_hough_dp = 1.2
        try:
            self._hud_hough_param1 = float(os.environ.get("METABONK_HUD_HOUGH_PARAM1", "100"))
        except Exception:
            self._hud_hough_param1 = 100.0
        try:
            self._hud_hough_param2 = float(os.environ.get("METABONK_HUD_HOUGH_PARAM2", "30"))
        except Exception:
            self._hud_hough_param2 = 30.0
        try:
            self._hud_min_radius_frac = float(os.environ.get("METABONK_HUD_MIN_RADIUS_FRAC", "0.08"))
            self._hud_max_radius_frac = float(os.environ.get("METABONK_HUD_MAX_RADIUS_FRAC", "0.40"))
            self._hud_min_dist_frac = float(os.environ.get("METABONK_HUD_MIN_DIST_FRAC", "0.40"))
        except Exception:
            self._hud_min_radius_frac = 0.08
            self._hud_max_radius_frac = 0.40
            self._hud_min_dist_frac = 0.40
        self._hud_present = False
        self._hud_on_count = 0
        self._hud_off_count = 0
        self._hud_last_phase: Optional[str] = None
        try:
            self._hud_pulse_log_s = float(os.environ.get("METABONK_HUD_PULSE_LOG_S", "0") or 0.0)
        except Exception:
            self._hud_pulse_log_s = 0.0
        self._hud_pulse_last_ts = 0.0
        self._hud_phase_log = str(os.environ.get("METABONK_HUD_PHASE_LOG", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._phase_model_path = str(os.environ.get("METABONK_PHASE_MODEL_PATH", "") or "").strip()
        self._phase_model_enabled = bool(self._phase_model_path)
        try:
            self._phase_conf_thresh = float(os.environ.get("METABONK_PHASE_CONF_THRESH", "0.6"))
        except Exception:
            self._phase_conf_thresh = 0.6
        try:
            self._phase_infer_every_s = float(os.environ.get("METABONK_PHASE_INFER_EVERY_S", "0.5"))
        except Exception:
            self._phase_infer_every_s = 0.5
        try:
            self._phase_clip_frames = int(os.environ.get("METABONK_PHASE_CLIP_FRAMES", "12"))
        except Exception:
            self._phase_clip_frames = 12
        try:
            self._phase_input_size = int(os.environ.get("METABONK_PHASE_INPUT_SIZE", "224"))
        except Exception:
            self._phase_input_size = 224
        self._phase_labels = [
            s.strip()
            for s in str(os.environ.get("METABONK_PHASE_LABELS", "main_menu,char_select,map_select,loading,gameplay")).split(",")
            if s.strip()
        ]
        self._phase_menu_labels = {
            s.strip()
            for s in str(
                os.environ.get("METABONK_PHASE_MENU_LABELS", "main_menu,char_select,map_select,menu")
            ).split(",")
            if s.strip()
        }
        self._phase_block_labels = {
            s.strip()
            for s in str(os.environ.get("METABONK_PHASE_BLOCK_LABELS", "loading")).split(",")
            if s.strip()
        }
        self._phase_model = None
        self._phase_model_device = str(os.environ.get("METABONK_PHASE_MODEL_DEVICE", "cpu") or "cpu").strip()
        self._phase_model_warned = False
        self._phase_label: Optional[str] = None
        self._phase_conf: float = 0.0
        self._phase_source: Optional[str] = None
        self._phase_effective_label: Optional[str] = None
        self._phase_effective_source: Optional[str] = None
        self._phase_last_infer_ts: float = 0.0
        self._phase_gameplay = False
        self._gameplay_phase_active = False
        self._phase_dataset_enabled = os.environ.get("METABONK_PHASE_DATASET", "0") in ("1", "true", "True")
        self._phase_dataset_dir = str(os.environ.get("METABONK_PHASE_DATASET_DIR", "temp/phase_dataset") or "")
        try:
            self._phase_dataset_every_s = float(os.environ.get("METABONK_PHASE_DATASET_EVERY_S", "2.0"))
        except Exception:
            self._phase_dataset_every_s = 2.0
        try:
            self._phase_dataset_clip_frames = int(os.environ.get("METABONK_PHASE_DATASET_CLIP_FRAMES", "12"))
        except Exception:
            self._phase_dataset_clip_frames = 12
        self._phase_dataset_allow_unknown = os.environ.get("METABONK_PHASE_DATASET_ALLOW_UNKNOWN", "0") in (
            "1",
            "true",
            "True",
        )
        try:
            self._phase_dataset_max_per_label = int(
                os.environ.get("METABONK_PHASE_DATASET_MAX_PER_LABEL", "0")
            )
        except Exception:
            self._phase_dataset_max_per_label = 0
        self._phase_dataset_counts: Dict[str, int] = {}
        self._phase_dataset_last_ts = 0.0
        self._phase_dataset_forced_label: Optional[str] = None
        self._phase_dataset_forced_until_ts: float = 0.0
        self._phase_dataset_loading_active = False
        self._phase_dataset_loading_deadline = 0.0
        self._phase_dataset_loading_since = 0.0
        try:
            self._phase_dataset_post_play_s = float(
                os.environ.get("METABONK_PHASE_DATASET_POST_PLAY_S", "4.0")
            )
        except Exception:
            self._phase_dataset_post_play_s = 4.0
        try:
            self._phase_dataset_loading_timeout_s = float(
                os.environ.get("METABONK_PHASE_DATASET_LOADING_TIMEOUT_S", "12.0")
            )
        except Exception:
            self._phase_dataset_loading_timeout_s = 12.0
        try:
            self._phase_dataset_loading_restart_s = float(
                os.environ.get("METABONK_PHASE_DATASET_LOADING_RESTART_S", "0.0")
            )
        except Exception:
            self._phase_dataset_loading_restart_s = 0.0
        self._last_valid_frame: Optional[dict] = None
        self._gameplay_started: bool = False
        self._gameplay_start_ts: float = 0.0
        # Hybrid gameplay detector for robust game-agnostic detection.
        # Prevents false positives when stuck in menus/loading screens.
        # Higher variance threshold (2000) filters out animated menus.
        # Higher confidence threshold (0.75) requires stronger signal.
        try:
            gd_variance = float(os.environ.get("METABONK_GAMEPLAY_VARIANCE_THRESHOLD", "2000"))
        except Exception:
            gd_variance = 2000.0
        try:
            gd_confidence = float(os.environ.get("METABONK_GAMEPLAY_MIN_CONFIDENCE", "0.90"))
        except Exception:
            gd_confidence = 0.90
        self._gameplay_detector = HybridGameplayDetector(
            variance_threshold=gd_variance,
            min_confidence=gd_confidence,
        )
        self._gameplay_confidence: float = 0.0
        self._action_source = self._normalize_action_source(
            os.environ.get("METABONK_ACTION_SOURCE", "policy")
        )
        self._action_guard_enabled = os.environ.get("METABONK_ACTION_GUARD", "0") in ("1", "true", "True")
        self._action_guard_path = os.environ.get("METABONK_ACTION_GUARD_PATH", "")
        self._action_guard_violation: Optional[str] = None
        self._pure_vision_mode = str(os.environ.get("METABONK_PURE_VISION_MODE", "0") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._visual_exploration = VisualExplorationReward() if VisualExplorationReward is not None else None
        try:
            self._visual_exploration_interval_s = float(
                os.environ.get("METABONK_VISUAL_EXPLORATION_INTERVAL_S", "0.5") or 0.0
            )
        except Exception:
            self._visual_exploration_interval_s = 0.5
        self._visual_exploration_last_update_ts: float = 0.0
        self._visual_exploration_last_scene_hash: Optional[str] = None
        self._visual_exploration_last_scene_change_ts: float = 0.0

        # Optional UnityBridge embodiment (BepInEx plugin).
        self._use_bridge = os.environ.get("METABONK_USE_UNITY_BRIDGE", "0") in ("1", "true", "True")
        self.bridge: Optional["UnityBridge"] = None
        self._bridge_loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_disc_action: Optional[int] = None
        self._bridge_action_map = str(os.environ.get("METABONK_BRIDGE_ACTION_MAP", "") or "").strip().lower()
        if self._pure_vision_mode:
            # Disallow action-map shortcuts in pure-vision runs.
            self._bridge_action_map = ""
        self._bridge_held_keys: set[str] = set()
        # Optional OS-level input backend (uinput) for real keyboard/mouse injection.
        self._input_backend_name = str(os.environ.get("METABONK_INPUT_BACKEND", "") or "").strip().lower()
        self._input_backend = None
        self._input_buttons: List[dict] = []
        self._input_held_keys: set[str] = set()
        self._input_held_mouse: set[str] = set()
        self._input_cursor_pos: Optional[tuple[int, int]] = None
        try:
            self._input_mouse_scale = float(os.environ.get("METABONK_INPUT_MOUSE_SCALE", "100.0"))
        except Exception:
            self._input_mouse_scale = 100.0
        self._input_mouse_mode = str(os.environ.get("METABONK_INPUT_MOUSE_MODE", "scaled") or "scaled").strip().lower()
        try:
            self._input_scroll_scale = float(os.environ.get("METABONK_INPUT_SCROLL_SCALE", "3.0"))
        except Exception:
            self._input_scroll_scale = 3.0
        # Delay X11 input backend init until after the game (gamescope/Xwayland) display exists.
        # Optional BonkLink bridge (BepInEx 6 IL2CPP).
        self._use_bonklink = os.environ.get("METABONK_USE_BONKLINK", "0") in ("1", "true", "True")
        self._bonklink = None
        self._bonklink_capture_size: Optional[tuple[int, int]] = None
        try:
            self._bonklink_retry_s = float(os.environ.get("METABONK_BONKLINK_RETRY_S", "5.0"))
        except Exception:
            self._bonklink_retry_s = 5.0
        self._bonklink_last_attempt = 0.0
        # Optional causal Scientist for item/build discovery.
        self._use_causal_scientist = os.environ.get("METABONK_USE_CAUSAL", "0") in (
            "1",
            "true",
            "True",
        )
        self._causal_graph = None
        self._pending_intervention = None
        if self._use_causal_scientist:
            try:
                from src.learner.causal_rl import CausalGraph

                self._causal_graph = CausalGraph()
            except Exception:
                self._use_causal_scientist = False
                self._causal_graph = None
        # Optional SIMA2 dual-system controller backend (inference-first).
        self._use_sima2 = os.environ.get("METABONK_USE_SIMA2", "0") in ("1", "true", "True")
        self._sima2_controller = None
        self._sima2_push_rollouts = os.environ.get("METABONK_SIMA2_PUSH_ROLLOUTS", "0") in (
            "1",
            "true",
            "True",
        )
        if self._use_sima2:
            try:
                from src.sima2.controller import SIMA2Controller, SIMA2ControllerConfig

                cfg = SIMA2ControllerConfig(
                    log_reasoning=os.environ.get("METABONK_SIMA2_LOG", "1")
                    not in ("0", "false", "False"),
                )
                self._sima2_controller = SIMA2Controller(cfg)
                goal = os.environ.get("METABONK_SIMA2_GOAL")
                if goal:
                    self._sima2_controller.set_goal(goal)
            except Exception:
                self._use_sima2 = False
                self._sima2_controller = None
        # Optional MetaBonk2 tripartite controller backend (inference-first).
        self._use_metabonk2 = os.environ.get("METABONK_USE_METABONK2", "0") in ("1", "true", "True")
        self._metabonk2_controller = None
        self._metabonk2_time_budget_ms = None
        try:
            raw_budget = os.environ.get("METABONK2_TIME_BUDGET_MS")
            if raw_budget is not None and str(raw_budget).strip():
                self._metabonk2_time_budget_ms = float(raw_budget)
        except Exception:
            self._metabonk2_time_budget_ms = None
        # Avoid ambiguous dual overrides; MetaBonk2 takes precedence.
        if self._use_metabonk2 and self._use_sima2:
            self._use_sima2 = False
            self._sima2_controller = None
        # Optional ResearchPlugin shared memory (deterministic MMF).
        self._use_research_shm = os.environ.get("METABONK_USE_RESEARCH_SHM", "0") in ("1", "true", "True")
        self._research_shm = None

        self.streamer: Optional[NVENCStreamer] = None
        self._fifo_stream_enabled = str(os.environ.get("METABONK_FIFO_STREAM", "") or os.environ.get("METABONK_GO2RTC", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._stream_require_zero_copy = str(os.environ.get("METABONK_STREAM_REQUIRE_ZERO_COPY", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        sb = str(os.environ.get("METABONK_STREAM_BACKEND", "auto") or "").strip().lower()
        self._stream_backend = "ffmpeg" if sb == "obs" else sb
        self._stream_wants_cuda_frames = self._stream_backend in ("cuda_appsrc", "synthetic_eye", "eye", "cuda")
        # When FIFO/go2rtc is enabled we generally need at least 2 stream clients:
        # - one for the browser MP4 endpoint (/stream.mp4) used by the dev UI / warmup probes
        # - one for the FIFO publisher feeding go2rtc
        # Avoid hard failures due to max_clients=1 unless the user explicitly set a higher/lower value.
        if self._fifo_stream_enabled:
            try:
                cur = int(os.environ.get("METABONK_STREAM_MAX_CLIENTS", "0") or 0)
            except Exception:
                cur = 0
            if cur <= 0:
                # Mirror legacy default but make it usable with FIFO distribution + UI.
                os.environ["METABONK_STREAM_MAX_CLIENTS"] = "3"
            elif cur < 3:
                os.environ["METABONK_STREAM_MAX_CLIENTS"] = "3"
        self._fifo_stream_path: Optional[str] = None
        self._fifo_publisher: Optional[FifoH264Publisher] = None
        self._stream_epoch: int = 0
        self._stream_enabled = os.environ.get("METABONK_STREAM", "1") in ("1", "true", "True")
        # Hard requirement: when enabled, the worker must use PipeWire+NVENC for stream.
        # No MJPEG/CPU streaming fallback is allowed in this mode.
        require_raw = os.environ.get("METABONK_REQUIRE_PIPEWIRE_STREAM")
        self._require_pipewire_stream = str(require_raw or "1") in ("1", "true", "True")
        if self._stream_backend == "x11grab":
            # x11grab is explicitly non-PipeWire capture; don't gate startup on PipeWire discovery.
            self._require_pipewire_stream = False
        if require_raw is None and self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf"):
            # Synthetic Eye doesn't rely on PipeWire for observations; avoid gating stream readiness on it by default.
            # In strict zero-copy runs, streaming can be provided via the Synthetic Eye CUDA appsrc backend.
            self._require_pipewire_stream = False
        self._pipewire_node_ok = False
        self._pipewire_node = os.environ.get("PIPEWIRE_NODE")
        self._stream_error: Optional[str] = None
        self._featured_slot: Optional[str] = None
        self._featured_role: Optional[str] = None
        dash_stream = str(os.environ.get("METABONK_DASHBOARD_STREAM", "auto") or "").strip().lower()
        if dash_stream == "auto":
            dash_stream = "synthetic_eye" if self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf") else "pipewire"
        self._dashboard_stream = dash_stream
        try:
            self._dashboard_fps = float(os.environ.get("METABONK_DASHBOARD_FPS", "12"))
        except Exception:
            self._dashboard_fps = 12.0
        try:
            self._dashboard_scale = float(os.environ.get("METABONK_DASHBOARD_SCALE", "0.5"))
        except Exception:
            self._dashboard_scale = 0.5
        self._dashboard_last_ts = 0.0

        self.highlight: Optional[HighlightRecorder] = None
        if os.environ.get("METABONK_HIGHLIGHTS", "0") in ("1", "true", "True"):
            try:
                ds_raw = str(os.environ.get("METABONK_HIGHLIGHT_DOWNSCALE", "480x270") or "").strip().lower()
                downscale = (480, 270)
                if ds_raw and ("x" in ds_raw or "," in ds_raw):
                    sep = "x" if "x" in ds_raw else ","
                    parts = [p.strip() for p in ds_raw.split(sep) if p.strip()]
                    if len(parts) >= 2:
                        downscale = (max(64, int(parts[0])), max(64, int(parts[1])))
                hcfg = HighlightConfig(
                    fps=int(os.environ.get("METABONK_HIGHLIGHT_FPS", "30")),
                    max_seconds=int(os.environ.get("METABONK_HIGHLIGHT_SECONDS", "180")),
                    downscale=downscale,
                    speed=float(os.environ.get("METABONK_HIGHLIGHT_SPEED", "3.0")),
                    codec=str(os.environ.get("METABONK_HIGHLIGHT_CODEC", "h264_nvenc") or "h264_nvenc"),
                    bitrate=str(os.environ.get("METABONK_HIGHLIGHT_BITRATE", "4M") or "4M"),
                )
                out_root = os.environ.get("METABONK_HIGHLIGHTS_DIR", "highlights")
                self.highlight = HighlightRecorder(out_root=out_root, cfg=hcfg)
            except Exception:
                self.highlight = None
        self._best_reward_local = 0.0
        self._episode_start_ts = time.time()
        self._episode_idx = 0
        self._last_done_sent = False
        self._last_telemetry_ts = 0.0
        self._max_hit_local = 0.0
        self._last_hit_sig = None
        self._last_loot_sig = None
        self._last_heal_sig = None
        self._last_pipewire_refresh_ts: float = 0.0
        self._last_inventory_items: Optional[list[str]] = None

        # Episode metadata for milestone pinning (best-effort).
        self._last_stage: Optional[int] = None
        self._last_biome: Optional[str] = None
        self._last_end_reason: Optional[str] = None

        # Learned reward-from-video (optional; avoids hand-authored shaping).
        self._use_learned_reward = os.environ.get("METABONK_USE_LEARNED_REWARD", "1") in ("1", "true", "True")
        if self._pure_vision_mode:
            # Pure-vision validation forbids any external/shaped reward sources.
            self._use_learned_reward = False
        self._reward_model = None
        self._reward_device = None
        self._reward_frame_size = (224, 224)
        self._reward_scale = float(os.environ.get("METABONK_LEARNED_REWARD_SCALE", "1.0"))
        self._prev_progress_score: Optional[float] = None
        # Latest JPEG frame bytes (e.g., from BonkLink) for UI streaming fallback.
        self._latest_jpeg_bytes: Optional[bytes] = None
        self._latest_frame_ts: float = 0.0
        self._latest_thumb_b64: Optional[str] = None
        # Flight recorder: last N actions + thumbnails for Context Drawer.
        try:
            cap = int(os.environ.get("METABONK_TELEMETRY_HISTORY", "64"))
        except Exception:
            cap = 64
        self._telemetry_capacity = max(1, cap)
        self._telemetry_history: deque[dict] = deque(maxlen=self._telemetry_capacity)
        self._telemetry_lock = threading.Lock()
        # Optional low-rate JPEG preview (CPU) for UI fallback/debug.
        self._preview_stream: Optional[CaptureStream] = None
        self._preview_stop = threading.Event()
        self._preview_thread: Optional[threading.Thread] = None
        # When enabled, analytics/telemetry must be derived from visuals only (no game memory/state).
        self._visual_only = os.environ.get("METABONK_VISUAL_ONLY", "0") in ("1", "true", "True")
        # When enabled, do not attempt PipeWire capture (use BonkLink JPEG fallback for streaming if available).
        self._capture_disabled = os.environ.get("METABONK_CAPTURE_DISABLED", "0") in ("1", "true", "True")
        # "Hard" disable is derived from env at process start; orchestration may still toggle
        # capture on/off for featured slots. FIFO demand-paged streaming should remain
        # available even when capture is toggled off, unless hard-disabled via env.
        self._capture_disabled_env = bool(self._capture_disabled)
        # gamescope PipeWire streams can enter an "out of buffers" failure mode if no
        # consumer is attached. Keep a tiny DMABuf drain running to stabilize instances
        # even when no viewer is connected (does not encode, drops frames).
        self._pipewire_drain_enabled = os.environ.get("METABONK_PIPEWIRE_DRAIN", "0") in ("1", "true", "True")
        if self._visual_only:
            # Enforce "no memory access" posture: do not read ResearchPlugin shared memory.
            self._use_research_shm = False
            self._research_shm = None

        if self._fifo_stream_enabled:
            fifo_dir = str(os.environ.get("METABONK_STREAM_FIFO_DIR", "temp/streams") or "temp/streams").strip()
            if not fifo_dir:
                fifo_dir = "temp/streams"
            fifo_container = str(os.environ.get("METABONK_FIFO_CONTAINER", "mpegts") or "mpegts").strip().lower()
            if fifo_container in ("ts", "mpegts"):
                fifo_container = "mpegts"
                fifo_ext = "ts"
            else:
                fifo_container = "h264"
                fifo_ext = "h264"
            self._fifo_stream_container = fifo_container
            self._fifo_stream_path = os.path.join(fifo_dir, f"{self.instance_id}.{fifo_ext}")

    def _ensure_pipewire_node(self) -> Optional[str]:
        """Ensure PIPEWIRE_NODE is set (best-effort) for GPU capture/streaming."""
        override = (
            os.environ.get("METABONK_PIPEWIRE_TARGET_OVERRIDE")
            or os.environ.get("PIPEWIRE_NODE_OVERRIDE")
            or ""
        ).strip()
        if override and override.lower() in ("auto", "discover"):
            override = ""
        if override:
            # Validate override: PipeWire nodes/ports are frequently recreated by gamescope.
            # A stale override can silently yield black frames or buffer starvation.
            try:
                ok = bool(self.launcher.pipewire_target_exists(str(override), timeout_s=0.5))
            except Exception:
                ok = False
            if ok:
                os.environ["PIPEWIRE_NODE"] = override
                try:
                    self.stream.pipewire_node = override
                except Exception:
                    pass
                self._pipewire_node_ok = True
                self._pipewire_node = override
                self._stream_error = None
                return override
            # Ignore stale override and fall back to discovery.
            try:
                os.environ.pop("PIPEWIRE_NODE", None)
            except Exception:
                pass
            self._pipewire_node_ok = False
            self._pipewire_node = None
        node = os.environ.get("PIPEWIRE_NODE")
        # If we only have a generic gamescope capture path, try to upgrade to the
        # concrete node id once gamescope logs it (more reliable and portable).
        try:
            if isinstance(node, str) and node.startswith("gamescope:capture_"):
                from pathlib import Path
                import re

                lp = getattr(self.launcher, "log_path", None)
                log_path = Path(str(lp)) if lp else None
                if log_path and log_path.exists():
                    data = log_path.read_bytes()[-200000:]
                    m = re.findall(rb"stream available on node ID:\s*([0-9]+)", data)
                    if m:
                        nid = m[-1].decode("ascii", "ignore")
                        if nid and nid.isdigit():
                            try:
                                target = self.launcher._resolve_pipewire_target_object(int(nid), timeout_s=0.5)
                            except Exception:
                                target = None
                            if target:
                                try:
                                    if self.launcher.pipewire_target_exists(str(target), timeout_s=0.2):
                                        os.environ["PIPEWIRE_NODE"] = str(target)
                                        node = str(target)
                                except Exception:
                                    pass
        except Exception:
            pass
        # Guard against stale targets (gamescope can recreate its PipeWire ports on restart).
        if node:
            try:
                if not self.launcher.pipewire_target_exists(str(node), timeout_s=0.2):
                    os.environ.pop("PIPEWIRE_NODE", None)
                    node = None
            except Exception:
                pass
        if node:
            try:
                self.stream.pipewire_node = node
            except Exception:
                pass
            self._pipewire_node_ok = True
            self._pipewire_node = node
            return node
        node = None
        try:
            node = self.launcher.discover_pipewire_node()
        except Exception:
            node = None
        # Fallback: parse the instance log directly (helps if the launcher object did
        # not retain log_path for some reason).
        if not node:
            try:
                from pathlib import Path
                import re

                log_dir = os.environ.get("MEGABONK_LOG_DIR") or ""
                if log_dir:
                    lp = Path(log_dir).expanduser() / f"{self.instance_id}.log"
                    m = []
                    if lp.exists():
                        tail_bytes = int(os.environ.get("METABONK_PIPEWIRE_LOG_TAIL_BYTES", "2000000"))
                        data = lp.read_bytes()
                        if tail_bytes > 0 and len(data) > tail_bytes:
                            data = data[-tail_bytes:]
                        m = re.findall(rb"stream available on node ID:\s*([0-9]+)", data)
                    if m:
                        node_id_s = m[-1].decode("ascii", "ignore")
                        # Prefer unique port serial for GStreamer (per-instance safe).
                        try:
                            node_id = int(node_id_s)
                        except Exception:
                            node_id = None
                        if node_id is not None:
                            try:
                                node = self.launcher._resolve_pipewire_target_object(node_id, timeout_s=0.5)
                            except Exception:
                                node = None
            except Exception:
                node = None
        if node:
            os.environ["PIPEWIRE_NODE"] = str(node)
            try:
                self.stream.pipewire_node = str(node)
            except Exception:
                pass
            self._pipewire_node_ok = True
            self._pipewire_node = str(node)
            self._stream_error = None
        else:
            self._pipewire_node_ok = False
            self._pipewire_node = None
        return node

    def _ensure_streamer(self) -> None:
        if not self._stream_enabled:
            return
        # Allow the launcher/orchestrator to disable streaming entirely for low-priority instances.
        # This reduces per-instance RAM/VRAM overhead (encoder + buffers) without affecting training.
        if str(os.environ.get("METABONK_STREAMER_ENABLED", "1") or "").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return
        if self.streamer is not None:
            return
        backend = (os.environ.get("METABONK_STREAM_BACKEND") or "").strip().lower()
        # Back-compat: "obs" means "use OBS-like ffmpeg encoder selection" (no OBS required).
        if backend == "obs":
            backend = "ffmpeg"
        if backend and backend not in (
            "gst",
            "gstreamer",
            "gst-launch",
            "ffmpeg",
            "x11grab",
            "auto",
            "pixel_obs",
            "pixels",
            "cuda_appsrc",
            "cuda",
            "eye",
            "synthetic_eye",
        ):
            self._stream_error = (
                f"unsupported stream backend '{backend}' (expected auto|gst|cuda_appsrc|ffmpeg|x11grab|pixel_obs|obs)"
            )
            return
        node = None
        if backend != "x11grab":
            node = self._ensure_pipewire_node()
            if not node:
                if self._require_pipewire_stream:
                    self._stream_error = "PIPEWIRE_NODE not found (PipeWire stream required)"
                    # If FIFO streaming is enabled, keep going so we can become stream-ready later
                    # once gamescope/pipewire exposes a concrete node id.
                    if not self._fifo_stream_enabled:
                        return
                else:
                    # Best-effort: allow streamer init even without a node; iter_chunks will
                    # refresh PIPEWIRE_NODE per-connection.
                    pass
        try:
            pixel_provider = None
            cuda_provider = None
            if self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf"):
                # Prefer the GPU-resident Synthetic Eye frame path when available.
                # This avoids the ffmpeg/rawvideo pixel_obs path (CPU + NVENC fragility) and keeps
                # the spectator tiles readable (true spectator surface, not upscaled obs).
                cuda_provider = self._get_latest_spectator_cuda_frame
                # Keep the CPU pixel provider only as a non-strict fallback/debug hook.
                if not bool(getattr(self, "_stream_require_zero_copy", False)):
                    pixel_provider = self._get_latest_spectator_stream_frame
            scfg = NVENCConfig(
                pipewire_node=node,
                codec=os.environ.get("METABONK_STREAM_CODEC", "h264"),
                bitrate=os.environ.get("METABONK_STREAM_BITRATE", "6M"),
                fps=int(os.environ.get("METABONK_STREAM_FPS", "60")),
                # Default GOP to 1s worth of frames (avoid long keyframe intervals when FPS is lowered).
                gop=int(os.environ.get("METABONK_STREAM_GOP", os.environ.get("METABONK_STREAM_FPS", "60"))),
                preset=str(os.environ.get("METABONK_STREAM_PRESET", "p4") or "p4"),
                tune=str(os.environ.get("METABONK_STREAM_TUNE", "ll") or "ll"),
                container=os.environ.get("METABONK_STREAM_CONTAINER", "mp4"),
                pixel_frame_provider=pixel_provider,
                cuda_frame_provider=cuda_provider,
            )
            if cuda_provider is not None:
                self._stream_wants_cuda_frames = True
            try:
                self._stream_epoch = int(getattr(self, "_stream_epoch", 0) or 0) + 1
            except Exception:
                self._stream_epoch = 1
            self.streamer = NVENCStreamer(scfg)
            if self._fifo_stream_enabled and self._fifo_stream_path:
                try:
                    pipe_bytes = int(os.environ.get("METABONK_FIFO_PIPE_BYTES", "1048576"))
                except Exception:
                    pipe_bytes = 0
                self._fifo_publisher = FifoH264Publisher(
                    cfg=FifoPublishConfig(
                        fifo_path=self._fifo_stream_path,
                        pipe_size_bytes=pipe_bytes,
                        container=str(getattr(self, "_fifo_stream_container", "h264") or "h264"),
                    ),
                    streamer=self.streamer,
                )
            self._stream_error = None
        except Exception as e:
            self.streamer = None
            self._fifo_publisher = None
            if self._require_pipewire_stream:
                self._stream_error = f"failed to initialize streamer ({e})"

    def stream_meta(self) -> dict:
        """Return minimal stream metadata for UI self-heal + diagnostics."""
        out: dict = {
            "instance_id": self.instance_id,
            "epoch": int(getattr(self, "_stream_epoch", 0) or 0),
            "frame_source": getattr(self, "_frame_source", None),
            "stream_enabled": bool(getattr(self, "_stream_enabled", False)),
            "capture_enabled": not bool(getattr(self, "_capture_disabled", False)),
            "stream_require_pipewire": bool(getattr(self, "_require_pipewire_stream", False)),
            "pipewire_node": os.environ.get("PIPEWIRE_NODE") or getattr(self, "_pipewire_node", None),
            "pipewire_node_ok": bool(getattr(self, "_pipewire_node_ok", False)),
            "requested_backend": str(os.environ.get("METABONK_STREAM_BACKEND", "auto") or "auto").strip().lower(),
            "container": str(os.environ.get("METABONK_STREAM_CONTAINER", "mp4") or "mp4").strip().lower(),
            "codec": str(os.environ.get("METABONK_STREAM_CODEC", "h264") or "h264").strip().lower(),
        }
        try:
            out["fps"] = int(os.environ.get("METABONK_STREAM_FPS", "60") or 60)
        except Exception:
            out["fps"] = None
        try:
            out["gop"] = int(os.environ.get("METABONK_STREAM_GOP", os.environ.get("METABONK_STREAM_FPS", "60")) or 60)
        except Exception:
            out["gop"] = None
        try:
            if out.get("fps") and out.get("gop"):
                out["keyframe_interval_s"] = float(out["gop"]) / float(out["fps"])
        except Exception:
            out["keyframe_interval_s"] = None
        # Target output size (ffmpeg scaling).
        target = str(os.environ.get("METABONK_STREAM_NVENC_TARGET_SIZE", "") or "").strip().lower()
        if target and "x" in target:
            try:
                a, b = [p.strip() for p in target.split("x", 1)]
                out["out_width"] = int(a)
                out["out_height"] = int(b)
            except Exception:
                out["out_width"] = None
                out["out_height"] = None
        else:
            out["out_width"] = None
            out["out_height"] = None

        # Source/derived products (best-effort).
        try:
            src = getattr(self, "_latest_spectator_src_size", None)
            if isinstance(src, tuple) and len(src) >= 2:
                out["src_width"] = int(src[0])
                out["src_height"] = int(src[1])
        except Exception:
            pass
        if out.get("src_width") is None:
            try:
                src = getattr(self, "_latest_pixel_src_size", None)
                if isinstance(src, tuple) and len(src) >= 2:
                    out["src_width"] = int(src[0])
                    out["src_height"] = int(src[1])
            except Exception:
                pass
        out.setdefault("src_width", None)
        out.setdefault("src_height", None)
        out["spectator_width"] = int(getattr(self, "_spectator_w", 0) or 0) if bool(getattr(self, "_spectator_enabled", False)) else None
        out["spectator_height"] = int(getattr(self, "_spectator_h", 0) or 0) if bool(getattr(self, "_spectator_enabled", False)) else None
        out["pixel_obs_width"] = int(getattr(self, "_pixel_obs_w", 0) or 0) if bool(getattr(self, "_pixel_obs_enabled", False)) else None
        out["pixel_obs_height"] = int(getattr(self, "_pixel_obs_h", 0) or 0) if bool(getattr(self, "_pixel_obs_enabled", False)) else None
        # Transform hints (explicitly metadata, not applied here).
        out["vflip"] = bool(getattr(self, "_synthetic_eye_vflip", False))
        out["pix_fmt_in"] = "rgb24" if self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf") else None

        # Streamer stats (best-effort).
        out.setdefault("backend", None)
        out.setdefault("active_clients", None)
        out.setdefault("max_clients", None)
        out.setdefault("last_keyframe_ts", None)
        out.setdefault("last_chunk_ts", None)
        out.setdefault("last_error", None)
        out.setdefault("last_error_ts", None)
        if self.streamer is not None:
            try:
                out["backend"] = getattr(self.streamer, "backend", None)
            except Exception:
                pass
            try:
                out["active_clients"] = self.streamer.active_clients() if hasattr(self.streamer, "active_clients") else None
            except Exception:
                pass
            try:
                out["max_clients"] = self.streamer.max_clients() if hasattr(self.streamer, "max_clients") else None
            except Exception:
                pass
            try:
                out["last_keyframe_ts"] = float(getattr(self.streamer, "last_keyframe_ts", 0.0) or 0.0) or None
            except Exception:
                pass
            try:
                ts = out.get("last_keyframe_ts")
                if ts:
                    out["last_keyframe_age_ms"] = int(max(0.0, (time.time() - float(ts)) * 1000.0))
            except Exception:
                out["last_keyframe_age_ms"] = None
            try:
                out["last_chunk_ts"] = float(getattr(self.streamer, "last_chunk_ts", 0.0) or 0.0) or None
            except Exception:
                pass
            try:
                out["last_error"] = getattr(self.streamer, "last_error", None)
            except Exception:
                pass
            try:
                out["last_error_ts"] = float(getattr(self.streamer, "last_error_ts", 0.0) or 0.0) or None
            except Exception:
                pass
        return out

    def set_capture_enabled(self, enabled: bool) -> None:
        """Enable/disable PipeWire capture at runtime (best-effort)."""
        enabled = bool(enabled)
        new_disabled = not enabled
        is_synthetic_eye = self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf")
        if bool(getattr(self, "_capture_disabled", False)) == new_disabled:
            if not new_disabled:
                self._ensure_preview_jpeg()
            return
        self._capture_disabled = new_disabled
        if new_disabled:
            # Synthetic Eye is the vision path; it must keep consuming frames even when
            # "featured capture"/preview is toggled off by the orchestrator.
            if not is_synthetic_eye:
                if not bool(getattr(self, "_pipewire_drain_enabled", False)):
                    try:
                        self.stream.stop()
                    except Exception:
                        pass
            try:
                self._stop_preview_jpeg()
            except Exception:
                pass
            # Keep FIFO demand-paged streaming available when capture is toggled off
            # for non-featured agents, unless capture was hard-disabled via env.
            if not (self._fifo_stream_enabled and not getattr(self, "_capture_disabled_env", False)):
                if self.streamer is not None:
                    try:
                        self.streamer.stop()
                    except Exception:
                        pass
                    self.streamer = None
                if self._fifo_publisher is not None:
                    try:
                        self._fifo_publisher.stop()
                    except Exception:
                        pass
                    self._fifo_publisher = None
        else:
            self._ensure_pipewire_node()
            if is_synthetic_eye:
                try:
                    self.stream.start()
                except Exception:
                    pass
            elif bool(getattr(self, "_pipewire_drain_enabled", False)) or self._gst_capture_enabled:
                try:
                    self.stream.start()
                except Exception:
                    pass
            self._ensure_preview_jpeg()
            # Only featured slots should pay for GPU encoding.
            self._ensure_streamer()
            if self._fifo_publisher is not None:
                try:
                    self._fifo_publisher.start()
                except Exception:
                    pass

    def _ensure_reward_model(self) -> None:
        if not self._use_learned_reward:
            return
        if self._reward_model is not None:
            return
        try:
            import torch  # type: ignore
            from src.imitation.video_pretraining import TemporalRankRewardModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Learned reward requested (METABONK_USE_LEARNED_REWARD=1) but torch/reward model code is unavailable."
            ) from e

        ckpt_path = os.environ.get("METABONK_VIDEO_REWARD_CKPT", "checkpoints/video_reward_model.pt")
        if not os.path.exists(ckpt_path):
            raise RuntimeError(
                f"Learned reward requested but checkpoint missing: {ckpt_path}. "
                "Train it with `python scripts/video_pretrain.py --phase reward_train`."
            )

        from src.common.device import resolve_device

        dev_s = (
            os.environ.get("METABONK_LEARNED_REWARD_DEVICE")
            or os.environ.get("METABONK_REWARD_DEVICE")
            or ""
        )
        dev = torch.device(resolve_device(dev_s, context="learned reward"))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        conf = ckpt.get("config") or {}
        self._reward_frame_size = tuple(conf.get("frame_size") or (224, 224))
        embed_dim = int(conf.get("embed_dim") or 256)
        model = TemporalRankRewardModel(frame_size=self._reward_frame_size, embed_dim=embed_dim).to(dev)
        model.load_state_dict(ckpt.get("model_state_dict") or {})
        model.eval()

        self._reward_model = model
        self._reward_device = dev
        self._prev_progress_score = None

    def _learned_reward(self, frame_hwc: "Any") -> float:
        """Compute reward as delta(progress_score) from the learned reward model."""
        if not self._use_learned_reward:
            raise RuntimeError("learned reward disabled")
        self._ensure_reward_model()
        if self._reward_model is None or self._reward_device is None:
            raise RuntimeError("reward model not initialized")
        try:
            import numpy as np  # type: ignore
            import torch  # type: ignore
            import torch.nn.functional as F  # type: ignore
        except Exception as e:
            raise RuntimeError("torch/numpy required for learned reward computation") from e

        arr = np.asarray(frame_hwc)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise RuntimeError("invalid frame for learned reward (expected HWC RGB)")
        arr = arr[:, :, :3].astype(np.uint8, copy=False)

        f = torch.from_numpy(arr).to(device=self._reward_device, dtype=torch.uint8)
        f = f.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255.0
        f = F.interpolate(
            f,
            size=(int(self._reward_frame_size[0]), int(self._reward_frame_size[1])),
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            score = float(self._reward_model(f).detach().to("cpu").float().item())

        if self._prev_progress_score is None:
            self._prev_progress_score = score
            return 0.0

        r = (score - float(self._prev_progress_score)) * float(self._reward_scale)
        self._prev_progress_score = score
        return float(r)

    def start(self):
        # libxdo/xdotool require XTEST. Ensure the gamescope-spawned Xwayland is configured
        # to export it even when the launcher doesn't explicitly set the env.
        try:
            backend = str(os.environ.get("METABONK_INPUT_BACKEND") or "").strip().lower()
        except Exception:
            backend = ""
        if backend in ("libxdo", "xdotool", "xdo", ""):
            try:
                existing = str(os.environ.get("GAMESCOPE_XWAYLAND_ARGS") or "").strip()
            except Exception:
                existing = ""
            if "XTEST" not in existing.upper():
                want = "+extension XTEST"
                os.environ["GAMESCOPE_XWAYLAND_ARGS"] = (existing + " " + want).strip() if existing else want

        self.launcher.launch()
        # If DISPLAY was explicitly provided (e.g., Smithay compositor.env handshake), honor it and
        # do not attempt gamescope log discovery.
        if self.display:
            disp = str(self.display)
            os.environ["DISPLAY"] = disp
            os.environ["METABONK_INPUT_DISPLAY"] = disp
            # XWayland under Synthetic Eye can restart during early boot (or after compositor resets).
            # Wait briefly for the X server to become responsive before failing hard.
            try:
                xtest_wait_s = float(
                    os.environ.get(
                        "METABONK_XTEST_WAIT_S",
                        os.environ.get("METABONK_INPUT_XTEST_WAIT_S", "20"),
                    )
                )
            except Exception:
                xtest_wait_s = 20.0
            xtest_wait_s = max(0.0, float(xtest_wait_s))
            xtest_ok, xtest_diag = self._check_xtest(display=disp)
            if not xtest_ok and xtest_wait_s > 0.0:
                deadline = time.time() + xtest_wait_s
                while time.time() < deadline and not xtest_ok:
                    time.sleep(0.25)
                    xtest_ok, xtest_diag = self._check_xtest(display=disp)
            if not xtest_ok:
                diag = (xtest_diag or "").strip()
                if len(diag) > 800:
                    diag = diag[-800:]
                hint = (
                    "Ensure the nested Xwayland is running and exports XTEST. "
                    "For gamescope, try setting GAMESCOPE_XWAYLAND_ARGS='+extension XTEST'."
                )
                raise RuntimeError(
                    f"[worker:{self.instance_id}] XTEST extension not available on {disp}. "
                    f"{hint}\nxdpyinfo:\n{diag}"
                )
            self._upsert_isolation_log(
                {
                    "DISPLAY": disp,
                    "METABONK_INPUT_DISPLAY": disp,
                    "XTEST": "1" if xtest_ok else "0",
                }
            )
            self._init_input_backend()
            self._init_metabonk2_controller()
            self._bind_input_window()
            # Synthetic Eye: start the socket reader thread early so the producer does not block on accept()
            # and so we can begin consuming frames immediately.
            if self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf"):
                try:
                    self.stream.start()
                except Exception:
                    pass
                self._start_synthetic_eye_drain()
            self._ensure_pipewire_node()
            if bool(getattr(self, "_pipewire_drain_enabled", False)) and not getattr(self, "_capture_disabled_env", False):
                try:
                    self.stream.start()
                except Exception:
                    pass
            if self._fifo_stream_enabled and not getattr(self, "_capture_disabled_env", False):
                try:
                    self._ensure_streamer()
                except Exception:
                    pass
                if self._fifo_publisher is not None:
                    try:
                        self._fifo_publisher.start()
                    except Exception:
                        pass
            if not getattr(self, "_capture_disabled", False):
                self._ensure_streamer()
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
            threading.Thread(target=self._rollout_loop, daemon=True).start()
            return

        # Otherwise discover the gamescope-provided Xwayland display from launcher logs so libxdo/xdotool
        # connects to the correct isolated X server (never the host :0).
        try:
            disp = self._discover_gamescope_xwayland_display()
            input_backend = str(os.environ.get("METABONK_INPUT_BACKEND") or "").strip().lower()
            if not disp and input_backend in ("libxdo", "xdotool", ""):
                lp = getattr(self.launcher, "log_path", None)
                tail = ""
                try:
                    if lp and Path(str(lp)).exists():
                        tail = Path(str(lp)).read_text(errors="replace")[-4000:]
                except Exception:
                    tail = ""
                raise RuntimeError(
                    f"gamescope Xwayland display not discovered (launcher log={lp}). "
                    f"Last log tail:\n{tail}"
                )
            if disp:
                self.display = disp
                os.environ["DISPLAY"] = disp
                os.environ["METABONK_INPUT_DISPLAY"] = disp
                xtest_ok, xtest_diag = self._check_xtest(display=disp)
                if not xtest_ok:
                    diag = (xtest_diag or "").strip()
                    if len(diag) > 800:
                        diag = diag[-800:]
                    hint = (
                        "Ensure the nested Xwayland is running and exports XTEST. "
                        "For gamescope, try setting GAMESCOPE_XWAYLAND_ARGS='+extension XTEST'."
                    )
                    raise RuntimeError(f"XTEST extension not available on {disp}. {hint}\nxdpyinfo:\n{diag}")
                self._upsert_isolation_log(
                    {
                        "DISPLAY": disp,
                        "METABONK_INPUT_DISPLAY": disp,
                        "XTEST": "1" if xtest_ok else "0",
                    }
                )
        except Exception as e:
            raise RuntimeError(f"[worker:{self.instance_id}] failed to discover gamescope Xwayland display: {e}") from e
        self._init_input_backend()
        self._init_metabonk2_controller()
        self._bind_input_window()
        # Synthetic Eye: start the socket reader thread early so the producer does not block on accept()
        # and so we can begin consuming frames immediately.
        if self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf"):
            try:
                self.stream.start()
            except Exception:
                pass
            self._start_synthetic_eye_drain()
        # If this worker is going to stream via GPU, we need to discover the node after launch.
        self._ensure_pipewire_node()
        if bool(getattr(self, "_pipewire_drain_enabled", False)) and not getattr(self, "_capture_disabled_env", False):
            # Keep PipeWire drained so gamescope doesn't hit "out of buffers" when idle.
            try:
                self.stream.start()
            except Exception:
                pass
        # Initialize the streamer early so go2rtc FIFO demand-paging can work even if
        # the orchestrator later toggles capture off for non-featured agents.
        if self._fifo_stream_enabled and not getattr(self, "_capture_disabled_env", False):
            try:
                self._ensure_streamer()
            except Exception:
                pass
            if self._fifo_publisher is not None:
                try:
                    self._fifo_publisher.start()
                except Exception:
                    pass
        if not getattr(self, "_capture_disabled", False):
            # Featured capture: enable GPU streamer (encoding happens on demand per-client).
            self._ensure_streamer()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        threading.Thread(target=self._rollout_loop, daemon=True).start()

    def _start_synthetic_eye_drain(self) -> None:
        if self._synthetic_eye_drain_enabled:
            return
        if self._frame_source not in ("synthetic_eye", "smithay", "smithay_dmabuf"):
            return
        if self._synthetic_eye_ingestor is None:
            return
        self._synthetic_eye_drain_enabled = True
        threading.Thread(target=self._synthetic_eye_drain_loop, daemon=True).start()

    def _synthetic_eye_drain_loop(self) -> None:
        """Continuously service Synthetic Eye acquire/release fences.

        This keeps the compositor buffer pool flowing even when the main rollout loop is busy.
        """
        try:
            from .cuda_interop import _ensure_ctx

            _ensure_ctx()
            try:
                import torch  # type: ignore

                torch.cuda._lazy_init()
            except Exception:
                pass
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                frames = getattr(self.stream, "drain")() if hasattr(self.stream, "drain") else [self.stream.read()]
            except Exception:
                frames = []
            frames = [f for f in frames if f is not None]
            if not frames:
                time.sleep(0.005)
                continue
            # Drain loop policy: always service fences for every queued frame, but only perform
            # expensive CUDA import/normalization on the *newest* frame. This prevents the drain
            # thread from becoming compute-bound when the compositor outpaces the agent obs rate.
            primary = frames[-1]
            for frame in frames:
                try:
                    if (
                        SyntheticEyeFrame is not None
                        and isinstance(frame, SyntheticEyeFrame)
                        and self._synthetic_eye_ingestor is not None
                    ):
                        # Fast path for backlog: keep the compositor buffer pool flowing by servicing
                        # acquire/release fences without importing payloads for all but the newest frame.
                        if frame is not primary:
                            self._synthetic_eye_ingestor.handshake_only(frame)
                            try:
                                ts = float(getattr(frame, "timestamp", 0.0) or 0.0) or time.time()
                                self._latest_frame_ts = ts
                                self._record_obs_frame_ts(ts)
                            except Exception:
                                pass
                            continue
                        try:
                            # Important distinction:
                            # - `_capture_disabled` is a runtime toggle used by the orchestrator to stop
                            #   *streaming/preview* for background workers (featured slots only).
                            # - `_capture_disabled_env` is a "hard" disable for soak runs.
                            #
                            # Regardless of capture toggles, we must keep servicing acquire/release
                            # fences (otherwise the compositor stalls and the watchdog restarts the
                            # worker in a loop).
                            capture_disabled = bool(getattr(self, "_capture_disabled", False))
                            hard_disabled = bool(getattr(self, "_capture_disabled_env", False))

                            # For soak runs we may "hard" disable capture/inference entirely. In that
                            # mode we still must service fences, but we skip expensive CUDA imports.
                            want_pixels = (not hard_disabled) and bool(getattr(self, "_pixel_obs_enabled", False))
                            want_pixels_update = want_pixels
                            if want_pixels:
                                try:
                                    interval = float(getattr(self, "_pixel_obs_interval_s", 0.0) or 0.0)
                                    last_emit = float(getattr(self, "_pixel_obs_last_emit_ts", 0.0) or 0.0)
                                    now_ts = time.time()
                                    want_pixels_update = interval <= 0.0 or (now_ts - last_emit) >= float(interval) - 1e-4
                                except Exception:
                                    want_pixels_update = True
                            # Spectator frames are primarily used for human streaming/preview, but System2
                            # (centralized VLM) also needs readable frames to navigate UI flows. Historically
                            # we avoided generating spectator frames unless there was an active stream
                            # client, which meant System2 silently fell back to 128x128 pixel obs and could
                            # not localize UI buttons.
                            spectator_ok = (
                                (not hard_disabled)
                                and bool(getattr(self, "_spectator_enabled", False))
                                and int(getattr(self, "_spectator_w", 0) or 0) > 0
                                and int(getattr(self, "_spectator_h", 0) or 0) > 0
                            )
                            want_spectator_stream = spectator_ok and (not capture_disabled)
                            want_spectator_system2 = False
                            if spectator_ok and getattr(self, "_cognitive_client", None) is not None:
                                want_spectator_system2 = str(
                                    os.environ.get("METABONK_SYSTEM2_FORCE_SPECTATOR", "1") or "1"
                                ).strip().lower() in ("1", "true", "yes", "on")

                            stream_has_consumer = False
                            if want_spectator_stream:
                                if bool(getattr(self, "_fifo_stream_enabled", False)):
                                    # FIFO demand-paged streaming should only request expensive spectator frames
                                    # when a reader is connected (go2rtc exec:cat). Otherwise keep the Synthetic
                                    # Eye ingest loop cheap so it can service fences at full rate.
                                    try:
                                        pub = getattr(self, "_fifo_publisher", None)
                                        if pub is not None and hasattr(pub, "has_reader"):
                                            stream_has_consumer = bool(pub.has_reader())
                                        else:
                                            stream_has_consumer = False
                                    except Exception:
                                        stream_has_consumer = False
                                    if not stream_has_consumer:
                                        want_spectator_stream = False
                                else:
                                    try:
                                        s = getattr(self, "streamer", None)
                                        active = (
                                            int(s.active_clients() or 0)
                                            if (s is not None and hasattr(s, "active_clients"))
                                            else 0
                                        )
                                    except Exception:
                                        active = 0
                                    stream_has_consumer = active > 0
                                    if not stream_has_consumer:
                                        want_spectator_stream = False

                            want_spectator = bool(want_spectator_stream or want_spectator_system2)
                            want_spectator_update = False
                            if want_spectator:
                                try:
                                    if stream_has_consumer:
                                        interval = float(getattr(self, "_spectator_interval_s", 0.0) or 0.0)
                                    else:
                                        interval = float(
                                            os.environ.get("METABONK_SYSTEM2_FRAME_INTERVAL_S", "0.5") or "0.5"
                                        )
                                    last_emit = float(getattr(self, "_spectator_last_emit_ts", 0.0) or 0.0)
                                    now_ts = time.time()
                                    want_spectator_update = interval <= 0.0 or (now_ts - last_emit) >= float(interval) - 1e-4
                                except Exception:
                                    want_spectator_update = True
                            want_audit = (
                                (not hard_disabled)
                                and os.environ.get("METABONK_VISION_AUDIT", "0") == "1"
                                and int(frame.frame_id) % 60 == 0
                            )
                            if want_pixels_update or want_audit or want_spectator_update:
                                    h = None
                                    try:
                                        h = self._synthetic_eye_ingestor.begin(frame)
                                    except Exception as e:
                                        print(
                                            f"[VISION] worker={self.instance_id} frame={int(frame.frame_id)} "
                                            f"IMPORT_FAIL err={e} fd={int(frame.dmabuf_fd)} size={int(frame.size_bytes)} "
                                            f"fourcc=0x{int(frame.drm_fourcc) & 0xFFFFFFFF:08x} "
                                            f"modifier=0x{int(frame.modifier) & 0xFFFFFFFFFFFFFFFF:016x}",
                                            flush=True,
                                        )
                                        try:
                                            self._synthetic_eye_ingestor.handshake_only(frame)
                                        except Exception as e2:
                                            print(
                                                f"[VISION] worker={self.instance_id} frame={int(frame.frame_id)} "
                                                f"handshake_fallback_failed: {e2}",
                                                flush=True,
                                            )
                                    else:
                                        try:
                                            from src.agent.tensor_bridge import tensor_from_external_frame

                                            offset_bytes = int(frame.offset) if int(frame.modifier) == 0 else 0
                                            try:
                                                import torch  # type: ignore

                                                ext_stream = torch.cuda.ExternalStream(int(h.stream))
                                            except Exception:
                                                ext_stream = None

                                            if ext_stream is not None:
                                                raw_tensor = tensor_from_external_frame(
                                                    h.ext_frame,
                                                    width=frame.width,
                                                    height=frame.height,
                                                    stride_bytes=frame.stride,
                                                    offset_bytes=offset_bytes,
                                                    stream=h.stream,
                                                    sync=False,
                                                    pack=True,
                                                )
                                                # Ensure downstream work on the current torch stream waits for the
                                                # external CUstream copy to complete. This keeps the fence stream
                                                # free from heavy normalization ops while preserving correctness.
                                                try:
                                                    torch.cuda.current_stream(device=ext_stream.device).wait_stream(ext_stream)
                                                except Exception:
                                                    pass
                                            else:
                                                # Fallback: synchronize inside tensor_from_external_frame.
                                                raw_tensor = tensor_from_external_frame(
                                                    h.ext_frame,
                                                    width=frame.width,
                                                    height=frame.height,
                                                    stride_bytes=frame.stride,
                                                    offset_bytes=offset_bytes,
                                                    stream=h.stream,
                                                    pack=True,
                                                )

                                            # Signal the producer release fence as soon as the packed copy is queued.
                                            # Subsequent ops only touch our private tensor, not the DMA-BUF backing
                                            # store, so keeping the release semaphore gated behind normalization would
                                            # unnecessarily throttle the compositor.
                                            try:
                                                self._synthetic_eye_ingestor.end(h)
                                                h = None
                                            except Exception:
                                                # Keep `h` so the outer finally can retry (best-effort).
                                                pass

                                            raw_tensor = self._maybe_swizzle_channels(raw_tensor, frame.drm_fourcc)
                                            if bool(getattr(self, "_synthetic_eye_vflip", False)):
                                                try:
                                                    raw_tensor = raw_tensor.flip(0)
                                                except Exception:
                                                    pass
                                            raw_tensor_for_stream = None
                                            if want_spectator_update and self._should_cache_synthetic_eye_cuda_stream_frame():
                                                raw_tensor_for_stream = raw_tensor
                                                try:
                                                    if hasattr(raw_tensor_for_stream, "is_contiguous") and not raw_tensor_for_stream.is_contiguous():
                                                        raw_tensor_for_stream = raw_tensor_for_stream.contiguous()
                                                except Exception:
                                                    pass
                                            obs_u8 = raw_tensor.permute(2, 0, 1)[:3]

                                            # QUALITY: always record source frame timestamp for diagnostics.
                                            try:
                                                ts = float(getattr(frame, "timestamp", 0.0) or 0.0) or time.time()
                                            except Exception:
                                                ts = time.time()

                                            # Normalize frames for the agent and spectator separately.
                                            obs_small = None
                                            spectator_small = None
                                            try:
                                                from .frame_normalizer import normalize_obs_u8_chw, normalize_spectator_u8_chw

                                                def _compute_norm() -> tuple[object, object]:
                                                    o = None
                                                    s = None
                                                    if want_pixels_update:
                                                        cutile_obs = getattr(self, "_cutile_obs", None)
                                                        if getattr(self, "_obs_backend", "") == "cutile" and cutile_obs is not None:
                                                            o = cutile_obs.extract_from_chw_u8(obs_u8)
                                                        else:
                                                            o = normalize_obs_u8_chw(
                                                                obs_u8,
                                                                out_h=int(self._pixel_obs_h),
                                                                out_w=int(self._pixel_obs_w),
                                                                backend=str(
                                                                    getattr(self, "_pixel_preprocess_backend", "torch") or "torch"
                                                                ),
                                                            )
                                                    if want_spectator_update:
                                                        s = normalize_spectator_u8_chw(
                                                            obs_u8,
                                                            out_h=int(getattr(self, "_spectator_h", 540) or 540),
                                                            out_w=int(getattr(self, "_spectator_w", 960) or 960),
                                                        )
                                                    return o, s

                                                if ext_stream is not None:
                                                    import torch  # type: ignore

                                                    with torch.cuda.stream(ext_stream):
                                                        obs_small, spectator_small = _compute_norm()
                                                else:
                                                    obs_small, spectator_small = _compute_norm()
                                            except Exception:
                                                obs_small = None
                                                spectator_small = None

                                            if obs_small is not None or spectator_small is not None or raw_tensor_for_stream is not None:
                                                with self._pixel_lock:
                                                    if obs_small is not None:
                                                        self._latest_pixel_obs = obs_small
                                                        try:
                                                            self._latest_pixel_frame_id = int(frame.frame_id)
                                                        except Exception:
                                                            self._latest_pixel_frame_id = None
                                                        self._latest_pixel_ts = float(ts)
                                                        self._pixel_obs_last_emit_ts = float(ts)
                                                        try:
                                                            self._latest_pixel_src_size = (
                                                                int(frame.width),
                                                                int(frame.height),
                                                            )
                                                        except Exception:
                                                            self._latest_pixel_src_size = None
                                                    if spectator_small is not None:
                                                        self._latest_spectator_obs = spectator_small
                                                        try:
                                                            self._latest_spectator_frame_id = int(frame.frame_id)
                                                        except Exception:
                                                            self._latest_spectator_frame_id = None
                                                        self._latest_spectator_ts = float(ts)
                                                        try:
                                                            self._latest_spectator_src_size = (
                                                                int(frame.width),
                                                                int(frame.height),
                                                            )
                                                        except Exception:
                                                            self._latest_spectator_src_size = None
                                                        self._spectator_last_emit_ts = float(ts)
                                                    if raw_tensor_for_stream is not None:
                                                        # Full-res CUDA frame cache for Synthetic Eye NVENC streaming.
                                                        self._latest_spectator_cuda = raw_tensor_for_stream
                                                        self._latest_spectator_cuda_frame_id = int(frame.frame_id)
                                                        self._latest_spectator_cuda_ts = float(ts)
                                                        self._latest_spectator_cuda_src_size = (int(frame.width), int(frame.height))

                                            # One-shot quality debug log (source + derived products).
                                            if not hasattr(self, "_logged_quality_debug"):
                                                self._logged_quality_debug = True
                                                try:
                                                    src_w, src_h = int(frame.width), int(frame.height)
                                                except Exception:
                                                    src_w, src_h = 0, 0
                                                print(
                                                    "[QUALITY_DEBUG] "
                                                    f"worker={self.instance_id} "
                                                    f"src={src_w}x{src_h} "
                                                    f"pixel_obs={int(getattr(self, '_pixel_obs_w', 0) or 0)}x{int(getattr(self, '_pixel_obs_h', 0) or 0)} "
                                                    f"spectator={int(getattr(self, '_spectator_w', 0) or 0)}x{int(getattr(self, '_spectator_h', 0) or 0)} "
                                                    f"target={str(os.environ.get('METABONK_STREAM_NVENC_TARGET_SIZE', '') or '').strip() or 'none'} "
                                                    f"stream_fps={str(os.environ.get('METABONK_STREAM_FPS', '') or '').strip() or 'unset'}",
                                                    flush=True,
                                                )

                                            if want_audit:
                                                try:
                                                    obs_f32 = obs_u8.float().div(255.0)
                                                    mean_val = float(obs_f32.mean().item())
                                                    std_val = float(obs_f32.std().item())
                                                    self._dump_frame_to_png(
                                                        obs_f32, int(frame.frame_id), mean_val, std_val
                                                    )
                                                    print(
                                                        f"[VISION] worker={self.instance_id} frame={int(frame.frame_id)} "
                                                        f"shape={tuple(obs_f32.shape)} mean={mean_val:.4f} std={std_val:.4f} "
                                                        f"device={obs_f32.device}",
                                                        flush=True,
                                                    )
                                                except Exception:
                                                    pass
                                        except Exception as e:
                                            print(
                                                f"[VISION] worker={self.instance_id} tensor bridge failed: {e}",
                                                flush=True,
                                            )
                                    finally:
                                        if h is not None:
                                            self._synthetic_eye_ingestor.end(h)
                            else:
                                self._synthetic_eye_ingestor.handshake_only(frame)
                            try:
                                ts = float(getattr(frame, "timestamp", 0.0) or 0.0) or time.time()
                                self._latest_frame_ts = ts
                                self._record_obs_frame_ts(ts)
                            except Exception:
                                pass
                        except Exception as e:
                            now = time.time()
                            if (now - self._synthetic_eye_drain_last_error_ts) > 5.0:
                                self._synthetic_eye_drain_last_error_ts = now
                                print(
                                    f"[worker:{self.instance_id}] synthetic_eye handshake_only failed: {e}",
                                    flush=True,
                                )
                finally:
                    try:
                        frame.close()
                    except Exception:
                        pass

    def stop(self):
        self._stop.set()
        self.stream.stop()
        self.launcher.shutdown()
        try:
            self._stop_preview_jpeg()
        except Exception:
            pass
        if self._cognitive_client is not None:
            try:
                self._system2_maybe_log_outcome(now=time.time(), reason="stop", done=False)
            except Exception:
                pass
            try:
                self._cognitive_client.cleanup()
            except Exception:
                pass
        if self.streamer:
            self.streamer.stop()
        # Best-effort release held keys (avoid leaving the game stuck moving).
        if self._use_bridge and self.bridge and self._bridge_loop and self._bridge_held_keys:
            try:
                for k in list(self._bridge_held_keys):
                    try:
                        self._bridge_loop.run_until_complete(self.bridge.send_button(str(k), False))
                    except Exception:
                        pass
            except Exception:
                pass
            self._bridge_held_keys.clear()
        if self._fifo_publisher is not None:
            try:
                self._fifo_publisher.stop()
            except Exception:
                pass

    def _discover_gamescope_xwayland_display(self) -> Optional[str]:
        """Parse gamescope logs to find the Xwayland DISPLAY (e.g. ':1')."""
        log_path = getattr(self.launcher, "log_path", None)
        if not log_path:
            return None
        print(f"[worker:{self.instance_id}] waiting for gamescope Xwayland display in {log_path}", flush=True)
        try:
            wait_s = float(os.environ.get("METABONK_INPUT_DISPLAY_WAIT_S", "20.0"))
        except Exception:
            wait_s = 20.0
        deadline = time.time() + max(0.1, float(wait_s))
        pat = re.compile(r"Starting Xwayland on :(\d+)")
        last: Optional[str] = None
        while time.time() < deadline:
            try:
                if log_path.exists():
                    data = log_path.read_text(errors="replace")
                    m = pat.findall(data)
                    if m:
                        last = m[-1]
                        break
            except Exception:
                pass
            time.sleep(0.2)
        if not last:
            return None
        try:
            disp = f":{int(last)}"
        except Exception:
            return None
        print(f"[worker:{self.instance_id}] gamescope Xwayland display: {disp}", flush=True)
        return disp

    def _check_xtest(self, *, display: str) -> tuple[bool, str]:
        """Require XTEST extension for X11 input injection."""
        if not shutil.which("xdpyinfo"):
            raise RuntimeError("xdpyinfo is required to validate XTEST (install xorg-x11-utils)")
        try:
            wait_s = float(os.environ.get("METABONK_INPUT_XTEST_WAIT_S", "5.0"))
        except Exception:
            wait_s = 5.0
        deadline = time.time() + max(0.1, float(wait_s))
        last_out = ""
        while time.time() < deadline:
            try:
                result = subprocess.run(
                    ["xdpyinfo", "-display", str(display), "-ext", "XTEST"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                time.sleep(0.2)
                continue
            out = (result.stdout or "") + (result.stderr or "")
            last_out = out
            out_l = out.lower()
            if result.returncode == 0 and "not supported" not in out_l and "unable to open display" not in out_l:
                return True, out
            time.sleep(0.2)
        return False, last_out

    def _upsert_isolation_log(self, kv: dict[str, str]) -> None:
        try:
            run_dir = os.environ.get("METABONK_RUN_DIR") or os.environ.get("MEGABONK_LOG_DIR") or ""
            if not run_dir:
                return
            worker_id = os.environ.get("METABONK_WORKER_ID")
            if not worker_id:
                try:
                    worker_id = str(int(str(self.instance_id).rsplit("-", 1)[-1]))
                except Exception:
                    worker_id = None
            if not worker_id:
                return
            log_path = os.path.join(run_dir, "logs", f"worker_{worker_id}_isolation.log")
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except FileNotFoundError:
                lines = []
            updated: set[str] = set()
            for idx, line in enumerate(lines):
                if "=" not in line:
                    continue
                key = line.split("=", 1)[0].strip()
                if key in kv and key not in updated:
                    lines[idx] = f"{key}={kv[key]}\n"
                    updated.add(key)
            for key, value in kv.items():
                if key in updated:
                    continue
                if any(l.startswith(f"{key}=") for l in lines):
                    continue
                lines.append(f"{key}={value}\n")
            with open(log_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception:
            return

    def _bind_input_window(self) -> None:
        if not self._input_backend:
            return
        pid = getattr(self.launcher.proc, "pid", None)
        if not pid:
            return
        try:
            if hasattr(self._input_backend, "set_window_pid"):
                self._input_backend.set_window_pid(int(pid))
                os.environ["METABONK_INPUT_XDO_PID"] = str(int(pid))
                print(f"[worker:{self.instance_id}] bound input to pid {int(pid)}", flush=True)
                self._log_input_binding(int(pid))
                if os.environ.get("METABONK_INPUT_AUDIT", "0") == "1" and not bool(
                    getattr(self, "_input_audit_started", False)
                ):
                    self._input_audit_started = True
                    threading.Thread(target=self._run_input_audit, daemon=True).start()
        except Exception:
            pass

    def _run_input_audit(self) -> None:
        try:
            backend = getattr(self, "_input_backend", None)
            if not backend:
                print(f"[INPUT] worker={self.instance_id} audit SKIP backend=None", flush=True)
                return

            # For X11 backends, allow falling back to the active window during audit
            # so we can still validate injection even if PID-based lookup is delayed.
            try:
                if hasattr(backend, "allow_active_fallback"):
                    backend.allow_active_fallback = True
            except Exception:
                pass

            deadline = time.time() + float(os.environ.get("METABONK_INPUT_AUDIT_WAIT_S", "15.0") or "15.0")
            wid = None
            before_scene = None
            before_started = bool(getattr(self, "_gameplay_started", False))
            while time.time() < deadline:
                try:
                    if hasattr(backend, "get_window_id"):
                        wid = backend.get_window_id()
                except Exception:
                    wid = None
                try:
                    ve = getattr(self, "_visual_exploration", None)
                    before_scene = getattr(ve, "last_scene_hash", None) if ve is not None else None
                except Exception:
                    before_scene = None
                if wid is not None or before_scene is not None:
                    break
                time.sleep(0.1)

            print(
                f"[INPUT] worker={self.instance_id} audit BEGIN display={os.environ.get('DISPLAY')} wid={wid} "
                f"before_scene_hash={before_scene} gameplay_started={before_started}",
                flush=True,
            )

            # Optional: verify we can move the pointer on this DISPLAY as a concrete effect.
            pointer_before = None
            pointer_after = None
            pointer_moved = False
            try:
                if shutil.which("xdotool") and os.environ.get("DISPLAY"):
                    env = os.environ.copy()
                    if getattr(backend, "xauth", None):
                        env["XAUTHORITY"] = str(getattr(backend, "xauth"))
                    out = subprocess.check_output(
                        ["xdotool", "getmouselocation", "--shell"],
                        env=env,
                        stderr=subprocess.DEVNULL,
                    ).decode("utf-8", "replace")
                    kv = {}
                    for line in out.splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            kv[k.strip()] = v.strip()
                    if "X" in kv and "Y" in kv:
                        pointer_before = (int(kv["X"]), int(kv["Y"]))
            except Exception:
                pointer_before = None

            # Minimal, non-destructive sequence: tap Escape then Return.
            send_fail = False
            for key in ("Escape", "Return"):
                try:
                    backend.key_down(key)
                    time.sleep(0.03)
                    backend.key_up(key)
                except Exception as e:
                    send_fail = True
                    print(f"[INPUT] worker={self.instance_id} audit key={key} SEND_FAIL err={e}", flush=True)
                time.sleep(0.15)

            try:
                backend.mouse_move(40, 0)
                time.sleep(0.05)
                backend.mouse_move(-40, 0)
            except Exception:
                pass

            try:
                if shutil.which("xdotool") and os.environ.get("DISPLAY"):
                    env = os.environ.copy()
                    if getattr(backend, "xauth", None):
                        env["XAUTHORITY"] = str(getattr(backend, "xauth"))
                    out = subprocess.check_output(
                        ["xdotool", "getmouselocation", "--shell"],
                        env=env,
                        stderr=subprocess.DEVNULL,
                    ).decode("utf-8", "replace")
                    kv = {}
                    for line in out.splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            kv[k.strip()] = v.strip()
                    if "X" in kv and "Y" in kv:
                        pointer_after = (int(kv["X"]), int(kv["Y"]))
            except Exception:
                pointer_after = None

            if pointer_before is not None and pointer_after is not None:
                dx = abs(int(pointer_after[0]) - int(pointer_before[0]))
                dy = abs(int(pointer_after[1]) - int(pointer_before[1]))
                pointer_moved = (dx + dy) >= 5

            time.sleep(float(os.environ.get("METABONK_INPUT_AUDIT_POST_S", "1.0") or "1.0"))
            try:
                ve = getattr(self, "_visual_exploration", None)
                after_scene = getattr(ve, "last_scene_hash", None) if ve is not None else None
            except Exception:
                after_scene = None
            after_started = bool(getattr(self, "_gameplay_started", False))
            changed = (
                (before_scene is not None)
                and (after_scene is not None)
                and (str(after_scene) != str(before_scene))
            )
            print(
                f"[INPUT] worker={self.instance_id} audit END after_scene_hash={after_scene} scene_changed={changed} "
                f"pointer_moved={pointer_moved} send_fail={send_fail} "
                f"mouse_before={pointer_before} mouse_after={pointer_after}",
                flush=True,
            )
        except Exception as e:
            print(f"[INPUT] worker={self.instance_id} audit FAILED err={e}", flush=True)

    def _log_input_binding(self, pid: Optional[int]) -> None:
        try:
            wid = None
            if hasattr(self._input_backend, "get_window_id"):
                wid = self._input_backend.get_window_id()
            if wid:
                os.environ["METABONK_INPUT_XDO_WID"] = str(wid)
            updates: dict[str, str] = {}
            if pid:
                updates["GAME_PID"] = str(int(pid))
            if wid:
                updates["WINDOW_ID"] = str(wid)
            if updates:
                self._upsert_isolation_log(updates)
        except Exception:
            return

    def _set_latest_jpeg(self, data: Optional[bytes]) -> None:
        if not data:
            return
        self._latest_jpeg_bytes = data
        self._latest_frame_ts = time.time()
        self._record_obs_frame_ts(self._latest_frame_ts)
        try:
            import io
            from PIL import Image

            img = Image.open(io.BytesIO(data)).convert("RGB")
            thumb = img.copy()
            thumb.thumbnail((256, 144))
            buf_thumb = io.BytesIO()
            thumb.save(buf_thumb, format="JPEG", quality=60)
            self._latest_thumb_b64 = base64.b64encode(buf_thumb.getvalue()).decode("ascii")
        except Exception:
            self._latest_thumb_b64 = None

    def _record_obs_frame_ts(self, ts: float) -> None:
        try:
            self._obs_frame_times.append(float(ts))
            if len(self._obs_frame_times) >= 2:
                dt = float(self._obs_frame_times[-1]) - float(self._obs_frame_times[0])
                if dt > 0:
                    self._obs_fps = float((len(self._obs_frame_times) - 1) / dt)
        except Exception:
            pass

    def _get_latest_pixel_obs(self):
        if not bool(getattr(self, "_pixel_obs_enabled", False)):
            return None, None, None, None
        try:
            lock = getattr(self, "_pixel_lock", None)
        except Exception:
            lock = None
        if lock is None:
            return None, None, None, None
        with lock:
            obs = getattr(self, "_latest_pixel_obs", None)
            frame_id = getattr(self, "_latest_pixel_frame_id", None)
            ts = float(getattr(self, "_latest_pixel_ts", 0.0) or 0.0)
            src_size = getattr(self, "_latest_pixel_src_size", None)
        return obs, frame_id, ts, src_size

    def _get_latest_spectator_obs(self):
        if not bool(getattr(self, "_spectator_enabled", False)):
            return None, None, None, None
        try:
            lock = getattr(self, "_pixel_lock", None)
        except Exception:
            lock = None
        if lock is None:
            return None, None, None, None
        with lock:
            obs = getattr(self, "_latest_spectator_obs", None)
            frame_id = getattr(self, "_latest_spectator_frame_id", None)
            ts = float(getattr(self, "_latest_spectator_ts", 0.0) or 0.0)
            src_size = getattr(self, "_latest_spectator_src_size", None)
        return obs, frame_id, ts, src_size

    def _should_cache_synthetic_eye_cuda_stream_frame(self) -> bool:
        if self._frame_source not in ("synthetic_eye", "smithay", "smithay_dmabuf"):
            return False
        if not bool(getattr(self, "_stream_enabled", False)):
            return False
        if bool(getattr(self, "_capture_disabled", False)):
            return False
        if bool(getattr(self, "_fifo_stream_enabled", False)):
            return True
        s = getattr(self, "streamer", None)
        if s is not None:
            try:
                if hasattr(s, "active_clients") and int(s.active_clients() or 0) > 0:
                    return True
            except Exception:
                pass
        # If the configured backend is explicitly CUDA-based, keep a hot cache so the first
        # viewer gets an init segment + IDR quickly without waiting for a slow path.
        if bool(getattr(self, "_stream_wants_cuda_frames", False)) or bool(getattr(self, "_stream_require_zero_copy", False)):
            return s is not None
        return False

    def _get_latest_spectator_cuda_frame(self) -> Optional[CudaFrame]:
        """Return the latest full-res Synthetic Eye frame as a CUDA tensor (best-effort)."""
        if self._frame_source not in ("synthetic_eye", "smithay", "smithay_dmabuf"):
            return None
        try:
            lock = getattr(self, "_pixel_lock", None)
        except Exception:
            lock = None
        if lock is None:
            return None
        with lock:
            t = getattr(self, "_latest_spectator_cuda", None)
            frame_id = getattr(self, "_latest_spectator_cuda_frame_id", None)
            ts = float(getattr(self, "_latest_spectator_cuda_ts", 0.0) or 0.0)
            src_size = getattr(self, "_latest_spectator_cuda_src_size", None)
            stream_ptr = int(getattr(self, "_synthetic_eye_cu_stream_ptr", 0) or 0)
        if t is None:
            return None
        try:
            import torch  # type: ignore

            if hasattr(t, "device") and str(getattr(t, "device", "")).startswith("cuda") and stream_ptr:
                ext = torch.cuda.ExternalStream(int(stream_ptr))
                torch.cuda.current_stream(device=ext.device).wait_stream(ext)
        except Exception:
            pass
        try:
            w, h = (int(src_size[0]), int(src_size[1])) if isinstance(src_size, tuple) and len(src_size) >= 2 else (0, 0)
        except Exception:
            w, h = (0, 0)
        if w <= 0 or h <= 0:
            try:
                shape = getattr(t, "shape", None)
                if shape is not None and len(shape) >= 2:
                    h = int(shape[0])
                    w = int(shape[1])
            except Exception:
                w, h = (0, 0)
        if w <= 0 or h <= 0:
            return None
        return CudaFrame(
            tensor=t,
            width=w,
            height=h,
            frame_id=int(frame_id) if frame_id is not None else None,
            timestamp=float(ts or time.time()),
            format="RGBA",
        )

    def _get_latest_spectator_stream_frame(self) -> Optional[PixelFrame]:
        """Return the latest spectator frame as raw RGB24 bytes (best-effort)."""
        obs, frame_id, ts, _src = self._get_latest_spectator_obs()
        if obs is None:
            # Fallback to pixel obs stream if spectator isn't ready yet.
            try:
                return self._get_latest_pixel_stream_frame()
            except Exception:
                return None
        try:
            fid = int(frame_id) if frame_id is not None else None
        except Exception:
            fid = None
        try:
            cached_id = getattr(self, "_spectator_stream_cache_frame_id", None)
            cached_bytes = getattr(self, "_spectator_stream_cache_bytes", None)
            cached_size = getattr(self, "_spectator_stream_cache_size", None)
            if fid is not None and cached_id is not None and int(fid) == int(cached_id) and cached_bytes and cached_size:
                w, h = int(cached_size[0]), int(cached_size[1])
                return PixelFrame(
                    data=cached_bytes,
                    width=w,
                    height=h,
                    pix_fmt="rgb24",
                    frame_id=fid,
                    timestamp=float(ts or time.time()),
                )
        except Exception:
            pass

        # Convert to HWC uint8 on CPU for ffmpeg rawvideo.
        try:
            t = obs
            try:
                if hasattr(t, "detach"):
                    t = t.detach()
            except Exception:
                pass
            try:
                if hasattr(t, "device") and str(getattr(t, "device", "")).startswith("cuda"):
                    try:
                        import torch  # type: ignore

                        stream_ptr = int(getattr(self, "_synthetic_eye_cu_stream_ptr", 0) or 0)
                        if stream_ptr:
                            ext = torch.cuda.ExternalStream(int(stream_ptr))
                            torch.cuda.current_stream(device=ext.device).wait_stream(ext)
                    except Exception:
                        pass
                    t = t.cpu()
                elif hasattr(t, "cpu"):
                    t = t.cpu()
            except Exception:
                if hasattr(t, "cpu"):
                    t = t.cpu()
            # Expect CHW uint8 (3xHxW).
            if hasattr(t, "ndim") and int(getattr(t, "ndim", 0) or 0) == 3:
                try:
                    t = t[:3].permute(1, 2, 0).contiguous()
                except Exception:
                    pass
            arr = t.numpy() if hasattr(t, "numpy") else None
            if arr is None or getattr(arr, "ndim", 0) != 3:
                return None
            h, w, c = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
            if w <= 0 or h <= 0 or c < 3:
                return None
            if c != 3:
                arr = arr[:, :, :3]
            data = arr.tobytes()
        except Exception:
            return None

        try:
            self._spectator_stream_cache_frame_id = fid
            self._spectator_stream_cache_bytes = data
            self._spectator_stream_cache_size = (int(w), int(h))
            self._spectator_stream_cache_ts = float(ts or time.time())
        except Exception:
            pass

        return PixelFrame(
            data=data,
            width=int(w),
            height=int(h),
            pix_fmt="rgb24",
            frame_id=fid,
            timestamp=float(ts or time.time()),
        )

    def _get_latest_pixel_stream_frame(self) -> Optional[PixelFrame]:
        """Return the latest pixel observation as raw RGB24 bytes (best-effort)."""
        obs, frame_id, ts, _src = self._get_latest_pixel_obs()
        if obs is None:
            return None
        try:
            fid = int(frame_id) if frame_id is not None else None
        except Exception:
            fid = None
        try:
            cached_id = getattr(self, "_pixel_stream_cache_frame_id", None)
            cached_bytes = getattr(self, "_pixel_stream_cache_bytes", None)
            cached_size = getattr(self, "_pixel_stream_cache_size", None)
            if fid is not None and cached_id is not None and int(fid) == int(cached_id) and cached_bytes and cached_size:
                w, h = int(cached_size[0]), int(cached_size[1])
                return PixelFrame(
                    data=cached_bytes,
                    width=w,
                    height=h,
                    pix_fmt="rgb24",
                    frame_id=fid,
                    timestamp=float(ts or time.time()),
                )
        except Exception:
            pass

        # Convert to HWC uint8 on CPU for ffmpeg rawvideo.
        #
        # IMPORTANT: Do not upscale to a large target size (e.g., 1920x1080) here. That forces
        # massive rawvideo writes (W*H*3) into ffmpeg and can induce stutter/backpressure across
        # multiple featured tiles. Output scaling is applied in ffmpeg's filter graph.
        try:
            t = obs
            try:
                if hasattr(t, "detach"):
                    t = t.detach()
            except Exception:
                pass
            try:
                if hasattr(t, "device") and str(getattr(t, "device", "")).startswith("cuda"):
                    try:
                        import torch  # type: ignore

                        stream_ptr = int(getattr(self, "_synthetic_eye_cu_stream_ptr", 0) or 0)
                        if stream_ptr:
                            ext = torch.cuda.ExternalStream(int(stream_ptr))
                            torch.cuda.current_stream(device=ext.device).wait_stream(ext)
                    except Exception:
                        pass
                    t = t.cpu()
                elif hasattr(t, "cpu"):
                    t = t.cpu()
            except Exception:
                if hasattr(t, "cpu"):
                    t = t.cpu()
            # Expect CHW uint8 (3xHxW).
            if hasattr(t, "ndim") and int(getattr(t, "ndim", 0) or 0) == 3:
                try:
                    t = t[:3].permute(1, 2, 0).contiguous()
                except Exception:
                    pass
            arr = t.numpy() if hasattr(t, "numpy") else None
            if arr is None:
                return None
            if getattr(arr, "ndim", 0) != 3:
                return None
            h, w, c = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
            if w <= 0 or h <= 0 or c < 3:
                return None
            if c != 3:
                arr = arr[:, :, :3]

            # NVENC encoders often reject very small frames. Upscale for streaming only
            # (keeps training/inference resolution unchanged).
            try:
                min_dim = int(os.environ.get("METABONK_STREAM_NVENC_MIN_DIM", "256"))
            except Exception:
                min_dim = 256
            if min_dim > 0 and (w < min_dim or h < min_dim):
                scale = max(float(min_dim) / float(w), float(min_dim) / float(h))
                out_w = max(min_dim, int(round(float(w) * scale)))
                out_h = max(min_dim, int(round(float(h) * scale)))
                # NVENC typically requires even dimensions.
                if out_w % 2:
                    out_w += 1
                if out_h % 2:
                    out_h += 1
                try:
                    from PIL import Image

                    img = Image.fromarray(arr)
                    img = img.resize((int(out_w), int(out_h)), resample=getattr(Image, "NEAREST", 0))
                    arr = img.tobytes()  # type: ignore[assignment]
                    w, h = int(out_w), int(out_h)
                except Exception:
                    # If PIL resize fails, fall back to encoding the original size (may fail on some NVENC stacks).
                    pass
            data = arr if isinstance(arr, (bytes, bytearray, memoryview)) else arr.tobytes()
        except Exception:
            return None

        try:
            self._pixel_stream_cache_frame_id = fid
            self._pixel_stream_cache_bytes = data
            self._pixel_stream_cache_size = (int(w), int(h))
            self._pixel_stream_cache_ts = float(ts or time.time())
        except Exception:
            pass

        return PixelFrame(
            data=data,
            width=int(w),
            height=int(h),
            pix_fmt="rgb24",
            frame_id=fid,
            timestamp=float(ts or time.time()),
        )

    def _lockstep_request_next_frame(self, baseline_frame_id: Optional[int]) -> None:
        if not bool(getattr(self, "_synthetic_eye_lockstep", False)):
            return
        try:
            if not hasattr(self.stream, "request_frame"):
                return
        except Exception:
            return
        try:
            ok = bool(getattr(self.stream, "request_frame")())
        except Exception:
            ok = False
        if not ok:
            return
        try:
            wait_s = float(getattr(self, "_synthetic_eye_lockstep_wait_s", 0.0) or 0.0)
        except Exception:
            wait_s = 0.0
        if wait_s <= 0:
            return
        deadline = time.time() + wait_s
        while not self._stop.is_set():
            _obs, frame_id, _ts, _src = self._get_latest_pixel_obs()
            if frame_id is not None:
                try:
                    if baseline_frame_id is None or int(frame_id) != int(baseline_frame_id):
                        break
                except Exception:
                    break
            if time.time() >= deadline:
                break
            self._stop.wait(0.001)

    @staticmethod
    def _parse_grid_spec(spec: str, *, default_rows: int = 8, default_cols: int = 8) -> tuple[int, int]:
        raw = str(spec or "").strip().lower()
        if not raw:
            return int(default_rows), int(default_cols)
        try:
            parts = raw.split("x")
            rows = int(parts[0])
            cols = int(parts[1]) if len(parts) > 1 else rows
            rows = max(1, rows)
            cols = max(1, cols)
            return rows, cols
        except Exception:
            return int(default_rows), int(default_cols)

    @staticmethod
    def _maybe_swizzle_channels(raw_tensor, drm_fourcc: int):
        """Swap channel order for common little-endian DRM formats (XR24/AR24)."""
        try:
            fmt = int(drm_fourcc) & 0xFFFFFFFF
        except Exception:
            return raw_tensor
        if fmt in (0x34325258, 0x34325241):  # XRGB8888 / ARGB8888
            try:
                return raw_tensor[..., [2, 1, 0, 3]]
            except Exception:
                return raw_tensor
        return raw_tensor

    def _dump_frame_to_png(self, obs_tensor, frame_id: int, mean_val: float, std_val: float) -> None:
        if os.environ.get("METABONK_DUMP_FRAMES") != "1":
            return
        try:
            import numpy as np
            from PIL import Image
        except Exception:
            return
        out_dir = Path(os.environ.get("METABONK_DUMP_FRAMES_DIR", "/tmp/metabonk_frames"))
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        try:
            frame_np = (
                obs_tensor.permute(1, 2, 0)
                .detach()
                .float()
                .clamp(0.0, 1.0)
                .cpu()
                .numpy()
            )
            frame_np = (frame_np * 255.0).astype(np.uint8)
            filename = f"frame_{int(frame_id):06d}_mean{mean_val:.4f}_std{std_val:.4f}.png"
            Image.fromarray(frame_np).save(out_dir / filename)
        except Exception:
            return

    def _record_flight_frame(
        self,
        *,
        action_label: str,
        input_vector: Optional[List[float]] = None,
        model_entropy: Optional[float] = None,
    ) -> None:
        if self._telemetry_capacity <= 0:
            return
        entry = {
            "step_id": int(getattr(self.trainer, "step_count", 0) or 0),
            "timestamp_ms": int(time.time() * 1000),
            "action_label": str(action_label),
            "input_vector": [float(x) for x in (input_vector or [])],
            "frame_thumbnail_b64": self._latest_thumb_b64,
            "model_entropy": float(model_entropy) if model_entropy is not None else None,
        }
        try:
            with self._telemetry_lock:
                self._telemetry_history.append(entry)
        except Exception:
            pass

    def telemetry_history(self, limit: int = 50) -> list[dict]:
        limit = max(1, int(limit))
        with self._telemetry_lock:
            hist = list(self._telemetry_history)
        return hist[-limit:]

    def _normalize_action_source(self, raw: Optional[str]) -> str:
        val = str(raw or "").strip().lower()
        if val in ("", "policy", "ppo", "learned", "agent", "default"):
            return "policy"
        if val in ("random", "rand", "baseline_random"):
            return "random"
        return "policy"

    def _set_action_source(self, raw: Optional[str]) -> None:
        self._action_source = self._normalize_action_source(raw)

    def _sample_random_action(self, action_mask: Optional[List[int]]) -> tuple[List[float], List[int]]:
        cfg = getattr(self.trainer, "cfg", None)
        cont_dim = int(getattr(cfg, "continuous_dim", 2) or 2)
        cont = [random.uniform(-1.0, 1.0) for _ in range(cont_dim)]
        disc: List[int] = []
        branches = list(getattr(cfg, "discrete_branches", (1,)) or (1,))
        for b in branches:
            try:
                b = int(b)
            except Exception:
                b = 1
            b = max(1, b)
            if action_mask is not None and len(action_mask) == b:
                valid = [i for i, m in enumerate(action_mask) if m == 1]
                if valid:
                    disc.append(int(random.choice(valid)))
                else:
                    disc.append(b - 1)
            else:
                disc.append(int(random.randrange(b)))
        return cont, disc

    def _update_gameplay_state(
        self,
        game_state: dict,
        *,
        hud_present: bool = False,
        phase_gameplay: bool = False,
        frame: "Any" = None,
        step_increment: int = 0,
        action_taken: bool = False,
    ) -> None:
        if self._gameplay_started:
            return
        
        # ONLY use hybrid detector for gameplay detection.
        # No HUD boost, no legacy fallbacks - pure vision only.
        if frame is not None and hasattr(self, "_gameplay_detector"):
            try:
                import numpy as np
                # Ensure frame is numpy array
                if hasattr(frame, "numpy"):
                    frame_arr = frame.cpu().numpy() if hasattr(frame, "cpu") else frame.numpy()
                elif hasattr(frame, "detach"):
                    frame_arr = frame.detach().cpu().numpy()
                elif isinstance(frame, np.ndarray):
                    frame_arr = frame
                else:
                    frame_arr = np.asarray(frame)
                
                detected = self._gameplay_detector.update(
                    frame=frame_arr,
                    step_increment=step_increment,
                    action_taken=action_taken,
                )
                self._gameplay_confidence = self._gameplay_detector.confidence
                
                if detected:
                    self._gameplay_started = True
                    self._gameplay_start_ts = time.time()
            except Exception:
                pass

    def _flag_action_guard_violation(self, reason: str) -> None:
        if self._action_guard_violation:
            return
        msg = f"[action-guard] {reason}"
        self._action_guard_violation = msg
        path = self._action_guard_path.strip() if self._action_guard_path else ""
        if not path:
            path = str(Path("temp") / "action_guard_violations" / f"{self.instance_id}.txt")
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(msg)
        except Exception:
            pass
        print(f"[worker:{self.instance_id}] {msg}", file=sys.stderr)
        try:
            os._exit(2)
        except Exception:
            raise RuntimeError(msg)

    def _action_guard_check(
        self,
        *,
        action_source: str,
        forced_ui_click: Optional[tuple[int, int]],
        input_bootstrap: bool,
        sima2_action: Optional[List[float]],
    ) -> None:
        if not self._action_guard_enabled:
            return
        if not self._gameplay_started:
            return
        try:
            from src.proof_harness.guards import action_guard_violation

            reason = action_guard_violation(
                gameplay_started=self._gameplay_started,
                action_source=action_source,
                forced_ui_click=forced_ui_click,
                input_bootstrap=input_bootstrap,
                sima2_action=sima2_action,
            )
            if reason:
                self._flag_action_guard_violation(reason)
        except Exception:
            return

    def _ensure_preview_jpeg(self) -> None:
        # Default-on: this runs at a low rate and keeps a cached JPEG for UI fallback/debug.
        if os.environ.get("METABONK_PREVIEW_JPEG", "0") in ("0", "false", "False"):
            return
        if self._preview_thread is not None and self._preview_thread.is_alive():
            return
        node = os.environ.get("PIPEWIRE_NODE") or self._pipewire_node
        if not node:
            return
        try:
            fps = float(os.environ.get("METABONK_PREVIEW_JPEG_FPS", "2.0"))
        except Exception:
            fps = 2.0
        if fps <= 0:
            return
        interval = 1.0 / max(0.25, fps)

        self._preview_stop.clear()
        try:
            self._preview_stream = CaptureStream(pipewire_node=str(node), use_dmabuf=False)
            self._preview_stream.start()
        except Exception:
            self._preview_stream = None
            return

        def _run() -> None:
            while not self._preview_stop.is_set():
                try:
                    stream = self._preview_stream
                    if stream is None:
                        break
                    frame = stream.read()
                    if frame is None or not getattr(frame, "cpu_rgb", None):
                        self._preview_stop.wait(0.05)
                        continue
                    try:
                        import io
                        from PIL import Image

                        img = Image.frombytes(
                            "RGB",
                            (int(frame.width), int(frame.height)),
                            frame.cpu_rgb,  # type: ignore[arg-type]
                        )
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=80)
                        self._set_latest_jpeg(buf.getvalue())
                    except Exception:
                        pass
                    try:
                        frame.close()
                    except Exception:
                        pass
                except Exception:
                    pass
                self._preview_stop.wait(interval)

            try:
                if self._preview_stream is not None:
                    self._preview_stream.stop()
            except Exception:
                pass
            self._preview_stream = None

        self._preview_thread = threading.Thread(target=_run, daemon=True)
        self._preview_thread.start()

    def _stop_preview_jpeg(self) -> None:
        self._preview_stop.set()
        try:
            if self._preview_stream is not None:
                self._preview_stream.stop()
        except Exception:
            pass
        self._preview_stream = None
        self._preview_thread = None

    def _public_base_url(self) -> str:
        """Best-effort public URL for this worker."""
        env_url = os.environ.get("WORKER_PUBLIC_URL")
        if env_url:
            return env_url.rstrip("/")
        public_host = os.environ.get("WORKER_PUBLIC_HOST") or self.host or "127.0.0.1"
        if public_host in ("0.0.0.0", "::"):
            public_host = "127.0.0.1"
        return f"http://{public_host}:{self.port}"

    def stream_url(self) -> str:
        # Stream HUD only advertises the GPU MP4 stream. No CPU/MJPEG fallback is advertised.
        if self.streamer is not None and not getattr(self, "_capture_disabled", False):
            return f"{self._public_base_url()}/stream.mp4"
        return ""

    def control_url(self) -> str:
        return self._public_base_url()

    def _init_unity_bridge(self):
        if not self._use_bridge or UnityBridge is None or BridgeConfig is None:
            return
        try:
            cfg = BridgeConfig()
            # Optional env overrides for capture shape.
            try:
                cfg.frame_width = int(os.environ.get("METABONK_BRIDGE_WIDTH", cfg.frame_width))
                cfg.frame_height = int(os.environ.get("METABONK_BRIDGE_HEIGHT", cfg.frame_height))
            except Exception:
                pass
            cfg.frame_format = os.environ.get("METABONK_BRIDGE_FORMAT", cfg.frame_format)
            self._bridge_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._bridge_loop)
            self.bridge = UnityBridge(cfg)
            ok = self._bridge_loop.run_until_complete(self.bridge.connect())
            if not ok:
                self.bridge = None
                self._use_bridge = False
        except Exception:
            self.bridge = None
            self._use_bridge = False

    def _parse_input_buttons(self) -> List[dict]:
        """Parse input button list for uinput backend."""
        raw = (
            os.environ.get("METABONK_INPUT_BUTTONS")
            or os.environ.get("METABONK_INPUT_KEYS")
            or os.environ.get("METABONK_BUTTON_KEYS")
            or ""
        )
        items = [s.strip() for s in str(raw).split(",") if s.strip()]
        mouse_names = {
            "LEFT",
            "RIGHT",
            "MIDDLE",
            "MOUSE_LEFT",
            "MOUSE_RIGHT",
            "MOUSE_MIDDLE",
            "BTN_LEFT",
            "BTN_RIGHT",
            "BTN_MIDDLE",
        }
        parsed: List[dict] = []
        for item in items:
            key = item.strip()
            if key.upper() in mouse_names:
                parsed.append({"kind": "mouse", "button": key})
            else:
                parsed.append({"kind": "key", "name": key})
        return parsed

    def _frame_stats_from_array(self, arr: "Any") -> Optional[tuple[float, float, float]]:
        try:
            import numpy as np

            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                data = arr
            else:
                data = np.asarray(arr)
            if data.ndim != 3 or data.shape[2] < 3:
                return None
            data = data[:, :, :3].astype("float32")
            luma = 0.2126 * data[:, :, 0] + 0.7152 * data[:, :, 1] + 0.0722 * data[:, :, 2]
            maxc = data.max(axis=2)
            minc = data.min(axis=2)
            delta = maxc - minc
            sat = np.zeros_like(maxc)
            mask = maxc > 1e-6
            sat[mask] = (delta[mask] / maxc[mask]) * 255.0
            mean = float(luma.mean())
            p99 = float(np.percentile(luma, 99))
            mean_s = float(sat.mean())
            return mean, p99, mean_s
        except Exception:
            return None

    def _frame_stats_from_bytes(self, payload: bytes) -> Optional[tuple[float, float, float]]:
        if not payload:
            return None
        try:
            import io
            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(payload)).convert("RGB")
            arr = np.asarray(img)
            return self._frame_stats_from_array(arr)
        except Exception:
            return None

    def _hud_roi_from_frame(self, img: "Any") -> Optional["Any"]:
        try:
            import numpy as np

            if isinstance(img, np.ndarray):
                arr = img
            else:
                arr = np.asarray(img)
            if arr.ndim != 3:
                return None
            h, w = arr.shape[:2]
            x0 = int(max(0, min(w - 1, round(float(self._hud_roi_x0) * w))))
            x1 = int(max(1, min(w, round(float(self._hud_roi_x1) * w))))
            y0 = int(max(0, min(h - 1, round(float(self._hud_roi_y0) * h))))
            y1 = int(max(1, min(h, round(float(self._hud_roi_y1) * h))))
            if x1 <= x0 or y1 <= y0:
                return None
            return arr[y0:y1, x0:x1]
        except Exception:
            return None

    def _detect_hud_minimap(self, jpeg_bytes: Optional[bytes]) -> bool:
        if not self._hud_enabled or not jpeg_bytes:
            return False
        try:
            import io
            import cv2  # type: ignore
            from PIL import Image

            img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            roi = self._hud_roi_from_frame(img)
            if roi is None:
                return False
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            mean_s = float(hsv[:, :, 1].mean())
            if mean_s < float(self._hud_sat_min):
                return False
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
            h, w = gray.shape[:2]
            min_dist = max(6, int(min(h, w) * float(self._hud_min_dist_frac)))
            min_radius = max(4, int(min(h, w) * float(self._hud_min_radius_frac)))
            max_radius = max(min_radius + 2, int(min(h, w) * float(self._hud_max_radius_frac)))
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                float(self._hud_hough_dp),
                float(min_dist),
                param1=float(self._hud_hough_param1),
                param2=float(self._hud_hough_param2),
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles is None:
                return False
            return len(circles[0]) > 0
        except Exception:
            return False

    def _detect_hud_minimap_array(self, frame_rgb: "Any") -> bool:
        """Detect HUD/minimap using an RGB ndarray (HWC) instead of JPEG bytes."""
        if not self._hud_enabled:
            return False
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            if frame_rgb is None:
                return False
            arr = frame_rgb if isinstance(frame_rgb, np.ndarray) else np.asarray(frame_rgb)
            if arr.ndim != 3 or int(arr.shape[2]) < 3:
                return False
            roi = self._hud_roi_from_frame(arr)
            if roi is None:
                return False
            if roi.dtype != np.uint8:
                roi = np.clip(roi, 0, 255).astype(np.uint8)
            if int(roi.shape[2]) != 3:
                roi = roi[:, :, :3]
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            mean_s = float(hsv[:, :, 1].mean())
            if mean_s < float(self._hud_sat_min):
                return False
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
            h, w = gray.shape[:2]
            min_dist = max(6, int(min(h, w) * float(self._hud_min_dist_frac)))
            min_radius = max(4, int(min(h, w) * float(self._hud_min_radius_frac)))
            max_radius = max(min_radius + 2, int(min(h, w) * float(self._hud_max_radius_frac)))
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                float(self._hud_hough_dp),
                float(min_dist),
                param1=float(self._hud_hough_param1),
                param2=float(self._hud_hough_param2),
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles is None:
                return False
            return len(circles[0]) > 0
        except Exception:
            return False

    def _update_hud_state(self, hud_present: bool) -> None:
        if hud_present:
            self._hud_on_count += 1
            self._hud_off_count = 0
        else:
            self._hud_off_count += 1
            self._hud_on_count = 0
        if not self._hud_present and self._hud_on_count >= max(1, int(self._hud_on_frames)):
            self._hud_present = True
            self._hud_on_count = 0
            self._hud_off_count = 0
        elif self._hud_present and self._hud_off_count >= max(1, int(self._hud_off_frames)):
            self._hud_present = False
            self._hud_on_count = 0
            self._hud_off_count = 0

    def _sample_frame_ring_bytes(self, n: int) -> List[bytes]:
        if not self._frame_ring or n <= 0:
            return []
        entries = list(self._frame_ring)[-n:]
        payloads: List[bytes] = []
        for entry in entries:
            payload = entry.get("bytes")
            if payload:
                payloads.append(payload)
        return payloads

    def _decode_jpeg_to_array(self, payload: bytes) -> Optional["Any"]:
        if not payload:
            return None
        try:
            import io
            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(payload)).convert("RGB")
            return np.asarray(img)
        except Exception:
            return None

    def _weak_phase_label(self, game_state: dict) -> str:
        """Best-effort coarse phase label (game-agnostic).

        Intentionally avoids any game-specific menu/scene taxonomy. For labeling
        and debugging, prefer `VisualExplorationReward.metrics()["scene_hash"]`.
        """
        if self._hud_present:
            return "gameplay"
        try:
            if game_state.get("isPlaying") is True:
                return "gameplay"
        except Exception:
            pass
        return "unknown"

    def _phase_dataset_label_count(self, label: str) -> int:
        if label in self._phase_dataset_counts:
            return self._phase_dataset_counts[label]
        try:
            out_dir = Path(self._phase_dataset_dir) / label
            if not out_dir.exists():
                self._phase_dataset_counts[label] = 0
                return 0
            count = 0
            for entry in out_dir.iterdir():
                if entry.is_file():
                    count += 1
            self._phase_dataset_counts[label] = count
            return count
        except Exception:
            return 0

    def _phase_dataset_can_write(self, label: str) -> bool:
        if self._phase_dataset_max_per_label <= 0:
            return True
        return self._phase_dataset_label_count(label) < int(self._phase_dataset_max_per_label)

    def _set_phase_dataset_forced_label(self, label: Optional[str], *, now: float, duration_s: float) -> None:
        if not label:
            self._phase_dataset_forced_label = None
            self._phase_dataset_forced_until_ts = 0.0
            return
        self._phase_dataset_forced_label = label
        self._phase_dataset_forced_until_ts = float(now + max(0.0, duration_s))

    def _arm_loading_window(self, *, now: float) -> None:
        # Dump the frames immediately preceding the transition. This is a game-agnostic
        # "pre-transition" slice that surfaces the otherwise invisible menu/lobby screens.
        try:
            self._maybe_dump_phase_sample(
                now=now,
                label="pre_loading",
                game_state={},
                force=True,
                event="arm_loading",
            )
        except Exception:
            pass
        self._phase_dataset_loading_active = True
        self._phase_dataset_loading_since = float(now)
        self._phase_dataset_loading_deadline = float(now + max(0.0, self._phase_dataset_loading_timeout_s))

    def _phase_dataset_label_override(
        self,
        *,
        now: float,
        base_label: str,
        game_state: dict,
        gameplay_now: bool,
    ) -> str:
        label = base_label
        if self._phase_dataset_forced_label and now <= float(self._phase_dataset_forced_until_ts):
            label = self._phase_dataset_forced_label
        if self._phase_dataset_loading_active:
            if gameplay_now or self._hud_present or (game_state.get("isPlaying") is True):
                self._phase_dataset_loading_active = False
                self._phase_dataset_loading_since = 0.0
            elif now > float(self._phase_dataset_loading_deadline):
                self._phase_dataset_loading_active = False
                self._phase_dataset_loading_since = 0.0
            else:
                label = "loading"
        return label

    def _terminate_launcher_proc(self, *, reason: str) -> None:
        proc = getattr(self.launcher, "proc", None)
        if proc is None:
            return
        try:
            pid = int(getattr(proc, "pid", 0) or 0)
        except Exception:
            pid = 0
        if pid <= 1:
            return
        try:
            alive = getattr(proc, "poll", lambda: 0)() is None
        except Exception:
            alive = False
        if not alive:
            return
        print(f"[launcher] terminating game (reason={reason}) pid={pid}", flush=True)
        try:
            proc.terminate()
        except Exception:
            pass
        deadline = time.time() + 1.0
        while time.time() < deadline:
            try:
                if getattr(proc, "poll", lambda: 0)() is not None:
                    break
            except Exception:
                break
            time.sleep(0.05)
        try:
            if getattr(proc, "poll", lambda: 0)() is None:
                proc.kill()
        except Exception:
            pass

    def _maybe_dump_phase_sample(
        self,
        *,
        now: float,
        label: str,
        game_state: dict,
        force: bool = False,
        event: Optional[str] = None,
    ) -> None:
        if not self._phase_dataset_enabled:
            return
        if not self._phase_dataset_dir:
            return
        if (not force) and (now - self._phase_dataset_last_ts) < float(self._phase_dataset_every_s):
            return
        if (label == "unknown") and (not self._phase_dataset_allow_unknown):
            return
        if not self._phase_dataset_can_write(label):
            return
        payloads = self._sample_frame_ring_bytes(int(self._phase_dataset_clip_frames))
        if not payloads:
            return
        try:
            import json
            import numpy as np
            from pathlib import Path

            frames = []
            base_size = None
            for payload in payloads:
                arr = self._decode_jpeg_to_array(payload)
                if arr is None:
                    continue
                if base_size is None:
                    base_size = (arr.shape[1], arr.shape[0])
                else:
                    if arr.shape[1] != base_size[0] or arr.shape[0] != base_size[1]:
                        from PIL import Image

                        img = Image.fromarray(arr.astype("uint8"))
                        img = img.resize(base_size)
                        arr = np.asarray(img)
                frames.append(arr)
            if not frames:
                return
            out_dir = Path(self._phase_dataset_dir) / label
            out_dir.mkdir(parents=True, exist_ok=True)
            ts_ms = int(now * 1000)
            out_path = out_dir / f"{self.instance_id}_{ts_ms}.npz"
            scene_hash = None
            try:
                if self._visual_exploration is not None:
                    scene_hash = (self._visual_exploration.metrics() or {}).get("scene_hash")
            except Exception:
                scene_hash = None
            meta = {
                "instance": self.instance_id,
                "scene_hash": str(scene_hash) if scene_hash is not None else None,
                "hud_present": bool(self._hud_present),
                "isPlaying": bool(game_state.get("isPlaying")),
                "phase_label": self._phase_label,
                "phase_conf": self._phase_conf,
                "phase_source": self._phase_source,
                "event": str(event) if event else None,
            }
            np.savez_compressed(out_path, frames=np.stack(frames), label=label, meta=json.dumps(meta))
            self._phase_dataset_counts[label] = self._phase_dataset_counts.get(label, 0) + 1
            self._phase_dataset_last_ts = now
        except Exception:
            self._phase_dataset_last_ts = now

    def _load_phase_model(self):
        if not self._phase_model_enabled:
            return None
        if self._phase_model is not None:
            return self._phase_model
        if not self._phase_model_path:
            return None
        try:
            import torch

            model = torch.jit.load(self._phase_model_path, map_location="cpu")
            model.eval()
            device = self._phase_model_device.lower()
            if device and device != "cpu":
                try:
                    model.to(device)
                except Exception:
                    device = "cpu"
            self._phase_model_device = device
            self._phase_model = model
            return model
        except Exception as exc:
            if not self._phase_model_warned:
                print(f"[phase] WARN: failed to load model {self._phase_model_path}: {exc}", flush=True)
                self._phase_model_warned = True
            self._phase_model_enabled = False
            return None

    def _infer_phase_model(self, payloads: List[bytes]) -> Optional[tuple[str, float]]:
        if not self._phase_model_enabled:
            return None
        model = self._load_phase_model()
        if model is None:
            return None
        if not payloads:
            return None
        try:
            import io
            import numpy as np
            import torch
            from PIL import Image

            size = int(self._phase_input_size or 224)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
            frames = []
            for payload in payloads:
                img = Image.open(io.BytesIO(payload)).convert("RGB")
                if size > 0:
                    img = img.resize((size, size))
                arr = np.asarray(img).astype("float32") / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)
                t = (t - mean) / std
                frames.append(t)
            if not frames:
                return None
            clip = torch.stack(frames, dim=0).unsqueeze(0)
            device = self._phase_model_device.lower() if self._phase_model_device else "cpu"
            if device and device != "cpu":
                clip = clip.to(device)
            with torch.no_grad():
                out = model(clip)
            if isinstance(out, dict):
                if "phase_logits" in out:
                    out = out["phase_logits"]
                elif "logits" in out:
                    out = out["logits"]
            logits = out.squeeze(0)
            if logits.ndim > 1:
                logits = logits[0]
            probs = torch.softmax(logits, dim=0)
            conf, idx = torch.max(probs, dim=0)
            idx_int = int(idx.item())
            label = (
                self._phase_labels[idx_int]
                if self._phase_labels and idx_int < len(self._phase_labels)
                else f"class_{idx_int}"
            )
            return label, float(conf.item())
        except Exception:
            return None

    def _frame_is_black(self, stats: Optional[tuple[float, float, float]]) -> bool:
        if not stats:
            return True
        mean = float(stats[0])
        p99 = float(stats[1])
        mean_s = float(stats[2]) if len(stats) > 2 else 255.0
        dark = (mean < float(self._frame_black_mean)) and (p99 < float(self._frame_black_p99))
        desat = mean_s < float(self._frame_black_sat)
        return dark or desat

    def _load_confirm_template(self) -> Optional["Any"]:
        if self._menu_teacher_confirm_gray is not None:
            return self._menu_teacher_confirm_gray
        if not self._menu_teacher_confirm_enabled:
            return None
        path = self._menu_teacher_confirm_template
        if not path:
            return None
        try:
            import cv2
            import numpy as np

            data = Path(path).read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                raise RuntimeError("template decode failed")
            self._menu_teacher_confirm_gray = img
            return self._menu_teacher_confirm_gray
        except Exception as e:
            if not self._menu_teacher_confirm_warned:
                print(f"[TEACHER] WARN: confirm template unavailable ({e})", flush=True)
                self._menu_teacher_confirm_warned = True
            self._menu_teacher_confirm_gray = None
            return None

    def _match_confirm_template(self, payload: bytes) -> Optional[tuple[float, float, float]]:
        if not payload:
            return None
        tmpl = self._load_confirm_template()
        if tmpl is None:
            return None
        try:
            import cv2
            import numpy as np

            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                return None
            scale = float(self._menu_teacher_confirm_scale or 1.0)
            if scale and scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            th, tw = int(tmpl.shape[0]), int(tmpl.shape[1])
            if img.shape[0] < th or img.shape[1] < tw:
                return None
            res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val < float(self._menu_teacher_confirm_thresh):
                return None
            cx = float(max_loc[0] + tw * 0.5)
            cy = float(max_loc[1] + th * 0.5)
            if scale and scale != 1.0:
                cx /= scale
                cy /= scale
            return float(max_val), cx, cy
        except Exception:
            return None

    def _find_confirm_button_rect(
        self, payload: bytes
    ) -> Optional[tuple[float, float, float, List[tuple[int, int, int, int, float]], tuple[int, int, int, int]]]:
        if not payload:
            return None
        try:
            import cv2
            import numpy as np

            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                return None
            h, w = img.shape[:2]

            def _scan(rx0f: float, rx1f: float, ry0f: float, ry1f: float, *, require_center: bool = False):
                rx0 = int(w * rx0f)
                rx1 = int(w * rx1f)
                ry0 = int(h * ry0f)
                ry1 = int(h * ry1f)
                rx1 = max(rx1, rx0 + 1)
                ry1 = max(ry1, ry0 + 1)
                roi = img[ry0:ry1, rx0:rx1]
                roi_h, roi_w = roi.shape[:2]
                roi_area = float(max(1, roi_h * roi_w))
                edges = cv2.Canny(roi, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidates: List[tuple[int, int, int, int, float]] = []
                for cnt in contours:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if ch <= 0 or cw <= 0:
                        continue
                    area = float(cw * ch)
                    frac = area / roi_area
                    # Confirm buttons are medium sized and wide; exclude huge panels and tiny glyphs.
                    if frac < 0.004 or frac > 0.18:
                        continue
                    aspect = float(cw) / float(ch)
                    if aspect < 1.8 or aspect > 10.0:
                        continue
                    # Exclude extremely thin lines; UI buttons tend to be reasonably tall.
                    if float(ch) < 0.05 * float(roi_h):
                        continue
                    ax = rx0 + x
                    ay = ry0 + y
                    candidates.append((ax, ay, cw, ch, frac))
                if not candidates:
                    return None
                # Prefer lowest button in the region (common for CONFIRM), then larger.
                candidates.sort(key=lambda r: (-r[1], -r[4]))
                best = candidates[0]
                cx = float(best[0] + best[2] * 0.5)
                cy = float(best[1] + best[3] * 0.5)
                return best[4], cx, cy, candidates, (rx0, ry0, rx1, ry1)

            # Prefer the right-side panel region (character/map selection), fallback to centered bottom (warnings).
            res = _scan(0.55, 0.97, 0.42, 0.95)
            if res is None:
                res = _scan(0.30, 0.70, 0.45, 0.95)
            return res
        except Exception:
            return None

    def _find_play_button_rect(
        self, payload: bytes
    ) -> Optional[tuple[float, float, float, List[tuple[int, int, int, int, float]], tuple[int, int, int, int]]]:
        if not payload:
            return None
        try:
            import cv2
            import numpy as np

            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                return None
            h, w = img.shape[:2]

            def _scan(rx0f: float, rx1f: float, ry0f: float, ry1f: float):
                rx0 = int(w * rx0f)
                rx1 = int(w * rx1f)
                ry0 = int(h * ry0f)
                ry1 = int(h * ry1f)
                rx1 = max(rx1, rx0 + 1)
                ry1 = max(ry1, ry0 + 1)
                roi = img[ry0:ry1, rx0:rx1]
                roi_h, roi_w = roi.shape[:2]
                roi_area = float(max(1, roi_h * roi_w))
                edges = cv2.Canny(roi, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidates: List[tuple[int, int, int, int, float]] = []
                for cnt in contours:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if ch <= 0 or cw <= 0:
                        continue
                    area = float(cw * ch)
                    frac = area / roi_area
                    # The main menu has many large panels (title/leaderboard). We're trying
                    # to find the smaller "PLAY" button in the central stack.
                    if frac < 0.008 or frac > 0.16:
                        continue
                    aspect = float(cw) / float(ch)
                    if aspect < 2.0 or aspect > 10.0:
                        continue
                    if float(ch) < 0.05 * float(roi_h):
                        continue
                    ax = rx0 + x
                    ay = ry0 + y
                    if require_center:
                        cx_abs = float(ax) + float(cw) * 0.5
                        if abs(cx_abs - (float(w) * 0.5)) > (float(w) * 0.14):
                            continue
                    candidates.append((ax, ay, cw, ch, frac))
                if not candidates:
                    return None
                # Prefer the highest button within the stack (PLAY is above UNLOCKS/QUESTS/SHOP).
                candidates.sort(key=lambda r: (r[1], -r[4]))
                best = candidates[0]
                cx = float(best[0] + best[2] * 0.5)
                cy = float(best[1] + best[3] * 0.5)
                return best[4], cx, cy, candidates, (rx0, ry0, rx1, ry1)

            # Prefer the centered "stack of buttons" region first; fallback to left column.
            # Start lower to avoid matching the big title header.
            res = _scan(0.28, 0.78, 0.34, 0.86, require_center=True)
            if res is None:
                res = _scan(0.12, 0.62, 0.34, 0.90, require_center=True)
            return res
        except Exception as exc:
            if not self._menu_teacher_play_rect_warned:
                print(f"[TEACHER] WARN: play rect detector unavailable ({exc})", flush=True)
                self._menu_teacher_play_rect_warned = True
            return None

    def _load_play_template(self) -> Optional["Any"]:
        if self._menu_teacher_play_gray is not None:
            return self._menu_teacher_play_gray
        if not self._menu_teacher_play_enabled:
            return None
        path = self._menu_teacher_play_template
        if not path:
            return None
        try:
            import cv2
            import numpy as np

            data = Path(path).read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                raise RuntimeError("template decode failed")
            self._menu_teacher_play_gray = img
            return self._menu_teacher_play_gray
        except Exception as e:
            if not self._menu_teacher_play_warned:
                print(f"[TEACHER] WARN: play template unavailable ({e})", flush=True)
                self._menu_teacher_play_warned = True
            self._menu_teacher_play_gray = None
            return None

    def _match_play_template(self, payload: bytes) -> Optional[tuple[float, float, float]]:
        if not payload:
            return None
        tmpl = self._load_play_template()
        if tmpl is None:
            return None
        try:
            import cv2
            import numpy as np

            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                return None
            scale = float(self._menu_teacher_play_scale or 1.0)
            if scale and scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            th, tw = int(tmpl.shape[0]), int(tmpl.shape[1])
            if img.shape[0] < th or img.shape[1] < tw:
                return None
            res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val < float(self._menu_teacher_play_thresh):
                return None
            cx = float(max_loc[0] + tw * 0.5)
            cy = float(max_loc[1] + th * 0.5)
            if scale and scale != 1.0:
                cx /= scale
                cy /= scale
            return float(max_val), cx, cy
        except Exception:
            return None

    def _push_frame_ring(self, payload: Optional[bytes], *, source: str, ts: float) -> None:
        if not self._frame_ring_enabled or not payload:
            return
        stats = self._frame_stats_from_bytes(payload)
        entry = {
            "ts": float(ts),
            "bytes": payload,
            "source": source,
            "stats": stats,
        }
        self._frame_ring.append(entry)
        if stats and (not self._frame_is_black(stats)):
            self._last_valid_frame = entry

    def _init_input_backend(self) -> None:
        name = self._input_backend_name
        if name not in ("uinput", "xdotool", "xdo", "libxdo"):
            return
        strict_input = os.environ.get("METABONK_REQUIRE_INPUT", "1") in ("1", "true", "True")
        self._input_buttons = self._parse_input_buttons()
        if not self._input_buttons:
            print(f"[worker:{self.instance_id}] input backend enabled but no METABONK_INPUT_BUTTONS set")
        if name == "uinput":
            if UInputBackend is None:
                print(f"[worker:{self.instance_id}] uinput backend unavailable (missing dependency)")
                return
            try:
                key_names = [b["name"] for b in self._input_buttons if b.get("kind") == "key"]
                key_names = list({str(k) for k in key_names if str(k).strip()})
                self._input_backend = UInputBackend(keys=key_names)
                print(f"[worker:{self.instance_id}] uinput backend enabled (focused window receives input)")
            except UInputError as e:
                print(f"[worker:{self.instance_id}] uinput backend failed: {e}")
                self._input_backend = None
            return
        # libxdo backend (direct X11 input)
        if name == "libxdo":
            if LibXDoBackend is None:
                print(f"[worker:{self.instance_id}] libxdo backend unavailable (missing dependency)")
                return
            try:
                wait_s = float(os.environ.get("METABONK_INPUT_DISPLAY_WAIT_S", "5.0"))
            except Exception:
                wait_s = 5.0
            deadline = time.time() + max(0.0, float(wait_s))
            last_err: Optional[Exception] = None
            try:
                display = os.environ.get("METABONK_INPUT_DISPLAY") or self.display
                xauth = os.environ.get("METABONK_INPUT_XAUTHORITY") or os.environ.get("XAUTHORITY")
                window_name = os.environ.get("METABONK_INPUT_XDO_WINDOW")
                window_class = os.environ.get("METABONK_INPUT_XDO_CLASS")
                while True:
                    try:
                        self._input_backend = LibXDoBackend(
                            display=display,
                            xauth=xauth,
                            window_name=window_name,
                            window_class=window_class,
                        )
                        print(f"[worker:{self.instance_id}] libxdo backend enabled (DISPLAY={display})")
                        last_err = None
                        break
                    except LibXDoError as e:
                        last_err = e
                        if time.time() >= deadline:
                            raise
                        time.sleep(0.2)
            except LibXDoError as e:
                if strict_input:
                    raise RuntimeError(
                        f"[worker:{self.instance_id}] libxdo backend failed (strict input): {e}"
                    ) from e
                print(f"[worker:{self.instance_id}] libxdo backend failed: {e}")
                self._input_backend = None
            return
        # xdotool backend (per-display input)
        if XDoToolBackend is None:
            print(f"[worker:{self.instance_id}] xdotool backend unavailable (missing dependency)")
            return
        try:
            display = os.environ.get("METABONK_INPUT_DISPLAY") or self.display
            xauth = os.environ.get("METABONK_INPUT_XAUTHORITY") or os.environ.get("XAUTHORITY")
            window_name = os.environ.get("METABONK_INPUT_XDO_WINDOW")
            window_class = os.environ.get("METABONK_INPUT_XDO_CLASS")
            try:
                wait_s = float(os.environ.get("METABONK_INPUT_DISPLAY_WAIT_S", "5.0"))
            except Exception:
                wait_s = 5.0
            deadline = time.time() + max(0.0, float(wait_s))
            while True:
                try:
                    self._input_backend = XDoToolBackend(
                        display=display,
                        xauth=xauth,
                        window_name=window_name,
                        window_class=window_class,
                    )
                    print(f"[worker:{self.instance_id}] xdotool backend enabled (DISPLAY={display})")
                    break
                except XDoToolError as e:
                    if time.time() >= deadline:
                        raise
                    time.sleep(0.2)
        except XDoToolError as e:
            if strict_input:
                raise RuntimeError(
                    f"[worker:{self.instance_id}] xdotool backend failed (strict input): {e}"
                ) from e
            print(f"[worker:{self.instance_id}] xdotool backend failed: {e}")
            self._input_backend = None

    def _init_metabonk2_controller(self) -> None:
        if not getattr(self, "_use_metabonk2", False) or self._metabonk2_controller is not None:
            return
        try:
            from src.metabonk2.controller import MetaBonk2Controller, MetaBonk2ControllerConfig

            specs = self._input_buttons or self._parse_input_buttons()
            cfg = MetaBonk2ControllerConfig(
                log_reasoning=os.environ.get("METABONK2_LOG", "0") in ("1", "true", "True"),
                override_discrete=os.environ.get("METABONK2_OVERRIDE_DISCRETE", "1") in ("1", "true", "True"),
            )
            self._metabonk2_controller = MetaBonk2Controller(button_specs=specs, cfg=cfg)
        except Exception:
            self._use_metabonk2 = False
            self._metabonk2_controller = None

    def _input_send_actions(self, a_cont: List[float], a_disc: List[int]) -> None:
        if not self._input_backend:
            return
        # Mouse move (relative).
        if a_cont:
            if self._input_mouse_mode == "direct":
                dx = int(round(float(a_cont[0]))) if len(a_cont) > 0 else 0
                dy = int(round(float(a_cont[1]))) if len(a_cont) > 1 else 0
            else:
                dx = int(round(float(a_cont[0]) * self._input_mouse_scale)) if len(a_cont) > 0 else 0
                dy = int(round(float(a_cont[1]) * self._input_mouse_scale)) if len(a_cont) > 1 else 0
            if dx or dy:
                try:
                    self._input_backend.mouse_move(dx, dy)
                except Exception:
                    pass
        # Mouse scroll (optional, uses third continuous dim).
        if len(a_cont) >= 3:
            try:
                steps = int(round(float(a_cont[2]) * self._input_scroll_scale))
                if steps:
                    self._input_backend.mouse_scroll(steps)
            except Exception:
                pass
        # Buttons as per discrete branches (binary per branch).
        desired_keys: set[str] = set()
        desired_mouse: set[str] = set()
        for i, spec in enumerate(self._input_buttons):
            if i >= len(a_disc):
                break
            try:
                pressed = int(a_disc[i]) == 1
            except Exception:
                pressed = False
            if not pressed:
                continue
            if spec.get("kind") == "mouse":
                desired_mouse.add(str(spec.get("button")))
            else:
                desired_keys.add(str(spec.get("name")))
        # Key transitions.
        for k in sorted(desired_keys - self._input_held_keys):
            try:
                self._input_backend.key_down(k)
            except Exception:
                pass
        for k in sorted(self._input_held_keys - desired_keys):
            try:
                self._input_backend.key_up(k)
            except Exception:
                pass
        self._input_held_keys = desired_keys
        # Mouse button transitions.
        for b in sorted(desired_mouse - self._input_held_mouse):
            try:
                self._input_backend.mouse_button(b, True)
            except Exception:
                pass
        for b in sorted(self._input_held_mouse - desired_mouse):
            try:
                self._input_backend.mouse_button(b, False)
            except Exception:
                pass
        self._input_held_mouse = desired_mouse

    def _system2_build_agent_state(
        self,
        *,
        game_state: dict,
        detections: List[dict],
        frame_size: Optional[tuple[int, int]],
        stuck: bool,
        now: float,
    ) -> dict:
        health: Optional[float] = None
        max_health: Optional[float] = None
        health_ratio: Optional[float] = None
        if isinstance(game_state, dict):
            if game_state.get("playerHealth") is not None:
                try:
                    health = float(game_state.get("playerHealth"))
                except Exception:
                    health = None
            if game_state.get("playerMaxHealth") is not None:
                try:
                    max_health = float(game_state.get("playerMaxHealth"))
                except Exception:
                    max_health = None
        if health is not None and max_health is not None:
            try:
                denom = float(max_health)
                if denom > 0:
                    health_ratio = float(health / denom)
            except Exception:
                health_ratio = None
        try:
            gtime = float(game_state.get("gameTime") or float(getattr(self.trainer, "step_count", 0) or 0))
        except Exception:
            gtime = float(getattr(self.trainer, "step_count", 0) or 0)
        w, h = frame_size if frame_size else (0, 0)

        # Provide System2 with a lightweight, vision-derived UI summary so it can
        # reliably choose `interact` targets even when button text is hard to read
        # in the 3x3 temporal grid.
        ui_elements: list[dict[str, float | str]] = []
        try:
            from .perception import parse_detections, build_ui_elements, CLASS_NAMES

            if frame_size is not None:
                dets_parsed = parse_detections(detections or [])
                ui, mask, _ = build_ui_elements(dets_parsed, frame_size=frame_size)
                # Keep only interactables; preserve priority order.
                for i, row in enumerate(ui):
                    if i >= len(mask) or mask[i] != 1:
                        continue
                    try:
                        cls = int(row[4])
                    except Exception:
                        cls = -1
                    name = CLASS_NAMES.get(cls, f"cls_{cls}")
                    ui_elements.append(
                        {
                            "name": str(name),
                            "cx": float(row[0]),
                            "cy": float(row[1]),
                            "w": float(row[2]),
                            "h": float(row[3]),
                            "conf": float(row[5]),
                        }
                    )
                    if len(ui_elements) >= 8:
                        break
        except Exception:
            ui_elements = []

        # If we have no UI detections (common in early boot / UI screens), optionally
        # attach OCR-derived UI candidates from the most recent full-res spectator frame.
        #
        # This remains vision-only (text is read from pixels) while avoiding hardcoded
        # menu scripts in the worker. System2 decides what to click based on the list.
        if (not ui_elements) and bool(stuck):
            try:
                ocr_enabled = str(
                    os.environ.get("METABONK_SYSTEM2_OCR_UI_ELEMENTS", "1" if self._pure_vision_mode else "0")
                    or ""
                ).strip().lower() in ("1", "true", "yes", "on")
            except Exception:
                ocr_enabled = False
            if ocr_enabled:
                try:
                    interval_s = float(os.environ.get("METABONK_SYSTEM2_OCR_UI_INTERVAL_S", "1.0") or 1.0)
                except Exception:
                    interval_s = 1.0
                interval_s = max(0.0, float(interval_s))
                try:
                    last_ocr_ts = float(getattr(self, "_system2_ocr_ui_ts", 0.0) or 0.0)
                except Exception:
                    last_ocr_ts = 0.0
                cached = None
                try:
                    cached = getattr(self, "_system2_ocr_ui_elements", None)
                except Exception:
                    cached = None
                if cached and interval_s > 0.0 and (float(now) - float(last_ocr_ts)) < interval_s:
                    try:
                        ui_elements = list(cached)
                    except Exception:
                        ui_elements = []
                else:
                    try:
                        import numpy as np  # type: ignore
                        from PIL import Image

                        from src.worker.ocr import ocr_boxes

                        frame_hwc = getattr(self, "_system2_last_full_frame_hwc", None)
                        ts_f = float(getattr(self, "_system2_last_full_frame_ts", 0.0) or 0.0)
                        if frame_hwc is not None and ts_f > 0 and (float(now) - ts_f) <= 2.0:
                            arr = np.asarray(frame_hwc)
                            if arr.ndim == 3 and int(arr.shape[2]) >= 3:
                                o_w = int(arr.shape[1])
                                o_h = int(arr.shape[0])
                                if o_w > 0 and o_h > 0:
                                    # OCR is unreliable on tiny UI text. Focus on the bottom portion
                                    # of the frame (where primary buttons often live) and upscale.
                                    y0 = int(max(0, min(o_h - 1, int(round(float(o_h) * 0.35)))))
                                    roi = arr[y0:o_h, :, :3]
                                    roi_img = Image.fromarray(roi.astype("uint8"))
                                    scale = 3
                                    try:
                                        up_w = int(roi_img.size[0] * scale)
                                        up_h = int(roi_img.size[1] * scale)
                                        roi_img = roi_img.resize((up_w, up_h), resample=Image.BICUBIC)
                                    except Exception:
                                        scale = 1
                                    boxes = ocr_boxes(roi_img, min_conf=10, min_len=2, psm=6)
                                    # Prefer boxes lower in the ROI (likely buttons), then confidence.
                                    def _score(b: dict) -> tuple[float, float]:
                                        bbox = b.get("bbox") or (0, 0, 0, 0)
                                        try:
                                            _y2 = float(bbox[3])
                                        except Exception:
                                            _y2 = 0.0
                                        try:
                                            _conf = float(b.get("conf", 0.0) or 0.0)
                                        except Exception:
                                            _conf = 0.0
                                        return (_y2, _conf)

                                    boxes = sorted(boxes, key=_score, reverse=True)
                                    out_boxes: list[dict[str, float | str]] = []
                                    for b in boxes:
                                        txt = " ".join(str(b.get("text") or "").split())
                                        if not txt:
                                            continue
                                        # Keep the System2 prompt small: OCR on splash screens can emit
                                        # long paragraphs, which bloats token counts and can exceed the
                                        # VLM context length. The model still has access to pixels; the
                                        # text field is only a hint.
                                        if len(txt) > 48:
                                            txt = txt[:48]
                                        bbox = b.get("bbox")
                                        if not bbox or len(bbox) != 4:
                                            continue
                                        x1, y1, x2, y2 = bbox
                                        try:
                                            conf = float(b.get("conf", 0.0) or 0.0)
                                        except Exception:
                                            conf = 0.0
                                        # Map ROI (possibly upscaled) coords -> original frame.
                                        denom = float(scale) if scale > 0 else 1.0
                                        ox1 = float(x1) / denom
                                        oy1 = float(y1) / denom + float(y0)
                                        ox2 = float(x2) / denom
                                        oy2 = float(y2) / denom + float(y0)
                                        cx = (ox1 + ox2) * 0.5 / float(o_w)
                                        cy = (oy1 + oy2) * 0.5 / float(o_h)
                                        ww = max(0.0, ox2 - ox1) / float(o_w)
                                        hh = max(0.0, oy2 - oy1) / float(o_h)
                                        if ww <= 0.005 or hh <= 0.005:
                                            continue
                                        cx = max(0.0, min(1.0, float(cx)))
                                        cy = max(0.0, min(1.0, float(cy)))
                                        out_boxes.append(
                                            {
                                                "name": txt,
                                                "cx": float(cx),
                                                "cy": float(cy),
                                                "w": float(ww),
                                                "h": float(hh),
                                                "conf": float(conf) / 100.0,
                                            }
                                        )
                                        if len(out_boxes) >= 12:
                                            break
                                    ui_elements = out_boxes
                                    try:
                                        setattr(self, "_system2_ocr_ui_ts", float(now))
                                        setattr(self, "_system2_ocr_ui_elements", list(ui_elements))
                                    except Exception:
                                        pass
                    except Exception:
                        ui_elements = ui_elements

        # Always provide System2 with at least a coarse, game-agnostic UI candidate list so it
        # can localize an "advance/proceed" click even when detectors/OCR are unavailable.
        if (not ui_elements) and frame_size is not None:
            try:
                from .perception import build_grid_ui_elements

                grid, mask = build_grid_ui_elements(frame_size, max_elements=16, rows=4, cols=4)
                candidates: list[dict[str, float | str]] = []
                for i, row in enumerate(grid):
                    if i >= len(mask) or int(mask[i]) != 1:
                        continue
                    candidates.append(
                        {
                            "name": "",
                            "cx": float(row[0]),
                            "cy": float(row[1]),
                            "w": float(row[2]),
                            "h": float(row[3]),
                            "conf": float(row[5]),
                        }
                    )
                    if len(candidates) >= 12:
                        break
                if candidates:
                    ui_elements = candidates
            except Exception:
                pass

        # Vision-only exploration context (pure signals, no scene labels).
        scene_hash = None
        visual_novelty = None
        scenes_discovered = None
        stuck_score = None
        try:
            ve = getattr(self, "_visual_exploration", None)
            if ve is not None:
                metrics = ve.metrics()
                scene_hash = metrics.get("scene_hash")
                visual_novelty = metrics.get("visual_novelty")
                scenes_discovered = metrics.get("scenes_discovered")
                stuck_score = metrics.get("stuck_score")
        except Exception:
            scene_hash = None
            visual_novelty = None
            scenes_discovered = None
            stuck_score = None

        return {
            "instance_id": self.instance_id,
            "ts": float(now),
            "game_time": gtime,
            "health": float(health) if health is not None else None,
            "max_health": float(max_health) if max_health is not None else None,
            "health_ratio": float(health_ratio) if health_ratio is not None else None,
            "enemies_nearby": int(len(detections or [])),
            "frame_w": int(w),
            "frame_h": int(h),
            "gameplay_started": bool(getattr(self, "_gameplay_started", False)),
            "gameplay_confidence": float(getattr(self, "_gameplay_confidence", 0.0) or 0.0),
            "stuck": bool(stuck),
            "stuck_score": stuck_score,
            "visual_novelty": visual_novelty,
            "scene_hash": scene_hash,
            "scenes_discovered": scenes_discovered,
            "ui_elements": ui_elements,
        }

    def _system2_on_strategy_response(self, response: dict, *, now: float) -> None:
        # Close out previous directive if it was active.
        self._system2_maybe_log_outcome(now=now, reason="replaced", done=False)

        self._system2_active_reward_sum = 0.0
        self._system2_active_started_ts = float(now)
        # New directive: clear "applied" latch so we count at most once.
        try:
            self._system2_active_applied = False
        except Exception:
            pass
        try:
            self._system2_last_response = dict(response or {})
            self._system2_last_response_ts = float(now)
        except Exception:
            self._system2_last_response = None
            self._system2_last_response_ts = 0.0
        # Count responses for external validation harnesses (even if the directive is later ignored).
        try:
            self._vlm_hints_used = int(getattr(self, "_vlm_hints_used", 0) or 0) + 1
        except Exception:
            self._vlm_hints_used = 1
        try:
            goal = str((response or {}).get("goal") or "").strip()
            reasoning = str((response or {}).get("reasoning") or "").strip()
            conf = (response or {}).get("confidence")
            conf_f = float(conf) if isinstance(conf, (int, float)) else None
            if reasoning:
                msg = f"{goal}: {reasoning}" if goal else reasoning
            else:
                msg = goal or "directive"
            if conf_f is not None:
                msg = f"{msg} (conf={conf_f:.2f})"
            self._system2_reasoning_trace.append(msg[:300])
        except Exception:
            pass

        if self._system2_rl_logger is not None and self._system2_last_request is not None:
            try:
                decision_id = self._system2_rl_logger.log_decision(
                    agent_id=self.instance_id,
                    request_data=self._system2_last_request,
                    response_data=response,
                )
                self._system2_active_decision_id = str(decision_id)
            except Exception:
                self._system2_active_decision_id = None
        else:
            self._system2_active_decision_id = None

    def _system2_maybe_log_outcome(self, *, now: float, reason: str, done: bool) -> None:
        if self._system2_rl_logger is None:
            return
        if self._system2_active_decision_id is None:
            return

        outcome = {
            "reason": str(reason),
            "done": bool(done),
            "reward_sum": float(self._system2_active_reward_sum),
            "duration_s": float(max(0.0, float(now) - float(self._system2_active_started_ts or now))),
        }
        try:
            self._system2_rl_logger.log_outcome(
                agent_id=self.instance_id,
                decision_id=str(self._system2_active_decision_id),
                outcome=outcome,
            )
        except Exception:
            pass

        self._system2_active_decision_id = None
        self._system2_active_reward_sum = 0.0
        self._system2_active_started_ts = 0.0

    def _system2_apply_continuous_directive(
        self,
        *,
        a_cont: List[float],
        directive: dict,
        frame_size: tuple[int, int],
    ) -> List[float]:
        try:
            data = directive.get("directive") or {}
            action = str(data.get("action") or "").strip().lower()
            target = data.get("target")
            if not (isinstance(target, (list, tuple)) and len(target) >= 2):
                return a_cont
            tx = float(target[0])
            ty = float(target[1])
        except Exception:
            return a_cont

        w, h = frame_size
        if 0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0 and w > 0 and h > 0:
            tx *= float(w)
            ty *= float(h)

        cx = float(w) / 2.0
        cy = float(h) / 2.0
        dx = float(tx - cx)
        dy = float(ty - cy)
        denom = max(1.0, max(abs(dx), abs(dy)))
        vx = dx / denom
        vy = dy / denom

        if action == "retreat":
            vx = -vx
            vy = -vy

        base = [float(x) for x in (a_cont or [])]
        while len(base) < 2:
            base.append(0.0)
        alpha = float(self._system2_blend)
        base[0] = (1.0 - alpha) * base[0] + alpha * float(vx)
        base[1] = (1.0 - alpha) * base[1] + alpha * float(vy)
        return base

    def _bridge_read_frame(self) -> Optional["GameFrame"]:
        if not self.bridge or not self._bridge_loop:
            return None
        try:
            return self._bridge_loop.run_until_complete(self.bridge.read_frame())
        except Exception:
            return None

    def _bridge_send_actions(
        self,
        a_cont: List[float],
        a_disc: List[int],
        forced_ui_click: Optional[tuple[int, int]],
        ui_elements: Optional[List[List[float]]],
        action_mask: Optional[List[int]],
        frame_size: Optional[tuple[int, int]],
    ):
        if not self.bridge or not self._bridge_loop:
            return

        # Forced click (System2/UI bootstrap) takes precedence.
        if forced_ui_click is not None and frame_size is not None:
            try:
                w, h = frame_size
                x = max(0, min(int(w - 1), int(forced_ui_click[0]))) if w > 0 else int(forced_ui_click[0])
                y = max(0, min(int(h - 1), int(forced_ui_click[1]))) if h > 0 else int(forced_ui_click[1])
                self._bridge_loop.run_until_complete(self.bridge.send_mouse_click(int(x), int(y), 0))
                return
            except Exception:
                pass

        # Discrete UI click on rising edge.
        if (
            a_disc
            and ui_elements is not None
            and action_mask is not None
            and frame_size is not None
        ):
            try:
                idx = int(a_disc[0])
                if idx != self._last_disc_action and 0 <= idx < len(ui_elements) and action_mask[idx] == 1:
                    w, h = frame_size
                    cx, cy = ui_elements[idx][0], ui_elements[idx][1]
                    x = int(cx * w)
                    y = int(cy * h)
                    self._bridge_loop.run_until_complete(self.bridge.send_mouse_click(x, y, 0))
                self._last_disc_action = idx
            except Exception:
                pass

    def _heartbeat_loop(self):
        next_retry = 0.0
        while not self._stop.is_set():
            now = time.time()

            # Synthetic Eye control: drop buffered rollout on compositor/XWayland resets to avoid
            # poisoning training with frozen frames.
            try:
                if self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf") and hasattr(self.stream, "pop_reset"):
                    r = self.stream.pop_reset()  # type: ignore[attr-defined]
                else:
                    r = None
            except Exception:
                r = None
            if r is not None:
                try:
                    reason = int(getattr(r, "reason", 0) or 0)
                except Exception:
                    reason = 0
                print(f"[worker] synthetic_eye reset detected (reason={reason}); dropping buffered rollout", flush=True)
                try:
                    self.rollout.reset()
                except Exception:
                    # Fall back to flush() semantics (does not reset episode counters).
                    try:
                        _ = self.rollout.flush()
                    except Exception:
                        pass
                # Best-effort: clear any pending CUDA fence cleanup so we don't accumulate stale
                # external semaphore handles across compositor resets.
                try:
                    if getattr(self, "_synthetic_eye_ingestor", None) is not None:
                        self._synthetic_eye_ingestor.on_reset()  # type: ignore[attr-defined]
                except Exception:
                    pass
                self._episode_start_ts = now
                self._episode_idx += 1
                # Only restart the worker for "hard" resets (e.g. XWayland restart). For output resets
                # (reason=2) we must keep draining and signaling release fences; restarting here can
                # strand producer waits and trigger a deadlock loop.
                if self._synthetic_eye_reset_restart and reason in (1,):
                    print(
                        f"[worker] synthetic_eye reset (reason={reason}) -> restarting worker to recover vision",
                        flush=True,
                    )
                    try:
                        self.launcher.shutdown()
                    except Exception:
                        pass
                    os._exit(3)

            # Synthetic Eye stall detector: if frames stop advancing, restart worker (GPU-only policy).
            if (
                self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf")
                and float(self._synthetic_eye_stall_restart_s or 0.0) > 0.0
                and (now - float(self._boot_ts or now)) > 30.0
            ):
                last_ts = float(getattr(self, "_latest_frame_ts", 0.0) or 0.0)
                if last_ts > 0 and (now - last_ts) > float(self._synthetic_eye_stall_restart_s):
                    print(
                        f"[worker] synthetic_eye stalled (no frames for {now - last_ts:.1f}s) -> restarting worker",
                        flush=True,
                    )
                    try:
                        self.launcher.shutdown()
                    except Exception:
                        pass
                    os._exit(4)
            if (now - float(self._pipewire_health_ts or 0.0)) >= 5.0:
                try:
                    self._pipewire_daemon_ok = bool(GameLauncher.pipewire_daemon_ok(timeout_s=0.4))
                except Exception:
                    self._pipewire_daemon_ok = None
                try:
                    self._pipewire_session_ok = bool(GameLauncher.pipewire_session_ok(timeout_s=0.4))
                except Exception:
                    self._pipewire_session_ok = None
                self._pipewire_health_ts = now
            launcher_proc = getattr(self.launcher, "proc", None)
            launcher_pid = None
            launcher_alive = False
            try:
                if launcher_proc is not None:
                    launcher_pid = int(getattr(launcher_proc, "pid", 0) or 0) or None
                    launcher_alive = getattr(launcher_proc, "poll", lambda: 0)() is None
            except Exception:
                launcher_pid = None
                launcher_alive = False
            if (
                self._game_restart_enabled
                and self._game_restart_possible
                and not launcher_alive
                and not self._game_restart_failed
            ):
                try:
                    max_restarts = int(os.environ.get("METABONK_GAME_RESTART_MAX", "3"))
                except Exception:
                    max_restarts = 3
                try:
                    backoff_s = float(os.environ.get("METABONK_GAME_RESTART_BACKOFF_S", "5.0"))
                except Exception:
                    backoff_s = 5.0
                if max_restarts < 0:
                    max_restarts = 0
                if backoff_s < 0.5:
                    backoff_s = 0.5
                if self._game_restart_count >= max_restarts:
                    self._game_restart_failed = True
                elif (now - self._last_game_restart_ts) >= backoff_s:
                    self._last_game_restart_ts = now
                    self._game_restart_count += 1
                    try:
                        self.launcher.launch()
                        self._bind_input_window()
                        # Force PipeWire target rediscovery after game restart.
                        try:
                            os.environ.pop("PIPEWIRE_NODE", None)
                            self._pipewire_node_ok = False
                            self._pipewire_node = None
                        except Exception:
                            pass
                    except Exception:
                        self._game_restart_failed = True
            # If we get stuck on a loading screen for too long, terminate the game so the
            # normal restart loop can relaunch it and we can harvest another transition.
            try:
                stall_s = float(self._phase_dataset_loading_restart_s or 0.0)
            except Exception:
                stall_s = 0.0
            if (
                stall_s > 0.0
                and launcher_alive
                and self._phase_dataset_loading_active
                and float(self._phase_dataset_loading_since or 0.0) > 0.0
                and (now - float(self._phase_dataset_loading_since)) >= float(stall_s)
            ):
                self._phase_dataset_loading_active = False
                self._phase_dataset_loading_since = 0.0
                self._phase_dataset_loading_deadline = 0.0
                self._terminate_launcher_proc(reason="loading_stall")
            # Opportunistically refresh PipeWire selection; gamescope may initially expose only
            # `gamescope:capture_*` then later publish a concrete node id in logs.
            try:
                cur = os.environ.get("PIPEWIRE_NODE") or ""
                if (
                    isinstance(cur, str)
                    and cur.startswith("gamescope:capture_")
                    and (now - float(getattr(self, "_last_pipewire_refresh_ts", 0.0) or 0.0)) >= 2.0
                ):
                    self._last_pipewire_refresh_ts = now
                    try:
                        self._ensure_pipewire_node()
                    except Exception:
                        pass
            except Exception:
                pass
            # In strict GPU-stream mode, keep retrying discovery/streamer startup.
            if (
                self._stream_enabled
                and not getattr(self, "_capture_disabled", False)
                and self._require_pipewire_stream
                and self.streamer is None
                and now >= next_retry
            ):
                try:
                    self._ensure_streamer()
                except Exception:
                    pass
                next_retry = now + 2.0
            try:
                last = float(getattr(self, "_latest_frame_ts", 0.0) or 0.0)
            except Exception:
                last = 0.0
            try:
                enc_last = float(getattr(self.streamer, "last_chunk_ts", 0.0) or 0.0) if self.streamer else 0.0
            except Exception:
                enc_last = 0.0
            try:
                active_clients = int(self.streamer.active_clients()) if self.streamer and hasattr(self.streamer, "active_clients") else 0
            except Exception:
                active_clients = 0
            try:
                max_clients = int(self.streamer.max_clients()) if self.streamer and hasattr(self.streamer, "max_clients") else int(os.environ.get("METABONK_STREAM_MAX_CLIENTS", "1"))
            except Exception:
                max_clients = 1
            ok_ttl = float(os.environ.get("METABONK_STREAM_OK_TTL_S", "10.0"))
            stream_ok_age = bool((last > 0 and (now - last) <= ok_ttl) or (enc_last > 0 and (now - enc_last) <= ok_ttl))
            last_any = max(last, enc_last)
            # Optional stream-quality watchdog (uses cached JPEG, so no extra capture cost).
            try:
                black_var = float(os.environ.get("METABONK_STREAM_BLACK_VAR", "1.0"))
            except Exception:
                black_var = 1.0
            try:
                black_s = float(os.environ.get("METABONK_STREAM_BLACK_S", "8.0"))
            except Exception:
                black_s = 8.0
            try:
                black_check_s = float(os.environ.get("METABONK_STREAM_BLACKCHECK_S", "5.0"))
            except Exception:
                black_check_s = 5.0
            try:
                frozen_diff = float(os.environ.get("METABONK_STREAM_FROZEN_DIFF", "1.0"))
            except Exception:
                frozen_diff = 1.0
            try:
                frozen_s = float(os.environ.get("METABONK_STREAM_FROZEN_S", "8.0"))
            except Exception:
                frozen_s = 8.0
            try:
                frozen_check_s = float(os.environ.get("METABONK_STREAM_FROZENCHECK_S", str(black_check_s)))
            except Exception:
                frozen_check_s = black_check_s
            quality_intervals = []
            if black_var > 0 and black_s > 0 and black_check_s > 0:
                quality_intervals.append(float(black_check_s))
            if frozen_diff > 0 and frozen_s > 0 and frozen_check_s > 0:
                quality_intervals.append(float(frozen_check_s))
            quality_check_s = min(quality_intervals) if quality_intervals else 0.0
            if quality_check_s > 0 and (now - self._last_frame_var_ts) >= quality_check_s:
                self._last_frame_var_ts = now
                self._stream_health_checks += 1
                thumb = None
                var = None
                diff = None

                # Prefer a GPU-derived luma thumbnail when possible (works in strict zero-copy).
                try:
                    import torch  # type: ignore

                    pix, _fid, _ts, _src = self._get_latest_pixel_obs()
                    if isinstance(pix, torch.Tensor) and bool(getattr(pix, "is_cuda", False)) and pix.ndim == 3:
                        t = pix
                        if int(t.shape[0]) >= 3:
                            r = t[0].to(dtype=torch.float32)
                            g = t[1].to(dtype=torch.float32)
                            b = t[2].to(dtype=torch.float32)
                            luma_t = (0.299 * r) + (0.587 * g) + (0.114 * b)
                        else:
                            luma_t = t[0].to(dtype=torch.float32)
                        prev_t = self._last_frame_luma if isinstance(self._last_frame_luma, torch.Tensor) else None
                        try:
                            var = float(luma_t.var(unbiased=False).item())
                        except Exception:
                            var = None
                        if prev_t is not None and getattr(prev_t, "shape", None) == getattr(luma_t, "shape", None):
                            try:
                                diff = float((luma_t - prev_t).abs().mean().item())
                            except Exception:
                                diff = None
                        self._last_frame_luma = luma_t
                        thumb = "gpu"
                except Exception:
                    thumb = None

                # Fallback to cached JPEG thumbnail (best-effort).
                if thumb is None:
                    thumb = jpeg_luma_thumbnail(self._latest_jpeg_bytes or b"")
                    var = float(thumb.var()) if thumb is not None else None
                    diff = luma_mean_abs_diff(thumb, self._last_frame_luma) if thumb is not None else None
                    self._last_frame_luma = thumb

                self._last_frame_var = var
                self._last_frame_luma_diff = diff

                if black_var > 0 and black_s > 0 and var is not None and var <= black_var:
                    if self._black_frame_since <= 0:
                        self._black_frame_since = now
                else:
                    self._black_frame_since = 0.0

                if frozen_diff > 0 and frozen_s > 0 and diff is not None and diff <= frozen_diff:
                    if self._frozen_frame_since <= 0:
                        self._frozen_frame_since = now
                else:
                    self._frozen_frame_since = 0.0

            quality_ok = bool(self._black_frame_since <= 0 and self._frozen_frame_since <= 0)
            stream_ok = bool(stream_ok_age and quality_ok)
            if stream_ok:
                self._last_stream_ok_ts = now
            heal_s = float(os.environ.get("METABONK_STREAM_SELF_HEAL_S", "20.0"))
            if (
                heal_s > 0
                and self._stream_enabled
                and self.streamer is not None
            ):
                err_ts = float(getattr(self.streamer, "last_error_ts", 0.0) or 0.0)
                should_heal = False
                reason = ""
                if err_ts and (now - err_ts) >= heal_s:
                    reason = str(getattr(self.streamer, "last_error", "") or "stream error")
                    should_heal = True
                elif last_any > 0 and (now - last_any) >= heal_s and (now - self._last_stream_ok_ts) >= heal_s:
                    reason = "stream stale (no frames)"
                    should_heal = True
                elif (
                    self._black_frame_since > 0
                    and (now - self._black_frame_since) >= black_s
                    and (now - self._last_stream_ok_ts) >= black_s
                ):
                    reason = f"stream black frames (var={self._last_frame_var})"
                    should_heal = True
                elif (
                    frozen_diff > 0
                    and frozen_s > 0
                    and self._frozen_frame_since > 0
                    and (now - self._frozen_frame_since) >= frozen_s
                    and (now - self._last_stream_ok_ts) >= frozen_s
                ):
                    reason = f"stream frozen frames (diff={self._last_frame_luma_diff})"
                    should_heal = True
                # Avoid tearing down/recreating the streamer while multiple clients are actively consuming it.
                # In go2rtc FIFO mode there is typically a persistent reader; allow healing with a single
                # client so frozen streams can recover automatically.
                max_clients_for_heal = 0
                try:
                    raw_max = str(os.environ.get("METABONK_STREAM_SELF_HEAL_MAX_ACTIVE_CLIENTS", "") or "").strip()
                    if raw_max:
                        max_clients_for_heal = int(raw_max)
                    else:
                        max_clients_for_heal = 1 if bool(getattr(self, "_fifo_stream_enabled", False)) else 0
                except Exception:
                    max_clients_for_heal = 1 if bool(getattr(self, "_fifo_stream_enabled", False)) else 0

                if should_heal and active_clients <= max_clients_for_heal and (now - self._last_stream_heal_ts) >= heal_s:
                    self._last_stream_heal_ts = now
                    self._stream_heals += 1
                    self._stream_error = reason
                    self.streamer = None
                    try:
                        if self._require_pipewire_stream:
                            # Force PipeWire target rediscovery on heal; gamescope often recreates
                            # nodes/ports when it stalls ("out of buffers") or restarts internally.
                            os.environ.pop("PIPEWIRE_NODE", None)
                            self._pipewire_node_ok = False
                            self._pipewire_node = None
                        self._ensure_streamer()
                    except Exception:
                        pass
            worker_device = str(getattr(self.trainer, "device", "") or os.environ.get("METABONK_WORKER_DEVICE") or os.environ.get("METABONK_DEVICE") or "")
            vision_device = str(os.environ.get("METABONK_VISION_DEVICE", "") or "")
            learned_reward_device = str(os.environ.get("METABONK_LEARNED_REWARD_DEVICE", "") or "")
            reward_device = str(os.environ.get("METABONK_REWARD_DEVICE", "") or "")
            stream_fps = None
            stream_keyframe_ts = None
            stream_keyframe_count = None
            if self.streamer is not None:
                try:
                    stream_fps = float(getattr(self.streamer, "stream_fps", None)) if getattr(self.streamer, "stream_fps", None) is not None else None
                except Exception:
                    stream_fps = None
                try:
                    kts = float(getattr(self.streamer, "last_keyframe_ts", 0.0) or 0.0)
                    stream_keyframe_ts = kts if kts > 0 else None
                except Exception:
                    stream_keyframe_ts = None
                try:
                    kc = int(getattr(self.streamer, "keyframe_count", 0) or 0)
                    stream_keyframe_count = kc if kc > 0 else None
                except Exception:
                    stream_keyframe_count = None
            action_entropy = None
            try:
                action_entropy = float(getattr(self.trainer, "last_entropy", None))
            except Exception:
                action_entropy = None
            bonk_confidence = None
            if action_entropy is not None:
                try:
                    entropy_max = float(os.environ.get("METABONK_ENTROPY_MAX", "3.0"))
                except Exception:
                    entropy_max = 3.0
                if entropy_max <= 0:
                    entropy_max = 3.0
                bonk_confidence = max(0.0, min(1.0, 1.0 - float(action_entropy) / entropy_max))
            step_now = int(getattr(self.trainer, "step_count", 0) or 0)
            if step_now != self._last_step_seen:
                self._last_step_seen = step_now
                self._last_step_ts = now
            step_age_s = (now - self._last_step_ts) if self._last_step_ts > 0 else None
            stuck_score = None
            try:
                ve = getattr(self, "_visual_exploration", None)
                if ve is not None and hasattr(ve, "stuck_score"):
                    stuck_score = float(ve.stuck_score())
            except Exception:
                stuck_score = None

            # Memory telemetry (best-effort; do not fail the heartbeat on /proc parsing issues).
            def _kb_from_status(pid: int) -> tuple[Optional[float], Optional[float]]:
                try:
                    p = f"/proc/{int(pid)}/status"
                    rss_kb: Optional[float] = None
                    vms_kb: Optional[float] = None
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.startswith("VmRSS:"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    rss_kb = float(parts[1])
                            elif line.startswith("VmSize:"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    vms_kb = float(parts[1])
                            if rss_kb is not None and vms_kb is not None:
                                break
                    return rss_kb, vms_kb
                except Exception:
                    return None, None

            def _kb_from_meminfo() -> tuple[Optional[float], Optional[float], Optional[float]]:
                try:
                    mem_total: Optional[float] = None
                    mem_avail: Optional[float] = None
                    swap_free: Optional[float] = None
                    with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    mem_total = float(parts[1])
                            elif line.startswith("MemAvailable:"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    mem_avail = float(parts[1])
                            elif line.startswith("SwapFree:"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    swap_free = float(parts[1])
                            if mem_total is not None and mem_avail is not None and swap_free is not None:
                                break
                    return mem_total, mem_avail, swap_free
                except Exception:
                    return None, None, None

            def _mb(kb: Optional[float]) -> Optional[float]:
                if kb is None:
                    return None
                try:
                    mb = float(kb) / 1024.0
                    if not _math.isfinite(mb):
                        return None
                    return mb
                except Exception:
                    return None

            rss_kb, vms_kb = _kb_from_status(int(os.getpid()))
            launcher_rss_kb, launcher_vms_kb = _kb_from_status(int(launcher_pid or 0)) if launcher_pid else (None, None)
            mem_total_kb, mem_avail_kb, swap_free_kb = _kb_from_meminfo()
            hb = Heartbeat(
                schema_version=int(HEARTBEAT_SCHEMA_VERSION),
                run_id=os.environ.get("METABONK_RUN_ID"),
                instance_id=self.instance_id,
                port=int(self.port),
                policy_name=self.policy_name,
                policy_version=self._policy_version,
                step=step_now,
                reward=self.trainer.last_reward,
                steam_score=self.trainer.last_reward,
                episode_idx=int(self._episode_idx),
                episode_t=float(time.time() - float(self._episode_start_ts)),
                status="no_pipewire"
                if (self._require_pipewire_stream and self._stream_enabled and not self._pipewire_node_ok)
                else "running",
                stream_url=self.stream_url() or None,
                stream_type="mp4"
                if (self.streamer is not None and not getattr(self, "_capture_disabled", False))
                else "none",
                stream_ok=stream_ok,
                stream_last_frame_ts=last_any if last_any > 0 else None,
                stream_error=self._stream_error,
                streamer_last_error=getattr(self.streamer, "last_error", None) if self.streamer else None,
                stream_backend=getattr(self.streamer, "backend", None) if self.streamer else None,
                nvenc_sessions_used=getattr(self.streamer, "nvenc_sessions_used_last", None) if self.streamer else None,
                stream_active_clients=active_clients,
                stream_max_clients=max_clients,
                stream_fps=stream_fps,
                stream_keyframe_ts=stream_keyframe_ts,
                stream_keyframe_count=stream_keyframe_count,
                stream_frame_var=self._last_frame_var,
                stream_black_since_s=(now - self._black_frame_since) if self._black_frame_since > 0 else None,
                stream_frame_diff=self._last_frame_luma_diff,
                stream_frozen_since_s=(now - self._frozen_frame_since) if self._frozen_frame_since > 0 else None,
                fifo_stream_enabled=bool(getattr(self, "_fifo_stream_enabled", False)),
                fifo_stream_path=getattr(self, "_fifo_stream_path", None),
                fifo_stream_last_error=(self._fifo_publisher.last_error() if self._fifo_publisher is not None else None),
                go2rtc_stream_name=self.instance_id if bool(getattr(self, "_fifo_stream_enabled", False)) else None,
                # Only advertise go2rtc when public streaming is enabled; otherwise the UI should
                # use the per-worker MP4 endpoint directly.
                go2rtc_base_url=os.environ.get("METABONK_GO2RTC_URL")
                if (
                    bool(getattr(self, "_fifo_stream_enabled", False))
                    and str(os.environ.get("METABONK_ENABLE_PUBLIC_STREAM", "0") or "0").strip().lower()
                    in ("1", "true", "yes", "on")
                )
                else None,
                pipewire_node_ok=bool(getattr(self, "_pipewire_node_ok", False)),
                pipewire_ok=self._pipewire_daemon_ok,
                pipewire_session_ok=self._pipewire_session_ok,
                stream_require_pipewire=bool(getattr(self, "_require_pipewire_stream", False)),
                frame_source=self._frame_source,
                worker_device=worker_device or None,
                vision_device=vision_device or None,
                learned_reward_device=learned_reward_device or None,
                reward_device=reward_device or None,
                worker_pid=os.getpid(),
                launcher_pid=launcher_pid,
                launcher_alive=launcher_alive,
                game_restart_count=self._game_restart_count,
                game_restart_failed=self._game_restart_failed,
                step_age_s=step_age_s,
                control_url=self.control_url(),
                obs_fps=self._obs_fps,
                act_hz=self._act_hz,
                gameplay_started=bool(getattr(self, "_gameplay_started", False)),
                actions_total=int(getattr(self, "_actions_total", 0) or 0),
                vlm_hints_used=int(getattr(self, "_vlm_hints_used", 0) or 0),
                vlm_hints_applied=int(getattr(self, "_vlm_hints_applied", 0) or 0),
                action_entropy=action_entropy,
                bonk_confidence=bonk_confidence,
                stuck_score=stuck_score,
                rss_mb=_mb(rss_kb),
                vms_mb=_mb(vms_kb),
                launcher_rss_mb=_mb(launcher_rss_kb),
                launcher_vms_mb=_mb(launcher_vms_kb),
                mem_total_mb=_mb(mem_total_kb),
                mem_available_mb=_mb(mem_avail_kb),
                swap_free_mb=_mb(swap_free_kb),
            )
            if requests:
                try:
                    requests.post(f"{self.orch_url}/heartbeat", json=hb.model_dump(), timeout=1.0)
                except Exception:
                    pass
            hb_s = float(os.environ.get("METABONK_HEARTBEAT_S", "5.0"))
            self._stop.wait(max(0.2, hb_s))

    def _rollout_loop(self):
        # Register and warm weights.
        caps: dict = {}
        reg_obs_dim: Optional[int] = self.trainer.obs_dim
        if bool(getattr(self, "_pixel_obs_enabled", False)):
            reg_obs_dim = None
            caps.update(
                {
                    "obs_kind": "pixels",
                    "obs_width": int(getattr(self, "_pixel_obs_w", 0) or 0),
                    "obs_height": int(getattr(self, "_pixel_obs_h", 0) or 0),
                    "obs_channels": 3,
                    "obs_dtype": "uint8",
                }
            )
        self.learner.register(self.instance_id, self.policy_name, obs_dim=reg_obs_dim, capabilities=caps)
        last_pull = 0.0
        loop_t0 = time.time()
        learned_reward_grace_s = float(os.environ.get("METABONK_LEARNED_REWARD_GRACE_S", "5.0"))

        # Initial config from orchestrator (PBT assignment / hparams).
        if requests:
            try:
                r = requests.get(f"{self.orch_url}/config/{self.instance_id}", timeout=2.0)
                if r.ok:
                    cfg = InstanceConfig(**r.json())
                    self.set_config(cfg)
            except Exception:
                pass

        # Optional bridge init in this thread.
        if self._use_bridge:
            self._init_unity_bridge()
        if self._use_bonklink and self._bonklink is None:
            try:
                from src.bridge.bonklink_client import BonkLinkClient

                self._bonklink = BonkLinkClient(
                    host=os.environ.get("METABONK_BONKLINK_HOST", "127.0.0.1"),
                    port=int(os.environ.get("METABONK_BONKLINK_PORT", "5560")),
                    use_named_pipe=os.environ.get("METABONK_BONKLINK_USE_PIPE", "0")
                    in ("1", "true", "True"),
                    pipe_name=os.environ.get("METABONK_BONKLINK_PIPE_NAME", "BonkLink"),
                )
                if not self._bonklink.connect(timeout_s=2.0):
                    self._bonklink = None
                    self._bonklink_last_attempt = time.time()
            except Exception:
                self._bonklink = None
                self._bonklink_last_attempt = time.time()
        if self._use_research_shm and self._research_shm is None:
            try:
                from src.bridge.research_shm import ResearchSharedMemoryClient

                self._research_shm = ResearchSharedMemoryClient(self.instance_id)
                if not self._research_shm.open():
                    self._research_shm = None
                    self._use_research_shm = False
            except Exception:
                self._research_shm = None
                self._use_research_shm = False

        while not self._stop.is_set():
            now = time.time()
            if self._use_bonklink and self._bonklink is None:
                if (now - float(getattr(self, "_bonklink_last_attempt", 0.0))) >= float(
                    getattr(self, "_bonklink_retry_s", 5.0)
                ):
                    try:
                        from src.bridge.bonklink_client import BonkLinkClient

                        self._bonklink_last_attempt = now
                        self._bonklink = BonkLinkClient(
                            host=os.environ.get("METABONK_BONKLINK_HOST", "127.0.0.1"),
                            port=int(os.environ.get("METABONK_BONKLINK_PORT", "5560")),
                            use_named_pipe=os.environ.get("METABONK_BONKLINK_USE_PIPE", "0")
                            in ("1", "true", "True"),
                            pipe_name=os.environ.get("METABONK_BONKLINK_PIPE_NAME", "BonkLink"),
                        )
                        if not self._bonklink.connect(timeout_s=2.0):
                            self._bonklink = None
                    except Exception:
                        self._bonklink = None
            if now - last_pull > float(getattr(self, "_config_poll_s", 30.0)):
                # Refresh config/hparams from orchestrator.
                if requests:
                    try:
                        r = requests.get(f"{self.orch_url}/config/{self.instance_id}", timeout=1.0)
                        if r.ok:
                            cfg = InstanceConfig(**r.json())
                            self.set_config(cfg)
                    except Exception:
                        pass
                last_pull = now
                self._last_policy_fetch_ts = now
                since = int(self._policy_version) if self._policy_version is not None else -1
                w, ver = self.learner.get_weights_with_version(self.policy_name, since_version=since)
                if w:
                    try:
                        self.trainer.set_weights_b64(w)
                        if ver is not None:
                            self._policy_version = int(ver)
                        self._last_policy_update_ts = time.time()
                    except Exception:
                        pass
                policy_stale_s = float(os.environ.get("METABONK_POLICY_STALE_S", "900.0"))
                if policy_stale_s > 0 and self._last_policy_fetch_ts > 0:
                    base_ts = self._last_policy_update_ts or self._last_policy_fetch_ts
                    if (now - base_ts) >= policy_stale_s:
                        warn_s = float(os.environ.get("METABONK_POLICY_STALE_WARN_S", "120.0"))
                        if (now - self._last_policy_warn_ts) >= warn_s:
                            age_s = int(max(0.0, now - base_ts))
                            print(f"[worker] WARN: policy {self.policy_name} weights stale for {age_s}s; forcing refresh")
                            self._last_policy_warn_ts = now
                        w2, ver2 = self.learner.get_weights_with_version(self.policy_name, since_version=-1)
                        if w2:
                            try:
                                self.trainer.set_weights_b64(w2)
                                if ver2 is not None:
                                    self._policy_version = int(ver2)
                                self._last_policy_update_ts = time.time()
                            except Exception:
                                pass
                # Optional CPQE motor weights for SIMA2 controller.
                if (
                    self._use_sima2
                    and self._sima2_controller is not None
                    and requests
                    and os.environ.get("METABONK_SIMA2_LOAD_CPQE", "0")
                    in ("1", "true", "True")
                ):
                    try:
                        import base64
                        import io
                        import torch

                        r2 = requests.get(
                            f"{self.learner_url}/get_cpqe_weights",
                            params={"policy_name": self.policy_name},
                            timeout=1.0,
                        )
                        if r2.ok:
                            wb = r2.json().get("weights_b64")
                            if wb and getattr(self._sima2_controller, "motor", None) is not None:
                                raw = base64.b64decode(wb)
                                state = torch.load(io.BytesIO(raw), map_location="cpu")
                                try:
                                    self._sima2_controller.motor.load_state_dict(state)
                                    self._sima2_controller.motor.eval()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                # Pull curriculum from orchestrator if available.
                if requests:
                    try:
                        r = requests.get(f"{self.orch_url}/curriculum", timeout=1.0)
                        if r.ok:
                            self.curriculum = CurriculumConfig(**r.json())
                    except Exception:
                        pass
                last_pull = now

            detections: List[dict] = []
            vision_metrics: dict = {}
            frame_size: Optional[tuple[int, int]] = None
            game_state: dict = {}
            reward_from_game: Optional[float] = None
            done_from_game: bool = False
            used_pixels: bool = False
            latest_image_bytes: Optional[bytes] = None
            latest_image_source: Optional[str] = None
            reward_frame_hwc = None
            forced_ui_click: Optional[tuple[int, int]] = None
            suppress_policy_clicks: bool = False
            sima2_action: Optional[List[float]] = None

            # ResearchPlugin shared memory path (highest priority if enabled).
            if self._use_research_shm and self._research_shm is not None:
                try:
                    obs_pack = self._research_shm.read_observation(timeout_ms=16)
                except Exception:
                    obs_pack = None
                if obs_pack is not None:
                    pixels, header = obs_pack
                    used_pixels = True
                    reward_from_game = float(header.reward)
                    done_from_game = bool(header.done) or header.flag == 4
                    client_w = int(getattr(self._research_shm, "obs_width", 84) or 84)
                    client_h = int(getattr(self._research_shm, "obs_height", 84) or 84)
                    frame_w = int(header.obs_width or client_w)
                    frame_h = int(header.obs_height or client_h)
                    if frame_w * frame_h * 3 != len(pixels):
                        frame_w, frame_h = client_w, client_h
                    frame_size = (frame_w, frame_h)
                    try:
                        from PIL import Image

                        img = Image.frombytes("RGB", frame_size, pixels)
                        try:
                            import numpy as np

                            reward_frame_hwc = np.asarray(img)
                        except Exception:
                            reward_frame_hwc = None
                        if self.highlight:
                            self.highlight.add_frame(img.tobytes(), frame_size[0], frame_size[1])  # type: ignore[index]
                        if requests:
                            import base64, io

                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=80)
                            latest_image_bytes = buf.getvalue()
                            latest_image_source = "research_shm"
                            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                            vr = requests.post(
                                f"{self.vision_url}/predict",
                                json={"image_b64": b64},
                                timeout=1.0,
                            )
                            if vr.ok:
                                data = vr.json() or {}
                                detections = data.get("detections", []) or []
                                if isinstance(data.get("metrics"), dict):
                                    vision_metrics.update(data.get("metrics") or {})
                    except Exception:
                        detections = []
                else:
                    self._stop.wait(0.001)
                    continue

            # BonkLink (state + optional JPEG frame).
            if self._use_bonklink and self._bonklink is not None:
                try:
                    pkt = self._bonklink.read_state_frame(timeout_ms=16)
                except Exception:
                    pkt = None
                if pkt is not None:
                    state_obj, jpeg_bytes = pkt
                    # BonkLink UI click coords are expressed in the capture (frame) coordinate space.
                    # Cache the capture resolution so we can map other vision-derived coordinates
                    # (e.g., Synthetic Eye 640x360) into the expected click space.
                    if jpeg_bytes:
                        try:
                            import struct

                            if len(jpeg_bytes) >= 16 and jpeg_bytes[:4] == b"MBRF":
                                w_cap, h_cap, _c = struct.unpack_from("<3i", jpeg_bytes, 4)
                                if int(w_cap) > 0 and int(h_cap) > 0:
                                    self._bonklink_capture_size = (int(w_cap), int(h_cap))
                        except Exception:
                            pass
                    if not self._visual_only:
                        game_state.update(state_obj.to_dict())
                        self._last_state_ts = now
                    if (not used_pixels) and (not detections) and jpeg_bytes:
                        try:
                            import base64
                            import io
                            import struct
                            from PIL import Image

                            if len(jpeg_bytes) >= 16 and jpeg_bytes[:4] == b"MBRF":
                                w, h, c = struct.unpack_from("<3i", jpeg_bytes, 4)
                                raw = jpeg_bytes[16:]
                                if w > 0 and h > 0 and c in (3, 4) and len(raw) >= (w * h * c):
                                    try:
                                        import numpy as np

                                        arr = np.frombuffer(raw[: w * h * c], dtype=np.uint8).reshape((h, w, c))
                                        if c == 4:
                                            arr = arr[:, :, :3]
                                        img = Image.fromarray(arr, mode="RGB")
                                        frame_size = (w, h)
                                        reward_frame_hwc = arr
                                    except Exception:
                                        img = Image.frombytes("RGB", (w, h), raw[: w * h * 3])
                                        frame_size = (w, h)
                                        reward_frame_hwc = None
                                    buf = io.BytesIO()
                                    img.save(buf, format="JPEG", quality=80)
                                    latest_image_bytes = buf.getvalue()
                                    latest_image_source = "bonklink"
                                else:
                                    latest_image_bytes = jpeg_bytes
                                    img = Image.open(io.BytesIO(jpeg_bytes))
                                    frame_size = img.size
                                    reward_frame_hwc = None
                                    latest_image_source = "bonklink"
                            else:
                                latest_image_bytes = jpeg_bytes
                                img = Image.open(io.BytesIO(jpeg_bytes))
                                frame_size = img.size
                                reward_frame_hwc = None
                                latest_image_source = "bonklink"

                            self._set_latest_jpeg(latest_image_bytes)
                            used_pixels = True
                            try:
                                if reward_frame_hwc is None:
                                    import numpy as np

                                    reward_frame_hwc = np.asarray(img.convert("RGB"))
                            except Exception:
                                reward_frame_hwc = None
                            if self.highlight:
                                try:
                                    self.highlight.add_frame(
                                        img.convert("RGB").tobytes(),
                                        frame_size[0],
                                        frame_size[1],
                                    )  # type: ignore[index]
                                except Exception:
                                    pass
                            if requests:
                                b64 = base64.b64encode(latest_image_bytes).decode("ascii")
                                vr = requests.post(
                                    f"{self.vision_url}/predict",
                                    json={"image_b64": b64},
                                    timeout=1.0,
                                )
                                if vr.ok:
                                    data = vr.json() or {}
                                    detections = data.get("detections", []) or []
                                    if isinstance(data.get("metrics"), dict):
                                        vision_metrics.update(data.get("metrics") or {})
                        except Exception:
                            detections = []

            # Primary: UnityBridge if enabled and connected.
            if (not used_pixels) and (not detections) and self._use_bridge and self.bridge and self._bridge_loop:
                gf = self._bridge_read_frame()
                if gf is not None and getattr(gf, "pixels", None) is not None:
                    game_state = {} if self._visual_only else (getattr(gf, "state", {}) or {})
                    if not self._visual_only:
                        self._last_state_ts = now
                    pixels = gf.pixels
                    try:
                        import base64
                        import io
                        from PIL import Image
                        import numpy as np

                        if isinstance(pixels, Image.Image):
                            img = pixels.convert("RGB")
                            frame_size = img.size
                        else:
                            arr = np.asarray(pixels)
                            if arr.ndim == 3:
                                h, w = arr.shape[:2]
                                frame_size = (w, h)
                                img = Image.fromarray(arr.astype("uint8"))
                            else:
                                img = None

                        if self.highlight and img is not None:
                            self.highlight.add_frame(img.tobytes(), frame_size[0], frame_size[1])  # type: ignore[index]

                        if requests and img is not None:
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=80)
                            latest_image_bytes = buf.getvalue()
                            latest_image_source = "unity_bridge"
                            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                            vr = requests.post(
                                f"{self.vision_url}/predict",
                                json={"image_b64": b64},
                                timeout=1.0,
                            )
                            if vr.ok:
                                data = vr.json() or {}
                                detections = data.get("detections", []) or []
                                if isinstance(data.get("metrics"), dict):
                                    vision_metrics.update(data.get("metrics") or {})
                    except Exception:
                        detections = []

            # Fallback: visual capture (PipeWire or Synthetic Eye) when no bridge frames exist.
            # Synthetic Eye is the GPU-only vision sensor: always consume frames so the producer
            # can make forward progress (release fences must be serviced), even if other sources
            # also provide observations.
            if (
                (
                    self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf")
                    and not bool(getattr(self, "_synthetic_eye_drain_enabled", False))
                )
                or (
                    (not detections)
                    and (not used_pixels)
                    and (not getattr(self, "_capture_disabled", False))
                    and not (self._use_bridge and self.bridge and self._bridge_loop)
                    and not (
                        self._frame_source in ("synthetic_eye", "smithay", "smithay_dmabuf")
                        and bool(getattr(self, "_synthetic_eye_drain_enabled", False))
                    )
                )
            ):
                frame = self.stream.read()
                if frame is not None:
                    try:
                        self._latest_frame_ts = float(getattr(frame, "timestamp", 0.0) or time.time())
                    except Exception:
                        self._latest_frame_ts = time.time()
                    self._record_obs_frame_ts(self._latest_frame_ts)
                    # Synthetic Eye frames are GPU-only (DMA-BUF + fences). For now, the worker
                    # only performs explicit-sync handshake (wait acquire / signal release) to
                    # keep the compositor's buffer pool flowing. Vision inference on DMA-BUF is
                    # wired separately (see src/worker/synthetic_eye_cuda.py).
                    if SyntheticEyeFrame is not None and isinstance(frame, SyntheticEyeFrame):
                        try:
                            if self._synthetic_eye_ingestor is not None:
                                h = None
                                try:
                                    h = self._synthetic_eye_ingestor.begin(frame)
                                    try:
                                        from src.agent.tensor_bridge import tensor_from_external_frame

                                        offset_bytes = int(frame.offset) if int(frame.modifier) == 0 else 0
                                        try:
                                            import torch  # type: ignore

                                            ext_stream = torch.cuda.ExternalStream(int(h.stream))
                                        except Exception:
                                            ext_stream = None

                                        if ext_stream is not None:
                                            with torch.cuda.stream(ext_stream):
                                                raw_tensor = tensor_from_external_frame(
                                                    h.ext_frame,
                                                    width=frame.width,
                                                    height=frame.height,
                                                    stride_bytes=frame.stride,
                                                    offset_bytes=offset_bytes,
                                                    stream=h.stream,
                                                    sync=False,
                                                )
                                                raw_tensor = self._maybe_swizzle_channels(raw_tensor, frame.drm_fourcc)
                                                if bool(getattr(self, "_synthetic_eye_vflip", False)):
                                                    try:
                                                        raw_tensor = raw_tensor.flip(0)
                                                    except Exception:
                                                        pass
                                                obs = raw_tensor.permute(2, 0, 1)[:3].float().div(255.0)

                                                if (
                                                    self._dashboard_stream == "synthetic_eye"
                                                    and self._dashboard_fps > 0
                                                    and (now - float(self._dashboard_last_ts or 0.0))
                                                    >= (1.0 / float(self._dashboard_fps))
                                                ):
                                                    try:
                                                        import io
                                                        from PIL import Image
                                                        import torch.nn.functional as F

                                                        img_t = raw_tensor[..., :3].permute(2, 0, 1).unsqueeze(0).float()
                                                        if 0 < self._dashboard_scale < 1:
                                                            h_out = max(2, int(img_t.shape[2] * self._dashboard_scale))
                                                            w_out = max(2, int(img_t.shape[3] * self._dashboard_scale))
                                                            img_t = F.interpolate(
                                                                img_t, size=(h_out, w_out), mode="bilinear", align_corners=False
                                                            )
                                                        img_t = img_t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().to("cpu")
                                                        img = Image.fromarray(img_t.numpy())
                                                        buf = io.BytesIO()
                                                        img.save(buf, format="JPEG", quality=80)
                                                        latest_image_bytes = buf.getvalue()
                                                        latest_image_source = "synthetic_eye"
                                                        self._set_latest_jpeg(latest_image_bytes)
                                                        self._dashboard_last_ts = now
                                                    except Exception:
                                                        pass
                                                if os.environ.get("METABONK_VISION_AUDIT", "0") == "1":
                                                    if int(frame.frame_id) % 60 == 0:
                                                        mean_val = float(obs.mean().item())
                                                        std_val = float(obs.std().item())
                                                        print(
                                                            f"[VISION] worker={self.instance_id} frame={int(frame.frame_id)} "
                                                            f"shape={tuple(obs.shape)} mean={mean_val:.4f} std={std_val:.4f} "
                                                            f"device={obs.device}",
                                                            flush=True,
                                                        )
                                        else:
                                            raw_tensor = tensor_from_external_frame(
                                                h.ext_frame,
                                                width=frame.width,
                                                height=frame.height,
                                                stride_bytes=frame.stride,
                                                offset_bytes=offset_bytes,
                                                stream=h.stream,
                                            )
                                            raw_tensor = self._maybe_swizzle_channels(raw_tensor, frame.drm_fourcc)
                                            if bool(getattr(self, "_synthetic_eye_vflip", False)):
                                                try:
                                                    raw_tensor = raw_tensor.flip(0)
                                                except Exception:
                                                    pass
                                            obs = raw_tensor.permute(2, 0, 1)[:3].float().div(255.0)
                                        if (
                                            self._dashboard_stream == "synthetic_eye"
                                            and self._dashboard_fps > 0
                                            and (now - float(self._dashboard_last_ts or 0.0))
                                            >= (1.0 / float(self._dashboard_fps))
                                        ):
                                            try:
                                                import io
                                                from PIL import Image
                                                import torch.nn.functional as F

                                                img_t = raw_tensor[..., :3].permute(2, 0, 1).unsqueeze(0).float()
                                                if 0 < self._dashboard_scale < 1:
                                                    h_out = max(2, int(img_t.shape[2] * self._dashboard_scale))
                                                    w_out = max(2, int(img_t.shape[3] * self._dashboard_scale))
                                                    img_t = F.interpolate(
                                                        img_t, size=(h_out, w_out), mode="bilinear", align_corners=False
                                                    )
                                                img_t = img_t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().to("cpu")
                                                img = Image.fromarray(img_t.numpy())
                                                buf = io.BytesIO()
                                                img.save(buf, format="JPEG", quality=80)
                                                latest_image_bytes = buf.getvalue()
                                                latest_image_source = "synthetic_eye"
                                                self._set_latest_jpeg(latest_image_bytes)
                                                self._dashboard_last_ts = now
                                            except Exception:
                                                pass
                                        if os.environ.get("METABONK_VISION_AUDIT", "0") == "1":
                                            if int(frame.frame_id) % 60 == 0:
                                                mean_val = float(obs.mean().item())
                                                std_val = float(obs.std().item())
                                                print(
                                                    f"[VISION] worker={self.instance_id} frame={int(frame.frame_id)} "
                                                    f"shape={tuple(obs.shape)} mean={mean_val:.4f} std={std_val:.4f} "
                                                    f"device={obs.device}",
                                                    flush=True,
                                                )
                                    except Exception as e:
                                        print(
                                            f"[VISION] worker={self.instance_id} tensor bridge failed: {e}",
                                            flush=True,
                                        )
                                finally:
                                    if h is not None:
                                        self._synthetic_eye_ingestor.end(h)
                            else:
                                # If CUDA ingest is unavailable, avoid deadlocking the producer by
                                # dropping the frame immediately (release fence remains unsignaled).
                                # This configuration is unsupported for Synthetic Eye runs.
                                pass
                        except Exception:
                            # Do not propagate; keep worker alive even if a single frame import fails.
                            pass
                        try:
                            frame.close()
                        except Exception:
                            pass
                    else:
                        frame_size = (frame.width, frame.height)
                        if self.highlight and frame.cpu_rgb:
                            self.highlight.add_frame(frame.cpu_rgb, frame.width, frame.height)
                        if frame.cpu_rgb:
                            try:
                                import numpy as np

                                reward_frame_hwc = np.frombuffer(frame.cpu_rgb, dtype=np.uint8).reshape(
                                    (int(frame.height), int(frame.width), 3)
                                )
                            except Exception:
                                reward_frame_hwc = None
                        if requests and frame.cpu_rgb:
                            try:
                                import base64
                                import io
                                from PIL import Image

                                img = Image.frombytes("RGB", (frame.width, frame.height), frame.cpu_rgb)
                                buf = io.BytesIO()
                                img.save(buf, format="JPEG", quality=80)
                                latest_image_bytes = buf.getvalue()
                                latest_image_source = "pipewire"
                                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                                vr = requests.post(
                                    f"{self.vision_url}/predict",
                                    json={"image_b64": b64},
                                    timeout=1.0,
                                )
                                if vr.ok:
                                    data = vr.json() or {}
                                    detections = data.get("detections", []) or []
                                    if isinstance(data.get("metrics"), dict):
                                        vision_metrics.update(data.get("metrics") or {})
                            except Exception:
                                detections = []

            # Persist latest frame bytes for UI fallback/debug endpoints.
            if latest_image_bytes is not None:
                self._set_latest_jpeg(latest_image_bytes)
            elif (
                str(os.environ.get("METABONK_FRAME_SNAPSHOT_FALLBACK", "1")).lower() in ("1", "true", "yes")
                and self.streamer is not None
                and hasattr(self.streamer, "capture_jpeg")
            ):
                # Best-effort snapshot when no live frames are available.
                try:
                    snap = self.streamer.capture_jpeg(
                        timeout_s=float(os.environ.get("METABONK_FRAME_JPEG_TIMEOUT_S", "1.5"))
                    )
                except Exception:
                    snap = None
                if snap:
                    latest_image_bytes = snap
                    latest_image_source = "streamer_snapshot"
                    self._set_latest_jpeg(snap)
            elif (
                latest_image_bytes is None
                and reward_frame_hwc is not None
                and self._frame_ring_enabled
            ):
                try:
                    import io
                    from PIL import Image
                    import numpy as np

                    arr = np.asarray(reward_frame_hwc)
                    img = Image.fromarray(arr.astype("uint8"))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=75)
                    latest_image_bytes = buf.getvalue()
                    latest_image_source = latest_image_source or "frame_rgb"
                    self._set_latest_jpeg(latest_image_bytes)
                except Exception:
                    pass

            if latest_image_bytes is not None:
                if not latest_image_source:
                    latest_image_source = "unknown"
                try:
                    ts = self._latest_frame_ts or now
                    self._push_frame_ring(latest_image_bytes, source=latest_image_source, ts=ts)
                except Exception:
                    pass

            # Vision-only intrinsic exploration signals (game-agnostic).
            # These are used to (a) provide reward in pure-vision mode and (b) increase
            # exploration pressure when the screen is static ("stuck"), without any menu/UI
            # semantics.
            stuck = False
            if self._visual_exploration is not None:
                try:
                    interval_s = float(getattr(self, "_visual_exploration_interval_s", 0.0) or 0.0)
                except Exception:
                    interval_s = 0.0
                should_update = interval_s <= 0.0 or (now - float(self._visual_exploration_last_update_ts or 0.0)) >= interval_s
                if should_update:
                    frame_for_exploration = reward_frame_hwc
                    if frame_for_exploration is None and latest_image_bytes is not None:
                        # Fallback decode when only JPEG bytes are available.
                        try:
                            import io
                            import numpy as np
                            from PIL import Image

                            img = Image.open(io.BytesIO(latest_image_bytes)).convert("RGB")
                            frame_for_exploration = np.asarray(img)
                        except Exception:
                            frame_for_exploration = None
                    if frame_for_exploration is not None:
                        try:
                            self._visual_exploration_last_update_ts = float(now)
                            self._visual_exploration.update(frame_for_exploration)
                        except Exception:
                            pass
                try:
                    stuck = bool(self._visual_exploration.is_stuck())
                except Exception:
                    stuck = False

            hud_rise = False
            prev_hud = bool(self._hud_present)
            hud_present = self._hud_present
            if latest_image_bytes is not None:
                try:
                    hud_present = self._detect_hud_minimap(latest_image_bytes)
                except Exception:
                    hud_present = False
            if (not hud_present) and str(getattr(self, "_frame_source", "") or "") in (
                "synthetic_eye",
                "smithay",
                "smithay_dmabuf",
            ):
                # Synthetic Eye runs often lack high-res JPEG snapshots (frame.jpg can be tiny).
                # Prefer the most recent full-res spectator frame we already copied for System2.
                try:
                    ts_full = float(getattr(self, "_system2_last_full_frame_ts", 0.0) or 0.0)
                    frame_full = getattr(self, "_system2_last_full_frame_hwc", None)
                    if frame_full is not None and ts_full > 0 and (now - ts_full) <= 1.5:
                        hud_present = bool(self._detect_hud_minimap_array(frame_full))
                except Exception:
                    pass
                self._update_hud_state(bool(hud_present))
                hud_rise = (not prev_hud) and bool(self._hud_present)
                phase = "gameplay" if self._hud_present else "lobby"
                if self._hud_phase_log and phase != self._hud_last_phase:
                    print(
                        f"[HUD] instance={self.instance_id} phase={phase} hud={self._hud_present}",
                        flush=True,
                    )
                    self._hud_last_phase = phase

            phase_confident = False
            phase_label = self._phase_label
            phase_conf = float(self._phase_conf or 0.0)
            phase_menu = False
            phase_block = False
            if self._phase_model_enabled and (now - self._phase_last_infer_ts) >= float(self._phase_infer_every_s):
                payloads = self._sample_frame_ring_bytes(int(self._phase_clip_frames))
                result = self._infer_phase_model(payloads)
                if result:
                    label, conf = result
                    self._phase_label = label
                    self._phase_conf = float(conf)
                    self._phase_source = "model"
                self._phase_last_infer_ts = now
            phase_label = self._phase_label
            phase_conf = float(self._phase_conf or 0.0)
            if phase_label is not None and phase_conf >= float(self._phase_conf_thresh):
                phase_confident = True
                phase_menu = phase_label in self._phase_menu_labels
                phase_block = phase_label in self._phase_block_labels
                self._phase_gameplay = phase_label == "gameplay"
            else:
                self._phase_gameplay = False

            if phase_confident:
                gameplay_now = bool(self._phase_gameplay)
                phase_effective_label = phase_label
                phase_effective_source = "model"
            else:
                gameplay_now = bool(self._hud_present)
                phase_effective_label = "gameplay" if gameplay_now else None
                phase_effective_source = "hud" if gameplay_now else None
            gameplay_rise = (not self._gameplay_phase_active) and gameplay_now
            self._gameplay_phase_active = gameplay_now
            self._phase_effective_label = phase_effective_label
            self._phase_effective_source = phase_effective_source

            # Pure vision: no menu heuristics (no scene/UI taxonomy).
            if gameplay_now and self._hud_pulse_log_s > 0.0:
                if (now - self._hud_pulse_last_ts) >= float(self._hud_pulse_log_s):
                    try:
                        pv = game_state.get("playerVelocity") or (0.0, 0.0, 0.0)
                        pp = game_state.get("playerPosition") or (0.0, 0.0, 0.0)
                        gt = game_state.get("gameTime")
                        try:
                            vx, vy, vz = pv
                        except Exception:
                            vx = vy = vz = 0.0
                        try:
                            px, py, pz = pp
                        except Exception:
                            px = py = pz = 0.0
                        try:
                            gt_val = float(gt) if gt is not None else None
                        except Exception:
                            gt_val = None
                        print(
                            f"[HUD_PULSE] instance={self.instance_id} "
                            f"pos=({px:.2f},{py:.2f},{pz:.2f}) "
                            f"vel=({vx:.2f},{vy:.2f},{vz:.2f}) "
                            f"gameTime={gt_val if gt_val is not None else 'None'}",
                            flush=True,
                        )
                    except Exception:
                        pass
                    self._hud_pulse_last_ts = now

            weak_label = self._weak_phase_label(game_state)
            dataset_label = self._phase_dataset_label_override(
                now=now,
                base_label=weak_label,
                game_state=game_state,
                gameplay_now=gameplay_now,
            )
            self._maybe_dump_phase_sample(now=now, label=dataset_label, game_state=game_state)

            # Get pixel observations FIRST so we can feed them to the gameplay detector.
            pixel_tensor = None
            pixel_frame_id = None
            pixel_src_size = None
            if bool(getattr(self, "_pixel_obs_enabled", False)):
                try:
                    pixel_tensor, pixel_frame_id, _, pixel_src_size = self._get_latest_pixel_obs()
                except Exception:
                    pixel_tensor = None
                    pixel_frame_id = None
                    pixel_src_size = None
                if frame_size is None and pixel_src_size is not None:
                    try:
                        frame_size = (int(pixel_src_size[0]), int(pixel_src_size[1]))
                    except Exception:
                        frame_size = frame_size
                if (
                    self._use_learned_reward
                    and reward_frame_hwc is None
                    and pixel_tensor is not None
                ):
                    try:
                        import numpy as np  # type: ignore

                        reward_frame_hwc = (
                            pixel_tensor.detach()
                            .to(device="cpu", non_blocking=False)
                            .permute(1, 2, 0)
                            .contiguous()
                            .numpy()
                        )
                        if reward_frame_hwc is not None:
                            reward_frame_hwc = np.asarray(reward_frame_hwc, dtype=np.uint8)
                    except Exception:
                        reward_frame_hwc = None
                if pixel_tensor is None:
                    if bool(getattr(self, "_synthetic_eye_lockstep", False)):
                        # Prime the lock-step pipeline: request the first frame.
                        self._lockstep_request_next_frame(None)
                    now_wait = time.time()
                    last_warn = float(getattr(self, "_pixel_wait_warn_ts", 0.0) or 0.0)
                    if (now_wait - last_warn) > 5.0:
                        self._pixel_wait_warn_ts = now_wait
                        print(
                            f"[worker:{self.instance_id}] waiting for pixel observations from Synthetic Eye...",
                            flush=True,
                        )
                    self._stop.wait(0.01)
                    continue

            # Update gameplay state with frame data for hybrid detector.
            # This is called AFTER we have pixel_tensor so the detector can analyze visuals.
            self._update_gameplay_state(
                game_state,
                hud_present=self._hud_present,
                phase_gameplay=gameplay_now,
                frame=pixel_tensor,
                step_increment=1,  # Each iteration is a step
                action_taken=True,  # We're in the action loop
            )

            # Optional System 2/3 centralized VLM loop (non-blocking).
            if self._cognitive_client is not None:
                try:
                    # System2/VLM needs readable frames for UI flows. BonkLink/PipeWire snapshots
                    # can be low-res (and Synthetic Eye often has higher-res spectator frames).
                    #
                    # Throttle frame ingestion to avoid 60Hz CPU encode/decode overhead.
                    last_add = float(self._system2_last_frame_add_ts or 0.0)
                    try:
                        system2_frame_interval_s = float(os.environ.get("METABONK_SYSTEM2_FRAME_INTERVAL_S", "0.5") or "0.5")
                    except Exception:
                        system2_frame_interval_s = 0.5
                    system2_frame_interval_s = max(0.0, float(system2_frame_interval_s))

                    fed = False
                    spectator_fed = False
                    if system2_frame_interval_s <= 0.0 or (float(now) - last_add) >= system2_frame_interval_s:
                        # Prefer Synthetic Eye spectator frames when available: they're derived
                        # from the GPU-only path and preserve UI readability.
                        prefer_spectator = (
                            str(getattr(self, "_frame_source", "") or "") in ("synthetic_eye", "smithay", "smithay_dmabuf")
                            and bool(getattr(self, "_spectator_enabled", False))
                        )
                        if prefer_spectator:
                            try:
                                import numpy as np  # type: ignore
                                obs_t, _fid, _ts, _src_size = self._get_latest_spectator_obs()
                                if obs_t is not None:
                                    t = obs_t
                                    try:
                                        if hasattr(t, "detach"):
                                            t = t.detach()
                                    except Exception:
                                        pass
                                    try:
                                        if hasattr(t, "device") and str(getattr(t, "device", "")).startswith("cuda"):
                                            try:
                                                import torch  # type: ignore

                                                stream_ptr = int(getattr(self, "_synthetic_eye_cu_stream_ptr", 0) or 0)
                                                if stream_ptr:
                                                    ext = torch.cuda.ExternalStream(int(stream_ptr))
                                                    torch.cuda.current_stream(device=ext.device).wait_stream(ext)
                                            except Exception:
                                                pass
                                            t = t.cpu()
                                        elif hasattr(t, "cpu"):
                                            t = t.cpu()
                                    except Exception:
                                        if hasattr(t, "cpu"):
                                            t = t.cpu()
                                    # Expect CHW uint8 (3xHxW).
                                    try:
                                        if hasattr(t, "ndim") and int(getattr(t, "ndim", 0) or 0) == 3:
                                            t = t[:3].permute(1, 2, 0).contiguous()
                                    except Exception:
                                        pass
                                    arr = t.numpy() if hasattr(t, "numpy") else None
                                    if arr is not None and getattr(arr, "ndim", 0) == 3:
                                        if int(arr.shape[2]) != 3:
                                            arr = arr[:, :, :3]
                                        arr_u8 = np.asarray(arr, dtype=np.uint8)
                                        self._cognitive_client.add_frame(arr_u8)
                                        fed = True
                                        spectator_fed = True
                            except Exception:
                                fed = False
                                spectator_fed = False

                        # Fallback to whatever snapshot bytes we have (PipeWire/BonkLink).
                        if (not fed) and latest_image_bytes is not None:
                            try:
                                import io
                                import numpy as np  # type: ignore
                                from PIL import Image

                                img = Image.open(io.BytesIO(latest_image_bytes)).convert("RGB")
                                self._cognitive_client.add_frame(np.asarray(img, dtype=np.uint8))
                                fed = True
                            except Exception:
                                fed = False

                        # Last resort: use a reward frame if provided.
                        if (not fed) and reward_frame_hwc is not None:
                            try:
                                self._cognitive_client.add_frame(reward_frame_hwc)
                                fed = True
                            except Exception:
                                fed = False

                        if fed:
                            self._system2_last_frame_add_ts = float(now)
                        if spectator_fed:
                            try:
                                setattr(self, "_system2_spectator_ready", True)
                            except Exception:
                                pass
                            try:
                                setattr(self, "_system2_last_full_frame_hwc", arr_u8)
                                setattr(self, "_system2_last_full_frame_ts", float(now))
                            except Exception:
                                pass

                    # Fallback: some paths may not produce spectator/snapshots early. Feed System2
                    # from pixel observations (small CHW tensor) with separate throttling to avoid
                    # a 60Hz GPU->CPU copy.
                    if (not fed) and pixel_tensor is not None and not bool(getattr(self, "_system2_spectator_ready", False)):
                        if (float(now) - float(self._system2_last_frame_add_ts or 0.0)) >= 0.2:
                            import numpy as np  # type: ignore

                            arr = pixel_tensor
                            try:
                                if hasattr(arr, "detach"):
                                    arr = arr.detach()
                                if hasattr(arr, "to"):
                                    arr = arr.to(device="cpu", non_blocking=False)
                                if hasattr(arr, "contiguous"):
                                    arr = arr.contiguous()
                                if hasattr(arr, "numpy"):
                                    arr = arr.numpy()
                            except Exception:
                                arr = arr
                            try:
                                arr = np.asarray(arr)
                                if arr.ndim == 3:
                                    # Normalize to HWC RGB uint8.
                                    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                                        arr = np.transpose(arr, (1, 2, 0))
                                    if arr.shape[-1] == 4:
                                        arr = arr[..., :3]
                                    elif arr.shape[-1] == 1:
                                        arr = np.repeat(arr, 3, axis=-1)
                                    arr_u8 = np.asarray(arr, dtype=np.uint8)
                                    self._cognitive_client.add_frame(arr_u8)
                                    self._system2_last_frame_add_ts = float(now)
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    resp = self._cognitive_client.poll_response()
                    if resp is not None:
                        self._system2_on_strategy_response(resp, now=now)
                except Exception:
                    pass
                try:
                    if self._cognitive_client.should_request_strategy():
                        agent_state = self._system2_build_agent_state(
                            game_state=game_state,
                            detections=detections,
                            frame_size=frame_size,
                            stuck=stuck,
                            now=now,
                        )
                        req = self._cognitive_client.request_strategy(agent_state=agent_state)
                        if req is not None:
                            self._system2_last_request = req
                except Exception:
                    pass

            ui_elements_cache = None
            action_mask_cache = None
            try:
                from .perception import (
                    build_grid_ui_elements,
                    build_saliency_ui_elements,
                    build_ui_elements,
                    parse_detections,
                )

                ui_mode = str(os.environ.get("METABONK_UI_CANDIDATES_MODE", "") or "").strip().lower()
                if not ui_mode:
                    ui_mode = "saliency" if bool(getattr(self, "_pure_vision_mode", False)) else "detections"

                # UI candidate grids: default to 8x4 to match the fixed 32-slot UI branch
                # without discarding half the screen by truncation.
                rows, cols = self._parse_grid_spec(
                    str(getattr(self, "_pixel_ui_grid", "")),
                    default_rows=8,
                    default_cols=4,
                )

                if ui_mode in ("saliency", "attention"):
                    # Game-agnostic: derive click candidates from pixel saliency (no class taxonomy).
                    if reward_frame_hwc is not None and frame_size is not None:
                        ui_elements_cache, action_mask_cache = build_saliency_ui_elements(
                            reward_frame_hwc,
                            frame_size,
                            max_elements=32,
                            grid_rows=rows,
                            grid_cols=cols,
                        )
                    else:
                        ui_elements_cache, action_mask_cache = build_grid_ui_elements(
                            frame_size,
                            max_elements=32,
                            rows=rows,
                            cols=cols,
                        )
                elif bool(getattr(self, "_pixel_obs_enabled", False)) and not detections:
                    ui_elements_cache, action_mask_cache = build_grid_ui_elements(
                        frame_size,
                        max_elements=32,
                        rows=rows,
                        cols=cols,
                    )
                else:
                    dets_parsed = parse_detections(detections)
                    ui_elements_cache, action_mask_cache, _ = build_ui_elements(
                        dets_parsed,
                        frame_size=frame_size,
                    )
                    # If the detector produced no clickable candidates, fall back to saliency
                    # so the discrete click branch can still explore.
                    try:
                        if (
                            reward_frame_hwc is not None
                            and frame_size is not None
                            and action_mask_cache is not None
                            and sum(1 for m in action_mask_cache[:-1] if int(m) == 1) <= 0
                        ):
                            ui_elements_cache, action_mask_cache = build_saliency_ui_elements(
                                reward_frame_hwc,
                                frame_size,
                                max_elements=32,
                                grid_rows=rows,
                                grid_cols=cols,
                            )
                    except Exception:
                        pass
            except Exception:
                ui_elements_cache = None
                action_mask_cache = None

            obs, action_mask = construct_observation(
                detections,
                obs_dim=self._obs_dim_raw,
                pixel_tensor=pixel_tensor,
                frame_size=frame_size,
                ui_override=ui_elements_cache,
                action_mask_override=action_mask_cache,
            )
            obs = self._stack_observation(obs)

            # Default policy action (PPO).
            action_mask_for_policy = action_mask if self._input_backend is None else None
            if action_mask_for_policy is not None:
                try:
                    branches = list(getattr(self.trainer.cfg, "discrete_branches", []) or [])
                    expected = int(branches[0]) if branches else None
                    if expected is not None and len(action_mask_for_policy) != expected:
                        action_mask_for_policy = None
                except Exception:
                    action_mask_for_policy = None
            action_source = self._action_source if self._gameplay_started else "policy"
            if self._gameplay_started and action_source == "random":
                a_cont, a_disc = self._sample_random_action(action_mask_for_policy or action_mask)
                lp = 0.0
                val = 0.0
            else:
                try:
                    a_cont, a_disc, lp, val = self.trainer.act(obs, action_mask=action_mask_for_policy)
                    # Hardening: guard against NaNs/Infs from unstable policy weights. A crash here can
                    # stall Synthetic Eye by leaving release fences unsignaled.
                    if any((not _math.isfinite(float(x))) for x in (list(a_cont) if a_cont else [])):
                        raise ValueError("non-finite continuous action")
                    if not _math.isfinite(float(lp)) or not _math.isfinite(float(val)):
                        raise ValueError("non-finite lp/val")
                except Exception as e:
                    try:
                        self.trainer.reset_state()
                    except Exception:
                        pass
                    a_cont, a_disc = self._sample_random_action(action_mask_for_policy or action_mask)
                    lp = 0.0
                    val = 0.0
                    now_dbg = time.time()
                    last_warn = float(getattr(self, "_last_policy_nan_warn_ts", 0.0) or 0.0)
                    if (now_dbg - last_warn) > 5.0:
                        self._last_policy_nan_warn_ts = now_dbg
                        print(
                            f"[worker:{self.instance_id}] WARN: policy.act failed ({e}); using random action and resetting state",
                            flush=True,
                        )
            action_mask_for_rollout = action_mask_for_policy

            # Optional MetaBonk2 backend override (tripartite mind).
            if (
                self._use_metabonk2
                and self._metabonk2_controller is not None
                and latest_image_bytes is not None
                and not (self._gameplay_started and action_source == "random")
            ):
                try:
                    import io
                    import numpy as np
                    from PIL import Image

                    img = Image.open(io.BytesIO(latest_image_bytes)).convert("RGB")
                    frame_arr = np.asarray(img)
                    step_now = int(getattr(self.trainer, "step_count", 0) or 0)
                    mb2_cont, mb2_disc = self._metabonk2_controller.step(
                        frame_arr,
                        game_state,
                        time_budget_ms=self._metabonk2_time_budget_ms,
                        step=step_now,
                    )

                    # Override discrete controls only when using OS-level input backend. In BonkLink
                    # mode discrete actions can be semantic (e.g., UI click index).
                    if (
                        self._input_backend is not None
                        and getattr(self._metabonk2_controller, "cfg", None) is not None
                        and bool(getattr(self._metabonk2_controller.cfg, "override_discrete", False))
                        and mb2_disc is not None
                    ):
                        if len(a_disc) > 0:
                            disc = [int(x) for x in list(mb2_disc)]
                            disc = disc[: len(a_disc)] + [0] * max(0, len(a_disc) - len(disc))
                            a_disc = disc
                        else:
                            a_disc = [int(x) for x in list(mb2_disc)]

                    # Continuous overrides are opt-in to avoid clobbering aim policies.
                    if os.environ.get("METABONK2_OVERRIDE_CONT", "0") in ("1", "true", "True"):
                        if mb2_cont is not None and len(mb2_cont) > 0:
                            cont = [float(x) for x in list(mb2_cont)]
                            if len(a_cont) > 0:
                                merged = list(a_cont)
                                for i in range(min(len(merged), len(cont))):
                                    merged[i] = float(cont[i])
                                a_cont = merged
                            else:
                                a_cont = cont
                    lp = 0.0
                    val = 0.0
                except Exception:
                    pass

            # Optional SIMA2 backend override (inference).
            if (
                self._use_sima2
                and self._sima2_controller is not None
                and latest_image_bytes is not None
                and not (self._gameplay_started and action_source == "random")
            ):
                try:
                    import io
                    import numpy as np
                    from PIL import Image

                    img = Image.open(io.BytesIO(latest_image_bytes)).convert("RGB")
                    frame_arr = np.asarray(img)
                    act_vec = self._sima2_controller.step(
                        frame_arr,
                        game_state,
                        detections=detections,
                        frame_size=frame_size,
                    )
                    # Ensure python list
                    sima2_action = [float(x) for x in act_vec.reshape(-1).tolist()]
                    # For compatibility with recovery rollout shapes, keep movement dims.
                    if len(sima2_action) >= 2:
                        a_cont = sima2_action[:2]
                    # Disable discrete clicks under SIMA2 unless VLM/menu supplies one.
                    a_disc = a_disc if suppress_policy_clicks else a_disc
                    lp = 0.0
                    val = 0.0
                except Exception:
                    sima2_action = None

            # Optional UI exploration: epsilon-greedy UI clicks when the screen has been static
            # for a long time ("stuck"). This is game-agnostic: no scene/menu labels, just
            # "change the screen" exploration pressure.
            pre_gameplay_click = False
            try:
                if not bool(getattr(self, "_gameplay_started", False)):
                    grace_s = float(os.environ.get("METABONK_UI_PRE_GAMEPLAY_GRACE_S", "2.0") or 2.0)
                    pre_gameplay_click = (float(now) - float(loop_t0)) >= max(0.0, float(grace_s))
            except Exception:
                pre_gameplay_click = False
            if (
                (bool(stuck) or pre_gameplay_click)
                and action_mask
                and self._input_backend is None
                and (forced_ui_click is None)
                and (not suppress_policy_clicks)
            ):
                try:
                    if pre_gameplay_click:
                        eps = float(str(os.environ.get("METABONK_UI_PRE_GAMEPLAY_EPS", "0") or "0").strip() or 0.0)
                    else:
                        eps = float(str(os.environ.get("METABONK_UI_EPS", "0") or "0").strip() or 0.0)
                except Exception:
                    eps = 0.0
                if eps <= 0.0 and bool(getattr(self, "_pure_vision_mode", False)):
                    eps = 0.7 if pre_gameplay_click else 0.5
                if eps > 0.0 and random.random() < eps:
                    try:
                        valid = [i for i, m in enumerate(action_mask[:-1]) if int(m) == 1]
                        if valid and a_disc:
                            a_disc[0] = int(random.choice(valid))
                            action_source = f"{action_source}+ui_eps" if action_source else "ui_eps"
                    except Exception:
                        pass

            # Optional System 2 (centralized VLM) directive -> UI click via BonkLink/UnityBridge.
            # This keeps progression vision-only (no game state) while allowing System2 to target
            # visible affordances.
            if (
                forced_ui_click is None
                and self._cognitive_client is not None
                and frame_size is not None
                and self._input_backend is None
                and (bool(stuck) or not self._gameplay_started)
            ):
                try:
                    directive = getattr(self, "_system2_last_response", None)
                    if isinstance(directive, dict) and directive:
                        conf_min = None
                        try:
                            raw = str(os.environ.get("METABONK_SYSTEM2_CLICK_CONF_MIN", "") or "").strip()
                            if raw:
                                conf_min = float(raw)
                        except Exception:
                            conf_min = None
                        if conf_min is None:
                            conf_min = 0.5 if bool(getattr(self, "_pure_vision_mode", False)) else 0.0
                        try:
                            conf_f = float(directive.get("confidence") or 0.0)
                        except Exception:
                            conf_f = 0.0
                        if conf_f < float(conf_min):
                            data = None
                        else:
                            data = directive.get("directive") or {}
                        if isinstance(data, dict):
                            act = str(data.get("action") or "").strip().lower()
                        else:
                            act = ""
                        if act in ("interact", "click", "confirm", "select"):
                            target = data.get("target")
                            if isinstance(target, (list, tuple)) and len(target) >= 2:
                                try:
                                    tx = float(target[0])
                                    ty = float(target[1])
                                except Exception:
                                    tx = None
                                    ty = None
                                if tx is not None and ty is not None:
                                    w, h = frame_size
                                    if 0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0 and w > 0 and h > 0:
                                        x = int(tx * float(w))
                                        y = int(ty * float(h))
                                    else:
                                        x = int(tx)
                                        y = int(ty)
                                    if w > 0:
                                        x = max(0, min(int(w - 1), int(x)))
                                    if h > 0:
                                        y = max(0, min(int(h - 1), int(y)))
                                    # If System2 requests an interaction but provides a generic
                                    # center target (common when the VLM can't localize buttons),
                                    # try a vision-only OCR fallback to click the primary button.
                                    ocr_fallback = False
                                    try:
                                        ocr_fallback = str(
                                            os.environ.get("METABONK_SYSTEM2_OCR_CLICK_FALLBACK", "0") or "0"
                                        ).strip().lower() in ("1", "true", "yes", "on")
                                    except Exception:
                                        ocr_fallback = False
                                    if (
                                        ocr_fallback
                                        and bool(stuck)
                                        and (abs(float(tx) - 0.5) <= 0.12)
                                        and (abs(float(ty) - 0.5) <= 0.12)
                                    ):
                                        try:
                                            import numpy as np  # type: ignore
                                            from PIL import Image

                                            from src.worker.ocr import ocr_boxes

                                            frame_hwc = getattr(self, "_system2_last_full_frame_hwc", None)
                                            ts_f = float(getattr(self, "_system2_last_full_frame_ts", 0.0) or 0.0)
                                            if frame_hwc is not None and (now - ts_f) <= 2.0:
                                                arr = np.asarray(frame_hwc)
                                                if arr.ndim == 3 and int(arr.shape[2]) >= 3:
                                                    img = Image.fromarray(arr[:, :, :3].astype("uint8"))
                                                    boxes = ocr_boxes(img, min_conf=30, min_len=2, psm=6)
                                                    if boxes:
                                                        best = None
                                                        best_score = -1.0
                                                        import re

                                                        for b in boxes:
                                                            txt = str(b.get("text") or "").strip().lower()
                                                            if not txt:
                                                                continue
                                                            conf = float(b.get("conf", 0) or 0)
                                                            # Prefer higher OCR confidence (kept as an opt-in debug fallback).
                                                            tokens = re.findall(r"[a-z0-9]+", txt)
                                                            if not tokens:
                                                                continue
                                                            score = conf
                                                            if score > best_score:
                                                                best_score = score
                                                                best = b
                                                        if best and best.get("bbox"):
                                                            x1, y1, x2, y2 = best["bbox"]
                                                            ox = int((int(x1) + int(x2)) / 2)
                                                            oy = int((int(y1) + int(y2)) / 2)
                                                            # Map OCR-frame coords -> current frame_size coords.
                                                            try:
                                                                o_w = int(arr.shape[1])
                                                                o_h = int(arr.shape[0])
                                                            except Exception:
                                                                o_w = 0
                                                                o_h = 0
                                                            if o_w > 0 and o_h > 0:
                                                                x = int((float(ox) / float(o_w)) * float(w))
                                                                y = int((float(oy) / float(o_h)) * float(h))
                                                                x = max(0, min(int(w - 1), int(x)))
                                                                y = max(0, min(int(h - 1), int(y)))
                                                                action_source = (
                                                                    f"{action_source}+system2_ocr" if action_source else "system2_ocr"
                                                                )
                                        except Exception:
                                            pass
                                    last_resp_ts = float(getattr(self, "_system2_last_response_ts", 0.0) or 0.0)
                                    last_click_resp = float(getattr(self, "_system2_last_ui_click_resp_ts", 0.0) or 0.0)
                                    last_click_ts = float(getattr(self, "_system2_last_ui_click_ts", 0.0) or 0.0)
                                    cooldown_s = float(getattr(self, "_system2_ui_click_cooldown_s", 1.0) or 1.0)
                                    if last_resp_ts > 0 and last_resp_ts > last_click_resp and (now - last_click_ts) >= cooldown_s:
                                        forced_ui_click = (int(x), int(y))
                                        suppress_policy_clicks = True
                                        self._system2_last_ui_click_ts = float(now)
                                        self._system2_last_ui_click_resp_ts = float(last_resp_ts)
                                        try:
                                            if not bool(getattr(self, "_system2_active_applied", False)):
                                                self._vlm_hints_applied = int(getattr(self, "_vlm_hints_applied", 0) or 0) + 1
                                                self._system2_active_applied = True
                                        except Exception:
                                            self._vlm_hints_applied = 1
                                            self._system2_active_applied = True
                                        action_source = f"{action_source}+system2_click" if action_source else "system2_click"
                except Exception:
                    pass

            # Optional System 2 (centralized VLM) directive -> continuous control hint.
            if (
                self._cognitive_client is not None
                and self._system2_override_cont
                and frame_size is not None
                and bool(self._gameplay_started)
            ):
                try:
                    directive = self._cognitive_client.get_current_directive()
                    if directive is not None:
                        a_cont = self._system2_apply_continuous_directive(
                            a_cont=a_cont,
                            directive=directive,
                            frame_size=frame_size,
                        )
                        try:
                            if not bool(getattr(self, "_system2_active_applied", False)):
                                self._vlm_hints_applied = int(getattr(self, "_vlm_hints_applied", 0) or 0) + 1
                                self._system2_active_applied = True
                        except Exception:
                            self._vlm_hints_applied = 1
                            self._system2_active_applied = True
                        action_source = f"{action_source}+system2" if action_source else "system2"
                except Exception:
                    pass

            if self._action_clip_enabled and a_cont:
                raw_cont = [float(x) for x in (a_cont or [])]
                clipped = []
                for val in raw_cont:
                    if val < self._action_clip_min:
                        clipped.append(self._action_clip_min)
                    elif val > self._action_clip_max:
                        clipped.append(self._action_clip_max)
                    else:
                        clipped.append(val)
                if clipped != raw_cont:
                    a_cont = clipped
                    if not self._action_clip_logged:
                        self._action_clip_logged = True
                        print(
                            f"[worker:{self.instance_id}] action_clip raw={raw_cont} clipped={clipped} "
                            f"range=({self._action_clip_min},{self._action_clip_max})",
                            flush=True,
                        )

            input_bootstrap = False
            if self._input_backend is not None:
                self._input_send_actions(a_cont, a_disc)

            self._action_guard_check(
                action_source=action_source,
                forced_ui_click=forced_ui_click,
                input_bootstrap=input_bootstrap,
                sima2_action=sima2_action,
            )

            # Action cadence telemetry (for readiness checks / debugging "slow actions" reports).
            try:
                self._actions_total = int(getattr(self, "_actions_total", 0) or 0) + 1
                self._act_times.append(float(now))
                if len(self._act_times) >= 2:
                    dt = float(self._act_times[-1]) - float(self._act_times[0])
                    if dt > 0:
                        self._act_hz = float((len(self._act_times) - 1) / dt)
            except Exception:
                pass

            # Flight recorder: store last actions + thumbnails for Context Drawer.
            try:
                action_label = "random" if (self._gameplay_started and action_source == "random") else "policy"
                if forced_ui_click is not None:
                    action_label = "click_xy"
                elif a_disc:
                    action_label = f"disc:{int(a_disc[0])}"
                elif a_cont:
                    action_label = "cont"
                entropy = getattr(self.trainer, "last_entropy", None)
                input_vec = [float(x) for x in (a_cont or [])] + [float(x) for x in (a_disc or [])]
                if self._action_log_freq > 0:
                    step_now = int(getattr(self.trainer, "step_count", 0) or 0)
                    if step_now != self._last_action_log_step and (step_now % self._action_log_freq) == 0:
                        self._last_action_log_step = step_now
                        disc_on = [i for i, v in enumerate(a_disc or []) if int(v) == 1]
                        disc_preview = disc_on[:8]
                        disc_suffix = f"+{len(disc_on) - len(disc_preview)}" if len(disc_on) > len(disc_preview) else ""
                        cont_preview = [round(float(x), 3) for x in (a_cont or [])[:3]]
                        print(
                            f"[worker:{self.instance_id}] action step={step_now} src={action_source} "
                            f"label={action_label} stuck={bool(stuck)} cont={cont_preview} disc={disc_preview}{disc_suffix}",
                            flush=True,
                        )
                self._record_flight_frame(
                    action_label=action_label,
                    input_vector=input_vec,
                    model_entropy=entropy,
                )
            except Exception:
                pass

            # Optional causal Scientist updates for item/build discovery.
            if self._use_causal_scientist and self._causal_graph is not None:
                try:
                    import math
                    from src.learner.causal_rl import Observation, Intervention

                    pv = game_state.get("playerVelocity") or (0.0, 0.0, 0.0)
                    try:
                        vx, vy, vz = pv
                    except Exception:
                        vx = vy = vz = 0.0
                    speed = float(math.sqrt(vx * vx + vy * vy + vz * vz))
                    health = float(game_state.get("playerHealth") or 0.0)
                    max_health = float(game_state.get("playerMaxHealth") or 1.0)
                    health_ratio = health / max(max_health, 1e-6)
                    gtime = float(game_state.get("gameTime") or float(self.trainer.step_count))
                    stats_now = {
                        "speed": speed,
                        "health": health,
                        "health_ratio": health_ratio,
                    }
                    run_id = os.environ.get("METABONK_RUN_ID", "run-local")

                    horizon_s = float(os.environ.get("METABONK_CAUSAL_HORIZON_S", "5.0"))
                    delta_thresh = float(os.environ.get("METABONK_CAUSAL_DELTA_THRESH", "0.05"))

                    pending = self._pending_intervention
                    if pending is not None and (gtime - pending.get("t0", gtime)) >= horizon_s:
                        opt_name = str(pending.get("option", ""))
                        baseline = pending.get("baseline") or {}
                        obs_after = Observation(
                            items=[opt_name],
                            stats=stats_now,
                            outcome=0.0,
                            timestamp=gtime,
                            run_id=run_id,
                        )
                        self._causal_graph.record_intervention(
                            Intervention(opt_name, 1.0), obs_after
                        )
                        new_edges = []
                        for k, v0 in baseline.items():
                            v1 = stats_now.get(k, v0)
                            if abs(float(v1) - float(v0)) >= delta_thresh:
                                self._causal_graph.add_edge(opt_name, k)
                                new_edges.append((opt_name, k, float(v1) - float(v0)))
                        if new_edges and requests:
                            try:
                                from src.common.observability import Event

                                ev = Event(
                                    event_id="",
                                    run_id=run_id,
                                    instance_id=self.instance_id,
                                    event_type="Eureka",
                                    message=f"Causal link discovered from {opt_name}",
                                    payload={"edges": new_edges},
                                )
                                requests.post(
                                    f"{self.orch_url}/events",
                                    json=ev.model_dump(),
                                    timeout=1.0,
                                )
                            except Exception:
                                pass
                        self._pending_intervention = None

                    options = game_state.get("levelUpOptions") or []
                    if (
                        not self._pending_intervention
                        and options
                        and frame_size is not None
                        and (forced_ui_click is not None or (not suppress_policy_clicks and a_disc))
                    ):
                        selected_option = None
                        try:
                            from .perception import (
                                parse_detections,
                                sort_by_priority,
                                CARD_OFFER,
                            )

                            dets_sorted = sort_by_priority(parse_detections(detections))
                            dets_used = dets_sorted[:32]
                            chosen_i = None
                            if forced_ui_click is not None:
                                cx, cy = forced_ui_click
                                best_d = 1e18
                                best_i = None
                                for i, d in enumerate(dets_used):
                                    dd = (d.cx - cx) ** 2 + (d.cy - cy) ** 2
                                    if dd < best_d:
                                        best_d = dd
                                        best_i = i
                                chosen_i = best_i
                            else:
                                idx = int(a_disc[0]) if a_disc else None
                                if idx is not None and 0 <= idx < len(dets_used):
                                    chosen_i = idx
                            if (
                                chosen_i is not None
                                and chosen_i < len(dets_used)
                                and dets_used[chosen_i].cls == CARD_OFFER
                            ):
                                card_dets = [
                                    (i, d) for i, d in enumerate(dets_used) if d.cls == CARD_OFFER
                                ]
                                card_sorted = sorted(card_dets, key=lambda t: t[1].x1)
                                ranks = [i for i, _ in card_sorted]
                                rank = ranks.index(chosen_i) if chosen_i in ranks else 0
                                if rank >= len(options):
                                    rank = 0
                                selected_option = str(options[rank])
                        except Exception:
                            selected_option = None

                        if selected_option:
                            self._pending_intervention = {
                                "option": selected_option,
                                "t0": gtime,
                                "baseline": stats_now,
                            }
                            self._causal_graph.record_observation(
                                Observation(
                                    items=[selected_option],
                                    stats=stats_now,
                                    outcome=0.0,
                                    timestamp=gtime,
                                    run_id=run_id,
                                )
                            )
                except Exception:
                    pass

            # If ResearchPlugin SHM, send 6-float action (movement + 4 button slots).
            if self._use_research_shm and self._research_shm is not None and not input_bootstrap:
                try:
                    btns = [0.0, 0.0, 0.0, 0.0]
                    # Current action space doesn't align with ResearchPlugin buttons;
                    # we conservatively keep them off unless discrete outputs are binary.
                    if len(a_disc) >= 4 and all(x in (0, 1) for x in a_disc[:4]):
                        btns = [float(x) for x in a_disc[:4]]
                    action6 = (
                        float(a_cont[0]) if len(a_cont) > 0 else 0.0,
                        float(a_cont[1]) if len(a_cont) > 1 else 0.0,
                        btns[0], btns[1], btns[2], btns[3],
                    )
                    self._research_shm.write_action(action6)
                except Exception:
                    pass

            # If BonkLink, send continuous movement + optional UI click.
            if self._use_bonklink and self._bonklink is not None and self._input_backend is None and not input_bootstrap:
                try:
                    from src.bridge.bonklink_coords import map_click_top_left_to_bonklink
                    from src.bridge.bonklink_client import BonkLinkAction
                    from .perception import parse_detections, build_ui_elements

                    move_x = float(a_cont[0]) if len(a_cont) > 0 else 0.0
                    move_y = float(a_cont[1]) if len(a_cont) > 1 else 0.0
                    action = BonkLinkAction(move_x=move_x, move_y=move_y)

                    if forced_ui_click is not None:
                        action.ui_click = True
                        try:
                            cap = getattr(self, "_bonklink_capture_size", None)
                        except Exception:
                            cap = None
                        click_x_tl = int(forced_ui_click[0])
                        click_y_tl = int(forced_ui_click[1])
                        click_x, click_y = map_click_top_left_to_bonklink(
                            x_top_left=click_x_tl,
                            y_top_left=click_y_tl,
                            frame_size=frame_size,
                            capture_size=cap,
                        )
                        action.click_x = int(click_x)
                        action.click_y = int(click_y)
                        if os.environ.get("METABONK_LOG_UI_CLICKS", "0") in ("1", "true", "True"):
                            print(
                                f"[worker:{self.instance_id}] ui_click forced x={action.click_x} y={action.click_y}"
                            )
                    elif (not suppress_policy_clicks) and a_disc and frame_size is not None:
                        if ui_elements_cache is not None and action_mask_cache is not None:
                            ui_elements = ui_elements_cache
                            action_mask2 = action_mask_cache
                        else:
                            dets_parsed = parse_detections(detections)
                            ui_elements, action_mask2, _ = build_ui_elements(
                                dets_parsed, frame_size=frame_size
                            )
                        idx = int(a_disc[0])
                        allow_repeat = False
                        try:
                            if os.environ.get("METABONK_UI_ALLOW_REPEAT_CLICK", "0") in ("1", "true", "True"):
                                allow_repeat = True
                        except Exception:
                            allow_repeat = False
                        if (not allow_repeat) and bool(getattr(self, "_pure_vision_mode", False)) and bool(stuck):
                            # Pure-vision default: allow repeated UI clicks while stuck, but
                            # throttle them to avoid spamming the target app.
                            allow_repeat = True
                        repeat_s = 0.4
                        try:
                            repeat_s = float(os.environ.get("METABONK_UI_CLICK_REPEAT_S", "0.4") or 0.4)
                        except Exception:
                            repeat_s = 0.4
                        repeat_s = max(0.0, float(repeat_s))
                        last_click_ts = float(getattr(self, "_last_ui_click_ts", 0.0) or 0.0)
                        if (
                            (allow_repeat or idx != self._last_disc_action)
                            and 0 <= idx < len(ui_elements)
                            and action_mask2[idx] == 1
                        ):
                            should_click = True
                            if allow_repeat and idx == self._last_disc_action and repeat_s > 0.0:
                                if (now - last_click_ts) < repeat_s:
                                    # Too soon to repeat the same click.
                                    should_click = False
                            if should_click:
                                w, h = frame_size
                                cx, cy = ui_elements[idx][0], ui_elements[idx][1]
                                ew, eh = ui_elements[idx][2], ui_elements[idx][3]
                                cls = ui_elements[idx][4]
                                action.ui_click = True
                                try:
                                    cap = getattr(self, "_bonklink_capture_size", None)
                                except Exception:
                                    cap = None
                                click_x_tl = int(cx * w)
                                click_y_tl = int(cy * h)
                                # Coarse candidates (e.g., saliency/grid cells) benefit from
                                # sampling inside the candidate box so clicks can land on
                                # smaller buttons without hardcoding semantics.
                                try:
                                    jitter_raw = str(os.environ.get("METABONK_UI_CLICK_JITTER", "") or "").strip()
                                    if not jitter_raw:
                                        # Back-compat (deprecated)
                                        jitter_raw = str(os.environ.get("METABONK_UI_GRID_JITTER", "0") or "0").strip()
                                    jitter = float(jitter_raw or 0.0)
                                    if jitter <= 0.0 and bool(getattr(self, "_pure_vision_mode", False)) and bool(stuck):
                                        # Pure-vision default: coarse saliency/grid candidates (cls<0)
                                        # benefit from sampling across the entire candidate region to
                                        # hit smaller buttons without encoding game/UI semantics.
                                        jitter = 1.0 if float(cls) < 0 else 0.6
                                    jitter = max(0.0, min(1.0, jitter))
                                    if jitter > 0.0:
                                        # Only jitter for candidates with large boxes (normalized).
                                        if float(ew) >= 0.10 or float(eh) >= 0.10:
                                            dx = (random.random() - 0.5) * float(ew) * float(w) * jitter
                                            dy = (random.random() - 0.5) * float(eh) * float(h) * jitter
                                            click_x_tl = int((float(cx) * float(w)) + dx)
                                            click_y_tl = int((float(cy) * float(h)) + dy)
                                except Exception:
                                    pass
                                click_x, click_y = map_click_top_left_to_bonklink(
                                    x_top_left=click_x_tl,
                                    y_top_left=click_y_tl,
                                    frame_size=frame_size,
                                    capture_size=cap,
                                )
                                action.click_x = int(click_x)
                                action.click_y = int(click_y)
                                if os.environ.get("METABONK_LOG_UI_CLICKS", "0") in ("1", "true", "True"):
                                    print(
                                        f"[worker:{self.instance_id}] ui_click idx={idx} x={action.click_x} y={action.click_y}"
                                    )
                                self._last_ui_click_ts = float(now)
                            else:
                                if os.environ.get("METABONK_LOG_UI_CLICKS", "0") in ("1", "true", "True"):
                                    try:
                                        dt = float(now - last_click_ts)
                                    except Exception:
                                        dt = 0.0
                                    print(
                                        f"[worker:{self.instance_id}] ui_click throttled idx={idx} dt={dt:.3f}s < {repeat_s:.3f}s"
                                    )
                        self._last_disc_action = idx

                    self._bonklink.send_action(action)
                except Exception:
                    pass

            # If bridged, send actions into the game.
            if self._use_bridge and self.bridge and self._bridge_loop and not self._use_bonklink and self._input_backend is None and not input_bootstrap:
                try:
                    if ui_elements_cache is not None:
                        ui_elements = ui_elements_cache
                        action_mask_bridge = action_mask_cache
                    else:
                        from .perception import parse_detections, build_ui_elements

                        dets_parsed = parse_detections(detections)
                        ui_elements, action_mask_bridge, _ = build_ui_elements(
                            dets_parsed, frame_size=frame_size
                        )
                except Exception:
                    ui_elements = None
                    action_mask_bridge = None
                self._bridge_send_actions(
                    a_cont,
                    a_disc,
                    forced_ui_click,
                    ui_elements,
                    action_mask_bridge if action_mask_bridge is not None else action_mask,
                    frame_size,
                )

            # Reward: prefer learned reward-from-video (progress score delta).
            # In pure-vision mode we explicitly disable all extrinsic reward sources
            # and rely on intrinsic reward shaping in the learner.
            if self._pure_vision_mode:
                try:
                    reward = float(self._visual_exploration.last_reward) if self._visual_exploration is not None else 0.0
                except Exception:
                    reward = 0.0
            elif self._use_learned_reward:
                if reward_frame_hwc is None:
                    # Grace window to allow the game bridge / capture pipeline to warm up.
                    # Without this, the worker can crash before the first JPEG/capture arrives.
                    if time.time() - loop_t0 < max(0.0, learned_reward_grace_s):
                        self._stop.wait(0.05)
                        continue
                    if not self._warned_no_reward_frame:
                        self._warned_no_reward_frame = True
                        print(
                            "[worker] WARN: learned reward enabled but no RGB frame available; "
                            "falling back to 0 reward until frames arrive. "
                            "(check BonkLink/PipeWire capture)"
                        )
                    if reward_from_game is not None:
                        reward = float(reward_from_game)
                    else:
                        reward = 0.0
                else:
                    reward = float(self._learned_reward(reward_frame_hwc))
            elif reward_from_game is not None:
                reward = float(reward_from_game)
            else:
                raise RuntimeError(
                    "No reward available (game reward missing and learned reward disabled). "
                    "Enable learned reward (METABONK_USE_LEARNED_REWARD=1) or provide a reward-capable bridge."
                )

            reward_hit = False
            try:
                if self._visual_exploration is not None:
                    reward_hit = bool(
                        bool(getattr(self._visual_exploration, "last_new_scene", False))
                        or bool(getattr(self._visual_exploration, "last_transition", False))
                    )
            except Exception:
                reward_hit = False

            # Optional reward logging.
            if self._reward_log and abs(float(reward)) > 1e-9:
                try:
                    playing_flag = bool(game_state.get("isPlaying"))
                except Exception:
                    playing_flag = False
                scene_hash = None
                try:
                    if self._visual_exploration is not None:
                        scene_hash = self._visual_exploration.last_scene_hash
                except Exception:
                    scene_hash = None
                print(
                    f"[REWARD] instance={self.instance_id} reward={float(reward):.4f} "
                    f"scene_hash={scene_hash} playing={playing_flag} stuck={bool(stuck)}",
                    flush=True,
                )
            # Dump a frame on reward hit (for manual verification).
            if reward_hit and (not self._reward_hit_once or not self._reward_hit_saved):
                try:
                    base = Path(self._reward_hit_frame_path)
                    if base.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        out_path = base
                    else:
                        base.mkdir(parents=True, exist_ok=True)
                        out_path = base / f"{self.instance_id}_reward_hit.jpg"
                    strip_path = None
                    if out_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        strip_path = out_path.with_name(f"{out_path.stem}_strip.jpg")
                    elif base:
                        strip_path = base / f"{self.instance_id}_reward_hit_strip.jpg"
                    chosen = None
                    chosen_stats = None
                    chosen_source = None
                    chosen_idx = None
                    if self._frame_ring_enabled and self._frame_ring:
                        for offset, entry in enumerate(reversed(self._frame_ring)):
                            stats = entry.get("stats")
                            if not self._frame_is_black(stats):
                                chosen = entry
                                chosen_stats = stats
                                chosen_source = str(entry.get("source") or "ring")
                                chosen_idx = -(offset + 1)
                                break
                    if chosen is None and self._last_valid_frame is not None:
                        chosen = self._last_valid_frame
                        chosen_stats = chosen.get("stats")
                        chosen_source = str(chosen.get("source") or "last_valid")
                    if chosen is None and latest_image_bytes:
                        stats = self._frame_stats_from_bytes(latest_image_bytes)
                        if not self._frame_is_black(stats):
                            chosen = {
                                "bytes": latest_image_bytes,
                                "stats": stats,
                                "source": latest_image_source or "current",
                            }
                            chosen_stats = stats
                            chosen_source = str(latest_image_source or "current")
                    if chosen is None and self.streamer is not None and hasattr(self.streamer, "capture_jpeg"):
                        try:
                            snap = self.streamer.capture_jpeg(
                                timeout_s=float(os.environ.get("METABONK_FRAME_JPEG_TIMEOUT_S", "1.5"))
                            )
                        except Exception:
                            snap = None
                        if snap:
                            stats = self._frame_stats_from_bytes(snap)
                            if not self._frame_is_black(stats):
                                chosen = {"bytes": snap, "stats": stats, "source": "streamer_snapshot"}
                                chosen_stats = stats
                                chosen_source = "streamer_snapshot"
                    if chosen and chosen.get("bytes"):
                        out_path.write_bytes(chosen["bytes"])
                        self._reward_hit_saved = True
                        stat_msg = ""
                        if chosen_stats:
                            mean = float(chosen_stats[0])
                            p99 = float(chosen_stats[1])
                            mean_s = float(chosen_stats[2]) if len(chosen_stats) > 2 else 0.0
                            stat_msg = f" mean={mean:.1f} p99={p99:.1f} sat={mean_s:.1f}"
                        idx_msg = f" idx={chosen_idx}" if chosen_idx is not None else ""
                        src_msg = f" source={chosen_source}" if chosen_source else ""
                        print(f"[REWARD_HIT] saved={out_path}{src_msg}{idx_msg}{stat_msg}", flush=True)
                    elif not self._reward_hit_saved:
                        print("[REWARD_HIT] WARN: no non-black frame available", flush=True)
                    if strip_path and self._frame_ring:
                        try:
                            from PIL import Image
                            import io
                            import math

                            frames = list(self._frame_ring)[-16:]
                            images = []
                            for entry in frames:
                                payload = entry.get("bytes")
                                if not payload:
                                    continue
                                try:
                                    img = Image.open(io.BytesIO(payload)).convert("RGB")
                                    images.append(img)
                                except Exception:
                                    continue
                            if images:
                                cols = 4
                                rows = int(math.ceil(len(images) / float(cols)))
                                tile_w, tile_h = images[0].size
                                sheet = Image.new("RGB", (tile_w * cols, tile_h * rows), color=(0, 0, 0))
                                for idx, img in enumerate(images):
                                    r = idx // cols
                                    c = idx % cols
                                    if img.size != (tile_w, tile_h):
                                        img = img.resize((tile_w, tile_h))
                                    sheet.paste(img, (c * tile_w, r * tile_h))
                                sheet.save(strip_path, format="JPEG", quality=85)
                                print(f"[REWARD_HIT] strip={strip_path} frames={len(images)}", flush=True)
                        except Exception:
                            pass
                except Exception:
                    pass
            # Dump a frame on first gameplay (HUD detection).
            if gameplay_rise and (not self._gameplay_hit_once or not self._gameplay_hit_saved):
                try:
                    base = Path(self._gameplay_hit_frame_path)
                    if base.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        out_path = base
                    else:
                        base.mkdir(parents=True, exist_ok=True)
                        out_path = base / f"{self.instance_id}_gameplay_hit.jpg"
                    strip_path = None
                    if out_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        strip_path = out_path.with_name(f"{out_path.stem}_strip.jpg")
                    elif base:
                        strip_path = base / f"{self.instance_id}_gameplay_strip.jpg"
                    chosen = None
                    chosen_stats = None
                    chosen_source = None
                    chosen_idx = None
                    if latest_image_bytes:
                        stats = self._frame_stats_from_bytes(latest_image_bytes)
                        if not self._frame_is_black(stats):
                            chosen = {
                                "bytes": latest_image_bytes,
                                "stats": stats,
                                "source": latest_image_source or "current",
                            }
                            chosen_stats = stats
                            chosen_source = str(latest_image_source or "current")
                    if chosen is None and self._frame_ring:
                        for offset, entry in enumerate(reversed(self._frame_ring)):
                            stats = entry.get("stats")
                            if not self._frame_is_black(stats):
                                chosen = entry
                                chosen_stats = stats
                                chosen_source = str(entry.get("source") or "ring")
                                chosen_idx = -(offset + 1)
                                break
                    if chosen is None and self._last_valid_frame is not None:
                        chosen = self._last_valid_frame
                        chosen_stats = chosen.get("stats")
                        chosen_source = str(chosen.get("source") or "last_valid")
                    if chosen is None and self.streamer is not None and hasattr(self.streamer, "capture_jpeg"):
                        try:
                            snap = self.streamer.capture_jpeg(
                                timeout_s=float(os.environ.get("METABONK_FRAME_JPEG_TIMEOUT_S", "1.5"))
                            )
                        except Exception:
                            snap = None
                        if snap:
                            stats = self._frame_stats_from_bytes(snap)
                            if not self._frame_is_black(stats):
                                chosen = {"bytes": snap, "stats": stats, "source": "streamer_snapshot"}
                                chosen_stats = stats
                                chosen_source = "streamer_snapshot"
                    if chosen and chosen.get("bytes"):
                        out_path.write_bytes(chosen["bytes"])
                        self._gameplay_hit_saved = True
                        stat_msg = ""
                        if chosen_stats:
                            mean = float(chosen_stats[0])
                            p99 = float(chosen_stats[1])
                            mean_s = float(chosen_stats[2]) if len(chosen_stats) > 2 else 0.0
                            stat_msg = f" mean={mean:.1f} p99={p99:.1f} sat={mean_s:.1f}"
                        idx_msg = f" idx={chosen_idx}" if chosen_idx is not None else ""
                        src_msg = f" source={chosen_source}" if chosen_source else ""
                        print(f"[GAMEPLAY_HIT] saved={out_path}{src_msg}{idx_msg}{stat_msg}", flush=True)
                    elif not self._gameplay_hit_saved:
                        print("[GAMEPLAY_HIT] WARN: no non-black frame available", flush=True)
                    if strip_path and self._frame_ring:
                        try:
                            from PIL import Image
                            import io
                            import math

                            n_frames = max(1, int(self._gameplay_strip_n))
                            frames = list(self._frame_ring)[-n_frames:]
                            images = []
                            for entry in frames:
                                payload = entry.get("bytes")
                                if not payload:
                                    continue
                                try:
                                    img = Image.open(io.BytesIO(payload)).convert("RGB")
                                    images.append(img)
                                except Exception:
                                    continue
                            if images:
                                cols = 4
                                rows = int(math.ceil(len(images) / float(cols)))
                                tile_w, tile_h = images[0].size
                                sheet = Image.new("RGB", (tile_w * cols, tile_h * rows), color=(0, 0, 0))
                                for idx, img in enumerate(images):
                                    r = idx // cols
                                    c = idx % cols
                                    if img.size != (tile_w, tile_h):
                                        img = img.resize((tile_w, tile_h))
                                    sheet.paste(img, (c * tile_w, r * tile_h))
                                sheet.save(strip_path, format="JPEG", quality=85)
                                print(f"[GAMEPLAY_HIT] strip={strip_path} frames={len(images)}", flush=True)
                        except Exception:
                            pass
                except Exception:
                    pass
            # Episode done: when METABONK_VISUAL_ONLY=1, never use SHM/memory flags.
            # If the vision model exports a boolean done signal, consume it here.
            if self._visual_only and isinstance(vision_metrics, dict):
                try:
                    if "episode_done" in vision_metrics:
                        done_from_game = bool(vision_metrics.get("episode_done"))
                    elif "done" in vision_metrics:
                        done_from_game = bool(vision_metrics.get("done"))
                    elif "is_dead" in vision_metrics:
                        done_from_game = bool(vision_metrics.get("is_dead"))
                except Exception:
                    pass
            done = bool(done_from_game)
            # System 2 RL outcome tracking (append-only JSONL).
            if self._cognitive_client is not None and self._system2_active_decision_id is not None:
                try:
                    self._system2_active_reward_sum += float(reward)
                except Exception:
                    pass
                expired = False
                try:
                    expired = self._cognitive_client.get_current_directive() is None
                except Exception:
                    expired = False
                if done or expired:
                    self._system2_maybe_log_outcome(
                        now=now,
                        reason="done" if done else "expired",
                        done=bool(done),
                    )
            if done:
                try:
                    self.trainer.reset_state()
                except Exception:
                    pass
                self._obs_stack.clear()

            # Episode lifecycle + telemetry (real data only).
            if requests:
                try:
                    from src.common.observability import Event

                    # Telemetry cadence (keep light; used for survival model / choke meter).
                    telem_s = float(os.environ.get("METABONK_TELEMETRY_S", "0.5"))
                    if now - self._last_telemetry_ts >= max(0.1, telem_s):
                        health_ratio = None
                        enemy_count = None
                        player_pos = None
                        crit_chance = None
                        overcrit_tier = None
                        last_hit_damage = None
                        last_hit_is_crit = None
                        last_loot_rarity = None
                        last_loot_item = None
                        last_loot_source = None
                        last_chest_rarity = None
                        last_heal_amount = None
                        last_heal_item = None
                        last_heal_is_borgar = None
                        incoming_dps = None
                        clearing_dps = None
                        stage = None
                        biome = None
                        charge_level = None
                        charge_is_max = None
                        inventory_items = None
                        synergy_edges = None
                        evolution_recipes = None
                        # Prefer vision-derived metrics. If METABONK_VISUAL_ONLY=1,
                        # do not consume any game_state-derived values.
                        boss_visible = False
                        try:
                            v = vision_metrics.get("health_ratio")
                            if v is not None:
                                health_ratio = float(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("enemy_count")
                            if v is not None:
                                enemy_count = int(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("player_pos")
                            if isinstance(v, (list, tuple)) and len(v) >= 3:
                                player_pos = [float(v[0]), float(v[1]), float(v[2])]
                        except Exception:
                            pass
                        try:
                            from .perception import parse_detections, ICON_BOSS, MINIMAP_ICON_RED

                            dets_parsed = parse_detections(detections)
                            boss_visible = any(d.cls == ICON_BOSS for d in dets_parsed)
                            # If no explicit enemy_count, use a conservative visual proxy from minimap danger icons.
                            if enemy_count is None:
                                enemy_count = int(sum(1 for d in dets_parsed if d.cls in (MINIMAP_ICON_RED, ICON_BOSS)))
                        except Exception:
                            pass
                        if not self._visual_only:
                            try:
                                if health_ratio is None:
                                    hp = float(game_state.get("playerHealth") or 0.0)
                                    mhp = float(game_state.get("playerMaxHealth") or 1.0)
                                    health_ratio = float(hp / max(mhp, 1e-6))
                            except Exception:
                                pass
                            try:
                                if enemy_count is None:
                                    enemies = game_state.get("enemies") or []
                                    if isinstance(enemies, list):
                                        enemy_count = int(len(enemies))
                            except Exception:
                                pass
                            try:
                                if player_pos is None:
                                    pp = game_state.get("playerPosition")
                                    if isinstance(pp, (list, tuple)) and len(pp) >= 3:
                                        player_pos = [float(pp[0]), float(pp[1]), float(pp[2])]
                            except Exception:
                                pass
                        # Optional "juicy" combat telemetry (only if the game/plugin provides it).
                        # We do not synthesize these values.
                        # Prefer vision-derived metrics when present.
                        try:
                            v = vision_metrics.get("crit_chance")
                            if v is not None:
                                crit_chance = float(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("overcrit_tier")
                            if v is not None:
                                overcrit_tier = int(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("last_hit_damage")
                            if v is not None:
                                last_hit_damage = float(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("last_hit_is_crit")
                            if v is not None:
                                last_hit_is_crit = bool(v)
                        except Exception:
                            pass
                        if not self._visual_only:
                            try:
                                for k in ("critChance", "criticalChance", "crit_chance", "playerCritChance"):
                                    if k in game_state and game_state.get(k) is not None:
                                        crit_chance = float(game_state.get(k))
                                        break
                            except Exception:
                                pass
                            try:
                                for k in ("overcritTier", "critTier", "overcrit_tier"):
                                    if k in game_state and game_state.get(k) is not None:
                                        overcrit_tier = int(game_state.get(k))
                                        break
                            except Exception:
                                pass
                            try:
                                for k in ("lastHitDamage", "last_damage", "lastDamage", "last_hit_damage"):
                                    if k in game_state and game_state.get(k) is not None:
                                        last_hit_damage = float(game_state.get(k))
                                        break
                            except Exception:
                                pass
                            try:
                                for k in ("lastHitWasCrit", "last_hit_is_crit", "lastHitCrit", "lastCrit"):
                                    if k in game_state and game_state.get(k) is not None:
                                        last_hit_is_crit = bool(game_state.get(k))
                                        break
                            except Exception:
                                pass
                        try:
                            if overcrit_tier is None and crit_chance is not None:
                                # Same tier definition as spectator/fun_metrics.py.
                                if crit_chance < 100:
                                    overcrit_tier = 0
                                elif crit_chance < 200:
                                    overcrit_tier = 1
                                elif crit_chance < 300:
                                    overcrit_tier = 2
                                else:
                                    overcrit_tier = 3
                        except Exception:
                            pass

                        # Swarm / DPS pressure telemetry (optional; must come from real game/plugin state).
                        try:
                            v = vision_metrics.get("incoming_dps")
                            if v is not None:
                                incoming_dps = float(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("clearing_dps")
                            if v is not None:
                                clearing_dps = float(v)
                        except Exception:
                            pass
                        if not self._visual_only:
                            try:
                                for k in (
                                    "incomingDps",
                                    "incoming_dps",
                                    "enemyPotentialDps",
                                    "enemy_potential_dps",
                                    "totalEnemyDps",
                                    "total_enemy_dps",
                                ):
                                    if k in game_state and game_state.get(k) is not None:
                                        incoming_dps = float(game_state.get(k))
                                        break
                            except Exception:
                                pass

                        # Build/inventory telemetry (vision-only). We do not synthesize.
                        try:
                            v = vision_metrics.get("inventory_items")
                            if isinstance(v, list):
                                inventory_items = v
                        except Exception:
                            inventory_items = None
                        if isinstance(inventory_items, list):
                            self._last_inventory_items = list(inventory_items)
                        try:
                            v = vision_metrics.get("synergy_edges")
                            if isinstance(v, list):
                                synergy_edges = v
                        except Exception:
                            synergy_edges = None
                        try:
                            v = vision_metrics.get("evolution_recipes")
                            if isinstance(v, list):
                                evolution_recipes = v
                        except Exception:
                            evolution_recipes = None
                            try:
                                for k in (
                                    "clearingDps",
                                    "clearing_dps",
                                    "damageDealtDps",
                                    "damage_dealt_dps",
                                    "dps",
                                    "playerDps",
                                    "player_dps",
                                    "damage_dealt",
                                ):
                                    if k in game_state and game_state.get(k) is not None:
                                        clearing_dps = float(game_state.get(k))
                                        break
                            except Exception:
                                pass

                        # Loot telemetry (optional; must come from real game/plugin state).
                        try:
                            v = vision_metrics.get("last_loot_rarity")
                            if v is not None:
                                last_loot_rarity = str(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("last_loot_item")
                            if v is not None:
                                last_loot_item = str(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("last_loot_source")
                            if v is not None:
                                last_loot_source = str(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("last_chest_rarity")
                            if v is not None:
                                last_chest_rarity = str(v)
                        except Exception:
                            pass
                        if not self._visual_only:
                            try:
                                for k in ("lastLootRarity", "lootRarity", "last_loot_rarity", "lastChestRarity"):
                                    if k in game_state and game_state.get(k) is not None:
                                        last_loot_rarity = str(game_state.get(k))
                                        break
                            except Exception:
                                pass
                            try:
                                for k in ("lastLootItem", "lootItemName", "last_loot_item", "lastChestItem"):
                                    if k in game_state and game_state.get(k) is not None:
                                        last_loot_item = str(game_state.get(k))
                                        break
                            except Exception:
                                pass
                            try:
                                for k in ("lastLootSource", "lootSource", "last_loot_source"):
                                    if k in game_state and game_state.get(k) is not None:
                                        last_loot_source = str(game_state.get(k))
                                        break
                            except Exception:
                                pass
                            try:
                                for k in ("lastChestRarity", "chestRarity", "last_chest_rarity"):
                                    if k in game_state and game_state.get(k) is not None:
                                        last_chest_rarity = str(game_state.get(k))
                                        break
                            except Exception:
                                pass

                        # Progression gates / biome (must come from vision in visual-only mode).
                        try:
                            v = vision_metrics.get("stage")
                            if v is not None:
                                stage = int(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("biome")
                            if v is not None:
                                biome = str(v)
                        except Exception:
                            pass
                        if not self._visual_only:
                            try:
                                if stage is None and game_state.get("stage") is not None:
                                    stage = int(game_state.get("stage"))
                            except Exception:
                                pass
                            try:
                                if biome is None and game_state.get("biome") is not None:
                                    biome = str(game_state.get("biome"))
                            except Exception:
                                pass

                        # Cache for EpisodeEnd payloads.
                        try:
                            if stage is not None:
                                self._last_stage = int(stage)
                        except Exception:
                            pass
                        try:
                            if biome:
                                self._last_biome = str(biome)
                        except Exception:
                            pass

                        # Charge shrine (economy/power) telemetry.
                        try:
                            v = vision_metrics.get("charge_level")
                            if v is not None:
                                charge_level = float(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("charge_is_max")
                            if v is not None:
                                charge_is_max = bool(v)
                        except Exception:
                            pass
                        if not self._visual_only:
                            try:
                                if charge_level is None and game_state.get("chargeLevel") is not None:
                                    charge_level = float(game_state.get("chargeLevel"))
                            except Exception:
                                pass
                            try:
                                if charge_is_max is None and game_state.get("chargeIsMax") is not None:
                                    charge_is_max = bool(game_state.get("chargeIsMax"))
                            except Exception:
                                pass

                        # Optional end reason (vision-derived).
                        try:
                            v = vision_metrics.get("end_reason")
                            if v is not None:
                                self._last_end_reason = str(v)
                        except Exception:
                            pass

                        # Heal telemetry / Borgar (optional; must come from real game/plugin state).
                        try:
                            v = vision_metrics.get("last_heal_amount")
                            if v is not None:
                                last_heal_amount = float(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("last_heal_item")
                            if v is not None:
                                last_heal_item = str(v)
                        except Exception:
                            pass
                        try:
                            v = vision_metrics.get("last_heal_is_borgar")
                            if v is not None:
                                last_heal_is_borgar = bool(v)
                        except Exception:
                            pass
                        if not self._visual_only:
                            try:
                                for k in ("lastHealAmount", "healAmount", "last_heal_amount"):
                                    if k in game_state and game_state.get(k) is not None:
                                        last_heal_amount = float(game_state.get(k))
                                        break
                            except Exception:
                                pass
                            try:
                                for k in ("lastHealItem", "healItemName", "last_heal_item"):
                                    if k in game_state and game_state.get(k) is not None:
                                        last_heal_item = str(game_state.get(k))
                                        break
                            except Exception:
                                pass
                            try:
                                for k in ("lastHealWasBorgar", "isBorgarHeal", "last_heal_is_borgar"):
                                    if k in game_state and game_state.get(k) is not None:
                                        last_heal_is_borgar = bool(game_state.get(k))
                                        break
                            except Exception:
                                pass
                        try:
                            if last_heal_is_borgar is None and last_heal_item:
                                nm = str(last_heal_item).strip().lower()
                                if "borgar" in nm or "burger" in nm or "hamburger" in nm:
                                    last_heal_is_borgar = True
                        except Exception:
                            pass

                        # Fire "Overcrit" / "NewMaxHit" events when real telemetry indicates it.
                        # These drive impact frames + "Big Number" UI animations.
                        try:
                            hit_sig = None
                            if last_hit_damage is not None:
                                # Best-effort dedupe: (damage, crit, tier) at current step.
                                hit_sig = (
                                    float(last_hit_damage),
                                    bool(last_hit_is_crit) if last_hit_is_crit is not None else None,
                                    int(overcrit_tier) if overcrit_tier is not None else None,
                                    int(self.trainer.step_count),
                                )
                            if hit_sig is not None and hit_sig != self._last_hit_sig:
                                self._last_hit_sig = hit_sig
                                if (
                                    last_hit_damage is not None
                                    and last_hit_is_crit is True
                                    and (overcrit_tier or 0) > 0
                                ):
                                    ev_over = Event(
                                        event_id="",
                                        run_id=os.environ.get("METABONK_RUN_ID", "run-local"),
                                        instance_id=self.instance_id,
                                        event_type="Overcrit",
                                        message=f"OVERCRIT! {last_hit_damage:,.0f}",
                                        payload={
                                            "ts": now,
                                            "episode_idx": self._episode_idx,
                                            "damage": float(last_hit_damage),
                                            "crit_chance": float(crit_chance) if crit_chance is not None else None,
                                            "overcrit_tier": int(overcrit_tier) if overcrit_tier is not None else None,
                                        },
                                    )
                                    requests.post(f"{self.orch_url}/events", json=ev_over.model_dump(), timeout=1.0)
                                if last_hit_damage is not None and float(last_hit_damage) > self._max_hit_local + 1e-6:
                                    self._max_hit_local = float(last_hit_damage)
                                    ev_max = Event(
                                        event_id="",
                                        run_id=os.environ.get("METABONK_RUN_ID", "run-local"),
                                        instance_id=self.instance_id,
                                        event_type="NewMaxHit",
                                        message=f"NEW BIG NUMBER {self._max_hit_local:,.0f}",
                                        payload={
                                            "ts": now,
                                            "episode_idx": self._episode_idx,
                                            "max_hit": float(self._max_hit_local),
                                            "damage": float(last_hit_damage),
                                            "crit_chance": float(crit_chance) if crit_chance is not None else None,
                                            "overcrit_tier": int(overcrit_tier) if overcrit_tier is not None else None,
                                        },
                                    )
                                    requests.post(f"{self.orch_url}/events", json=ev_max.model_dump(), timeout=1.0)
                        except Exception:
                            pass

                        # Fire LootDrop + Heal events when real telemetry indicates it.
                        try:
                            # Dedupe per step.
                            if last_loot_rarity:
                                loot_sig = (str(last_loot_rarity), str(last_loot_item or ""), int(self.trainer.step_count))
                                if loot_sig != self._last_loot_sig:
                                    self._last_loot_sig = loot_sig
                                    ev_loot = Event(
                                        event_id="",
                                        run_id=os.environ.get("METABONK_RUN_ID", "run-local"),
                                        instance_id=self.instance_id,
                                        event_type="LootDrop",
                                        message=f"loot {str(last_loot_rarity).upper()} {str(last_loot_item or '').strip()}",
                                        payload={
                                            "ts": now,
                                            "episode_idx": self._episode_idx,
                                            "episode_t": float(now - self._episode_start_ts),
                                            "rarity": str(last_loot_rarity),
                                            "item_name": str(last_loot_item) if last_loot_item else None,
                                            "source": str(last_loot_source) if last_loot_source else None,
                                            "chest_rarity": str(last_chest_rarity) if last_chest_rarity else None,
                                        },
                                    )
                                    requests.post(f"{self.orch_url}/events", json=ev_loot.model_dump(), timeout=1.0)
                            if last_heal_amount is not None and (last_heal_item or last_heal_is_borgar is not None):
                                heal_sig = (
                                    float(last_heal_amount),
                                    str(last_heal_item or ""),
                                    bool(last_heal_is_borgar) if last_heal_is_borgar is not None else None,
                                    int(self.trainer.step_count),
                                )
                                if heal_sig != self._last_heal_sig:
                                    self._last_heal_sig = heal_sig
                                    ev_heal = Event(
                                        event_id="",
                                        run_id=os.environ.get("METABONK_RUN_ID", "run-local"),
                                        instance_id=self.instance_id,
                                        event_type="Heal",
                                        message=f"heal {float(last_heal_amount):.1f} {str(last_heal_item or '').strip()}",
                                        payload={
                                            "ts": now,
                                            "episode_idx": self._episode_idx,
                                            "amount": float(last_heal_amount),
                                            "item_name": str(last_heal_item) if last_heal_item else None,
                                            "is_borgar": bool(last_heal_is_borgar) if last_heal_is_borgar is not None else False,
                                        },
                                    )
                                    requests.post(f"{self.orch_url}/events", json=ev_heal.model_dump(), timeout=1.0)
                        except Exception:
                            pass

                        ev = Event(
                            event_id="",
                            run_id=os.environ.get("METABONK_RUN_ID", "run-local"),
                            instance_id=self.instance_id,
                            event_type="Telemetry",
                            message="telemetry",
                            payload={
                                "ts": now,
                                "episode_idx": self._episode_idx,
                                "episode_t": float(now - self._episode_start_ts),
                                "health_ratio": health_ratio,
                                "enemy_count": enemy_count,
                                "boss_visible": boss_visible,
                                "player_pos": player_pos,
                                "stage": stage,
                                "biome": biome,
                                "crit_chance": crit_chance,
                                "overcrit_tier": overcrit_tier,
                                "last_hit_damage": last_hit_damage,
                                "last_hit_is_crit": last_hit_is_crit,
                                "last_loot_rarity": last_loot_rarity,
                                "last_loot_item": last_loot_item,
                                "last_loot_source": last_loot_source,
                                "last_chest_rarity": last_chest_rarity,
                                "last_heal_amount": last_heal_amount,
                                "last_heal_item": last_heal_item,
                                "last_heal_is_borgar": last_heal_is_borgar,
                                "incoming_dps": incoming_dps,
                                "clearing_dps": clearing_dps,
                                "charge_level": charge_level,
                                "charge_is_max": charge_is_max,
                                "inventory_items": inventory_items,
                                "synergy_edges": synergy_edges,
                                "evolution_recipes": evolution_recipes,
                            },
                        )
                        requests.post(f"{self.orch_url}/events", json=ev.model_dump(), timeout=1.0)
                        self._last_telemetry_ts = now

                    # Episode boundaries from done flag (ResearchPlugin SHM).
                    if done and not self._last_done_sent:
                        self._last_done_sent = True
                        ev2 = Event(
                            event_id="",
                            run_id=os.environ.get("METABONK_RUN_ID", "run-local"),
                            instance_id=self.instance_id,
                            event_type="EpisodeEnd",
                            message=f"episode {self._episode_idx} ended",
                            payload={
                                "ts": now,
                                "episode_idx": self._episode_idx,
                                "duration_s": float(now - self._episode_start_ts),
                                "final_reward": float(reward),
                                "stage": int(self._last_stage) if self._last_stage is not None else None,
                                "biome": str(self._last_biome) if self._last_biome else None,
                                "end_reason": str(self._last_end_reason) if self._last_end_reason else None,
                            },
                        )
                        requests.post(f"{self.orch_url}/events", json=ev2.model_dump(), timeout=1.0)
                    if (not done) and self._last_done_sent:
                        # New episode detected.
                        self._last_done_sent = False
                        self._episode_idx += 1
                        self._episode_start_ts = now
                        if self.rollout.eval_mode and self.rollout.eval_seed is not None:
                            if self._action_source == "random":
                                try:
                                    random.seed(int(self.rollout.eval_seed) + int(self._episode_idx))
                                except Exception:
                                    pass
                        ev3 = Event(
                            event_id="",
                            run_id=os.environ.get("METABONK_RUN_ID", "run-local"),
                            instance_id=self.instance_id,
                            event_type="EpisodeStart",
                            message=f"episode {self._episode_idx} started",
                            payload={"ts": now, "episode_idx": self._episode_idx},
                        )
                        requests.post(f"{self.orch_url}/events", json=ev3.model_dump(), timeout=1.0)
                except Exception:
                    pass

            if not (self._use_sima2 and not self._sima2_push_rollouts):
                meaningful = self._is_meaningful_action(
                    a_cont,
                    a_disc,
                    action_mask_for_rollout,
                    forced_ui_click,
                    suppress_policy_clicks,
                )
                self.trainer.update(reward, meaningful=meaningful)
                self.rollout.add(obs, a_cont, a_disc, action_mask_for_rollout, reward, done, lp, val)

            # Optional eval clip on episode end (best-effort).
            if (
                done
                and self.rollout.eval_mode
                and self._eval_clip_on_done
                and self.highlight
                and requests
            ):
                try:
                    exp_id = os.environ.get("METABONK_EXPERIMENT_ID", "exp-local")
                    run_id = os.environ.get("METABONK_RUN_ID", "run-local")
                    clip_url = self.highlight.encode_clip(
                        experiment_id=exp_id,
                        run_id=run_id,
                        instance_id=self.instance_id,
                        score=reward,
                        speed=float(os.environ.get("METABONK_HIGHLIGHT_SPEED", "3.0")),
                        tag="eval",
                    )
                    if clip_url:
                        self.rollout.eval_clip_url = clip_url
                        self._post_buildlab_clip(clip_url=clip_url, score=float(reward), tag="eval")
                except Exception:
                    pass

            # Highlight capture on local PB.
            if self.highlight and reward > self._best_reward_local + 1e-6 and requests:
                self._best_reward_local = reward
                exp_id = os.environ.get("METABONK_EXPERIMENT_ID", "exp-local")
                run_id = os.environ.get("METABONK_RUN_ID", "run-local")
                clip_url = self.highlight.encode_clip(
                    experiment_id=exp_id,
                    run_id=run_id,
                    instance_id=self.instance_id,
                    score=reward,
                    speed=float(os.environ.get("METABONK_HIGHLIGHT_SPEED", "3.0")),
                    tag="pb",
                )
                if clip_url:
                    self._post_buildlab_clip(clip_url=clip_url, score=float(reward), tag="pb", run_id=run_id)
                    try:
                        from src.common.observability import Event

                        ev = Event(
                            event_id="",
                            run_id=run_id,
                            instance_id=self.instance_id,
                            event_type="NewBestRunClip",
                            message=f"{self.instance_id} new best {reward:.2f}",
                            payload={
                                "clip_url": clip_url,
                                "score": reward,
                                "speed": float(os.environ.get("METABONK_HIGHLIGHT_SPEED", "3.0")),
                                "agent_name": os.environ.get("MEGABONK_AGENT_NAME") or self.instance_id,
                            },
                        )
                        requests.post(f"{self.orch_url}/events", json=ev.model_dump(), timeout=1.0)
                    except Exception:
                        pass

            if not (self._use_sima2 and not self._sima2_push_rollouts):
                if done or self.rollout.ready():
                    batch = self.rollout.flush()
                    if hasattr(batch, "obs_zlib_b64"):
                        self.learner.push_rollout_pixels(batch)  # type: ignore[arg-type]
                    else:
                        self.learner.push_rollout(batch)  # type: ignore[arg-type]

            if bool(getattr(self, "_synthetic_eye_lockstep", False)) and bool(getattr(self, "_pixel_obs_enabled", False)):
                self._lockstep_request_next_frame(pixel_frame_id)
            else:
                self._stop.wait(0.05)

    def _post_buildlab_clip(
        self,
        *,
        clip_url: str,
        score: float,
        tag: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        if not requests:
            return
        rid = str(run_id or os.environ.get("METABONK_RUN_ID", "run-local"))
        payload: dict[str, object] = {
            "run_id": rid,
            "worker_id": self.instance_id,
            "timestamp": time.time(),
            "clip_url": clip_url,
            "final_score": float(score),
            "match_duration_sec": int(max(0.0, time.time() - float(self._episode_start_ts))),
            "episode_idx": int(getattr(self, "_episode_idx", 0) or 0),
            "policy_name": str(getattr(self, "policy_name", "") or ""),
            "policy_version": int(self._policy_version) if self._policy_version is not None else None,
            "agent_name": str(os.environ.get("MEGABONK_AGENT_NAME") or self.instance_id),
            "seed": str(os.environ.get("MEGABONK_SEED") or ""),
        }
        if self._last_inventory_items:
            payload["inventory_snapshot"] = list(self._last_inventory_items)
            payload["items"] = list(self._last_inventory_items)
        if tag:
            payload["tag"] = tag
        try:
            requests.post(f"{self.orch_url}/buildlab/runs", json=payload, timeout=1.0)
        except Exception:
            return

    def _stack_observation(self, obs):
        # Pixel observations are tensors; do not attempt vector stacking here.
        if not isinstance(obs, list):
            return obs
        if self._frame_stack <= 1:
            return obs
        self._obs_stack.append(list(obs))
        frames = list(self._obs_stack)
        if len(frames) < self._frame_stack:
            pad = [[0.0] * len(obs) for _ in range(self._frame_stack - len(frames))]
            frames = pad + frames
        stacked: List[float] = []
        for f in frames:
            stacked.extend(f)
        return stacked

    def _is_meaningful_action(
        self,
        a_cont: List[float],
        a_disc: List[int],
        action_mask: Optional[List[int]],
        forced_ui_click: Optional[tuple[int, int]],
        suppress_policy_clicks: bool,
    ) -> bool:
        try:
            deadband = float(os.environ.get("METABONK_STEP_ACTION_DEADBAND", "0.15"))
        except Exception:
            deadband = 0.15
        move_mag = 0.0
        if a_cont:
            try:
                move_mag = max(abs(float(a_cont[0])), abs(float(a_cont[1])) if len(a_cont) > 1 else 0.0)
            except Exception:
                move_mag = 0.0
        move_ok = move_mag >= deadband

        if self._input_backend is not None:
            key_ok = False
            try:
                key_ok = any(int(x) == 1 for x in a_disc)
            except Exception:
                key_ok = False
            scroll_ok = False
            try:
                if len(a_cont) >= 3 and abs(float(a_cont[2])) >= deadband:
                    scroll_ok = True
            except Exception:
                scroll_ok = False
            return bool(move_ok or key_ok or scroll_ok)

        click_ok = False
        if forced_ui_click is not None:
            click_ok = True
        elif (not suppress_policy_clicks) and a_disc and action_mask:
            try:
                idx = int(a_disc[0])
                noop_idx = len(action_mask) - 1 if action_mask else -1
                if 0 <= idx < len(action_mask) and idx != noop_idx and action_mask[idx] == 1 and idx != self._last_disc_action:
                    click_ok = True
            except Exception:
                click_ok = False

        return bool(move_ok or click_ok)

    def status(self):
        # Best-effort debug helpers for PipeWire/NVENC bringup.
        launcher_pid = None
        launcher_alive = False
        launcher_log = None
        launcher_last_node = None
        try:
            lp = getattr(self.launcher, "log_path", None)
            launcher_log = str(lp) if lp else None
        except Exception:
            launcher_log = None
        try:
            proc = getattr(self.launcher, "proc", None)
            if proc is not None:
                launcher_pid = int(getattr(proc, "pid", 0) or 0) or None
                launcher_alive = getattr(proc, "poll", lambda: 0)() is None
        except Exception:
            launcher_pid = None
            launcher_alive = False
        try:
            if launcher_log:
                from pathlib import Path
                import re

                data = Path(launcher_log).read_bytes()[-200000:]
                m = re.findall(rb"stream available on node ID:\s*([0-9]+)", data)
                if m:
                    launcher_last_node = m[-1].decode("ascii", "ignore")
        except Exception:
            launcher_last_node = None
        out = {
            "instance_id": self.instance_id,
            "policy_name": self.policy_name,
            "gameplay_started": bool(self._gameplay_started),
            "gameplay_confidence": float(getattr(self, "_gameplay_confidence", 0.0) or 0.0),
            "hud_present": bool(self._hud_present),
            "hud_phase": ("gameplay" if self._hud_present else "lobby"),
            "observation_backend": str(getattr(self, "_obs_backend", "") or ""),
            "observation_type": str(getattr(self, "_obs_backend", "") or ""),
            "pixel_preprocess_backend": str(getattr(self, "_pixel_preprocess_backend", "") or "")
            if bool(getattr(self, "_pixel_obs_enabled", False))
            else None,
            "vlm_hints_used": int(getattr(self, "_vlm_hints_used", 0) or 0),
            "vlm_hints_applied": int(getattr(self, "_vlm_hints_applied", 0) or 0),
            "act_hz": float(getattr(self, "_act_hz", 0.0) or 0.0),
            "actions_total": int(getattr(self, "_actions_total", 0) or 0),
            "phase_label": self._phase_label,
            "phase_conf": float(self._phase_conf or 0.0),
            "phase_source": self._phase_source,
            "phase_effective": self._phase_effective_label,
            "phase_effective_source": self._phase_effective_source,
            "phase_gameplay": bool(self._gameplay_phase_active),
            "display": self.display,
            "display_name": os.environ.get("MEGABONK_AGENT_NAME"),
            "hparams": self.hparams,
            "pipewire_node": os.environ.get("PIPEWIRE_NODE"),
            "pipewire_node_ok": bool(getattr(self, "_pipewire_node_ok", False)),
            "stream_enabled": bool(getattr(self, "_stream_enabled", False)),
            "stream_require_pipewire": bool(getattr(self, "_require_pipewire_stream", False)),
            "stream_require_zero_copy": bool(getattr(self, "_stream_require_zero_copy", False)),
            "stream_error": getattr(self, "_stream_error", None),
            "stream_epoch": int(getattr(self, "_stream_epoch", 0) or 0),
            "capture_enabled": not bool(getattr(self, "_capture_disabled", False)),
            "fifo_stream_enabled": bool(getattr(self, "_fifo_stream_enabled", False)),
            "fifo_stream_path": getattr(self, "_fifo_stream_path", None),
            "fifo_stream_last_error": (self._fifo_publisher.last_error() if self._fifo_publisher is not None else None),
            "go2rtc_stream_name": self.instance_id,
            "go2rtc_base_url": os.environ.get("METABONK_GO2RTC_URL"),
            "featured_slot": getattr(self, "_featured_slot", None),
            "featured_role": getattr(self, "_featured_role", None),
            "game_log_path": os.environ.get("MEGABONK_LOG_PATH") or os.environ.get("MEGABONK_LOG_DIR"),
            "launcher_pid": launcher_pid,
            "launcher_alive": launcher_alive,
            "launcher_log_path": launcher_log,
            "launcher_last_pipewire_node": launcher_last_node,
            **self.trainer.metrics(),
        }
        try:
            ve = getattr(self, "_visual_exploration", None)
            if ve is not None and hasattr(ve, "metrics"):
                out.update(ve.metrics())
        except Exception:
            pass
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                out["gpu_memory_gb"] = float(torch.cuda.memory_allocated() / 1e9)
            else:
                out["gpu_memory_gb"] = None
        except Exception:
            out["gpu_memory_gb"] = None
        # Stream quality diagnostics (best-effort).
        try:
            src = getattr(self, "_latest_pixel_src_size", None)
            if isinstance(src, tuple) and len(src) >= 2:
                out["pixel_src_width"] = int(src[0])
                out["pixel_src_height"] = int(src[1])
            else:
                out["pixel_src_width"] = None
                out["pixel_src_height"] = None
        except Exception:
            out["pixel_src_width"] = None
            out["pixel_src_height"] = None
        try:
            src = getattr(self, "_latest_spectator_src_size", None)
            if isinstance(src, tuple) and len(src) >= 2:
                out["spectator_src_width"] = int(src[0])
                out["spectator_src_height"] = int(src[1])
            else:
                out["spectator_src_width"] = None
                out["spectator_src_height"] = None
        except Exception:
            out["spectator_src_width"] = None
            out["spectator_src_height"] = None
        try:
            out["pixel_obs_width"] = int(getattr(self, "_pixel_obs_w", 0) or 0) if bool(getattr(self, "_pixel_obs_enabled", False)) else None
            out["pixel_obs_height"] = int(getattr(self, "_pixel_obs_h", 0) or 0) if bool(getattr(self, "_pixel_obs_enabled", False)) else None
        except Exception:
            out["pixel_obs_width"] = None
            out["pixel_obs_height"] = None
        try:
            out["spectator_width"] = int(getattr(self, "_spectator_w", 0) or 0) if bool(getattr(self, "_spectator_enabled", False)) else None
            out["spectator_height"] = int(getattr(self, "_spectator_h", 0) or 0) if bool(getattr(self, "_spectator_enabled", False)) else None
        except Exception:
            out["spectator_width"] = None
            out["spectator_height"] = None
        if self.stream is not None:
            try:
                out.update(self.stream.dmabuf_stats())
            except Exception:
                pass
        # Always include these keys so UIs can rely on them (None when streamer is not initialized).
        out.setdefault("stream_backend", None)
        out.setdefault("streamer_last_error", None)
        out.setdefault("stream_active_clients", None)
        out.setdefault("stream_max_clients", None)
        out.setdefault("stream_busy", None)
        out.setdefault("nvenc_sessions_used", None)
        # Optional diagnostic FPS fields (these are also used by validation harnesses).
        out.setdefault("stream_fps", None)
        out.setdefault("stream_target_fps", None)
        out.setdefault("obs_fps", None)
        if self.streamer is not None:
            try:
                out["stream_backend"] = getattr(self.streamer, "backend", None)
            except Exception:
                pass
            try:
                out["streamer_last_error"] = getattr(self.streamer, "last_error", None)
            except Exception:
                pass
            try:
                out["stream_active_clients"] = (
                    self.streamer.active_clients() if hasattr(self.streamer, "active_clients") else None
                )
            except Exception:
                pass
            try:
                out["stream_max_clients"] = (
                    self.streamer.max_clients() if hasattr(self.streamer, "max_clients") else None
                )
            except Exception:
                pass
            try:
                out["nvenc_sessions_used"] = getattr(self.streamer, "nvenc_sessions_used_last", None)
            except Exception:
                pass
            try:
                out["stream_busy"] = (
                    self.streamer.is_busy() if hasattr(self.streamer, "is_busy") else None
                )
            except Exception:
                pass
            try:
                sfps = getattr(self.streamer, "stream_fps", None)
                out["stream_fps"] = float(sfps) if sfps is not None else None
            except Exception:
                out["stream_fps"] = None
        elif bool(getattr(self, "_stream_enabled", False)) and not bool(getattr(self, "_capture_disabled", False)):
            # If capture is enabled but the streamer isn't initialized yet, return stable defaults.
            try:
                max_clients = int(os.environ.get("METABONK_STREAM_MAX_CLIENTS", "1"))
            except Exception:
                max_clients = 1
            if max_clients < 1:
                max_clients = 1
            out["stream_active_clients"] = 0
            out["stream_max_clients"] = int(max_clients)
            out["stream_busy"] = False
        try:
            out["obs_fps"] = float(getattr(self, "_obs_fps", 0.0) or 0.0) if getattr(self, "_obs_fps", None) is not None else None
        except Exception:
            out["obs_fps"] = None
        try:
            tfps = int(os.environ.get("METABONK_STREAM_FPS", "0") or 0)
            out["stream_target_fps"] = int(tfps) if int(tfps) > 0 else None
        except Exception:
            out["stream_target_fps"] = None
        out.setdefault("frames_fps", 0.0)
        # Preserve the original computed FPS (often capture/obs cadence) so diagnostics can
        # compare it against stream output/target FPS.
        try:
            out.setdefault("capture_fps", float(out.get("frames_fps") or 0.0))
        except Exception:
            out.setdefault("capture_fps", None)
        # Prefer stream FPS when available; otherwise fall back to target stream FPS and then obs FPS.
        try:
            sfps = out.get("stream_fps")
            if sfps is not None and float(sfps or 0.0) > 0.0:
                out["frames_fps"] = float(sfps)
            else:
                tfps = out.get("stream_target_fps")
                if tfps is not None and float(tfps or 0.0) > 0.0 and bool(out.get("stream_enabled", False)):
                    out["frames_fps"] = float(tfps)
                else:
                    ofps = out.get("obs_fps")
                    if ofps is not None and float(ofps or 0.0) > 0.0:
                        out["frames_fps"] = float(ofps)
        except Exception:
            pass
        # Expose whether the worker believes the capture/stream pipeline is GPU zero-copy.
        try:
            backend = str(out.get("stream_backend") or "")
            backend_l = backend.lower()
            zero_copy = False
            if bool(out.get("synthetic_eye")):
                zero_copy = True
            elif bool(out.get("stream_require_zero_copy")):
                zero_copy = True
            elif backend_l and ("cuda" in backend_l or "dmabuf" in backend_l):
                zero_copy = True
            out["zero_copy"] = bool(zero_copy)
        except Exception:
            out["zero_copy"] = False
        # System2/3 compatibility payload (vision-only; no menu/scene labels).
        try:
            resp = getattr(self, "_system2_last_response", None)
            if isinstance(resp, dict) and resp:
                req_ts = None
                try:
                    req_ts = float((self._system2_last_request or {}).get("timestamp") or 0.0)
                except Exception:
                    req_ts = None
                out["system2_reasoning"] = {
                    "goal": str(resp.get("goal") or ""),
                    "confidence": float(resp.get("confidence") or 0.0),
                    "reasoning": str(resp.get("reasoning") or ""),
                    "directive": resp.get("directive") if isinstance(resp.get("directive"), dict) else {},
                    "inference_time_ms": resp.get("inference_time_ms"),
                    "last_request_ts": req_ts,
                    "last_response_ts": float(getattr(self, "_system2_last_response_ts", 0.0) or 0.0),
                    "reasoning_trace": list(getattr(self, "_system2_reasoning_trace", []) or []),
                }
            else:
                out["system2_reasoning"] = None
        except Exception:
            out["system2_reasoning"] = None
        # Action-discovery telemetry (best-effort).
        try:
            buttons = list(getattr(self, "_input_buttons", []) or [])
            out["discovery_stats"] = {
                "actions_discovered": int(len(buttons)),
                "input_backend": str(getattr(self, "_input_backend_name", "") or ""),
                "pure_vision_mode": bool(getattr(self, "_pure_vision_mode", False)),
                "most_rewarding": "none",
            }
        except Exception:
            out["discovery_stats"] = {"actions_discovered": 0, "most_rewarding": "none"}
        if self._use_sima2 and self._sima2_controller is not None:
            try:
                out["sima2"] = self._sima2_controller.get_status()
            except Exception:
                out["sima2"] = {"enabled": True}
        if self._use_metabonk2 and self._metabonk2_controller is not None:
            try:
                out["metabonk2"] = self._metabonk2_controller.get_status()
            except Exception:
                out["metabonk2"] = {"enabled": True}
        # Stream health/quality (best-effort; mirrors heartbeat fields).
        try:
            now = time.time()
            out["stream_frame_var"] = getattr(self, "_last_frame_var", None)
            try:
                black_since = float(getattr(self, "_black_frame_since", 0.0) or 0.0)
            except Exception:
                black_since = 0.0
            out["stream_black_since_s"] = (now - black_since) if black_since > 0 else None
            out["stream_frame_diff"] = getattr(self, "_last_frame_luma_diff", None)
            try:
                frozen_since = float(getattr(self, "_frozen_frame_since", 0.0) or 0.0)
            except Exception:
                frozen_since = 0.0
            out["stream_frozen_since_s"] = (now - frozen_since) if frozen_since > 0 else None

            # Derive a stable stream_ok flag similar to the heartbeat loop.
            try:
                last = float(getattr(self, "_latest_frame_ts", 0.0) or 0.0)
            except Exception:
                last = 0.0
            try:
                enc_last = float(getattr(self.streamer, "last_chunk_ts", 0.0) or 0.0) if self.streamer else 0.0
            except Exception:
                enc_last = 0.0
            try:
                ok_ttl = float(os.environ.get("METABONK_STREAM_OK_TTL_S", "10.0"))
            except Exception:
                ok_ttl = 10.0
            stream_ok_age = bool((last > 0 and (now - last) <= ok_ttl) or (enc_last > 0 and (now - enc_last) <= ok_ttl))
            quality_ok = bool(black_since <= 0 and frozen_since <= 0)
            out["stream_ok"] = bool(stream_ok_age and quality_ok) if bool(out.get("stream_enabled", False)) else None
        except Exception:
            out.setdefault("stream_frame_var", None)
            out.setdefault("stream_black_since_s", None)
            out.setdefault("stream_frame_diff", None)
            out.setdefault("stream_frozen_since_s", None)
            out.setdefault("stream_ok", None)

        # Compatibility aliases for external validation harnesses (and older docs):
        # - frames/steps/fps
        # - stream_frozen/stream_diff/stream_frozen_checks/stream_heals
        try:
            out.setdefault("worker_id", out.get("instance_id"))
            out.setdefault("steps", out.get("step"))
            out.setdefault("frames", out.get("frames_ok"))
            out.setdefault("fps", out.get("frames_fps"))
            out.setdefault("streaming", out.get("stream_enabled"))
            out.setdefault("stream_diff", out.get("stream_frame_diff"))
            out.setdefault("stream_frozen", bool(out.get("stream_frozen_since_s") is not None))
            out.setdefault("stream_frozen_checks", int(getattr(self, "_stream_health_checks", 0)))
            out.setdefault("stream_heals", int(getattr(self, "_stream_heals", 0)))
            out.setdefault("system2_enabled", bool(getattr(self, "_system2_enabled", False)))
        except Exception:
            pass
        return out

    def get_config(self) -> InstanceConfig:
        return InstanceConfig(
            instance_id=self.instance_id,
            display=self.display,
            display_name=os.environ.get("MEGABONK_AGENT_NAME"),
            policy_name=self.policy_name,
            hparams=self.hparams,
        )

    def set_config(self, cfg: InstanceConfig):
        prev_policy = str(getattr(self, "policy_name", "") or "")
        prev_version = getattr(self, "_policy_version", None)

        self.policy_name = cfg.policy_name
        self.hparams = cfg.hparams
        self.trainer.policy_name = cfg.policy_name
        self.trainer.hparams = cfg.hparams
        self.rollout.policy_name = cfg.policy_name
        self.rollout.hparams = cfg.hparams
        # Do not reset the cached policy version unless the policy identity changes.
        # Resetting on every config poll forces full-weight pulls (`since_version=-1`),
        # which can stall action cadence.
        if str(cfg.policy_name or "") != prev_policy:
            self._policy_version = None
        else:
            self._policy_version = prev_version
        caps: dict = {}
        reg_obs_dim: Optional[int] = self.trainer.obs_dim
        if bool(getattr(self, "_pixel_obs_enabled", False)):
            reg_obs_dim = None
            caps.update(
                {
                    "obs_kind": "pixels",
                    "obs_width": int(getattr(self, "_pixel_obs_w", 0) or 0),
                    "obs_height": int(getattr(self, "_pixel_obs_h", 0) or 0),
                    "obs_channels": 3,
                    "obs_dtype": "uint8",
                }
            )
        self.learner.register(self.instance_id, cfg.policy_name, obs_dim=reg_obs_dim, capabilities=caps)
        try:
            if cfg.eval_mode is not None:
                self.rollout.eval_mode = bool(cfg.eval_mode)
        except Exception:
            pass
        try:
            if cfg.eval_seed is not None:
                self.rollout.eval_seed = int(cfg.eval_seed)
        except Exception:
            pass
        try:
            if cfg.config_poll_s is not None:
                self._config_poll_s = float(cfg.config_poll_s)
        except Exception:
            pass
        try:
            raw = None
            if hasattr(cfg, "action_source"):
                raw = getattr(cfg, "action_source")
            if raw is None:
                raw = cfg.model_dump().get("action_source")
            if raw is not None:
                self._set_action_source(str(raw))
        except Exception:
            pass
        try:
            self._featured_slot = str(cfg.featured_slot) if cfg.featured_slot is not None else None
        except Exception:
            self._featured_slot = None
        try:
            self._featured_role = str(cfg.featured_role) if cfg.featured_role is not None else None
        except Exception:
            self._featured_role = None
        try:
            if cfg.capture_enabled is not None:
                self.set_capture_enabled(bool(cfg.capture_enabled))
        except Exception:
            pass
        # Spectator output sizing can be updated at runtime by the orchestrator. This is used to
        # ensure featured tiles are truly 1080p (or configured) without upscaling tiny obs tensors.
        try:
            sw = getattr(cfg, "spectator_width", None)
            sh = getattr(cfg, "spectator_height", None)
            if sw is None or sh is None:
                raw = cfg.model_dump()
                sw = raw.get("spectator_width")
                sh = raw.get("spectator_height")
            if sw is not None and sh is not None:
                w = int(sw)
                h = int(sh)
                if w > 0 and h > 0:
                    if w % 2:
                        w += 1
                    if h % 2:
                        h += 1
                    self._spectator_w = int(w)
                    self._spectator_h = int(h)
        except Exception:
            pass


service: Optional[WorkerService] = None


if app:

    @app.get("/status")
    def status():
        if not service:
            raise HTTPException(status_code=503, detail="service not initialized")
        return service.status()

    @app.get("/stream")
    async def stream_video(request: Request):
        """NVENC MPEG-TS stream (read-only)."""
        if not service:
            raise HTTPException(status_code=404, detail="streaming disabled")
        if service.streamer is None and bool(getattr(service, "_stream_enabled", False)) and not bool(getattr(service, "_capture_disabled", False)):
            try:
                service._ensure_streamer()
            except Exception:
                pass
        if not service.streamer:
            raise HTTPException(status_code=404, detail="streaming disabled")
        if getattr(service.streamer, "is_busy", None) and service.streamer.is_busy():
            raise HTTPException(status_code=429, detail="stream busy")
        from fastapi.responses import StreamingResponse
        from starlette.concurrency import iterate_in_threadpool
        from src.worker.nvenc_session_manager import try_acquire_nvenc_slot

        allow_cpu = str(os.environ.get("METABONK_STREAM_ALLOW_CPU_FALLBACK", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        lease = None if allow_cpu else try_acquire_nvenc_slot(timeout_s=0.0, enforce_nvml=True)
        if (
            lease is None
            and not allow_cpu
            and str(os.environ.get("METABONK_NVENC_MAX_SESSIONS", "0") or "0").strip() not in ("0", "")
        ):
            raise HTTPException(status_code=503, detail="NVENC session limit reached")

        import threading

        stop_ev = threading.Event()

        def _gen():
            try:
                yield from service.streamer.iter_chunks(container="mpegts", stop_event=stop_ev)
            finally:
                if lease is not None:
                    lease.release()

        gen = _gen()

        async def _agen():  # type: ignore[no-untyped-def]
            async def _watch_disconnect() -> None:
                try:
                    while not stop_ev.is_set():
                        if await request.is_disconnected():
                            stop_ev.set()
                            break
                        await asyncio.sleep(0.25)
                except Exception:
                    stop_ev.set()

            watch_task = asyncio.create_task(_watch_disconnect())
            try:
                async for chunk in iterate_in_threadpool(gen):
                    if stop_ev.is_set():
                        break
                    yield chunk
            except asyncio.CancelledError:
                stop_ev.set()
                return
            finally:
                stop_ev.set()
                try:
                    watch_task.cancel()
                    await watch_task
                except BaseException:
                    pass

        return StreamingResponse(
            _agen(),
            media_type="video/MP2T",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/stream.mp4")
    async def stream_mp4(request: Request):
        """GPU NVENC fragmented MP4 stream (browser-friendly)."""
        if not service:
            raise HTTPException(status_code=404, detail="streaming disabled")
        if service.streamer is None and bool(getattr(service, "_stream_enabled", False)) and not bool(getattr(service, "_capture_disabled", False)):
            try:
                service._ensure_streamer()
            except Exception:
                pass
        if not service.streamer:
            raise HTTPException(status_code=404, detail="streaming disabled")
        if getattr(service.streamer, "is_busy", None) and service.streamer.is_busy():
            raise HTTPException(status_code=429, detail="stream busy")
        from fastapi.responses import StreamingResponse
        from starlette.concurrency import iterate_in_threadpool
        from src.worker.nvenc_session_manager import try_acquire_nvenc_slot

        allow_cpu = str(os.environ.get("METABONK_STREAM_ALLOW_CPU_FALLBACK", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        lease = None if allow_cpu else try_acquire_nvenc_slot(timeout_s=0.0, enforce_nvml=True)
        if (
            lease is None
            and not allow_cpu
            and str(os.environ.get("METABONK_NVENC_MAX_SESSIONS", "0") or "0").strip() not in ("0", "")
        ):
            raise HTTPException(status_code=503, detail="NVENC session limit reached")

        import threading

        stop_ev = threading.Event()

        def _gen():
            try:
                yield from service.streamer.iter_chunks(container="mp4", stop_event=stop_ev)
            finally:
                if lease is not None:
                    lease.release()

        gen = _gen()

        async def _agen():  # type: ignore[no-untyped-def]
            async def _watch_disconnect() -> None:
                try:
                    while not stop_ev.is_set():
                        if await request.is_disconnected():
                            stop_ev.set()
                            break
                        await asyncio.sleep(0.25)
                except Exception:
                    stop_ev.set()

            watch_task = asyncio.create_task(_watch_disconnect())
            try:
                async for chunk in iterate_in_threadpool(gen):
                    if stop_ev.is_set():
                        break
                    yield chunk
            except asyncio.CancelledError:
                stop_ev.set()
                return
            finally:
                stop_ev.set()
                try:
                    watch_task.cancel()
                    await watch_task
                except BaseException:
                    pass

        return StreamingResponse(
            _agen(),
            media_type="video/mp4",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/stream_meta.json")
    def stream_meta():
        """Stream metadata for UI self-heal + diagnostics (best-effort)."""
        if not service:
            raise HTTPException(status_code=503, detail="service not initialized")
        try:
            return service.stream_meta()
        except Exception:
            # Never crash the worker on metadata calls.
            return {"instance_id": getattr(service, "instance_id", None), "epoch": int(getattr(service, "_stream_epoch", 0) or 0)}

    @app.get("/frame.jpg")
    def frame_jpg():
        """Latest JPEG frame (best-effort).

        This is a lightweight UI fallback/debug endpoint for when MP4 streaming isn't
        available (or when a browser is still negotiating a stream).

        If no cached JPEG exists, we attempt an on-demand snapshot via GStreamer
        from the current PipeWire node.
        """
        if not service:
            raise HTTPException(status_code=404, detail="service not initialized")
        if os.environ.get("METABONK_STREAM_REQUIRE_ZERO_COPY", "0") in ("1", "true", "True"):
            raise HTTPException(status_code=404, detail="frame.jpg disabled in strict zero-copy mode")
        # Ensure streamer/node is initialized so we have a PipeWire target to snapshot.
        if (
            service.streamer is None
            and bool(getattr(service, "_stream_enabled", False))
            and not bool(getattr(service, "_capture_disabled", False))
        ):
            try:
                service._ensure_streamer()
            except Exception:
                pass

        now = time.time()
        data = getattr(service, "_latest_jpeg_bytes", None)
        ts = float(getattr(service, "_latest_frame_ts", 0.0) or 0.0)
        try:
            ttl = float(os.environ.get("METABONK_FRAME_JPEG_TTL_S", "1.0"))
        except Exception:
            ttl = 1.0
        if ttl < 0:
            ttl = 0.0

        # Cache hit.
        if data and ts > 0 and (ttl <= 0 or (now - ts) <= ttl):
            from fastapi.responses import Response

            return Response(
                content=data,
                media_type="image/jpeg",
                headers={"Cache-Control": "private, max-age=1"},
            )

        # Cache miss: attempt demand-paged snapshot (single frame).
        snap = None
        # Synthetic Eye (pixels obs) fallback: if we're running GPU-only observations and have a
        # recent pixel tensor cached, convert it into a small JPEG. This avoids relying on
        # PipeWire being available/healthy for the debug UI + smoke tests.
        if (
            getattr(service, "_frame_source", "") in ("synthetic_eye", "smithay", "smithay_dmabuf")
        ):
            try:
                obs_u8, _fid, _ts, _src_size = service._get_latest_spectator_obs()  # type: ignore[attr-defined]
            except Exception:
                obs_u8 = None
            if obs_u8 is None:
                try:
                    obs_u8, _fid, _ts, _src_size = service._get_latest_pixel_obs()  # type: ignore[attr-defined]
                except Exception:
                    obs_u8 = None
            if obs_u8 is not None:
                try:
                    import io
                    from PIL import Image

                    t = obs_u8
                    try:
                        # Move to CPU for JPEG encoding (debug endpoint only).
                        if hasattr(t, "detach"):
                            t = t.detach()
                        if hasattr(t, "device") and str(getattr(t, "device", "")).startswith("cuda"):
                            try:
                                import torch  # type: ignore

                                stream_ptr = int(getattr(service, "_synthetic_eye_cu_stream_ptr", 0) or 0)
                                if stream_ptr:
                                    ext = torch.cuda.ExternalStream(int(stream_ptr))
                                    torch.cuda.current_stream(device=ext.device).wait_stream(ext)
                            except Exception:
                                pass
                            t = t.cpu()
                        elif hasattr(t, "cpu"):
                            t = t.cpu()
                    except Exception:
                        if hasattr(t, "cpu"):
                            t = t.cpu()
                    # Expect CHW uint8 (3xHxW).
                    if hasattr(t, "ndim") and int(getattr(t, "ndim", 0)) == 3:
                        try:
                            t = t[:3].permute(1, 2, 0).contiguous()
                        except Exception:
                            pass
                        arr = t.numpy() if hasattr(t, "numpy") else None
                        if arr is not None:
                            img = Image.fromarray(arr)
                            buf = io.BytesIO()
                            try:
                                q = int(os.environ.get("METABONK_FRAME_JPEG_QUALITY", "80"))
                            except Exception:
                                q = 80
                            q = max(10, min(95, int(q)))
                            img.save(buf, format="JPEG", quality=q)
                            snap = buf.getvalue()
                except Exception:
                    snap = None
        if not snap:
            try:
                if service.streamer is not None and hasattr(service.streamer, "capture_jpeg"):
                    snap = service.streamer.capture_jpeg(timeout_s=float(os.environ.get("METABONK_FRAME_JPEG_TIMEOUT_S", "1.5")))
            except Exception:
                snap = None
        if snap:
            try:
                service._latest_jpeg_bytes = snap
                service._latest_frame_ts = now
            except Exception:
                pass
            from fastapi.responses import Response

            return Response(
                content=snap,
                media_type="image/jpeg",
                headers={"Cache-Control": "private, max-age=1"},
            )

        raise HTTPException(status_code=404, detail="no jpeg frame available")

    @app.get("/worker/{worker_id}/history")
    def worker_history(worker_id: str, limit: int = 50):
        """Return last N action+frame records for Context Drawer."""
        if not service:
            raise HTTPException(status_code=404, detail="service not initialized")
        if str(worker_id) != str(service.instance_id):
            raise HTTPException(status_code=404, detail="worker not found")
        hist = service.telemetry_history(limit=limit)
        return {
            "worker_id": service.instance_id,
            "buffer_capacity": service._telemetry_capacity,
            "history": hist,
        }

    @app.get("/telemetry/history")
    def telemetry_history(limit: int = 50):
        """Return last N action+frame records for this worker."""
        if not service:
            raise HTTPException(status_code=404, detail="service not initialized")
        hist = service.telemetry_history(limit=limit)
        return {
            "worker_id": service.instance_id,
            "buffer_capacity": service._telemetry_capacity,
            "history": hist,
        }

    @app.post("/highlight/clutch")
    def highlight_clutch():
        """Encode a short clutch highlight from the rolling buffer (if enabled)."""
        if not service or not service.highlight:
            raise HTTPException(status_code=404, detail="highlights disabled")
        exp_id = os.environ.get("METABONK_EXPERIMENT_ID", "exp-local")
        run_id = os.environ.get("METABONK_RUN_ID", "run-local")
        # Use survival probability as "score" proxy if provided; otherwise 0.
        try:
            score = float(getattr(service.trainer, "last_reward", 0.0) or 0.0)
        except Exception:
            score = 0.0
        clip_url = service.highlight.encode_clip(
            experiment_id=exp_id,
            run_id=run_id,
            instance_id=service.instance_id,
            score=score,
            speed=float(os.environ.get("METABONK_HIGHLIGHT_SPEED", "3.0")),
            tag="clutch",
        )
        if not clip_url:
            raise HTTPException(status_code=500, detail="clip encode failed")
        service._post_buildlab_clip(clip_url=clip_url, score=float(score), tag="clutch", run_id=run_id)
        return {"clip_url": clip_url}

    @app.post("/highlight/encode")
    def highlight_encode(req: dict):
        """Encode a highlight clip from the rolling buffer (if enabled).

        Request body (best-effort):
          - tag: string (e.g. "attract_fame", "attract_shame")
          - score: number (used in filename / metadata)
          - speed: number (playback speed, default env METABONK_HIGHLIGHT_SPEED)
          - experiment_id/run_id: optional overrides (defaults to env)
        """
        if not service or not service.highlight:
            raise HTTPException(status_code=404, detail="highlights disabled")

        tag = str(req.get("tag") or "clip")
        try:
            score = float(req.get("score") or 0.0)
        except Exception:
            score = 0.0
        try:
            speed = float(req.get("speed") or os.environ.get("METABONK_HIGHLIGHT_SPEED", "3.0"))
        except Exception:
            speed = float(os.environ.get("METABONK_HIGHLIGHT_SPEED", "3.0"))

        exp_id = str(req.get("experiment_id") or os.environ.get("METABONK_EXPERIMENT_ID", "exp-local"))
        run_id = str(req.get("run_id") or os.environ.get("METABONK_RUN_ID", "run-local"))

        clip_url = service.highlight.encode_clip(
            experiment_id=exp_id,
            run_id=run_id,
            instance_id=service.instance_id,
            score=score,
            speed=speed,
            tag=tag,
        )
        if not clip_url:
            raise HTTPException(status_code=500, detail="clip encode failed")
        service._post_buildlab_clip(clip_url=clip_url, score=float(score), tag=tag, run_id=run_id)
        return {"clip_url": clip_url}

    @app.get("/config")
    def get_config():
        if not service:
            raise HTTPException(status_code=503, detail="service not initialized")
        return service.get_config()

    @app.post("/config")
    def set_config(cfg: InstanceConfig):
        if not service:
            raise HTTPException(status_code=503, detail="service not initialized")
        service.set_config(cfg)
        return {"ok": True}


def resolve_gamescope_serial() -> bool:
    """Resolve the current PipeWire object.serial for the Gamescope Video/Source node."""
    if os.environ.get("METABONK_PIPEWIRE_TARGET_OVERRIDE"):
        return True
    if os.environ.get("METABONK_PIPEWIRE_RESOLVE_GAMESCOPE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    try:
        output = subprocess.check_output(["pw-cli", "info", "all"], text=True)
    except Exception as exc:  # pragma: no cover - depends on system PipeWire
        print(f"[RESOLVER] Failed to query PipeWire: {exc}")
        return False

    serial = ""
    for block in output.split("\n\n"):
        if 'media.class = "Video/Source"' not in block:
            continue
        # node.name can be "gamescope" or a namespaced variant like "gamescope:out_0".
        if "gamescope" not in block:
            continue
        match = re.search(r'object\\.serial\\s*=\\s*\"?(\\d+)\"?', block)
        if match:
            serial = match.group(1)
            break

    if not serial:
        print("[RESOLVER] Gamescope Video/Source serial not found.")
        return False

    os.environ["METABONK_PIPEWIRE_TARGET_MODE"] = "target-object"
    os.environ["METABONK_PIPEWIRE_TARGET_OVERRIDE"] = serial
    print(f"[RESOLVER] Found Gamescope serial: {serial}")
    return True


def main() -> int:
    if _import_error:
        raise RuntimeError(
            "fastapi/uvicorn not available; install requirements to run worker"
        ) from _import_error

    parser = argparse.ArgumentParser(description="MetaBonk worker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WORKER_PORT", "5000")))
    parser.add_argument("--instance-id", default=os.environ.get("INSTANCE_ID", "worker-0"))
    parser.add_argument("--policy-name", default=os.environ.get("POLICY_NAME", "Greed"))
    parser.add_argument("--orch-url", default=os.environ.get("ORCHESTRATOR_URL", "http://127.0.0.1:8040"))
    parser.add_argument("--learner-url", default=os.environ.get("LEARNER_URL", "http://127.0.0.1:8061"))
    parser.add_argument("--vision-url", default=os.environ.get("VISION_URL", "http://127.0.0.1:8050"))
    parser.add_argument("--obs-dim", type=int, default=int(os.environ.get("OBS_DIM", "204")))
    parser.add_argument("--frame-stack", type=int, default=int(os.environ.get("METABONK_FRAME_STACK", "4")))
    parser.add_argument("--display", default=os.environ.get("DISPLAY"))
    args = parser.parse_args()

    global service
    service = WorkerService(
        instance_id=args.instance_id,
        policy_name=args.policy_name,
        orch_url=args.orch_url,
        learner_url=args.learner_url,
        obs_dim=args.obs_dim,
        frame_stack=args.frame_stack,
        vision_url=args.vision_url,
        display=args.display,
        host=args.host,
        port=args.port,
    )
    service.start()

    loop_impl = os.environ.get("METABONK_UVICORN_LOOP")
    if os.environ.get("METABONK_DISABLE_IO_URING") == "1":
        os.environ["LIBUV_USE_IO_URING"] = "0"

    uvicorn_kwargs = dict(host=args.host, port=args.port, log_level="info")
    if loop_impl:
        uvicorn_kwargs["loop"] = loop_impl
        print(f"[worker] uvicorn loop override: {loop_impl}")
    http_impl = os.environ.get("METABONK_UVICORN_HTTP")
    if http_impl:
        uvicorn_kwargs["http"] = http_impl
        print(f"[worker] uvicorn http override: {http_impl}")

    uvicorn.run(app, **uvicorn_kwargs)  # type: ignore
    service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
