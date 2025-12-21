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
import os
import threading
import time
from collections import deque
from typing import List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except Exception as e:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    uvicorn = None  # type: ignore
    _import_error = e
else:
    _import_error = None

from src.common.schemas import CurriculumConfig, Heartbeat, InstanceConfig, TrainerConfig
from .launcher import GameLauncher
from .stream import CaptureStream
from .trainer import Trainer
from .perception import construct_observation
from .rollout import LearnerClient, RolloutBuffer
from .nvenc_streamer import NVENCConfig, NVENCStreamer
from .fifo_publisher import FifoH264Publisher, FifoPublishConfig
from .highlight_recorder import HighlightConfig, HighlightRecorder

try:
    from src.bridge.unity_bridge import UnityBridge, BridgeConfig, GameFrame
except Exception:  # pragma: no cover
    UnityBridge = None  # type: ignore
    BridgeConfig = None  # type: ignore
    GameFrame = None  # type: ignore


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
        from src.common.device import require_cuda

        use_dmabuf = os.environ.get("METABONK_CAPTURE_CPU", "0") not in ("1", "true", "True")
        if require_cuda():
            use_dmabuf = True
        self.stream = CaptureStream(
            pipewire_node=os.environ.get("PIPEWIRE_NODE"),
            use_dmabuf=use_dmabuf,
        )
        # Starting a GStreamer PipeWire capture pipeline can be fragile on some systems
        # (driver/gi/gstreamer mismatches). For stream HUD purposes we only need NVENC,
        # so keep capture opt-in and default it off.
        self._gst_capture_enabled = os.environ.get("METABONK_GST_CAPTURE", "0") in ("1", "true", "True")
        self._obs_dim_raw = int(obs_dim)
        env_stack = os.environ.get("METABONK_FRAME_STACK")
        try:
            stack = int(frame_stack) if frame_stack is not None else int(env_stack or 1)
        except Exception:
            stack = 1
        self._frame_stack = max(1, stack)
        self._obs_stack = deque(maxlen=self._frame_stack)
        stacked_dim = self._obs_dim_raw * self._frame_stack
        self.trainer = Trainer(policy_name=policy_name, hparams=self.hparams, obs_dim=stacked_dim)
        self.learner = LearnerClient(self.learner_url)
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
        self._config_poll_s = float(os.environ.get("METABONK_CONFIG_POLL_S", "30.0"))
        self._warned_no_reward_frame = False
        self._eval_clip_on_done = os.environ.get("METABONK_EVAL_CLIP_ON_DONE", "0") in ("1", "true", "True")
        self._policy_version: Optional[int] = None
        self._last_policy_update_ts: float = 0.0
        self._last_policy_fetch_ts: float = 0.0
        self._last_policy_warn_ts: float = 0.0
        self._last_stream_ok_ts: float = 0.0
        self._last_stream_heal_ts: float = 0.0

        # Optional UnityBridge embodiment (BepInEx plugin).
        self._use_bridge = os.environ.get("METABONK_USE_UNITY_BRIDGE", "0") in ("1", "true", "True")
        self.bridge: Optional["UnityBridge"] = None
        self._bridge_loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_disc_action: Optional[int] = None
        # When using UnityBridge, optionally map continuous PPO actions to keyboard movement.
        # This is disabled by default to avoid hard-coded input assumptions, but can be
        # enabled for practical end-to-end control testing.
        self._bridge_action_map = str(os.environ.get("METABONK_BRIDGE_ACTION_MAP", "") or "").strip().lower()
        if self._use_bridge and not self._bridge_action_map:
            # Practical default: if UnityBridge is enabled, map PPO continuous actions to WASD so the
            # agent can actually move without requiring extra glue code.
            self._bridge_action_map = "wasd"
        self._bridge_held_keys: set[str] = set()
        # Optional BonkLink bridge (BepInEx 6 IL2CPP).
        self._use_bonklink = os.environ.get("METABONK_USE_BONKLINK", "0") in ("1", "true", "True")
        self._bonklink = None
        # Optional VLM-driven menu navigation (Lobby Agent).
        self._use_vlm_menu = os.environ.get("METABONK_USE_VLM_MENU", "0") in ("1", "true", "True")
        self._vlm_menu_goal = os.environ.get("METABONK_MENU_GOAL", "Start Run")
        self._vlm_menu = None
        if self._use_vlm_menu:
            try:
                from .vlm_navigator import VLMNavigator, VLMConfig as _MenuVLMConfig

                model = os.environ.get("METABONK_VLM_MENU_MODEL")
                cfg = _MenuVLMConfig(model=model) if model else _MenuVLMConfig()
                self._vlm_menu = VLMNavigator(cfg)
            except Exception:
                self._vlm_menu = None
                self._use_vlm_menu = False
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
        self._stream_enabled = os.environ.get("METABONK_STREAM", "1") in ("1", "true", "True")
        # Hard requirement: when enabled, the worker must use PipeWire+NVENC for stream.
        # No MJPEG/CPU streaming fallback is allowed in this mode.
        self._require_pipewire_stream = os.environ.get("METABONK_REQUIRE_PIPEWIRE_STREAM", "1") in (
            "1",
            "true",
            "True",
        )
        try:
            sb = str(os.environ.get("METABONK_STREAM_BACKEND", "auto") or "").strip().lower()
            if sb == "obs":
                sb = "ffmpeg"
            if sb == "x11grab":
                # x11grab is explicitly non-PipeWire capture; don't gate startup on PipeWire discovery.
                self._require_pipewire_stream = False
        except Exception:
            pass
        self._pipewire_node_ok = False
        self._pipewire_node = os.environ.get("PIPEWIRE_NODE")
        self._stream_error: Optional[str] = None
        self._featured_slot: Optional[str] = None
        self._featured_role: Optional[str] = None

        self.highlight: Optional[HighlightRecorder] = None
        if os.environ.get("METABONK_HIGHLIGHTS", "0") in ("1", "true", "True"):
            try:
                hcfg = HighlightConfig(
                    fps=int(os.environ.get("METABONK_HIGHLIGHT_FPS", "30")),
                    max_seconds=int(os.environ.get("METABONK_HIGHLIGHT_SECONDS", "180")),
                    downscale=(480, 270),
                    speed=float(os.environ.get("METABONK_HIGHLIGHT_SPEED", "3.0")),
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

        # Episode metadata for milestone pinning (best-effort).
        self._last_stage: Optional[int] = None
        self._last_biome: Optional[str] = None
        self._last_end_reason: Optional[str] = None

        # Learned reward-from-video (optional but preferred; avoids hand-authored shaping).
        self._use_learned_reward = os.environ.get("METABONK_USE_LEARNED_REWARD", "1") in ("1", "true", "True")
        self._reward_model = None
        self._reward_device = None
        self._reward_frame_size = (224, 224)
        self._reward_scale = float(os.environ.get("METABONK_LEARNED_REWARD_SCALE", "1.0"))
        self._prev_progress_score: Optional[float] = None
        # Latest JPEG frame bytes (e.g., from BonkLink) for UI streaming fallback.
        self._latest_jpeg_bytes: Optional[bytes] = None
        self._latest_frame_ts: float = 0.0
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
            self._fifo_stream_path = os.path.join(fifo_dir, f"{self.instance_id}.h264")

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
        if self.streamer is not None:
            return
        backend = (os.environ.get("METABONK_STREAM_BACKEND") or "").strip().lower()
        # Back-compat: "obs" means "use OBS-like ffmpeg encoder selection" (no OBS required).
        if backend == "obs":
            backend = "ffmpeg"
        if backend and backend not in ("gst", "gstreamer", "gst-launch", "ffmpeg", "x11grab", "auto"):
            self._stream_error = f"unsupported stream backend '{backend}' (expected auto|gst|ffmpeg|x11grab|obs)"
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
            scfg = NVENCConfig(
                pipewire_node=node,
                codec=os.environ.get("METABONK_STREAM_CODEC", "h264"),
                bitrate=os.environ.get("METABONK_STREAM_BITRATE", "6M"),
                fps=int(os.environ.get("METABONK_STREAM_FPS", "60")),
                gop=int(os.environ.get("METABONK_STREAM_GOP", "60")),
                container=os.environ.get("METABONK_STREAM_CONTAINER", "mp4"),
            )
            self.streamer = NVENCStreamer(scfg)
            if self._fifo_stream_enabled and self._fifo_stream_path:
                try:
                    pipe_bytes = int(os.environ.get("METABONK_FIFO_PIPE_BYTES", "1048576"))
                except Exception:
                    pipe_bytes = 0
                self._fifo_publisher = FifoH264Publisher(
                    cfg=FifoPublishConfig(fifo_path=self._fifo_stream_path, pipe_size_bytes=pipe_bytes),
                    streamer=self.streamer,
                )
            self._stream_error = None
        except Exception as e:
            self.streamer = None
            self._fifo_publisher = None
            if self._require_pipewire_stream:
                self._stream_error = f"failed to initialize streamer ({e})"

    def set_capture_enabled(self, enabled: bool) -> None:
        """Enable/disable PipeWire capture at runtime (best-effort)."""
        enabled = bool(enabled)
        new_disabled = not enabled
        if bool(getattr(self, "_capture_disabled", False)) == new_disabled:
            if not new_disabled:
                self._ensure_preview_jpeg()
            return
        self._capture_disabled = new_disabled
        if new_disabled:
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
            if bool(getattr(self, "_pipewire_drain_enabled", False)) or self._gst_capture_enabled:
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
        self.launcher.launch()
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

    def stop(self):
        self._stop.set()
        self.stream.stop()
        self.launcher.shutdown()
        try:
            self._stop_preview_jpeg()
        except Exception:
            pass
        if self.streamer:
            self.streamer.stop()
        # Best-effort release held keys (avoid leaving the game stuck moving).
        if self._use_bridge and self.bridge and self._bridge_loop and self._bridge_held_keys:
            try:
                for k in list(self._bridge_held_keys):
                    try:
                        self._bridge_loop.run_until_complete(self.bridge.send_key(str(k), False))
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
                        self._latest_jpeg_bytes = buf.getvalue()
                        self._latest_frame_ts = time.time()
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
        ui_elements: Optional[List[List[float]]],
        action_mask: Optional[List[int]],
        frame_size: Optional[tuple[int, int]],
    ):
        if not self.bridge or not self._bridge_loop:
            return
        # Optional conservative mapping for end-to-end control bring-up:
        # - a_cont[0] = strafe (-1..1): A/D
        # - a_cont[1] = forward (-1..1): W/S
        #
        # Enable with: METABONK_BRIDGE_ACTION_MAP=wasd
        # Tuning: METABONK_BRIDGE_WASD_THRESH=0.25 (press), METABONK_BRIDGE_WASD_RELEASE=0.15 (release)
        if self._bridge_action_map in ("wasd", "wasd2d", "kbd_wasd"):
            try:
                thresh = float(os.environ.get("METABONK_BRIDGE_WASD_THRESH", "0.25"))
            except Exception:
                thresh = 0.25
            try:
                rel = float(os.environ.get("METABONK_BRIDGE_WASD_RELEASE", str(max(0.05, thresh * 0.6))))
            except Exception:
                rel = max(0.05, thresh * 0.6)
            x = float(a_cont[0]) if len(a_cont) > 0 else 0.0
            y = float(a_cont[1]) if len(a_cont) > 1 else 0.0
            desired: set[str] = set()
            if x >= thresh:
                desired.add("D")
            elif x <= -thresh:
                desired.add("A")
            if y >= thresh:
                desired.add("W")
            elif y <= -thresh:
                desired.add("S")
            # Add light hysteresis to reduce key spam when hovering near zero.
            if abs(x) <= rel:
                desired.discard("A")
                desired.discard("D")
            if abs(y) <= rel:
                desired.discard("W")
                desired.discard("S")
            try:
                for k in sorted(desired - self._bridge_held_keys):
                    self._bridge_loop.run_until_complete(self.bridge.send_key(str(k), True))
                for k in sorted(self._bridge_held_keys - desired):
                    self._bridge_loop.run_until_complete(self.bridge.send_key(str(k), False))
                self._bridge_held_keys = desired
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
            stream_ok = bool((last > 0 and (now - last) <= ok_ttl) or (enc_last > 0 and (now - enc_last) <= ok_ttl))
            last_any = max(last, enc_last)
            if stream_ok:
                self._last_stream_ok_ts = now
            heal_s = float(os.environ.get("METABONK_STREAM_SELF_HEAL_S", "20.0"))
            if (
                heal_s > 0
                and self._stream_enabled
                and self._require_pipewire_stream
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
                # Never tear down/recreate the streamer while a client is actively consuming it.
                # Doing so causes visible "cut out" flapping for the UI and can fight go2rtc/FIFO readers.
                if should_heal and active_clients <= 0 and (now - self._last_stream_heal_ts) >= heal_s:
                    self._last_stream_heal_ts = now
                    self._stream_error = reason
                    self.streamer = None
                    try:
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
            hb = Heartbeat(
                run_id=os.environ.get("METABONK_RUN_ID"),
                instance_id=self.instance_id,
                policy_name=self.policy_name,
                policy_version=self._policy_version,
                step=self.trainer.step_count,
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
                stream_active_clients=active_clients,
                stream_max_clients=max_clients,
                fifo_stream_enabled=bool(getattr(self, "_fifo_stream_enabled", False)),
                fifo_stream_path=getattr(self, "_fifo_stream_path", None),
                fifo_stream_last_error=(self._fifo_publisher.last_error() if self._fifo_publisher is not None else None),
                go2rtc_stream_name=self.instance_id,
                go2rtc_base_url=os.environ.get("METABONK_GO2RTC_URL"),
                pipewire_node_ok=bool(getattr(self, "_pipewire_node_ok", False)),
                worker_device=worker_device or None,
                vision_device=vision_device or None,
                learned_reward_device=learned_reward_device or None,
                reward_device=reward_device or None,
                control_url=self.control_url(),
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
        self.learner.register(self.instance_id, self.policy_name, obs_dim=self.trainer.obs_dim)
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
                    port=int(os.environ.get("METABONK_BONKLINK_PORT", "5555")),
                    use_named_pipe=os.environ.get("METABONK_BONKLINK_USE_PIPE", "0")
                    in ("1", "true", "True"),
                    pipe_name=os.environ.get("METABONK_BONKLINK_PIPE_NAME", "BonkLink"),
                )
                if not self._bonklink.connect(timeout_s=2.0):
                    self._bonklink = None
                    self._use_bonklink = False
            except Exception:
                self._bonklink = None
                self._use_bonklink = False
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
                    frame_size = (84, 84)
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
            if (
                (not used_pixels)
                and (not detections)
                and self._use_bonklink
                and self._bonklink is not None
            ):
                try:
                    pkt = self._bonklink.read_state_frame(timeout_ms=16)
                except Exception:
                    pkt = None
                if pkt is not None:
                    state_obj, jpeg_bytes = pkt
                    if not self._visual_only:
                        game_state.update(state_obj.to_dict())
                    if jpeg_bytes:
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
                                else:
                                    latest_image_bytes = jpeg_bytes
                                    img = Image.open(io.BytesIO(jpeg_bytes))
                                    frame_size = img.size
                                    reward_frame_hwc = None
                            else:
                                latest_image_bytes = jpeg_bytes
                                img = Image.open(io.BytesIO(jpeg_bytes))
                                frame_size = img.size
                                reward_frame_hwc = None

                            self._latest_jpeg_bytes = latest_image_bytes
                            self._latest_frame_ts = now
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

            # Fallback: PipeWire capture (only if bridge absent).
            if (
                (not detections)
                and (not used_pixels)
                and (not getattr(self, "_capture_disabled", False))
                and not (self._use_bridge and self.bridge and self._bridge_loop)
            ):
                frame = self.stream.read()
                if frame is not None:
                    try:
                        self._latest_frame_ts = float(getattr(frame, "timestamp", 0.0) or time.time())
                    except Exception:
                        self._latest_frame_ts = time.time()
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
                self._latest_jpeg_bytes = latest_image_bytes
                try:
                    if not float(getattr(self, "_latest_frame_ts", 0.0) or 0.0):
                        self._latest_frame_ts = now
                except Exception:
                    self._latest_frame_ts = now

            obs, action_mask = construct_observation(
                detections, obs_dim=self._obs_dim_raw, frame_size=frame_size
            )
            obs = self._stack_observation(obs)

            # Default policy action (PPO).
            a_cont, a_disc, lp, val = self.trainer.act(obs, action_mask=action_mask)

            # Optional SIMA2 backend override (inference).
            if (
                self._use_sima2
                and self._sima2_controller is not None
                and latest_image_bytes is not None
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

            # Optional Lobby Agent override for menus/level-up screens.
            if (
                self._use_vlm_menu
                and self._vlm_menu is not None
                and latest_image_bytes is not None
                and frame_size is not None
            ):
                menu_mode = False
                if isinstance(vision_metrics, dict):
                    try:
                        if vision_metrics.get("menu_mode") is not None:
                            menu_mode = bool(vision_metrics.get("menu_mode"))
                    except Exception:
                        pass
                try:
                    cm = game_state.get("currentMenu")
                    if isinstance(cm, str) and cm.lower() not in ("none", "combat", ""):
                        menu_mode = True
                    if game_state.get("levelUpOptions"):
                        menu_mode = True
                except Exception:
                    pass
                if not menu_mode:
                    try:
                        from .perception import parse_detections, derive_state_onehot

                        st = derive_state_onehot(parse_detections(detections))
                        if (len(st) > 0 and st[0] == 1.0) or (len(st) > 2 and st[2] == 1.0):
                            menu_mode = True
                    except Exception:
                        pass

                if menu_mode:
                    suppress_policy_clicks = True
                    try:
                        hint = str(game_state.get("currentMenu") or "")
                        act = self._vlm_menu.infer_action(
                            latest_image_bytes, self._vlm_menu_goal, hint
                        )
                        kind = act.get("action")
                        if kind == "click_xy":
                            forced_ui_click = (
                                int(act.get("x", 0)),
                                int(act.get("y", 0)),
                            )
                        elif kind == "click":
                            target = str(act.get("target_text", "")).strip().lower()
                            try:
                                from .perception import (
                                    parse_detections,
                                    build_ui_elements,
                                    CLASS_NAMES,
                                )

                                dets_parsed = parse_detections(detections)
                                ui_elements, mask2, _ = build_ui_elements(
                                    dets_parsed, frame_size=frame_size
                                )
                                best_idx = None
                                for i, ui in enumerate(ui_elements):
                                    if mask2[i] != 1:
                                        continue
                                    cls = int(ui[4])
                                    cname = CLASS_NAMES.get(cls, "").replace("_", " ")
                                    if target and target in cname:
                                        best_idx = i
                                        break
                                if best_idx is None:
                                    for i, ui in enumerate(ui_elements):
                                        if mask2[i] != 1:
                                            continue
                                        cls = int(ui[4])
                                        cname = CLASS_NAMES.get(cls, "").replace("_", " ")
                                        if any(tok in target for tok in cname.split()):
                                            best_idx = i
                                            break
                                if best_idx is None:
                                    for i in range(len(ui_elements)):
                                        if mask2[i] == 1:
                                            best_idx = i
                                            break
                                if best_idx is not None:
                                    w, h = frame_size
                                    forced_ui_click = (
                                        int(ui_elements[best_idx][0] * w),
                                        int(ui_elements[best_idx][1] * h),
                                    )
                            except Exception:
                                forced_ui_click = None
                    except Exception:
                        forced_ui_click = None

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
            if self._use_research_shm and self._research_shm is not None:
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
            if self._use_bonklink and self._bonklink is not None:
                try:
                    from src.bridge.bonklink_client import BonkLinkAction
                    from .perception import parse_detections, build_ui_elements

                    move_x = float(a_cont[0]) if len(a_cont) > 0 else 0.0
                    move_y = float(a_cont[1]) if len(a_cont) > 1 else 0.0
                    action = BonkLinkAction(move_x=move_x, move_y=move_y)

                    if forced_ui_click is not None:
                        action.ui_click = True
                        action.click_x = int(forced_ui_click[0])
                        action.click_y = int(forced_ui_click[1])
                    elif (not suppress_policy_clicks) and a_disc and frame_size is not None:
                        dets_parsed = parse_detections(detections)
                        ui_elements, action_mask2, _ = build_ui_elements(
                            dets_parsed, frame_size=frame_size
                        )
                        idx = int(a_disc[0])
                        if (
                            idx != self._last_disc_action
                            and 0 <= idx < len(ui_elements)
                            and action_mask2[idx] == 1
                        ):
                            w, h = frame_size
                            cx, cy = ui_elements[idx][0], ui_elements[idx][1]
                            action.ui_click = True
                            action.click_x = int(cx * w)
                            action.click_y = int(cy * h)
                        self._last_disc_action = idx

                    self._bonklink.send_action(action)
                except Exception:
                    pass

            # If bridged, send actions into the game.
            if self._use_bridge and self.bridge and self._bridge_loop and not self._use_bonklink:
                try:
                    from .perception import parse_detections, build_ui_elements

                    dets_parsed = parse_detections(detections)
                    ui_elements, _, _ = build_ui_elements(dets_parsed, frame_size=frame_size)
                except Exception:
                    ui_elements = None
                self._bridge_send_actions(a_cont, a_disc, ui_elements, action_mask, frame_size)

            # Reward: prefer learned reward-from-video (progress score delta). No placeholder shaping.
            if self._use_learned_reward:
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
                        menu_mode = None
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
                            v = vision_metrics.get("menu_mode")
                            if v is not None:
                                menu_mode = bool(v)
                        except Exception:
                            pass
                        try:
                            from .perception import parse_detections, ICON_BOSS, MINIMAP_ICON_RED

                            dets_parsed = parse_detections(detections)
                            boss_visible = any(d.cls == ICON_BOSS for d in dets_parsed)
                            # If no explicit enemy_count, use a conservative visual proxy from minimap danger icons.
                            if enemy_count is None:
                                enemy_count = int(sum(1 for d in dets_parsed if d.cls in (MINIMAP_ICON_RED, ICON_BOSS)))
                            # If no explicit menu_mode, infer from UI state.
                            if menu_mode is None:
                                try:
                                    from .perception import derive_state_onehot

                                    st = derive_state_onehot(dets_parsed)
                                    if (len(st) > 0 and st[0] == 1.0) or (len(st) > 2 and st[2] == 1.0):
                                        menu_mode = True
                                    else:
                                        menu_mode = False
                                except Exception:
                                    pass
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
                            try:
                                if menu_mode is None:
                                    cm = game_state.get("currentMenu")
                                    if isinstance(cm, str) and cm.lower() not in ("none", "combat", ""):
                                        menu_mode = True
                                    elif game_state.get("levelUpOptions"):
                                        menu_mode = True
                                    else:
                                        menu_mode = False
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
                                "menu_mode": menu_mode,
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
                    action_mask,
                    forced_ui_click,
                    suppress_policy_clicks,
                )
                self.trainer.update(reward, meaningful=meaningful)
                self.rollout.add(obs, a_cont, a_disc, action_mask, reward, done, lp, val)

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
                    self.learner.push_rollout(batch)

            self._stop.wait(0.05)

    def _stack_observation(self, obs: List[float]) -> List[float]:
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
            "display": self.display,
            "display_name": os.environ.get("MEGABONK_AGENT_NAME"),
            "hparams": self.hparams,
            "pipewire_node": os.environ.get("PIPEWIRE_NODE"),
            "pipewire_node_ok": bool(getattr(self, "_pipewire_node_ok", False)),
            "stream_enabled": bool(getattr(self, "_stream_enabled", False)),
            "stream_require_pipewire": bool(getattr(self, "_require_pipewire_stream", False)),
            "stream_error": getattr(self, "_stream_error", None),
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
        # Always include these keys so UIs can rely on them (None when streamer is not initialized).
        out.setdefault("stream_backend", None)
        out.setdefault("streamer_last_error", None)
        out.setdefault("stream_active_clients", None)
        out.setdefault("stream_max_clients", None)
        out.setdefault("stream_busy", None)
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
                out["stream_busy"] = (
                    self.streamer.is_busy() if hasattr(self.streamer, "is_busy") else None
                )
            except Exception:
                pass
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
        if self._use_sima2 and self._sima2_controller is not None:
            try:
                out["sima2"] = self._sima2_controller.get_status()
            except Exception:
                out["sima2"] = {"enabled": True}
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
        self.policy_name = cfg.policy_name
        self.hparams = cfg.hparams
        self.trainer.policy_name = cfg.policy_name
        self.trainer.hparams = cfg.hparams
        self.rollout.policy_name = cfg.policy_name
        self.rollout.hparams = cfg.hparams
        self._policy_version = None
        self.learner.register(self.instance_id, cfg.policy_name, obs_dim=self.trainer.obs_dim)
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


service: Optional[WorkerService] = None


if app:

    @app.get("/status")
    def status():
        if not service:
            raise HTTPException(status_code=503, detail="service not initialized")
        return service.status()

    @app.get("/stream")
    def stream_video():
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

        gen = service.streamer.iter_chunks(container="mpegts")
        return StreamingResponse(gen, media_type="video/MP2T")

    @app.get("/stream.mp4")
    def stream_mp4():
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

        gen = service.streamer.iter_chunks(container="mp4")
        return StreamingResponse(gen, media_type="video/mp4")

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

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")  # type: ignore
    service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
