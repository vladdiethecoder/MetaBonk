"""Centralized configuration loading for MetaBonk.

MetaBonk historically relied on many environment variables. This module provides a
small, typed config layer that:
  - reads defaults from a YAML config file (configs/streaming.yaml by default)
  - allows env vars / CLI to override those defaults
  - validates key streaming settings early (GPU-only contract)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import yaml


def _truthy(v: object) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _deep_get(d: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out.get(k), Mapping) and isinstance(v, Mapping):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class StreamingConfig:
    backend: str = "auto"  # auto|gst|ffmpeg|pixel_obs
    container: str = "mp4"  # mp4|mpegts|h264
    codec: str = "h264"  # h264|hevc|av1
    bitrate: str = "6M"
    fps: int = 60
    gop: int = 60
    max_clients: int = 3
    nvenc_max_sessions: int = 0
    nvml_gpu_index: Optional[int] = None
    allow_gst_mp4: bool = False
    require_zero_copy: bool = False
    pipewire_fastpath: bool = True
    gst_use_cuda_upload: bool = False

    def validate(self) -> None:
        b = str(self.backend or "").strip().lower()
        if b == "obs":
            b = "ffmpeg"
        if b not in ("auto", "gst", "gstreamer", "gst-launch", "ffmpeg", "pixel_obs", "pixels"):
            raise ValueError(f"stream backend must be auto|gst|ffmpeg|pixel_obs, got {self.backend!r}")
        c = str(self.container or "").strip().lower()
        if c not in ("mp4", "mpegts", "h264"):
            raise ValueError(f"stream container must be mp4|mpegts|h264, got {self.container!r}")
        codec = str(self.codec or "").strip().lower()
        if codec in ("avc",):
            codec = "h264"
        if codec not in ("h264", "hevc", "av1"):
            raise ValueError(f"stream codec must be h264|hevc|av1, got {self.codec!r}")
        if int(self.fps) < 1:
            raise ValueError("stream fps must be >= 1")
        if int(self.gop) < 1:
            raise ValueError("stream gop must be >= 1")
        if int(self.max_clients) < 1:
            raise ValueError("stream max_clients must be >= 1")
        if int(self.nvenc_max_sessions) < 0:
            raise ValueError("stream nvenc_max_sessions must be >= 0")


@dataclass(frozen=True)
class FifoStreamConfig:
    enabled: bool = False
    container: str = "mpegts"  # mpegts|h264
    fifo_dir: str = "temp/streams"

    def validate(self) -> None:
        c = str(self.container or "").strip().lower()
        if c in ("ts",):
            c = "mpegts"
        if c not in ("mpegts", "h264"):
            raise ValueError(f"fifo container must be mpegts|h264, got {self.container!r}")
        if not str(self.fifo_dir or "").strip():
            raise ValueError("fifo_dir must be non-empty")


@dataclass(frozen=True)
class Go2rtcConfig:
    enabled: bool = False
    url: str = "http://127.0.0.1:1984"
    mode: str = "fifo"  # fifo|exec

    def validate(self) -> None:
        m = str(self.mode or "").strip().lower()
        if m not in ("fifo", "exec"):
            raise ValueError(f"go2rtc mode must be fifo|exec, got {self.mode!r}")


@dataclass(frozen=True)
class MetaBonkConfig:
    profile: str
    streaming: StreamingConfig
    fifo: FifoStreamConfig
    go2rtc: Go2rtcConfig

    def validate(self) -> None:
        self.streaming.validate()
        self.fifo.validate()
        self.go2rtc.validate()

    @classmethod
    def load(
        cls,
        *,
        profile: str,
        path: Optional[str] = None,
    ) -> "MetaBonkConfig":
        cfg_path = Path(path or os.environ.get("METABONK_STREAM_CONFIG", "configs/streaming.yaml"))
        if not cfg_path.exists():
            raise FileNotFoundError(f"stream config not found: {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"invalid YAML root in {cfg_path} (expected mapping)")

        profiles = raw.get("profiles")
        if not isinstance(profiles, Mapping):
            raise ValueError(f"invalid YAML in {cfg_path}: missing profiles mapping")

        base = profiles.get("base") if isinstance(profiles.get("base"), Mapping) else {}
        prof = profiles.get(profile) if isinstance(profiles.get(profile), Mapping) else None
        if prof is None:
            raise ValueError(f"unknown stream profile {profile!r} (expected one of: {', '.join(sorted(profiles.keys()))})")

        merged = _deep_merge(base, prof) if base else dict(prof)

        raw_nvml_idx = _deep_get(merged, "streaming.nvml_gpu_index", None)
        nvml_gpu_index: Optional[int]
        if raw_nvml_idx is None:
            nvml_gpu_index = None
        else:
            s = str(raw_nvml_idx).strip()
            if not s:
                nvml_gpu_index = None
            else:
                nvml_gpu_index = int(s)

        stream = StreamingConfig(
            backend=str(_deep_get(merged, "streaming.backend", "auto") or "auto"),
            container=str(_deep_get(merged, "streaming.container", "mp4") or "mp4"),
            codec=str(_deep_get(merged, "streaming.codec", "h264") or "h264"),
            bitrate=str(_deep_get(merged, "streaming.bitrate", "6M") or "6M"),
            fps=int(_deep_get(merged, "streaming.fps", 60) or 60),
            gop=int(_deep_get(merged, "streaming.gop", 60) or 60),
            max_clients=int(_deep_get(merged, "streaming.max_clients", 3) or 3),
            nvenc_max_sessions=int(_deep_get(merged, "streaming.nvenc_max_sessions", 0) or 0),
            nvml_gpu_index=nvml_gpu_index,
            allow_gst_mp4=bool(_truthy(_deep_get(merged, "streaming.allow_gst_mp4", False))),
            require_zero_copy=bool(_truthy(_deep_get(merged, "streaming.require_zero_copy", False))),
            pipewire_fastpath=bool(_truthy(_deep_get(merged, "streaming.pipewire_fastpath", True))),
            gst_use_cuda_upload=bool(_truthy(_deep_get(merged, "streaming.gst_use_cuda_upload", False))),
        )

        fifo = FifoStreamConfig(
            enabled=bool(_truthy(_deep_get(merged, "fifo.enabled", False))),
            container=str(_deep_get(merged, "fifo.container", "mpegts") or "mpegts"),
            fifo_dir=str(_deep_get(merged, "fifo.dir", "temp/streams") or "temp/streams"),
        )

        go2rtc = Go2rtcConfig(
            enabled=bool(_truthy(_deep_get(merged, "go2rtc.enabled", False))),
            url=str(_deep_get(merged, "go2rtc.url", "http://127.0.0.1:1984") or "http://127.0.0.1:1984"),
            mode=str(_deep_get(merged, "go2rtc.mode", "fifo") or "fifo"),
        )

        cfg = cls(profile=str(profile), streaming=stream, fifo=fifo, go2rtc=go2rtc)
        cfg.validate()
        return cfg

    def apply_to_env(self, env: MutableMapping[str, str], *, overwrite: bool = False) -> None:
        def _set(k: str, v: str) -> None:
            if overwrite or not str(env.get(k, "")).strip():
                env[k] = v

        _set("METABONK_STREAM_BACKEND", str(self.streaming.backend))
        _set("METABONK_STREAM_CONTAINER", str(self.streaming.container))
        _set("METABONK_STREAM_CODEC", str(self.streaming.codec))
        _set("METABONK_STREAM_BITRATE", str(self.streaming.bitrate))
        _set("METABONK_STREAM_FPS", str(int(self.streaming.fps)))
        _set("METABONK_STREAM_GOP", str(int(self.streaming.gop)))
        _set("METABONK_STREAM_MAX_CLIENTS", str(int(self.streaming.max_clients)))
        _set("METABONK_NVENC_MAX_SESSIONS", str(int(self.streaming.nvenc_max_sessions)))
        if self.streaming.nvml_gpu_index is not None:
            _set("METABONK_NVML_GPU_INDEX", str(int(self.streaming.nvml_gpu_index)))
        _set("METABONK_STREAM_ALLOW_GST_MP4", "1" if self.streaming.allow_gst_mp4 else "0")
        _set("METABONK_STREAM_REQUIRE_ZERO_COPY", "1" if self.streaming.require_zero_copy else "0")
        _set("METABONK_STREAM_PIPEWIRE_FASTPATH", "1" if self.streaming.pipewire_fastpath else "0")
        _set("METABONK_GST_USE_CUDA_UPLOAD", "1" if self.streaming.gst_use_cuda_upload else "0")

        _set("METABONK_FIFO_STREAM", "1" if self.fifo.enabled else "0")
        _set("METABONK_FIFO_CONTAINER", str(self.fifo.container))
        _set("METABONK_STREAM_FIFO_DIR", str(self.fifo.fifo_dir))

        _set("METABONK_ENABLE_PUBLIC_STREAM", "1" if self.go2rtc.enabled else "0")
        # Back-compat: many scripts/worker code use METABONK_GO2RTC as the go2rtc/FIFO toggle.
        _set("METABONK_GO2RTC", "1" if self.go2rtc.enabled else "0")
        _set("METABONK_GO2RTC_URL", str(self.go2rtc.url))
        _set("METABONK_GO2RTC_MODE", str(self.go2rtc.mode))


def apply_streaming_profile(
    env: MutableMapping[str, str],
    *,
    mode: str,
    profile: Optional[str] = None,
    config_path: Optional[str] = None,
    overwrite: bool = False,
) -> MetaBonkConfig:
    """Load configs/streaming.yaml and apply a profile into env.

    Env vars/CLI should override config, so overwrite defaults to False.
    """
    chosen = str(profile or env.get("METABONK_STREAM_PROFILE") or "").strip()
    if not chosen:
        chosen = "prod" if str(mode or "").strip().lower() in ("train", "play") else "dev"
    cfg = MetaBonkConfig.load(profile=chosen, path=config_path)
    cfg.apply_to_env(env, overwrite=overwrite)
    return cfg


__all__ = ["MetaBonkConfig", "apply_streaming_profile"]
