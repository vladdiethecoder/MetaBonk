"""Gymnasium environment wrapper for MegaBonk (YOLO + masked RL).

This is a scaffold matching the hybrid architecture:
  - Capture frame (Gamescope/Sidecar or desktop)
  - Run YOLO to detect UI + gameplay entities
  - Convert detections to structured obs + action mask
  - Map discrete UI action index -> click target

Real capture/injection is platform specific and should be implemented in
`ScreenCapture` and `InputInjector`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

try:
    import torch
    import torch.nn.functional as F  # type: ignore

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    F = None  # type: ignore

from src.worker.perception import (
    build_ui_elements,
    construct_observation,
    parse_detections,
    TILE_CHAR,
    CARD_MAP,
)
from src.worker.menu import parse_character_select, parse_map_select


@dataclass
class EnvConfig:
    max_ui_elements: int = 32
    obs_dim: int = 204
    yolo_weights: str = "yolo11n.pt"
    frame_skip: int = 4
    # Capture overrides (best-effort; optional).
    capture_resize: Optional[Tuple[int, int]] = None
    capture_region: Optional[Tuple[int, int, int, int]] = None
    # Learned reward-from-video checkpoint (required; no placeholder rewards).
    reward_model_ckpt: str = "checkpoints/video_reward_model.pt"
    reward_scale: float = 1.0


class ScreenCapture:
    """Best-effort screen capture abstraction.

    Priority:
      1) `METABONK_CAPTURE_VIDEO` (OpenCV)
      2) `METABONK_CAPTURE_IMAGES_DIR` (PIL)
      3) `src.perception.capture` backends (DXGI/MSS where available)
      4) PIL.ImageGrab (where supported)
    """

    def __init__(
        self,
        resize_to: Optional[Tuple[int, int]] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.resize_to = resize_to
        self.region = region

        self._cap = None
        self._video = None
        self._images: Optional[List[str]] = None
        self._img_idx = 0

        video_path = os.environ.get("METABONK_CAPTURE_VIDEO")
        if video_path:
            try:
                import cv2  # type: ignore

                self._video = cv2.VideoCapture(video_path)
            except Exception:
                self._video = None

        images_dir = os.environ.get("METABONK_CAPTURE_IMAGES_DIR")
        if images_dir:
            try:
                from pathlib import Path

                exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
                paths = [str(p) for p in Path(images_dir).glob("*") if p.suffix.lower() in exts]
                paths.sort()
                if paths:
                    self._images = paths
            except Exception:
                self._images = None

        if self._video is None and self._images is None:
            try:
                from src.perception.capture import CaptureConfig, create_capture

                cfg = CaptureConfig(
                    target_fps=int(os.environ.get("METABONK_CAPTURE_FPS", "60")),
                    region=self.region,
                    resize_to=self.resize_to,
                )
                self._cap = create_capture(cfg)
            except Exception:
                self._cap = None

        # Optional overrides via env (string -> tuple parsing).
        if self.resize_to is None:
            rt = os.environ.get("METABONK_CAPTURE_RESIZE")
            if rt and "x" in rt:
                try:
                    w, h = rt.lower().split("x", 1)
                    self.resize_to = (int(w), int(h))
                except Exception:
                    pass
        if self.region is None:
            rg = os.environ.get("METABONK_CAPTURE_REGION")
            if rg and "," in rg:
                try:
                    parts = [int(p.strip()) for p in rg.split(",")]
                    if len(parts) == 4:
                        self.region = (parts[0], parts[1], parts[2], parts[3])
                except Exception:
                    pass

    def grab(self):
        # Video replay path.
        if self._video is not None:
            try:
                import cv2  # type: ignore

                ok, frame_bgr = self._video.read()
                if not ok or frame_bgr is None:
                    # Loop.
                    self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame_bgr = self._video.read()
                if not ok or frame_bgr is None:
                    return None, None
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if self.resize_to:
                    frame_rgb = cv2.resize(frame_rgb, self.resize_to)
                h, w = frame_rgb.shape[:2]
                return frame_rgb, (w, h)
            except Exception:
                return None, None

        # Image sequence replay path.
        if self._images:
            try:
                from PIL import Image

                p = self._images[self._img_idx % len(self._images)]
                self._img_idx += 1
                img = Image.open(p).convert("RGB")
                if self.resize_to:
                    img = img.resize(self.resize_to)
                w, h = img.size
                return img, (w, h)
            except Exception:
                return None, None

        # `src.perception.capture` path.
        if self._cap is not None:
            try:
                frame = None
                if hasattr(self._cap, "capture_sync"):
                    frame = self._cap.capture_sync()
                elif hasattr(self._cap, "get_frame"):
                    cf = self._cap.get_frame()
                    frame = getattr(cf, "frame", None) if cf else None
                if frame is None:
                    return None, None
                # frame can be numpy HWC.
                h = getattr(frame, "shape", (0, 0))[0] if hasattr(frame, "shape") else 0
                w = getattr(frame, "shape", (0, 0))[1] if hasattr(frame, "shape") else 0
                if w and h:
                    return frame, (int(w), int(h))
                if hasattr(frame, "size"):
                    w2, h2 = frame.size  # PIL
                    return frame, (int(w2), int(h2))
                return frame, None
            except Exception:
                return None, None

        # Final fallback: PIL.ImageGrab when supported.
        try:
            from PIL import ImageGrab

            img = ImageGrab.grab(bbox=self.region).convert("RGB")
            if self.resize_to:
                img = img.resize(self.resize_to)
            w, h = img.size
            return img, (w, h)
        except Exception:
            return None, None


class InputInjector:
    """Best-effort input injection.

    Uses `pyautogui` if installed; otherwise becomes a no-op.
    """

    def __init__(self):
        self._pg = None
        try:
            import pyautogui  # type: ignore

            self._pg = pyautogui
        except Exception:
            self._pg = None

    def click(self, x: float, y: float):
        if self._pg is None:
            return
        try:
            self._pg.click(x=int(x), y=int(y))
        except Exception:
            return

    def move(self, x: float, y: float):
        if self._pg is None:
            return
        try:
            self._pg.moveTo(int(x), int(y))
        except Exception:
            return


class MegaBonkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.cap = ScreenCapture(resize_to=self.cfg.capture_resize, region=self.cfg.capture_region)
        self.injector = InputInjector()

        self.yolo = YOLO(self.cfg.yolo_weights) if YOLO else None

        if not TORCH_AVAILABLE:
            raise RuntimeError("MegaBonkEnv requires torch to compute learned rewards (no placeholder rewards allowed).")
        from src.imitation.video_pretraining import TemporalRankRewardModel  # type: ignore

        ckpt_path = self.cfg.reward_model_ckpt
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Reward model checkpoint not found: {ckpt_path}. "
                "Train it with `python scripts/video_pretrain.py --phase reward_train`."
            )
        self._rm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        conf = ckpt.get("config") or {}
        self._rm_frame_size = tuple(conf.get("frame_size") or (224, 224))
        self._rm_embed_dim = int(conf.get("embed_dim") or 256)
        self._rm = TemporalRankRewardModel(frame_size=self._rm_frame_size, embed_dim=self._rm_embed_dim).to(self._rm_device)
        self._rm.load_state_dict(ckpt.get("model_state_dict") or {})
        self._rm.eval()
        self._prev_score: Optional[float] = None

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(-1.0, 1.0, shape=(self.cfg.obs_dim,), dtype=float),
                "action_mask": spaces.Box(0, 1, shape=(self.cfg.max_ui_elements + 1,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(self.cfg.max_ui_elements + 1)

        self._last_obs: List[float] = [0.0] * self.cfg.obs_dim
        self._last_mask: List[int] = [0] * (self.cfg.max_ui_elements + 1)
        self._last_ui_elements: List[List[float]] = [
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0] for _ in range(self.cfg.max_ui_elements)
        ]
        self._last_frame_size: Optional[Tuple[int, int]] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._last_obs = [0.0] * self.cfg.obs_dim
        self._last_mask = [0] * (self.cfg.max_ui_elements + 1)
        self._prev_score = None
        return {"obs": self._last_obs, "action_mask": self._last_mask}, {}

    def _progress_score(self, frame_rgb) -> float:
        """Compute learned progress score from a single RGB frame (PIL or np)."""
        assert TORCH_AVAILABLE and torch is not None and F is not None
        import numpy as np

        arr = frame_rgb if isinstance(frame_rgb, np.ndarray) else np.asarray(frame_rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        t = torch.from_numpy(arr).to(device=self._rm_device, dtype=torch.uint8)
        t = t.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255.0
        t = F.interpolate(t, size=(int(self._rm_frame_size[0]), int(self._rm_frame_size[1])), mode="bilinear", align_corners=False)
        with torch.no_grad():
            s = self._rm(t).detach().to("cpu").float().item()
        return float(s)

    def _reward_from_frame(self, frame_rgb) -> Tuple[float, float]:
        """Return (reward, score) from the current frame."""
        s = self._progress_score(frame_rgb)
        if self._prev_score is None:
            self._prev_score = s
            return 0.0, s
        r = (s - float(self._prev_score)) * float(self.cfg.reward_scale)
        self._prev_score = s
        return float(r), s

    def step(self, action: int):
        # Execute UI action if valid.
        if (
            action < self.cfg.max_ui_elements
            and self._last_mask[action] == 1
            and self._last_frame_size is not None
        ):
            try:
                w, h = self._last_frame_size
                cx, cy = self._last_ui_elements[action][0], self._last_ui_elements[action][1]
                x = int(max(0.0, min(1.0, float(cx))) * w)
                y = int(max(0.0, min(1.0, float(cy))) * h)
                self.injector.click(x, y)
            except Exception:
                pass

        # Frame skip: repeat action for N frames.
        detections: List[Dict[str, Any]] = []
        frame_size: Optional[Tuple[int, int]] = None
        last_frame = None
        for _ in range(self.cfg.frame_skip):
            frame, frame_size = self.cap.grab()
            if frame is None:
                continue
            last_frame = frame
            if self.yolo is None:
                continue
            res = self.yolo.predict(frame, verbose=False)
            # Ultralytics results to list of dicts.
            for r in res:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                for b in r.boxes:
                    detections.append(
                        {
                            "cls": int(b.cls.item()) if hasattr(b, "cls") else -1,
                            "conf": float(b.conf.item()) if hasattr(b, "conf") else 0.0,
                            "xyxy": b.xyxy.cpu().tolist()[0] if hasattr(b, "xyxy") else None,
                        }
                    )

        obs, mask = construct_observation(
            detections,
            obs_dim=self.cfg.obs_dim,
            frame_size=frame_size,
            max_elements=self.cfg.max_ui_elements,
        )
        # Store UI element centers for click mapping.
        try:
            dets_parsed = parse_detections(detections)
            ui_elements, mask2, _ = build_ui_elements(
                dets_parsed, frame_size=frame_size, max_elements=self.cfg.max_ui_elements
            )
        except Exception:
            ui_elements, mask2 = (
                [[0.0, 0.0, 0.0, 0.0, -1.0, 0.0] for _ in range(self.cfg.max_ui_elements)],
                mask,
            )
        self._last_obs = obs
        self._last_mask = mask2
        self._last_ui_elements = ui_elements
        self._last_frame_size = frame_size

        if last_frame is None:
            raise RuntimeError("No frames captured; cannot compute learned reward (placeholder rewards are not allowed).")
        reward, score = self._reward_from_frame(last_frame)
        terminated = False
        truncated = False
        info = {"action_mask": mask, "progress_score": score}

        # Optional structured menu state via OCR.
        if last_frame is not None:
            try:
                dets_parsed = parse_detections(detections)
                classes = {d.cls for d in dets_parsed}
                if TILE_CHAR in classes:
                    # Ensure PIL frame for OCR.
                    from PIL import Image
                    import numpy as np

                    pil_frame = (
                        last_frame
                        if isinstance(last_frame, Image.Image)
                        else Image.fromarray(np.asarray(last_frame))
                    )
                    info["menu_state"] = parse_character_select(pil_frame, dets_parsed)
                elif CARD_MAP in classes:
                    info["menu_state"] = parse_map_select(dets_parsed)
            except Exception:
                pass
        return {"obs": obs, "action_mask": mask}, reward, terminated, truncated, info
