"""Megabonk Custom Gymnasium Environment.

Encapsulates all game interaction:
- Ultra-low latency capture via DXGI
- Action execution via backend bridge (no OS-level key/mouse injection here)
- Reward computation via learned reward-from-video model (no hand-authored gameplay rules)

References:
- Gymnasium API specification
- Custom environment design patterns
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYM = True
    except ImportError:
        HAS_GYM = False


@dataclass
class MegabonkEnvConfig:
    """Configuration for Megabonk environment."""
    
    # Capture
    frame_size: Tuple[int, int] = (128, 128)
    frame_stack: int = 4
    capture_fps: int = 60
    
    # Action (generic, no hard-coded game semantics)
    action_repeat: int = 1  # repeat the same action for N frames
    button_keys: Tuple[str, ...] = field(default_factory=tuple)  # e.g. ("Space","E",...)
    mouse_scale_px: float = 40.0  # max pixels moved per step when look is +/-1
    
    # Timing
    step_duration_ms: float = 16.67  # 60 Hz
    max_episode_steps: int = 108000  # 30 minutes at 60 Hz
    
    # Reward
    reward_model_ckpt: str = "checkpoints/video_reward_model.pt"
    reward_scale: float = 1.0
    
    # Debug
    render_debug: bool = False


if HAS_GYM:
    class MegabonkEnv(gym.Env):
        """Megabonk Gymnasium environment."""
        
        metadata = {"render_modes": ["human", "rgb_array"]}
        
        def __init__(
            self,
            cfg: Optional[MegabonkEnvConfig] = None,
            render_mode: Optional[str] = None,
        ):
            super().__init__()
            self.cfg = cfg or MegabonkEnvConfig()
            self.render_mode = render_mode

            # Allow simple config via env var for dev (comma-separated).
            if not self.cfg.button_keys:
                env_keys = os.environ.get("METABONK_BUTTON_KEYS", "")
                if env_keys.strip():
                    self.cfg.button_keys = tuple([k.strip() for k in env_keys.split(",") if k.strip() != ""])

            # Observation: stacked RGB frames (CHW).
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(int(self.cfg.frame_stack), 3, int(self.cfg.frame_size[0]), int(self.cfg.frame_size[1])),
                dtype=np.uint8,
            )

            # Action: generic primitives only (no built-in "WASD"/aim semantics).
            self.action_space = spaces.Dict(
                {
                    "buttons": spaces.MultiBinary(len(self.cfg.button_keys)),
                    "look": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                    "click": spaces.MultiBinary(1),
                }
            )

            # Bridge/capture: require UnityBridge (real game backend). No simulated fallback.
            from src.bridge.unity_bridge import BridgeConfig, UnityBridge  # type: ignore

            bcfg = BridgeConfig(
                socket_path=os.environ.get("METABONK_BRIDGE_SOCKET", BridgeConfig().socket_path),
                pipe_name=os.environ.get("METABONK_BRIDGE_PIPE", BridgeConfig().pipe_name),
                use_socket=os.environ.get("METABONK_BRIDGE_USE_SOCKET", "1") in ("1", "true", "True"),
                frame_width=int(os.environ.get("METABONK_BRIDGE_FRAME_W", BridgeConfig().frame_width)),
                frame_height=int(os.environ.get("METABONK_BRIDGE_FRAME_H", BridgeConfig().frame_height)),
                frame_format=os.environ.get("METABONK_BRIDGE_FRAME_FMT", BridgeConfig().frame_format),
            )
            self._bridge_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._bridge_loop)
            self.bridge = UnityBridge(bcfg)
            ok = self._bridge_loop.run_until_complete(self.bridge.connect())
            if not ok:
                raise RuntimeError(
                    "Failed to connect to UnityBridge. Start MegaBonk with the MetaBonk bridge plugin enabled "
                    f"(socket={bcfg.socket_path!r})."
                )

            # Learned reward model (from video).
            try:
                import torch
                from src.imitation.video_pretraining import TemporalRankRewardModel  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "MegabonkEnv requires torch + the video reward model utilities. "
                    "Install requirements and ensure src/imitation/video_pretraining.py is available."
                ) from e

            self._torch = torch
            self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt_path = self.cfg.reward_model_ckpt
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Reward model checkpoint not found: {ckpt_path}. "
                    "Train it with `python scripts/video_pretrain.py --phase reward_train`."
                )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            conf = ckpt.get("config") or {}
            self._reward_frame_size = tuple(conf.get("frame_size") or (224, 224))
            self._reward_embed_dim = int(conf.get("embed_dim") or 256)
            self.reward_model = TemporalRankRewardModel(
                frame_size=self._reward_frame_size, embed_dim=self._reward_embed_dim
            ).to(self._torch_device)
            self.reward_model.load_state_dict(ckpt.get("model_state_dict") or {})
            self.reward_model.eval()

            # Internal buffers/state.
            self.frame_buffer: List[np.ndarray] = []
            self._prev_progress_score: Optional[float] = None
            self._mouse_x: float = float(self.bridge.cfg.frame_width) / 2.0
            self._mouse_y: float = float(self.bridge.cfg.frame_height) / 2.0
            self._key_state: Dict[str, bool] = {k: False for k in self.cfg.button_keys}
            self.steps = 0
            self.episode_start_time = 0.0
            self._last_frame = None
        
        def _read_frame(self):
            try:
                return self._bridge_loop.run_until_complete(self.bridge.read_frame())
            except Exception:
                return None

        def _resize_hwc(self, frame_hwc: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
            """Resize HWC uint8 frame to (H,W)."""
            th, tw = int(target_hw[0]), int(target_hw[1])
            if frame_hwc.shape[0] == th and frame_hwc.shape[1] == tw:
                return frame_hwc
            try:
                import cv2  # type: ignore

                return cv2.resize(frame_hwc, (tw, th), interpolation=cv2.INTER_AREA)
            except Exception:
                try:
                    from PIL import Image

                    img = Image.fromarray(frame_hwc)
                    img = img.resize((tw, th))
                    return np.asarray(img)
                except Exception:
                    # Last resort: nearest-neighbor via slicing (may distort).
                    return frame_hwc[:th, :tw]

        def _progress_score(self, frame_hwc: np.ndarray) -> float:
            torch = self._torch
            import torch.nn.functional as F  # type: ignore

            # uint8 HWC -> float BCHW [0,1]
            f = torch.from_numpy(frame_hwc).to(device=self._torch_device, dtype=torch.uint8)
            f = f.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255.0
            f = F.interpolate(f, size=(int(self._reward_frame_size[0]), int(self._reward_frame_size[1])), mode="bilinear", align_corners=False)
            with torch.no_grad():
                s = self.reward_model(f).detach().to("cpu").float().item()
            return float(s)
        
        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None,
        ) -> Tuple[np.ndarray, Dict]:
            """Reset environment."""
            super().reset(seed=seed)

            self.steps = 0
            self.episode_start_time = time.time()
            self._prev_progress_score = None
            self.frame_buffer.clear()
            self._mouse_x = float(self.bridge.cfg.frame_width) / 2.0
            self._mouse_y = float(self.bridge.cfg.frame_height) / 2.0
            self._key_state = {k: False for k in self.cfg.button_keys}

            # Prime frame stack from the real backend.
            first = None
            for _ in range(max(1, int(self.cfg.frame_stack))):
                fr = self._read_frame()
                if fr is not None:
                    first = fr
                    self._last_frame = fr
                    self._push_frame(fr)
            if first is None:
                raise RuntimeError("Failed to read initial frame from UnityBridge.")

            obs = np.stack(self.frame_buffer, axis=0)
            info = self._info_from_frame(first)
            return obs, info
        
        def step(
            self,
            action: Dict[str, Any],
        ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
            """Execute action and return new state."""

            self._apply_action(action)

            # Optional pacing (when the bridge does not stream at the desired rate).
            if os.environ.get("METABONK_ENV_NO_SLEEP", "1") not in ("1", "true", "True"):
                time.sleep(float(self.cfg.step_duration_ms) / 1000.0)

            reward_total = 0.0
            last = None
            n = max(1, int(self.cfg.action_repeat))
            for _ in range(n):
                fr = self._read_frame()
                if fr is None:
                    continue
                last = fr
                self._last_frame = fr
                reward_total += self._reward_from_frame(fr)
                self._push_frame(fr)

            if last is None:
                raise RuntimeError("Failed to read frame from UnityBridge during step().")

            self.steps += 1

            obs = np.stack(self.frame_buffer, axis=0)
            info = self._info_from_frame(last)

            # Termination (best-effort from extracted game state).
            terminated = bool(info.get("terminated", False))
            truncated = bool(self.steps >= int(self.cfg.max_episode_steps))
            return obs, float(reward_total), terminated, truncated, info
        
        def _push_frame(self, fr) -> None:
            """Append a frame to the stack buffer (CHW uint8)."""
            pixels = fr.pixels
            frame_hwc = np.asarray(pixels)
            if frame_hwc.dtype != np.uint8:
                frame_hwc = np.clip(frame_hwc, 0, 255).astype(np.uint8)
            frame_hwc = self._resize_hwc(frame_hwc, self.cfg.frame_size)
            frame_chw = np.transpose(frame_hwc, (2, 0, 1))
            self.frame_buffer.append(frame_chw)
            if len(self.frame_buffer) > int(self.cfg.frame_stack):
                self.frame_buffer.pop(0)
            while len(self.frame_buffer) < int(self.cfg.frame_stack):
                self.frame_buffer.insert(0, frame_chw)
        
        def _apply_action(self, action: Dict[str, Any]) -> None:
            """Send an action dict to the Unity bridge.

            Action format (generic primitives):
              - buttons: iterable of {0,1} with length == len(cfg.button_keys)
              - look: [dx, dy] in [-1,1] (mouse delta scaled by cfg.mouse_scale_px)
              - click: [0|1] (left click at current cursor position)
            """
            # Buttons -> key up/down edges.
            btn = action.get("buttons")
            if btn is not None and self.cfg.button_keys:
                try:
                    btn_arr = [bool(x) for x in list(btn)]
                except Exception:
                    btn_arr = []
                for i, key in enumerate(self.cfg.button_keys):
                    want = bool(btn_arr[i]) if i < len(btn_arr) else False
                    prev = bool(self._key_state.get(key, False))
                    if want != prev:
                        try:
                            self._bridge_loop.run_until_complete(self.bridge.send_key(str(key), pressed=want))
                        except Exception:
                            pass
                        self._key_state[key] = want

            # Look (mouse move) -> absolute cursor update.
            look = action.get("look")
            if look is not None:
                try:
                    dx = float(list(look)[0])
                    dy = float(list(look)[1])
                except Exception:
                    dx = dy = 0.0
                self._mouse_x = float(np.clip(self._mouse_x + dx * float(self.cfg.mouse_scale_px), 0.0, float(self.bridge.cfg.frame_width - 1)))
                self._mouse_y = float(np.clip(self._mouse_y + dy * float(self.cfg.mouse_scale_px), 0.0, float(self.bridge.cfg.frame_height - 1)))
                try:
                    self._bridge_loop.run_until_complete(
                        self.bridge.send_mouse_move(int(self._mouse_x), int(self._mouse_y))
                    )
                except Exception:
                    pass

            # Click (left).
            click = action.get("click")
            do_click = False
            if click is not None:
                try:
                    arr = list(click)
                    do_click = bool(arr[0]) if arr else bool(click)
                except Exception:
                    do_click = bool(click)
            if do_click:
                try:
                    self._bridge_loop.run_until_complete(
                        self.bridge.send_mouse_click(int(self._mouse_x), int(self._mouse_y), 0)
                    )
                except Exception:
                    pass
        
        def _reward_from_frame(self, fr) -> float:
            frame_hwc = np.asarray(fr.pixels)
            if frame_hwc.dtype != np.uint8:
                frame_hwc = np.clip(frame_hwc, 0, 255).astype(np.uint8)
            s = self._progress_score(frame_hwc)
            if self._prev_progress_score is None:
                self._prev_progress_score = s
                return 0.0
            r = (s - float(self._prev_progress_score)) * float(self.cfg.reward_scale)
            self._prev_progress_score = s
            return float(r)
        
        def _info_from_frame(self, fr) -> Dict[str, Any]:
            state = {}
            try:
                state = dict(fr.state or {})
            except Exception:
                state = {}
            hp = state.get("playerHealth") or state.get("hp")
            dead = state.get("isDead") or state.get("dead")
            terminated = False
            try:
                if dead is True:
                    terminated = True
                if hp is not None and float(hp) <= 0.0:
                    terminated = True
            except Exception:
                pass
            return {
                "timestamp": float(fr.timestamp),
                "frame_idx": int(getattr(fr, "frame_idx", 0) or 0),
                "steps": int(self.steps),
                "terminated": bool(terminated),
                "state": state,
                "ui_elements": getattr(fr, "ui_elements", None),
            }
        
        def render(self) -> Optional[np.ndarray]:
            """Render environment."""
            if self.render_mode == "rgb_array":
                if self.frame_buffer:
                    return np.transpose(self.frame_buffer[-1], (1, 2, 0))
            return None
        
        def close(self):
            """Clean up resources."""
            try:
                self._bridge_loop.run_until_complete(self.bridge.disconnect())
            except Exception:
                pass
            try:
                self._bridge_loop.close()
            except Exception:
                pass


def register_megabonk_env():
    """Register Megabonk environment with Gymnasium."""
    if HAS_GYM:
        gym.register(
            id="Megabonk-v1",
            entry_point="src.env.megabonk_gym:MegabonkEnv",
            max_episode_steps=108000,
        )
