"""Switching controller between PPO and System-2 menu override."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .menu_reasoner import MenuAction, MenuReasoner, MenuReasonerConfig
from src.vision.som_preprocess import SoMPreprocessor, SoMConfig
from src.vision.frame_filters import TemporalEMA


@dataclass
class SwitchingConfig:
    enabled: bool = True
    goal: str = os.environ.get("METABONK_MENU_GOAL", "Start Run")
    postrun_goal: str = os.environ.get("METABONK_MENU_POSTRUN_GOAL", "Continue")
    postrun_keywords: str = os.environ.get(
        "METABONK_MENU_POSTRUN_KEYWORDS",
        "death,gameover,game over,stats,stat,summary,results,run over,defeat",
    )
    menu_hold_s: float = float(os.environ.get("METABONK_MENU_HOLD_S", "0.2") or 0.2)
    action_cooldown_s: float = float(os.environ.get("METABONK_MENU_ACTION_COOLDOWN_S", "0.6") or 0.6)
    max_retries: int = int(os.environ.get("METABONK_MENU_MAX_RETRIES", "5") or 5)
    fallback_key: str = os.environ.get("METABONK_MENU_FALLBACK_KEY", "ENTER")
    dump_dir: str = os.environ.get("METABONK_SOM_DUMP_DIR", "")
    log: bool = os.environ.get("METABONK_MENU_LOG", "1") in ("1", "true", "True")
    frame_smooth_alpha: float = float(os.environ.get("METABONK_MENU_FRAME_ALPHA", "0") or 0)


class SwitchingController:
    def __init__(
        self,
        cfg: Optional[SwitchingConfig] = None,
        som_cfg: Optional[SoMConfig] = None,
        reasoner_cfg: Optional[MenuReasonerConfig] = None,
    ) -> None:
        self.cfg = cfg or SwitchingConfig()
        self.som = SoMPreprocessor(som_cfg)
        self.reasoner = MenuReasoner(reasoner_cfg)
        self._frame_filter = None
        if 0.0 < float(self.cfg.frame_smooth_alpha) < 1.0:
            self._frame_filter = TemporalEMA(alpha=float(self.cfg.frame_smooth_alpha))
        self._menu_since: float = 0.0
        self._last_action_ts: float = 0.0
        self._retry_count: int = 0

    def reset(self) -> None:
        self._menu_since = 0.0
        self._last_action_ts = 0.0
        self._retry_count = 0
        if self._frame_filter is not None:
            self._frame_filter.reset()

    def _log(self, msg: str) -> None:
        if self.cfg.log:
            print(f"[switch] {msg}")

    def _dump(self, overlay, mapping: List[Dict[str, Any]]) -> None:
        if not self.cfg.dump_dir:
            return
        try:
            ts = int(time.time() * 1000)
            out_dir = Path(self.cfg.dump_dir).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            img_path = out_dir / f"som_{ts}.jpg"
            json_path = out_dir / f"som_{ts}.json"
            overlay.save(img_path, format="JPEG", quality=85)
            json_path.write_text(json.dumps(mapping, indent=2))
        except Exception:
            pass

    def step(
        self,
        frame: Any,
        menu_hint: bool,
        *,
        detections: Optional[Sequence[Dict[str, Any]]] = None,
        hint: str = "",
    ) -> Optional[MenuAction]:
        if not self.cfg.enabled:
            return None
        if self._frame_filter is not None:
            try:
                frame = self._frame_filter.apply(frame)
            except Exception:
                pass
        now = time.time()
        if not menu_hint:
            if self._menu_since:
                self._log("menu cleared; resuming PPO")
            self.reset()
            return None
        if not self._menu_since:
            self._menu_since = now
            self._retry_count = 0
        if (now - self._menu_since) < self.cfg.menu_hold_s:
            return None
        if (now - self._last_action_ts) < self.cfg.action_cooldown_s:
            return None
        if self._retry_count >= self.cfg.max_retries:
            self._log("max retries reached; sending fallback key")
            self._last_action_ts = now
            self._retry_count += 1
            return MenuAction(kind="key", key=self.cfg.fallback_key, reason="retry_cap")

        overlay, _, mapping = self.som.process(frame, detections=detections)
        self._dump(overlay, mapping)
        goal = self.cfg.goal
        try:
            hint_l = (hint or "").strip().lower()
            if hint_l:
                keywords = [k.strip() for k in str(self.cfg.postrun_keywords).split(",") if k.strip()]
                if any(k in hint_l for k in keywords):
                    goal = self.cfg.postrun_goal
        except Exception:
            goal = self.cfg.goal
        act = self.reasoner.infer_action(overlay, mapping, goal=goal, hint=hint)
        if act is None:
            self._log("no VLM action; fallback key")
            self._last_action_ts = now
            self._retry_count += 1
            return MenuAction(kind="key", key=self.cfg.fallback_key, reason="no_action")

        self._last_action_ts = now
        self._retry_count += 1
        self._log(f"action={act.kind} id={act.target_id} reason={act.reason}")
        return act


__all__ = ["SwitchingController", "SwitchingConfig"]
