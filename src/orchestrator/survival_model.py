"""Survival probability model for the "Choke Meter".

This is a lightweight, online estimator:
  - Workers periodically send Telemetry events with raw features.
  - We discretize features into a compact key.
  - For each key, estimate P(survive next H seconds) via counts with smoothing.

We define "survive" over a fixed horizon (H seconds):
  - When a key observation ages out of the horizon without an episode ending,
    it counts as a survival.
  - If an episode ends within the horizon after an observation, it counts as a death.

This yields a real-time survival probability curve that can be visualized as
a Choke/Danger gauge and used for "clutch" detection (<1% then recovery).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now() -> float:
    return time.time()


def _bucket_enemy_count(n: Optional[int]) -> str:
    if n is None:
        return "u"
    n = int(n)
    if n <= 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "10+"


def _bucket_hp_ratio(x: Optional[float]) -> str:
    if x is None:
        return "u"
    try:
        x = float(x)
    except Exception:
        return "u"
    if x < 0:
        x = 0.0
    if x > 1:
        x = 1.0
    b = int(x * 10.0)  # 0..10
    if b >= 10:
        b = 9
    return str(b)  # decile bucket


def _cell_from_pos(pos: Any, grid: float = 10.0) -> Optional[Tuple[int, int]]:
    """Return coarse (x,z) cell if pos looks like (x,y,z)."""
    try:
        if pos is None:
            return None
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            x = float(pos[0])
            z = float(pos[2])
            return (int(round(x / grid)), int(round(z / grid)))
    except Exception:
        return None
    return None


def build_key(features: Dict[str, Any]) -> str:
    """Discretize raw telemetry -> stable key."""
    hp = _bucket_hp_ratio(features.get("health_ratio"))
    enemies = _bucket_enemy_count(features.get("enemy_count"))
    boss = "1" if bool(features.get("boss_visible")) else "0"
    menu = "1" if bool(features.get("menu_mode")) else "0"
    parts = [f"hp{hp}", f"en{enemies}", f"boss{boss}", f"menu{menu}"]
    cell = _cell_from_pos(features.get("player_pos"))
    if cell is not None:
        parts.append(f"cell{cell[0]},{cell[1]}")
    return "|".join(parts)


@dataclass
class KeyStats:
    survived: int = 0
    died: int = 0

    def p_survive(self, alpha: float = 2.0, beta: float = 2.0) -> float:
        s = float(self.survived)
        d = float(self.died)
        return float((s + alpha) / (s + d + alpha + beta))


@dataclass
class SurvivalModel:
    horizon_s: float = 10.0
    alpha: float = 2.0
    beta: float = 2.0
    # Per-key aggregates
    stats: Dict[str, KeyStats] = field(default_factory=dict)
    # Per-instance recent observations within horizon: list of (ts, key)
    _buffers: Dict[str, List[Tuple[float, str]]] = field(default_factory=dict)
    # Persistence
    path: Path = field(default_factory=lambda: Path(os.environ.get("METABONK_SURVIVAL_DB", "temp/survival_stats.json")))
    save_interval_s: float = 15.0
    _last_save_ts: float = 0.0

    def observe(self, instance_id: str, ts: Optional[float], features: Dict[str, Any]) -> float:
        """Ingest telemetry and return current survival probability for the discretized key."""
        t = float(ts or _now())
        key = build_key(features)
        buf = self._buffers.setdefault(instance_id, [])
        buf.append((t, key))
        self._flush_old(instance_id, now_t=t)
        p = self.stats.setdefault(key, KeyStats()).p_survive(self.alpha, self.beta)
        self._maybe_save(now_t=t)
        return p

    def episode_end(self, instance_id: str, ts: Optional[float]) -> None:
        """Mark a terminal event (death/episode end) at ts for buffered observations."""
        t_end = float(ts or _now())
        buf = self._buffers.get(instance_id) or []
        if not buf:
            return
        # Anything still in buffer is within horizon of "now" (or hasn't been flushed yet).
        # Count as died if within horizon_s of the end timestamp.
        keep: List[Tuple[float, str]] = []
        for t0, key in buf:
            if (t_end - t0) <= self.horizon_s:
                ks = self.stats.setdefault(key, KeyStats())
                ks.died += 1
            else:
                # Should have been flushed already, but be defensive.
                ks = self.stats.setdefault(key, KeyStats())
                ks.survived += 1
        self._buffers[instance_id] = keep  # clear
        self._maybe_save(now_t=t_end)

    def _flush_old(self, instance_id: str, now_t: float) -> None:
        buf = self._buffers.get(instance_id) or []
        if not buf:
            return
        cutoff = now_t - float(self.horizon_s)
        keep: List[Tuple[float, str]] = []
        for t0, key in buf:
            if t0 <= cutoff:
                ks = self.stats.setdefault(key, KeyStats())
                ks.survived += 1
            else:
                keep.append((t0, key))
        self._buffers[instance_id] = keep

    def _maybe_save(self, now_t: float) -> None:
        if (now_t - self._last_save_ts) < float(self.save_interval_s):
            return
        self.save()
        self._last_save_ts = float(now_t)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "horizon_s": self.horizon_s,
                "alpha": self.alpha,
                "beta": self.beta,
                "ts": _now(),
                "stats": {k: {"survived": v.survived, "died": v.died} for k, v in self.stats.items()},
            }
            self.path.write_text(json.dumps(payload, indent=2))
        except Exception:
            return

    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text())
            self.horizon_s = float(data.get("horizon_s", self.horizon_s))
            self.alpha = float(data.get("alpha", self.alpha))
            self.beta = float(data.get("beta", self.beta))
            raw = data.get("stats") or {}
            out: Dict[str, KeyStats] = {}
            for k, v in raw.items():
                try:
                    out[str(k)] = KeyStats(survived=int(v.get("survived", 0)), died=int(v.get("died", 0)))
                except Exception:
                    continue
            self.stats = out
        except Exception:
            return

    def snapshot(self, topk: int = 50) -> Dict[str, Any]:
        """Debug-friendly snapshot."""
        items = [
            (k, v.survived, v.died, v.p_survive(self.alpha, self.beta))
            for k, v in self.stats.items()
        ]
        items.sort(key=lambda x: (x[2] + x[1]), reverse=True)
        return {
            "horizon_s": self.horizon_s,
            "alpha": self.alpha,
            "beta": self.beta,
            "keys": [
                {"key": k, "survived": s, "died": d, "p_survive": p}
                for (k, s, d, p) in items[: max(1, int(topk))]
            ],
        }

