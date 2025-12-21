"""Spectator-style featured slot selection.

Goal:
  - Always expose 5 "featured" feeds for OBS/Stream HUD:
      - top 3 by hype
      - 2 most shamed
  - Avoid thrashing (hysteresis + cooldown).
  - Keep selection backend-driven so the UI can be dumb.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return float(default)
        return v
    except Exception:
        return float(default)


def _score_of(hb: Any) -> float:
    return _f(getattr(hb, "hype_score", None), 0.0)


def _shame_of(hb: Any) -> float:
    return _f(getattr(hb, "shame_score", None), 0.0)


def _tie_breaker(hb: Any) -> float:
    # Prefer higher "real" score when hype/shame are tied.
    return _f(getattr(hb, "steam_score", None), _f(getattr(hb, "reward", None), 0.0))

def _stream_ok(hb: Any) -> float:
    # Prefer candidates that appear to have a live stream.
    try:
        return 1.0 if bool(getattr(hb, "stream_ok", False)) else 0.0
    except Exception:
        return 0.0

def _is_gpu_stream(hb: Any) -> float:
    try:
        return 1.0 if str(getattr(hb, "stream_type", "") or "").lower() == "mp4" else 0.0
    except Exception:
        return 0.0


@dataclass
class FeaturedSnapshot:
    ts: float
    slots: Dict[str, Optional[str]]  # slot -> instance_id
    pending: Dict[str, Optional[str]]  # slot -> candidate being warmed (not yet switched)


class SpectatorDirector:
    def __init__(
        self,
        *,
        cooldown_s: float = 4.0,
        hype_margin: float = 1.0,
        shame_margin: float = 2.0,
        warmup_s: float = 2.5,
    ) -> None:
        self.cooldown_s = float(max(0.0, cooldown_s))
        self.hype_margin = float(max(0.0, hype_margin))
        self.shame_margin = float(max(0.0, shame_margin))
        self.warmup_s = float(max(0.0, warmup_s))
        self._slots: Dict[str, Optional[str]] = {"hype0": None, "hype1": None, "hype2": None, "shame0": None, "shame1": None}
        self._last_change_ts: Dict[str, float] = {k: 0.0 for k in self._slots.keys()}
        self._pending: Dict[str, Optional[str]] = {k: None for k in self._slots.keys()}

    @classmethod
    def from_env(cls) -> "SpectatorDirector":
        return cls(
            cooldown_s=float(os.environ.get("METABONK_FEATURED_COOLDOWN_S", "4.0")),
            hype_margin=float(os.environ.get("METABONK_FEATURED_HYPE_MARGIN", "1.0")),
            shame_margin=float(os.environ.get("METABONK_FEATURED_SHAME_MARGIN", "2.0")),
            warmup_s=float(os.environ.get("METABONK_FEATURED_WARMUP_S", "2.5")),
        )

    def snapshot(self, *, now: Optional[float] = None) -> FeaturedSnapshot:
        now = float(time.time() if now is None else now)
        return FeaturedSnapshot(ts=now, slots=dict(self._slots), pending=dict(self._pending))

    def _can_switch(self, slot: str, *, now: float) -> bool:
        return (now - float(self._last_change_ts.get(slot, 0.0))) >= float(self.cooldown_s)

    def _maybe_set(self, slot: str, new_iid: Optional[str], *, now: float) -> bool:
        if self._slots.get(slot) == new_iid:
            return False
        self._slots[slot] = new_iid
        self._last_change_ts[slot] = float(now)
        self._pending[slot] = None
        return True

    def pending_ids(self) -> set[str]:
        out: set[str] = set()
        for iid in self._pending.values():
            if iid:
                out.add(str(iid))
        return out

    def update(
        self,
        workers: Dict[str, Any],
        *,
        ready_since: Optional[Dict[str, float]] = None,
        now: Optional[float] = None,
    ) -> FeaturedSnapshot:
        now = float(time.time() if now is None else now)
        if not workers:
            self._slots = {"hype0": None, "hype1": None, "hype2": None, "shame0": None, "shame1": None}
            self._pending = {k: None for k in self._slots.keys()}
            return self.snapshot(now=now)
        ready_since = ready_since or {}

        # Rank candidates.
        by_hype = sorted(
            workers.values(),
            key=lambda hb: (_is_gpu_stream(hb), _stream_ok(hb), _score_of(hb), _tie_breaker(hb), str(getattr(hb, "instance_id", ""))),
            reverse=True,
        )
        by_shame = sorted(
            workers.values(),
            key=lambda hb: (_is_gpu_stream(hb), _stream_ok(hb), _shame_of(hb), _tie_breaker(hb), str(getattr(hb, "instance_id", ""))),
            reverse=True,
        )

        # Propose ideal set:
        # - shame slots are the top 2 by shame (even if they're also "hyped")
        # - hype slots are the top 3 by hype, excluding shame slots to keep feeds unique
        shame_ids: list[str] = []
        for hb in by_shame:
            iid = str(getattr(hb, "instance_id", "") or "")
            if iid:
                shame_ids.append(iid)
            if len(shame_ids) >= 2:
                break

        hype_ids: list[str] = []
        for hb in by_hype:
            iid = str(getattr(hb, "instance_id", "") or "")
            if not iid or iid in hype_ids:
                continue
            if iid in shame_ids:
                continue
            hype_ids.append(iid)
            if len(hype_ids) >= 3:
                break

        # If we have fewer than 3 hype feeds, allow the shame feeds to also be hype.
        if len(hype_ids) < 3 and shame_ids:
            for iid in shame_ids:
                if iid not in hype_ids:
                    hype_ids.append(iid)
                if len(hype_ids) >= 3:
                    break

        proposals: Dict[str, Optional[str]] = {
            "hype0": hype_ids[0] if len(hype_ids) > 0 else None,
            "hype1": hype_ids[1] if len(hype_ids) > 1 else None,
            "hype2": hype_ids[2] if len(hype_ids) > 2 else None,
            "shame0": shame_ids[0] if len(shame_ids) > 0 else None,
            "shame1": shame_ids[1] if len(shame_ids) > 1 else None,
        }

        # Hysteresis/cooldown per slot.
        for slot, proposed in proposals.items():
            cur = self._slots.get(slot)
            if cur and cur not in workers:
                self._maybe_set(slot, proposed, now=now)
                continue
            if cur is None:
                # Require warmup before first assignment too (prevents blank tiles).
                if proposed:
                    rs0 = float(ready_since.get(str(proposed), 0.0) or 0.0)
                    if rs0 > 0.0 and (now - rs0) >= self.warmup_s:
                        self._maybe_set(slot, proposed, now=now)
                    else:
                        self._pending[slot] = proposed
                else:
                    self._pending[slot] = None
                continue
            if proposed == cur:
                self._pending[slot] = None
                continue
            if not self._can_switch(slot, now=now):
                self._pending[slot] = proposed
                continue

            cur_hb = workers.get(cur)
            prop_hb = workers.get(proposed) if proposed else None

            # Warm-up gate: proposed must have had a live GPU stream for warmup_s.
            if proposed:
                rs = float(ready_since.get(str(proposed), 0.0) or 0.0)
                if rs <= 0.0 or (now - rs) < self.warmup_s:
                    self._pending[slot] = proposed
                    continue

            if slot.startswith("hype"):
                cur_s = _score_of(cur_hb) if cur_hb is not None else 0.0
                prop_s = _score_of(prop_hb) if prop_hb is not None else 0.0
                if prop_s >= (cur_s + self.hype_margin):
                    self._maybe_set(slot, proposed, now=now)
                else:
                    self._pending[slot] = proposed
            else:
                cur_s = _shame_of(cur_hb) if cur_hb is not None else 0.0
                prop_s = _shame_of(prop_hb) if prop_hb is not None else 0.0
                if prop_s >= (cur_s + self.shame_margin):
                    self._maybe_set(slot, proposed, now=now)
                else:
                    self._pending[slot] = proposed

        return self.snapshot(now=now)

    def annotate_workers(self, workers: Dict[str, Any], snap: FeaturedSnapshot) -> None:
        # Best-effort: add per-worker role/slot so the UI can filter without extra endpoints.
        featured_by_iid: Dict[str, str] = {}
        for slot, iid in (snap.slots or {}).items():
            if iid:
                featured_by_iid[str(iid)] = str(slot)

        for iid, hb in workers.items():
            slot = featured_by_iid.get(str(iid))
            try:
                setattr(hb, "featured_slot", slot)
                setattr(hb, "featured_role", "featured" if slot else "background")
            except Exception:
                pass
