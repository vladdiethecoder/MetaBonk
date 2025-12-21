"""Shame tracking for stream selection.

Shame is a lightweight, best-effort counterpart to hype:
- It reacts to "bad" events (early deaths, overruns, crashes).
- It decays over time so the "most shamed" slot rotates naturally.

This is used only for the Stream HUD. It is not used for training/PBT.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _label(score_0_100: float) -> str:
    s = float(score_0_100)
    if s >= 85:
        return "DISGRACE"
    if s >= 65:
        return "SHAME"
    if s >= 45:
        return "OOPS"
    if s >= 25:
        return "SUS"
    return "CLEAN"


@dataclass
class ShameState:
    value: float = 0.0
    last_ts: float = 0.0


class ShameTracker:
    def __init__(self, *, half_life_s: float = 16.0) -> None:
        self.half_life_s = float(max(1.0, half_life_s))
        self._by_instance: Dict[str, ShameState] = {}

    def _decay(self, st: ShameState, now: float) -> None:
        if st.last_ts <= 0:
            st.last_ts = now
            return
        dt = max(0.0, now - st.last_ts)
        if dt <= 0:
            return
        lam = math.log(2.0) / self.half_life_s
        st.value *= math.exp(-lam * dt)
        st.last_ts = now

    def bump(self, instance_id: str, amount: float, *, now: Optional[float] = None) -> float:
        now = float(time.time() if now is None else now)
        st = self._by_instance.get(instance_id) or ShameState(value=0.0, last_ts=now)
        self._decay(st, now)
        st.value = float(_clamp(st.value + float(amount), 0.0, 100.0))
        self._by_instance[instance_id] = st
        return st.value

    def score(self, instance_id: str, *, now: Optional[float] = None) -> float:
        now = float(time.time() if now is None else now)
        st = self._by_instance.get(instance_id)
        if st is None:
            return 0.0
        self._decay(st, now)
        return float(_clamp(st.value, 0.0, 100.0))

    def label(self, instance_id: str, *, now: Optional[float] = None) -> str:
        return _label(self.score(instance_id, now=now))

    def bump_from_event(self, instance_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> float:
        et = (event_type or "").lower()
        payload = payload or {}

        amt = 0.15

        if "crash" in et or "error" in et:
            amt = 20.0
        elif "overrunstart" in et:
            amt = 12.0
        elif "episodeend" in et:
            # Early deaths get shamed more.
            dur = payload.get("duration_s")
            try:
                dur_s = float(dur) if dur is not None else None
            except Exception:
                dur_s = None
            end_reason = str(payload.get("end_reason") or "").lower()
            if "death" in end_reason or "dead" in end_reason:
                amt = 18.0
            elif dur_s is not None:
                if dur_s < 45:
                    amt = 22.0
                elif dur_s < 90:
                    amt = 16.0
                elif dur_s < 160:
                    amt = 11.0
                else:
                    amt = 6.0
            else:
                amt = 8.0
        elif "rewardcollapse" in et:
            amt = 10.0

        return self.bump(str(instance_id), float(amt))

    def attach(self, hb: Any) -> Any:
        """Best-effort: mutate and return hb-like object."""
        try:
            iid = str(getattr(hb, "instance_id", "") or "")
            if not iid:
                return hb
            setattr(hb, "shame_score", self.score(iid))
            setattr(hb, "shame_label", self.label(iid))
        except Exception:
            return hb
        return hb

