"""Hype tracking for stream selection.

We keep this deliberately lightweight and "best effort":
- It must work even when only a subset of telemetry is available (visual-only mode).
- It should react quickly to exciting events and decay back toward baseline.

Hype is *not* used for training/PBT. It only drives the Stream HUD's feed ranking.
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
    if s >= 90:
        return "NUCLEAR"
    if s >= 75:
        return "INSANE"
    if s >= 55:
        return "HYPE"
    if s >= 35:
        return "SPICY"
    if s >= 18:
        return "COOKING"
    return "CHILL"


@dataclass
class HypeState:
    value: float = 0.0
    last_ts: float = 0.0


class HypeTracker:
    def __init__(self, *, half_life_s: float = 9.0) -> None:
        self.half_life_s = float(max(1.0, half_life_s))
        self._by_instance: Dict[str, HypeState] = {}

    def _decay(self, st: HypeState, now: float) -> None:
        if st.last_ts <= 0:
            st.last_ts = now
            return
        dt = max(0.0, now - st.last_ts)
        if dt <= 0:
            return
        # Exponential decay with a configurable half-life.
        lam = math.log(2.0) / self.half_life_s
        st.value *= math.exp(-lam * dt)
        st.last_ts = now

    def bump(self, instance_id: str, amount: float, *, now: Optional[float] = None) -> float:
        now = float(time.time() if now is None else now)
        st = self._by_instance.get(instance_id) or HypeState(value=0.0, last_ts=now)
        self._decay(st, now)
        st.value = float(_clamp(st.value + float(amount), 0.0, 100.0))
        self._by_instance[instance_id] = st
        return st.value

    def update_baseline(
        self,
        instance_id: str,
        *,
        survival_prob: Optional[float] = None,
        overrun: Optional[bool] = None,
        danger_level: Optional[float] = None,
        now: Optional[float] = None,
    ) -> float:
        """Pull hype toward a baseline derived from current danger."""
        now = float(time.time() if now is None else now)
        st = self._by_instance.get(instance_id) or HypeState(value=0.0, last_ts=now)
        self._decay(st, now)

        danger = None
        if danger_level is not None and isinstance(danger_level, (int, float)) and math.isfinite(float(danger_level)):
            danger = float(danger_level)
        if danger is None and survival_prob is not None and isinstance(survival_prob, (int, float)) and math.isfinite(float(survival_prob)):
            danger = 1.0 - float(survival_prob)
        danger = _clamp(danger if danger is not None else 0.0, 0.0, 1.0)

        # Baseline is mostly "danger", with an extra kick when mathematically overrun.
        base = 100.0 * (danger**0.85)
        if overrun:
            base = max(base, 65.0)

        # Gently pull toward baseline (keeps hype non-zero during tense moments).
        st.value = float(_clamp(0.88 * st.value + 0.12 * base, 0.0, 100.0))
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

        # Default: small bump for any event (keeps active runs from looking dead).
        amt = 0.35

        if "clutch" in et:
            amt = 18.0
        elif "overcrit" in et:
            amt = 10.0
        elif "newmaxhit" in et:
            amt = 7.0
        elif "bosskill" in et or "bosskilled" in et:
            amt = 9.0
        elif "levelup" in et:
            amt = 2.0
        elif "legendary" in et:
            amt = 6.0
        elif "lootdrop" in et:
            # If we can infer rarity, vary bump.
            r = str(payload.get("rarity") or "").lower()
            if "legend" in r:
                amt = 7.0
            elif "rare" in r:
                amt = 3.0
            else:
                amt = 1.0
        elif "overrunstart" in et:
            amt = 6.0
        elif "episodeend" in et:
            amt = 2.0

        return self.bump(str(instance_id), float(amt))

    def attach(self, hb: Any) -> Any:
        """Best-effort: mutate and return hb-like object."""
        try:
            iid = str(getattr(hb, "instance_id", "") or "")
            if not iid:
                return hb
            sp = getattr(hb, "survival_prob", None)
            dl = getattr(hb, "danger_level", None)
            ov = getattr(hb, "overrun", None)
            self.update_baseline(iid, survival_prob=sp, danger_level=dl, overrun=bool(ov))
            setattr(hb, "hype_score", self.score(iid))
            setattr(hb, "hype_label", self.label(iid))
        except Exception:
            return hb
        return hb

