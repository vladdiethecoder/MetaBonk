"""Qualia generator (debug/UX mapping for internal states).

This is a pragmatic, human-facing mapping layer that converts internal numeric
signals into multi-sensory "qualia-like" representations:

- color (for overlays / UI),
- tone (for optional sonification),
- valence/arousal (for status dashboards).

It is not a claim of consciousness; it's a UI/observability abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import math


@dataclass(frozen=True)
class Qualia:
    color_rgb: Tuple[int, int, int]
    tone_hz: float
    valence: float
    arousal: float
    tags: Tuple[str, ...] = ()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    h = float(h) % 1.0
    s = _clamp01(s)
    v = _clamp01(v)
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r, g, b = {
        0: (v, t, p),
        1: (q, v, p),
        2: (p, v, t),
        3: (p, q, v),
        4: (t, p, v),
        5: (v, p, q),
    }[i]
    return (int(r * 255), int(g * 255), int(b * 255))


class QualiaGenerator:
    def from_error(self, error: float) -> Qualia:
        e = abs(float(error))
        # Map error magnitude to red/orange and harsher tone.
        intensity = _clamp01(math.tanh(e))
        hue = 0.02 + 0.06 * (1.0 - intensity)  # red->orange
        rgb = _hsv_to_rgb(hue, 0.9, 0.35 + 0.55 * intensity)
        tone = 220.0 + 660.0 * intensity
        return Qualia(color_rgb=rgb, tone_hz=tone, valence=-intensity, arousal=0.6 + 0.4 * intensity, tags=("error",))

    def from_reward(self, reward: float) -> Qualia:
        r = float(reward)
        v = _clamp01(0.5 + 0.5 * math.tanh(r))
        hue = 0.33  # green
        rgb = _hsv_to_rgb(hue, 0.75, 0.25 + 0.7 * v)
        tone = 160.0 + 240.0 * v
        return Qualia(color_rgb=rgb, tone_hz=tone, valence=2.0 * v - 1.0, arousal=0.4 + 0.4 * v, tags=("reward",))

    def from_uncertainty(self, uncertainty: float) -> Qualia:
        u = _clamp01(float(uncertainty))
        # Uncertainty is blue/purple with a wobblier, higher tone.
        hue = 0.66 + 0.12 * u
        rgb = _hsv_to_rgb(hue, 0.8, 0.2 + 0.65 * u)
        tone = 300.0 + 500.0 * u
        return Qualia(color_rgb=rgb, tone_hz=tone, valence=-0.2 * u, arousal=0.3 + 0.6 * u, tags=("uncertainty",))

    def blend(self, qualia: Iterable[Qualia]) -> Optional[Qualia]:
        items = list(qualia)
        if not items:
            return None
        r = sum(q.color_rgb[0] for q in items) / len(items)
        g = sum(q.color_rgb[1] for q in items) / len(items)
        b = sum(q.color_rgb[2] for q in items) / len(items)
        tone = sum(float(q.tone_hz) for q in items) / len(items)
        valence = sum(float(q.valence) for q in items) / len(items)
        arousal = sum(float(q.arousal) for q in items) / len(items)
        tags = tuple({t for q in items for t in q.tags})
        return Qualia(color_rgb=(int(r), int(g), int(b)), tone_hz=float(tone), valence=float(valence), arousal=float(arousal), tags=tags)


__all__ = [
    "Qualia",
    "QualiaGenerator",
]

