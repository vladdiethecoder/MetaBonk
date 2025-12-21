"""Real-data fun stats for stream UI (no simulation).

This module aggregates per-instance "flavor metrics" from real telemetry events:
  - LuckGauge: compares actual loot value vs expected value (mean).
  - BorgarTracker: counts healing hamburger consumption (and general heals).

Workers are expected to emit events with these payload shapes:
  - LootDrop: {"rarity": "common|rare|legendary|...", "item_name": "...", "ts": ...}
  - Heal: {"amount": float, "item_name": "...", "is_borgar": bool, "ts": ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


@dataclass
class LuckGauge:
    # Point values (per user spec). Unknown rarities default to "common".
    rarity_points: Dict[str, float]
    # Expected probabilities (mean EV baseline). Can be overridden by env/config later.
    expected_probs: Dict[str, float]

    actual_value: float = 0.0
    expected_value: float = 0.0
    drop_count: int = 0
    legendary_count: int = 0

    def record_drop(self, rarity: str) -> None:
        r = _norm(rarity) or "common"
        if r not in self.rarity_points:
            r = "common"
        self.drop_count += 1
        self.actual_value += float(self.rarity_points.get(r, 1.0))

        # Expected value per drop = sum(points * prob)
        ev = 0.0
        for k, p in self.expected_probs.items():
            kk = _norm(k)
            pts = float(self.rarity_points.get(kk, self.rarity_points.get("common", 1.0)))
            try:
                ev += pts * float(p)
            except Exception:
                continue
        self.expected_value += float(ev)

        if r == "legendary":
            self.legendary_count += 1

    @property
    def luck_mult(self) -> float:
        if self.drop_count <= 0:
            return 1.0
        denom = max(self.expected_value, 1e-6)
        return float(self.actual_value / denom)

    @property
    def label(self) -> str:
        x = self.luck_mult
        if self.drop_count <= 0:
            return "NO DATA"
        if x > 2.0:
            return "GODLY"
        if x > 1.5:
            return "BLESSED"
        if x > 1.1:
            return "LUCKY"
        if x > 0.9:
            return "NORMAL"
        if x > 0.6:
            return "UNLUCKY"
        return "CURSED"

    @property
    def color(self) -> str:
        lab = self.label
        return {
            "NO DATA": "#888888",
            "GODLY": "#FFD700",
            "BLESSED": "#00FF88",
            "LUCKY": "#88FF88",
            "NORMAL": "#888888",
            "UNLUCKY": "#FF8844",
            "CURSED": "#FF0000",
        }.get(lab, "#888888")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "drop_count": int(self.drop_count),
            "legendary_count": int(self.legendary_count),
            "actual_value": float(self.actual_value),
            "expected_value": float(self.expected_value),
            "luck_mult": float(self.luck_mult),
            "label": self.label,
            "color": self.color,
        }


@dataclass
class BorgarTracker:
    borgars_consumed: int = 0
    total_heals: int = 0
    total_hp_healed: float = 0.0

    def record_heal(self, amount: float, is_borgar: bool) -> None:
        self.total_heals += 1
        try:
            self.total_hp_healed += float(amount)
        except Exception:
            pass
        if is_borgar:
            self.borgars_consumed += 1

    @property
    def label(self) -> str:
        n = self.borgars_consumed
        if n > 10:
            return "GLUTTON"
        if n > 5:
            return "HUNGRY"
        if n > 0:
            return "SNACKING"
        return "FASTING"

    @property
    def icon(self) -> str:
        n = self.borgars_consumed
        if n > 10:
            return "🍔🍔🍔"
        if n > 5:
            return "🍔🍔"
        if n > 0:
            return "🍔"
        return "🥗"

    def snapshot(self) -> Dict[str, Any]:
        return {
            "borgars_consumed": int(self.borgars_consumed),
            "total_heals": int(self.total_heals),
            "total_hp_healed": float(self.total_hp_healed),
            "label": self.label,
            "icon": self.icon,
        }


@dataclass
class FunStats:
    luck: LuckGauge
    borgar: BorgarTracker

    @classmethod
    def default(cls) -> "FunStats":
        # User-specified point values with minimal extensions for other rarities.
        rarity_points = {
            "common": 1.0,
            "uncommon": 1.0,
            "rare": 2.0,
            "epic": 3.0,
            "legendary": 5.0,
        }
        expected_probs = {
            "common": 0.50,
            "uncommon": 0.25,
            "rare": 0.15,
            "epic": 0.08,
            "legendary": 0.02,
        }
        return cls(luck=LuckGauge(rarity_points=rarity_points, expected_probs=expected_probs), borgar=BorgarTracker())

