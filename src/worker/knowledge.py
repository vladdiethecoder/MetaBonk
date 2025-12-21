"""Game knowledge base and lightweight decision helpers.

This module encodes domain knowledge from the MegaBonk report:
  - Upgrade/tome priorities
  - Simple shrine/chest heuristics

It is used by menu/offer logic once OCR + color rarity extraction is wired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class Goal(str, Enum):
    SURVIVAL = "survival"
    FARMING = "farming"
    SPEEDRUN = "speedrun"


class Rarity(str, Enum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    LEGENDARY = "legendary"


RARITY_BONUS: Dict[Rarity, float] = {
    Rarity.COMMON: 0.0,
    Rarity.UNCOMMON: 10.0,
    Rarity.RARE: 25.0,
    Rarity.LEGENDARY: 60.0,
}


S_TIER_KEYWORDS = [
    "luck tome",
    "luck",
    "xp tome",
    "xp",
    "cooldown tome",
    "cooldown",
    "quantity tome",
    "quantity",
]

SURVIVAL_KEYWORDS = [
    "armor",
    "shield",
    "regen",
    "health",
    "max hp",
]

FARMING_KEYWORDS = [
    "greed",
    "silver",
    "gold",
]


def score_upgrade(
    text: str,
    rarity: Optional[Rarity] = None,
    health_pct: Optional[float] = None,
    goal: Goal = Goal.SURVIVAL,
) -> float:
    """Score an upgrade/card offer from OCR text + rarity.

    Higher score = better pick.
    """
    t = (text or "").lower()
    score = 0.0

    if any(k in t for k in S_TIER_KEYWORDS):
        score += 100.0

    if health_pct is not None and health_pct < 0.5:
        if any(k in t for k in SURVIVAL_KEYWORDS):
            score += 50.0

    if goal == Goal.FARMING and any(k in t for k in FARMING_KEYWORDS):
        score += 30.0

    if rarity is not None:
        score += RARITY_BONUS.get(rarity, 0.0)

    return score


# --- Shrine / world heuristics ---


class ShrineType(str, Enum):
    CHARGE = "charge"
    SUCC = "succ"
    BOSS_CURSE = "boss_curse"
    CHALLENGE = "challenge"
    MICROWAVE = "microwave"
    MOAI = "moai"
    GREED = "greed"
    UNKNOWN = "unknown"


def should_activate_shrine(
    shrine: ShrineType,
    health_pct: float,
    build_strength: float = 0.5,
    goal: Goal = Goal.SURVIVAL,
) -> bool:
    """Rule-of-thumb shrine activation policy."""
    if shrine in (ShrineType.CHARGE, ShrineType.SUCC, ShrineType.MOAI):
        return True
    if shrine == ShrineType.CHALLENGE:
        return health_pct > 0.6
    if shrine == ShrineType.BOSS_CURSE:
        return health_pct > 0.7 and build_strength > 0.6
    if shrine == ShrineType.GREED:
        return goal == Goal.FARMING and health_pct > 0.6
    if shrine == ShrineType.MICROWAVE:
        # Needs inventory reasoning; default off unless farming.
        return goal == Goal.FARMING
    return False


class ChestType(str, Enum):
    WOOD_PAID = "wood_paid"
    WOOD_FREE = "wood_free"
    GOLDEN_FREE = "golden_free"
    UNKNOWN = "unknown"


def should_open_chest(
    chest: ChestType,
    gold: float,
    next_cost: float = 0.0,
) -> bool:
    if chest in (ChestType.WOOD_FREE, ChestType.GOLDEN_FREE):
        return True
    if chest == ChestType.WOOD_PAID:
        return gold >= next_cost
    return False

