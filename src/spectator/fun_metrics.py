"""Fun Metrics: Anthropomorphized AI Statistics.

Gamified metrics that give personality to the algorithm:
- Cowardice/Bravery Index
- Luck Gauge (RNG tracking)
- Overcrit Counter
- Spaghetti Factor (pathing)
- Borgar Count (healing)

References:
- AI anthropomorphism in Salty Bet
- "Flavor stats" from RPGs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FunMetric:
    """A single fun metric."""
    
    name: str
    value: float
    
    # Display
    display_value: str
    icon: str
    color: str
    
    # Thresholds for labels
    label: str  # e.g., "GODLY", "CURSED"
    
    # History for sparkline
    history: List[float] = field(default_factory=list)


class CowardiceIndex:
    """Measures AI tendency to avoid danger.
    
    High = stays far from enemies (Cowardly)
    Low = enters mob clusters (Brave/Reckless)
    """
    
    def __init__(self):
        self.distance_history: List[float] = []
        self.min_distance = float('inf')
        self.max_distance = 0.0
        
    def update(self, avg_enemy_distance: float):
        """Update with average distance to enemies."""
        self.distance_history.append(avg_enemy_distance)
        if len(self.distance_history) > 300:
            self.distance_history.pop(0)
        
        self.min_distance = min(self.min_distance, avg_enemy_distance)
        self.max_distance = max(self.max_distance, avg_enemy_distance)
    
    def get_metric(self) -> FunMetric:
        """Get the cowardice index metric."""
        if not self.distance_history:
            return FunMetric(
                name="Cowardice Index",
                value=50,
                display_value="50",
                icon="🐔",
                color="#888888",
                label="NORMAL",
            )
        
        avg = np.mean(self.distance_history[-60:])  # Last second
        
        # Normalize to 0-100
        value = min(100, max(0, avg * 5))  # Assumes avg distance in meters
        
        if value > 80:
            label, icon, color = "COWARD", "🐔", "#FFFF00"
        elif value > 60:
            label, icon, color = "CAUTIOUS", "🤔", "#88FF88"
        elif value > 40:
            label, icon, color = "NORMAL", "😐", "#888888"
        elif value > 20:
            label, icon, color = "BRAVE", "💪", "#FF8800"
        else:
            label, icon, color = "RECKLESS", "🔥", "#FF0000"
        
        return FunMetric(
            name="Cowardice Index",
            value=value,
            display_value=f"{value:.0f}",
            icon=icon,
            color=color,
            label=label,
            history=self.distance_history[-30:],
        )


class LuckGauge:
    """Tracks RNG luck (Actual vs Expected drops).
    
    Compares actual loot value against statistical expectation.
    """
    
    # Rarity values
    RARITY_VALUES = {
        "common": 1,
        "uncommon": 2,
        "rare": 5,
        "epic": 10,
        "legendary": 25,
    }
    
    # Expected values (probabilities)
    EXPECTED_PROBS = {
        "common": 0.50,
        "uncommon": 0.25,
        "rare": 0.15,
        "epic": 0.08,
        "legendary": 0.02,
    }
    
    def __init__(self):
        self.actual_value = 0.0
        self.expected_value = 0.0
        self.drop_count = 0
        self.legendary_count = 0
        
        self.luck_history: List[float] = []
    
    def record_drop(self, rarity: str):
        """Record a loot drop."""
        self.drop_count += 1
        
        # Actual value
        self.actual_value += self.RARITY_VALUES.get(rarity, 1)
        
        # Expected value (mean)
        exp = sum(
            self.RARITY_VALUES[r] * self.EXPECTED_PROBS[r]
            for r in self.RARITY_VALUES
        )
        self.expected_value += exp
        
        if rarity == "legendary":
            self.legendary_count += 1
        
        # Calculate luck ratio
        luck = self.actual_value / max(self.expected_value, 1)
        self.luck_history.append(luck)
    
    def get_metric(self) -> FunMetric:
        """Get the luck gauge metric."""
        if self.drop_count == 0:
            return FunMetric(
                name="Luck",
                value=1.0,
                display_value="N/A",
                icon="🎲",
                color="#888888",
                label="NO DATA",
            )
        
        luck = self.actual_value / max(self.expected_value, 1)
        
        if luck > 2.0:
            label, icon, color = "GODLY", "👼", "#FFD700"
        elif luck > 1.5:
            label, icon, color = "BLESSED", "✨", "#00FF88"
        elif luck > 1.1:
            label, icon, color = "LUCKY", "🍀", "#88FF88"
        elif luck > 0.9:
            label, icon, color = "NORMAL", "🎲", "#888888"
        elif luck > 0.6:
            label, icon, color = "UNLUCKY", "😢", "#FF8844"
        else:
            label, icon, color = "CURSED", "💀", "#FF0000"
        
        return FunMetric(
            name="Luck",
            value=luck,
            display_value=f"{luck:.2f}x",
            icon=icon,
            color=color,
            label=label,
            history=self.luck_history[-30:],
        )


class OvercritCounter:
    """Tracks overcrit (red crit) events and max damage.
    
    Overcrit = crit chance > 100% = exponential damage tiers.
    """
    
    def __init__(self):
        self.crit_chance = 100  # Base 100%
        self.overcrit_tier = 0  # 0 = normal, 1 = red, 2 = rainbow, etc.
        
        self.max_single_hit = 0.0
        self.total_overcrits = 0
        
        self.damage_history: List[float] = []
    
    def update_crit_chance(self, crit_chance: float):
        """Update current crit chance."""
        self.crit_chance = crit_chance
        
        # Calculate overcrit tier
        if crit_chance < 100:
            self.overcrit_tier = 0
        elif crit_chance < 200:
            self.overcrit_tier = 1
        elif crit_chance < 300:
            self.overcrit_tier = 2
        else:
            self.overcrit_tier = 3
    
    def record_damage(self, damage: float, is_crit: bool):
        """Record a damage event."""
        self.damage_history.append(damage)
        if len(self.damage_history) > 100:
            self.damage_history.pop(0)
        
        if damage > self.max_single_hit:
            self.max_single_hit = damage
        
        if is_crit and self.overcrit_tier > 0:
            self.total_overcrits += 1
    
    def get_metric(self) -> FunMetric:
        """Get the overcrit metric."""
        tier_labels = ["NORMAL", "RED CRIT", "RAINBOW", "ULTRA"]
        tier_icons = ["⚔️", "🔴", "🌈", "💥"]
        tier_colors = ["#888888", "#FF4444", "#FF88FF", "#FFFF00"]
        
        tier = min(self.overcrit_tier, 3)
        
        return FunMetric(
            name="Overcrit",
            value=self.crit_chance,
            display_value=f"{self.crit_chance:.0f}%",
            icon=tier_icons[tier],
            color=tier_colors[tier],
            label=tier_labels[tier],
            history=self.damage_history[-30:],
        )
    
    def get_big_number_metric(self) -> FunMetric:
        """Get the 'Big Number' max damage metric."""
        return FunMetric(
            name="Big Number",
            value=self.max_single_hit,
            display_value=f"{self.max_single_hit:,.0f}",
            icon="💥",
            color="#FF4444" if self.max_single_hit > 10000 else "#FFFFFF",
            label="MAX HIT",
        )


class SpaghettiFactory:
    """Measures pathing efficiency (Displacement / Distance).
    
    Low = efficient straight-line movement
    High = chaotic spaghetti pathing
    """
    
    def __init__(self):
        self.start_position: Optional[np.ndarray] = None
        self.total_distance = 0.0
        self.last_position: Optional[np.ndarray] = None
        
        self.spaghetti_history: List[float] = []
    
    def update(self, position: np.ndarray):
        """Update with current position."""
        if self.start_position is None:
            self.start_position = position.copy()
            self.last_position = position.copy()
            return
        
        # Track distance traveled
        step_dist = np.linalg.norm(position - self.last_position)
        self.total_distance += step_dist
        self.last_position = position.copy()
        
        # Calculate spaghetti factor
        displacement = np.linalg.norm(position - self.start_position)
        if displacement > 0.1:
            factor = self.total_distance / displacement
        else:
            factor = 1.0
        
        self.spaghetti_history.append(factor)
        if len(self.spaghetti_history) > 300:
            self.spaghetti_history.pop(0)
    
    def get_metric(self) -> FunMetric:
        """Get the spaghetti factor metric."""
        if not self.spaghetti_history:
            return FunMetric(
                name="Spaghetti",
                value=1.0,
                display_value="1.0",
                icon="🍝",
                color="#888888",
                label="NORMAL",
            )
        
        factor = np.mean(self.spaghetti_history[-60:])
        
        if factor > 5.0:
            label, icon, color = "CHAOS", "🌀", "#FF0000"
        elif factor > 3.0:
            label, icon, color = "SPAGHETTI", "🍝", "#FF8800"
        elif factor > 2.0:
            label, icon, color = "WIGGLY", "🐍", "#FFFF00"
        elif factor > 1.5:
            label, icon, color = "NORMAL", "➡️", "#888888"
        else:
            label, icon, color = "LASER", "🎯", "#00FF88"
        
        return FunMetric(
            name="Spaghetti",
            value=factor,
            display_value=f"{factor:.1f}x",
            icon=icon,
            color=color,
            label=label,
        )


class BorgarCounter:
    """Tracks healing item consumption (the "Gluttony Meter").
    
    Named after the healing hamburger item in Megabonk lore.
    """
    
    def __init__(self):
        self.total_heals = 0
        self.total_hp_healed = 0.0
        self.borgars_consumed = 0
        
        self.heal_timestamps: List[float] = []
    
    def record_heal(self, amount: float, is_borgar: bool = False, timestamp: float = 0):
        """Record a healing event."""
        self.total_heals += 1
        self.total_hp_healed += amount
        
        if is_borgar:
            self.borgars_consumed += 1
        
        self.heal_timestamps.append(timestamp)
    
    def get_metric(self) -> FunMetric:
        """Get the borgar count metric."""
        if self.borgars_consumed > 10:
            label, icon, color = "GLUTTON", "🍔🍔🍔", "#FF4444"
        elif self.borgars_consumed > 5:
            label, icon, color = "HUNGRY", "🍔🍔", "#FF8800"
        elif self.borgars_consumed > 0:
            label, icon, color = "SNACKING", "🍔", "#88FF88"
        else:
            label, icon, color = "FASTING", "🥗", "#888888"
        
        return FunMetric(
            name="Borgars",
            value=self.borgars_consumed,
            display_value=str(self.borgars_consumed),
            icon=icon,
            color=color,
            label=label,
        )
    
    def get_damage_taken_metric(self) -> FunMetric:
        """Inferred damage taken (heals needed)."""
        # More heals = more damage taken
        if self.total_heals > 20:
            label = "PUNCHING BAG"
        elif self.total_heals > 10:
            label = "TANKING"
        elif self.total_heals > 5:
            label = "NORMAL"
        else:
            label = "UNTOUCHABLE"
        
        return FunMetric(
            name="Damage Taken",
            value=self.total_hp_healed,
            display_value=f"{self.total_hp_healed:,.0f}",
            icon="💔" if self.total_heals > 10 else "💚",
            color="#FF4444" if self.total_heals > 10 else "#00FF88",
            label=label,
        )


class SwarmPressure:
    """Analyzes enemy swarm pressure vs clearing capability."""
    
    def __init__(self):
        self.enemy_count = 0
        self.incoming_dps = 0.0
        self.clearing_dps = 0.0
        
        self.pressure_history: List[float] = []
        self.overrun_time = 0.0
    
    def update(
        self,
        enemy_count: int,
        incoming_dps: float,
        clearing_dps: float,
        dt: float,
    ):
        """Update swarm pressure metrics."""
        self.enemy_count = enemy_count
        self.incoming_dps = incoming_dps
        self.clearing_dps = clearing_dps
        
        # Calculate pressure ratio
        if clearing_dps > 0:
            pressure = incoming_dps / clearing_dps
        else:
            pressure = float('inf') if incoming_dps > 0 else 0
        
        self.pressure_history.append(min(pressure, 10.0))
        if len(self.pressure_history) > 300:
            self.pressure_history.pop(0)
        
        # Track overrun time
        if pressure > 1.0:
            self.overrun_time += dt
    
    def get_metric(self) -> FunMetric:
        """Get the swarm pressure metric."""
        if not self.pressure_history:
            return FunMetric(
                name="Swarm",
                value=0,
                display_value="0",
                icon="🐜",
                color="#888888",
                label="CALM",
            )
        
        pressure = np.mean(self.pressure_history[-30:])
        
        if pressure > 2.0:
            label, icon, color = "OVERRUN", "🌊", "#FF0000"
        elif pressure > 1.5:
            label, icon, color = "DROWNING", "😰", "#FF8800"
        elif pressure > 1.0:
            label, icon, color = "STRUGGLING", "😓", "#FFFF00"
        elif pressure > 0.5:
            label, icon, color = "HOLDING", "😤", "#88FF88"
        else:
            label, icon, color = "CRUSHING", "😎", "#00FF88"
        
        return FunMetric(
            name="Swarm",
            value=pressure,
            display_value=f"{pressure:.1f}x",
            icon=icon,
            color=color,
            label=label,
            history=self.pressure_history[-30:],
        )


class FunMetricsCollector:
    """Collects all fun metrics."""
    
    def __init__(self):
        self.cowardice = CowardiceIndex()
        self.luck = LuckGauge()
        self.overcrit = OvercritCounter()
        self.spaghetti = SpaghettiFactory()
        self.borgar = BorgarCounter()
        self.swarm = SwarmPressure()
    
    def get_all_metrics(self) -> Dict[str, FunMetric]:
        """Get all fun metrics."""
        return {
            "cowardice": self.cowardice.get_metric(),
            "luck": self.luck.get_metric(),
            "overcrit": self.overcrit.get_metric(),
            "big_number": self.overcrit.get_big_number_metric(),
            "spaghetti": self.spaghetti.get_metric(),
            "borgar": self.borgar.get_metric(),
            "damage_taken": self.borgar.get_damage_taken_metric(),
            "swarm": self.swarm.get_metric(),
        }
    
    def get_metrics_for_display(self) -> List[Dict[str, Any]]:
        """Get metrics formatted for UI display."""
        metrics = self.get_all_metrics()
        
        return [
            {
                "name": m.name,
                "value": m.display_value,
                "icon": m.icon,
                "color": m.color,
                "label": m.label,
            }
            for m in metrics.values()
        ]
