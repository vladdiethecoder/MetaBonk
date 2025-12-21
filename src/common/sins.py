"""SinZero: Seven Deadly Sins definitions and default biases.

These are used to parameterize intrinsic rewards and exploration styles
for population-based training in MegaBonk.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Sin(str, Enum):
    GREED = "Greed"
    LUST = "Lust"
    WRATH = "Wrath"
    SLOTH = "Sloth"
    GLUTTONY = "Gluttony"
    ENVY = "Envy"
    PRIDE = "Pride"


@dataclass
class SinBias:
    """Default bias knobs for a Sin agent."""

    # Intrinsic reward multiplier λ_sin
    intrinsic_coef: float = 0.2
    # PPO entropy coefficient (exploration vs exploitation)
    entropy_coef: float = 0.01
    # Alpha-divergence setting; 1.0 ~= forward KL mass-covering, <0 mode-seeking
    alpha_divergence: float = 1.0
    # CVaR percentile for Pride/Sloth risk profiles (None = standard expected return)
    cvar_alpha: float | None = None


DEFAULT_SIN_BIASES: Dict[Sin, SinBias] = {
    Sin.LUST: SinBias(intrinsic_coef=0.6, entropy_coef=0.05, alpha_divergence=1.2),
    Sin.GLUTTONY: SinBias(intrinsic_coef=0.3, entropy_coef=0.02, alpha_divergence=1.0),
    Sin.GREED: SinBias(intrinsic_coef=0.3, entropy_coef=0.02, alpha_divergence=1.0),
    Sin.SLOTH: SinBias(intrinsic_coef=0.2, entropy_coef=0.005, alpha_divergence=0.8, cvar_alpha=0.2),
    Sin.WRATH: SinBias(intrinsic_coef=0.2, entropy_coef=0.01, alpha_divergence=-2.0),
    Sin.ENVY: SinBias(intrinsic_coef=0.1, entropy_coef=0.01, alpha_divergence=1.0),
    Sin.PRIDE: SinBias(intrinsic_coef=0.2, entropy_coef=0.01, alpha_divergence=-5.0, cvar_alpha=0.1),
}

