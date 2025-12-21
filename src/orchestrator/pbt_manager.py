"""Population-Based Training (PBT) manager.

The original project used PBT to evolve PPO hyperparameters across a
population of policies. This recovery version keeps the public surface area
and implements the core algorithm in-memory.
"""

from __future__ import annotations

import copy
import random
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.common.sins import Sin
from src.common.sin_presets import preset_for_policy
try:
    from src.cognitive.llm_weighting import LLMWeightComposer
except Exception:  # pragma: no cover
    LLMWeightComposer = None  # type: ignore


DEFAULT_HPARAMS = {
    "lr": 3e-4,
    "gamma": 0.99,
    "entropy_coef": 0.01,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "batch_size": 256,
    "minibatch_size": 64,
    "epochs": 3,
    "use_lstm": True,
    "seq_len": 32,
    "burn_in": 8,
}


@dataclass
class PolicyState:
    policy_name: str
    hparams: Dict[str, float] = field(default_factory=lambda: copy.deepcopy(DEFAULT_HPARAMS))
    steam_score: float = 0.0
    eval_score: float = 0.0
    policy_version: Optional[int] = None
    score_source: str = "live"
    last_update_ts: float = field(default_factory=time.time)
    last_eval_ts: Optional[float] = None
    last_mutation_ts: Optional[float] = None
    pbt_muted: bool = False
    # Lineage / phylogeny metadata for RCC.
    parents: List[str] = field(default_factory=list)
    mix_weights: Dict[str, float] = field(default_factory=dict)
    generation: int = 0


class PBTManager:
    """Simple in-memory PBT controller."""

    def __init__(
        self,
        population_size: int = 8,
        exploit_frac: float = 0.2,
        perturb_scale: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        self.population_size = population_size
        self.exploit_frac = exploit_frac
        self.perturb_scale = perturb_scale
        self.rng = random.Random(seed)
        self.population: Dict[str, PolicyState] = {}
        self._muted = os.environ.get("METABONK_PBT_MUTED", "0") in ("1", "true", "True")
        muted_list = os.environ.get("METABONK_PBT_MUTED_POLICIES", "")
        self._muted_policies = {n.strip() for n in muted_list.split(",") if n.strip()}
        self._llm_pbt_weights = os.environ.get("METABONK_USE_LLM_PBT_WEIGHTS", "1") in (
            "1",
            "true",
            "True",
        )
        self._composer = LLMWeightComposer() if (self._llm_pbt_weights and LLMWeightComposer) else None

        # Bootstrap population with Sin presets when available.
        sin_names = [s.value for s in Sin]
        for name in sin_names:
            hp = preset_for_policy(name) or copy.deepcopy(DEFAULT_HPARAMS)
            self.population[name] = PolicyState(policy_name=name, hparams=hp, pbt_muted=name in self._muted_policies)

        # If larger population requested, add generic policies.
        extra = max(0, population_size - len(sin_names))
        for i in range(extra):
            name = f"policy-{i}"
            self.population[name] = PolicyState(policy_name=name, pbt_muted=name in self._muted_policies)

    def is_muted(self) -> bool:
        return bool(self._muted)

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)

    def set_policy_muted(self, policy_name: str, muted: bool) -> None:
        self.register_policy(policy_name)
        st = self.population[policy_name]
        st.pbt_muted = bool(muted)
        if st.pbt_muted:
            self._muted_policies.add(policy_name)
        else:
            self._muted_policies.discard(policy_name)

    def muted_policies(self) -> List[str]:
        return [name for name, st in self.population.items() if st.pbt_muted]

    def register_policy(self, policy_name: str, hparams: Optional[Dict[str, float]] = None):
        if policy_name not in self.population:
            self.population[policy_name] = PolicyState(
                policy_name=policy_name,
                hparams=copy.deepcopy(hparams or DEFAULT_HPARAMS),
                pbt_muted=policy_name in self._muted_policies,
            )

    def update_score(self, policy_name: str, steam_score: float, policy_version: Optional[int] = None):
        self.register_policy(policy_name)
        st = self.population[policy_name]
        st.steam_score = float(steam_score)
        st.policy_version = policy_version if policy_version is not None else st.policy_version
        st.score_source = "live"
        st.last_update_ts = time.time()

    def update_eval_score(self, policy_name: str, eval_score: float):
        self.register_policy(policy_name)
        st = self.population[policy_name]
        st.eval_score = float(eval_score)
        st.last_eval_ts = time.time()
        st.score_source = "eval"

    def assign_policy(self, instance_id: str) -> PolicyState:
        """Pick a policy for a new worker."""
        # Round-robin by hashing instance id.
        names = sorted(self.population.keys())
        idx = abs(hash(instance_id)) % len(names)
        return self.population[names[idx]]

    def step(self) -> List[PolicyState]:
        """Run one PBT step, returning any mutated policies."""
        if self._muted or len(self.population) < 2:
            return []

        use_eval = os.environ.get("METABONK_PBT_USE_EVAL", "0") in ("1", "true", "True")

        def score_of(p: PolicyState) -> float:
            if use_eval and p.eval_score:
                return float(p.eval_score)
            return float(p.steam_score)

        members = sorted(self.population.values(), key=score_of, reverse=True)
        eligible = [m for m in members if not m.pbt_muted]
        if len(eligible) < 2:
            return []
        k = max(1, int(len(eligible) * self.exploit_frac))
        top = eligible[:k]
        bottom = eligible[-k:]

        mutated: List[PolicyState] = []
        for loser in bottom:
            winner = self.rng.choice(top)
            if winner.policy_name == loser.policy_name:
                continue
            new_hp = self._perturb_hparams(winner.hparams)

            # Optional LLM-derived intrinsic/reward weights per sin.
            if self._composer is not None:
                try:
                    metrics_snap = {
                        "winner": {
                            "policy": winner.policy_name,
                            "steam_score": winner.steam_score,
                            "hparams": winner.hparams,
                        },
                        "loser": {
                            "policy": loser.policy_name,
                            "steam_score": loser.steam_score,
                            "prev_hparams": loser.hparams,
                        },
                    }
                    rs = self._composer.propose_reward_shaping(
                        policy_name=loser.policy_name,
                        base_hparams=new_hp,
                        metrics_snapshot=metrics_snap,
                    )
                    if rs:
                        new_hp["reward_shaping"] = rs
                except Exception:
                    pass

            loser.hparams = new_hp
            loser.steam_score = winner.steam_score * 0.8  # slight penalty to avoid instant re-top.
            loser.last_update_ts = time.time()
            loser.last_mutation_ts = time.time()
            loser.parents = [winner.policy_name]
            loser.mix_weights = {winner.policy_name: 1.0}
            loser.generation = winner.generation + 1
            mutated.append(loser)

        return mutated

    def _perturb_hparams(self, hparams: Dict[str, float]) -> Dict[str, float]:
        new_hp = copy.deepcopy(hparams)
        for k, v in list(new_hp.items()):
            if not isinstance(v, (int, float)):
                continue
            # Multiply by random factor in [1-ps, 1+ps]
            factor = self.rng.uniform(1 - self.perturb_scale, 1 + self.perturb_scale)
            new_val = max(1e-8, float(v) * factor)
            new_hp[k] = new_val
        return new_hp
