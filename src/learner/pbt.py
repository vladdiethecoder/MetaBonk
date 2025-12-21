"""Population Based Training (PBT).

Evolutionary hyperparameter optimization:
- Multiple agents with different hyperparameters
- Periodic exploit (copy weights) + explore (mutate hparams)
- Automatic discovery of optimal learning strategy

References:
- DeepMind PBT (Jaderberg et al.)
- AlphaStar training
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PBTConfig:
    """Configuration for Population Based Training."""
    
    # Population
    population_size: int = 10
    
    # Evolution
    exploit_interval: int = 1_000_000  # Steps between evolution
    truncation_selection: float = 0.2  # Bottom 20% replaced by top 20%
    
    # Mutation
    mutation_prob: float = 0.8
    mutation_strength: float = 0.2
    resample_prob: float = 0.2  # Probability of resampling from scratch
    
    # Hyperparameter ranges
    hparam_ranges: Dict[str, Tuple[float, float, str]] = field(default_factory=lambda: {
        "learning_rate": (1e-5, 1e-3, "log"),
        "entropy_coef": (0.001, 0.1, "log"),
        "clip_range": (0.1, 0.4, "linear"),
        "gamma": (0.95, 0.999, "linear"),
        "gae_lambda": (0.9, 0.99, "linear"),
    })
    
    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints/pbt")


@dataclass
class PopulationMember:
    """A single member of the population."""
    
    member_id: int
    
    # Hyperparameters
    hparams: Dict[str, float]
    
    # Performance
    fitness: float = 0.0  # Survival time or score
    total_steps: int = 0
    total_episodes: int = 0
    
    # Model state (path or dict)
    model_path: Optional[Path] = None
    
    # History
    generation: int = 0
    parent_id: Optional[int] = None


class PBTScheduler:
    """Manages Population Based Training."""
    
    def __init__(self, cfg: Optional[PBTConfig] = None):
        self.cfg = cfg or PBTConfig()
        
        # Population
        self.population: List[PopulationMember] = []
        
        # Evolution tracking
        self.generation = 0
        self.total_exploits = 0
        
        # Initialize population
        self._init_population()
    
    def _init_population(self):
        """Initialize population with random hyperparameters."""
        for i in range(self.cfg.population_size):
            hparams = self._sample_hparams()
            member = PopulationMember(
                member_id=i,
                hparams=hparams,
                generation=0,
            )
            self.population.append(member)
    
    def _sample_hparams(self) -> Dict[str, float]:
        """Sample random hyperparameters."""
        hparams = {}
        
        for name, (low, high, scale) in self.cfg.hparam_ranges.items():
            if scale == "log":
                value = np.exp(np.random.uniform(np.log(low), np.log(high)))
            else:
                value = np.random.uniform(low, high)
            hparams[name] = value
        
        return hparams
    
    def _mutate_hparams(self, hparams: Dict[str, float]) -> Dict[str, float]:
        """Mutate hyperparameters."""
        mutated = {}
        
        for name, value in hparams.items():
            if random.random() < self.cfg.mutation_prob:
                low, high, scale = self.cfg.hparam_ranges.get(
                    name, (value * 0.5, value * 2.0, "log")
                )
                
                if random.random() < self.cfg.resample_prob:
                    # Resample from scratch
                    if scale == "log":
                        mutated[name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                    else:
                        mutated[name] = np.random.uniform(low, high)
                else:
                    # Perturb
                    if scale == "log":
                        log_value = np.log(value)
                        log_range = np.log(high) - np.log(low)
                        perturbation = np.random.normal(0, self.cfg.mutation_strength * log_range)
                        mutated[name] = np.exp(np.clip(
                            log_value + perturbation,
                            np.log(low),
                            np.log(high),
                        ))
                    else:
                        perturbation = np.random.normal(0, self.cfg.mutation_strength * (high - low))
                        mutated[name] = np.clip(value + perturbation, low, high)
            else:
                mutated[name] = value
        
        return mutated
    
    def update_fitness(self, member_id: int, fitness: float, steps: int, episodes: int):
        """Update member fitness."""
        member = self.population[member_id]
        member.fitness = fitness
        member.total_steps += steps
        member.total_episodes += episodes
    
    def should_evolve(self, global_steps: int) -> bool:
        """Check if we should run evolution."""
        return global_steps > 0 and global_steps % self.cfg.exploit_interval == 0
    
    def evolve(self) -> List[Tuple[int, int, Dict[str, float]]]:
        """Run one evolution step.
        
        Returns:
            List of (loser_id, winner_id, new_hparams) tuples
        """
        self.generation += 1
        
        # Sort by fitness (descending)
        sorted_pop = sorted(
            enumerate(self.population),
            key=lambda x: x[1].fitness,
            reverse=True,
        )
        
        num_truncate = int(self.cfg.population_size * self.cfg.truncation_selection)
        
        # Top performers (to copy from)
        top_ids = [idx for idx, _ in sorted_pop[:num_truncate]]
        
        # Bottom performers (to replace)
        bottom_ids = [idx for idx, _ in sorted_pop[-num_truncate:]]
        
        changes = []
        
        for loser_id in bottom_ids:
            # Select random winner
            winner_id = random.choice(top_ids)
            winner = self.population[winner_id]
            
            # Mutate winner's hyperparameters
            new_hparams = self._mutate_hparams(winner.hparams)
            
            # Update loser
            loser = self.population[loser_id]
            loser.hparams = new_hparams
            loser.generation = self.generation
            loser.parent_id = winner_id
            loser.fitness = winner.fitness  # Inherit fitness temporarily
            
            changes.append((loser_id, winner_id, new_hparams))
            self.total_exploits += 1
        
        return changes
    
    def get_hparams(self, member_id: int) -> Dict[str, float]:
        """Get hyperparameters for a member."""
        return self.population[member_id].hparams
    
    def get_best_member(self) -> PopulationMember:
        """Get the best performing member."""
        return max(self.population, key=lambda m: m.fitness)
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        fitnesses = [m.fitness for m in self.population]
        
        stats = {
            "generation": self.generation,
            "mean_fitness": np.mean(fitnesses),
            "max_fitness": np.max(fitnesses),
            "min_fitness": np.min(fitnesses),
            "std_fitness": np.std(fitnesses),
            "total_exploits": self.total_exploits,
            "best_hparams": self.get_best_member().hparams,
        }
        
        return stats
    
    def save_state(self, path: Optional[Path] = None):
        """Save PBT state."""
        path = path or self.cfg.checkpoint_dir / "pbt_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "generation": self.generation,
            "total_exploits": self.total_exploits,
            "population": [
                {
                    "member_id": m.member_id,
                    "hparams": m.hparams,
                    "fitness": m.fitness,
                    "total_steps": m.total_steps,
                    "total_episodes": m.total_episodes,
                    "generation": m.generation,
                    "parent_id": m.parent_id,
                }
                for m in self.population
            ],
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: Path):
        """Load PBT state."""
        with open(path) as f:
            state = json.load(f)
        
        self.generation = state["generation"]
        self.total_exploits = state["total_exploits"]
        
        self.population = []
        for m_data in state["population"]:
            member = PopulationMember(
                member_id=m_data["member_id"],
                hparams=m_data["hparams"],
                fitness=m_data["fitness"],
                total_steps=m_data["total_steps"],
                total_episodes=m_data["total_episodes"],
                generation=m_data["generation"],
                parent_id=m_data.get("parent_id"),
            )
            self.population.append(member)


class PBTTrainer:
    """PBT training orchestrator."""
    
    def __init__(
        self,
        policy_factory: Callable[[Dict[str, float]], Any],
        env_factory: Callable[[int], Any],
        cfg: Optional[PBTConfig] = None,
    ):
        self.policy_factory = policy_factory
        self.env_factory = env_factory
        self.cfg = cfg or PBTConfig()
        
        # PBT scheduler
        self.scheduler = PBTScheduler(self.cfg)
        
        # Policies and environments
        self.policies = []
        self.envs = []
        
        self._init_population()
    
    def _init_population(self):
        """Initialize policies and environments."""
        for member in self.scheduler.population:
            policy = self.policy_factory(member.hparams)
            env = self.env_factory(member.member_id)
            
            self.policies.append(policy)
            self.envs.append(env)
    
    def train(self, total_steps: int):
        """Main PBT training loop."""
        global_steps = 0
        
        while global_steps < total_steps:
            # Train each member
            for i, (member, policy, env) in enumerate(
                zip(self.scheduler.population, self.policies, self.envs)
            ):
                # Collect experience and train
                fitness, steps, episodes = self._train_member(
                    member, policy, env,
                    num_steps=self.cfg.exploit_interval // self.cfg.population_size,
                )
                
                # Update fitness
                self.scheduler.update_fitness(i, fitness, steps, episodes)
            
            global_steps += self.cfg.exploit_interval
            
            # Evolution
            if self.scheduler.should_evolve(global_steps):
                changes = self.scheduler.evolve()
                
                # Copy weights from winners to losers
                for loser_id, winner_id, new_hparams in changes:
                    self._copy_weights(winner_id, loser_id)
                    self._update_hparams(loser_id, new_hparams)
                
                # Log
                stats = self.scheduler.get_population_stats()
                print(f"[PBT] Gen {stats['generation']} | "
                      f"Best: {stats['max_fitness']:.1f} | "
                      f"Mean: {stats['mean_fitness']:.1f}")
        
        # Save final state
        self.scheduler.save_state()
    
    def _train_member(
        self,
        member: PopulationMember,
        policy: Any,
        env: Any,
        num_steps: int,
    ) -> Tuple[float, int, int]:
        """Train a single member.

        This implementation is intentionally generic so it can wrap:
          - torch-style policies with `act()` / `update()` methods
          - callable policies (policy(obs) -> action)
          - pure-random baselines via env.action_space.sample()
        """
        total_reward = 0.0
        episodes = 0
        episode_reward = 0.0

        obs, _ = env.reset()
        for _ in range(num_steps):
            # Choose action.
            raw_action: Any = None
            if hasattr(policy, "act") and callable(getattr(policy, "act")):
                try:
                    raw_action = policy.act(obs)
                except Exception:
                    raw_action = None
            elif callable(policy):
                try:
                    raw_action = policy(obs)
                except Exception:
                    raw_action = None

            # Adapt action into env format.
            if isinstance(raw_action, dict):
                env_action = raw_action
            elif hasattr(env, "action_space"):
                try:
                    env_action = env.action_space.sample()
                except Exception:
                    env_action = {"move": np.zeros(2), "buttons": np.zeros(4)}
            else:
                env_action = {"move": np.zeros(2), "buttons": np.zeros(4)}

            # If action is a flat vector (e.g. 6D), map into dict.
            if not isinstance(env_action, dict):
                try:
                    arr = np.asarray(env_action).astype(np.float32).reshape(-1)
                    # Action-agnostic split: first 2 dims for move, next 4 dims
                    # for button logits (no hard-coded semantics per button).
                    move = arr[:2] if arr.size >= 2 else np.zeros(2, dtype=np.float32)
                    btn_raw = arr[2:6] if arr.size >= 6 else np.zeros(4, dtype=np.float32)
                    env_action = {"move": move, "buttons": (btn_raw > 0.0).astype(np.int32)}
                except Exception:
                    env_action = {"move": np.zeros(2), "buttons": np.zeros(4)}

            next_obs, reward, done, truncated, info = env.step(env_action)
            total_reward += float(reward)
            episode_reward += float(reward)

            # Optional policy update hooks (online).
            if hasattr(policy, "observe") and callable(getattr(policy, "observe")):
                try:
                    policy.observe(obs, env_action, reward, next_obs, done or truncated, info)
                except Exception:
                    pass
            if hasattr(policy, "update") and callable(getattr(policy, "update")):
                try:
                    policy.update()
                except Exception:
                    pass

            obs = next_obs

            if done or truncated:
                episodes += 1
                episode_reward = 0.0
                obs, _ = env.reset()

        fitness = total_reward / max(1, episodes)
        return float(fitness), int(num_steps), int(episodes)
    
    def _copy_weights(self, source_id: int, target_id: int):
        """Copy weights from source to target policy."""
        src = self.policies[source_id]
        dst = self.policies[target_id]

        # Allow policy to define its own copy hook.
        if hasattr(dst, "copy_weights_from") and callable(getattr(dst, "copy_weights_from")):
            try:
                dst.copy_weights_from(src)
                return
            except Exception:
                pass

        # Torch-ish state_dict copy.
        src_obj = getattr(src, "net", src)
        dst_obj = getattr(dst, "net", dst)

        if hasattr(src_obj, "state_dict") and hasattr(dst_obj, "load_state_dict"):
            try:
                state = src_obj.state_dict()
                # Detach tensors if torch is present.
                try:
                    import torch  # type: ignore

                    state = {
                        k: (v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
                        for k, v in state.items()
                    }
                except Exception:
                    state = copy.deepcopy(state)
                dst_obj.load_state_dict(state)
                return
            except Exception:
                pass

        # Fallback: deep-copy the entire policy if possible.
        try:
            self.policies[target_id] = copy.deepcopy(src)
        except Exception:
            return
    
    def _update_hparams(self, member_id: int, hparams: Dict[str, float]):
        """Update hyperparameters for a policy."""
        policy = self.policies[member_id]

        # Allow policy-defined hook.
        if hasattr(policy, "update_hparams") and callable(getattr(policy, "update_hparams")):
            try:
                policy.update_hparams(hparams)
                return
            except Exception:
                pass

        # Best-effort optimizer / cfg mapping.
        lr = hparams.get("learning_rate", hparams.get("lr"))
        entropy_coef = hparams.get("entropy_coef")

        # PolicyLearner-style.
        cfg = getattr(policy, "cfg", None)
        if cfg is not None:
            try:
                if lr is not None and hasattr(cfg, "learning_rate"):
                    cfg.learning_rate = float(lr)
                if entropy_coef is not None and hasattr(cfg, "entropy_coef"):
                    cfg.entropy_coef = float(entropy_coef)
                for k in ("clip_range", "gamma", "gae_lambda", "batch_size", "minibatch_size", "num_epochs"):
                    if k in hparams and hasattr(cfg, k):
                        setattr(cfg, k, type(getattr(cfg, k))(hparams[k]))  # type: ignore[misc]
            except Exception:
                pass

        # Optimizer update.
        opt = getattr(policy, "opt", None) or getattr(policy, "optimizer", None)
        if opt is not None and lr is not None:
            try:
                for g in opt.param_groups:  # type: ignore[attr-defined]
                    g["lr"] = float(lr)
            except Exception:
                pass
