"""Agentic curriculum controller with difficulty strips."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStrip:
    strip_id: int
    difficulty: float
    min_success_rate: float = 0.7
    min_episodes: int = 100
    reward_multiplier: float = 1.0
    episodes_attempted: int = 0
    episodes_succeeded: int = 0
    total_reward: float = 0.0
    best_reward: float = float("-inf")

    def success_rate(self) -> float:
        if self.episodes_attempted == 0:
            return 0.0
        return self.episodes_succeeded / self.episodes_attempted

    def can_advance(self) -> bool:
        return (
            self.episodes_attempted >= self.min_episodes
            and self.success_rate() >= self.min_success_rate
        )

    def should_regress(self) -> bool:
        if self.episodes_attempted < 20:
            return False
        return self.success_rate() < 0.3


class AgenticCurriculum:
    """Self-regulating curriculum with up to N difficulty strips."""

    def __init__(
        self,
        *,
        num_strips: int = 16,
        auto_advance: bool = True,
        auto_regress: bool = True,
        reward_shaping: bool = True,
        checkpoint_dir: str = "checkpoints/curriculum",
    ) -> None:
        self.num_strips = max(1, int(num_strips))
        self.auto_advance = bool(auto_advance)
        self.auto_regress = bool(auto_regress)
        self.reward_shaping = bool(reward_shaping)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.strips = self._create_strips()
        self.current_strip_id = 0
        self.episode_count = 0
        self.total_steps = 0
        self.advancement_history: List[Dict] = []
        self.regression_history: List[Dict] = []

        logger.info(
            "AgenticCurriculum initialized: %s strips, auto_advance=%s, auto_regress=%s",
            self.num_strips,
            self.auto_advance,
            self.auto_regress,
        )

    def _create_strips(self) -> List[CurriculumStrip]:
        strips: List[CurriculumStrip] = []
        if self.num_strips == 1:
            return [CurriculumStrip(strip_id=0, difficulty=0.0)]
        for i in range(self.num_strips):
            difficulty = (i / (self.num_strips - 1)) ** 1.5
            min_success_rate = max(0.5, 0.7 - (i * 0.02))
            min_episodes = 100 + (i * 20)
            reward_multiplier = 1.0 + (i * 0.1)
            strips.append(
                CurriculumStrip(
                    strip_id=i,
                    difficulty=difficulty,
                    min_success_rate=min_success_rate,
                    min_episodes=min_episodes,
                    reward_multiplier=reward_multiplier,
                )
            )
        return strips

    @property
    def current_strip(self) -> CurriculumStrip:
        return self.strips[self.current_strip_id]

    def get_difficulty(self) -> float:
        return float(self.current_strip.difficulty)

    def record_episode(self, *, success: bool, episode_reward: float, episode_length: int) -> None:
        strip = self.current_strip
        strip.episodes_attempted += 1
        if success:
            strip.episodes_succeeded += 1
        strip.total_reward += float(episode_reward)
        strip.best_reward = max(strip.best_reward, float(episode_reward))

        self.episode_count += 1
        self.total_steps += int(episode_length)

        if strip.episodes_attempted % 50 == 0:
            logger.info(
                "Strip %s: %s eps, success=%.2f, avg_reward=%.2f",
                strip.strip_id,
                strip.episodes_attempted,
                strip.success_rate(),
                strip.total_reward / max(strip.episodes_attempted, 1),
            )

        if self.auto_advance and strip.can_advance():
            self._advance_strip()
        elif self.auto_regress and strip.should_regress():
            self._regress_strip()

    def _advance_strip(self) -> None:
        if self.current_strip_id >= self.num_strips - 1:
            logger.info("Already at max difficulty strip")
            return
        old_strip = self.current_strip_id
        self.current_strip_id += 1
        self.advancement_history.append(
            {
                "episode": self.episode_count,
                "from_strip": old_strip,
                "to_strip": self.current_strip_id,
                "success_rate": self.strips[old_strip].success_rate(),
            }
        )
        logger.info(
            "ADVANCED: strip %s -> %s (difficulty %.2f)",
            old_strip,
            self.current_strip_id,
            self.current_strip.difficulty,
        )

    def _regress_strip(self) -> None:
        if self.current_strip_id <= 0:
            logger.warning("Already at easiest strip, cannot regress")
            return
        old_strip = self.current_strip_id
        self.current_strip_id -= 1
        self.regression_history.append(
            {
                "episode": self.episode_count,
                "from_strip": old_strip,
                "to_strip": self.current_strip_id,
                "success_rate": self.strips[old_strip].success_rate(),
            }
        )
        logger.warning(
            "REGRESSED: strip %s -> %s (success rate %.2f)",
            old_strip,
            self.current_strip_id,
            self.strips[old_strip].success_rate(),
        )

    def shape_reward(self, *, raw_reward: float, success: bool, episode_length: int) -> float:
        if not self.reward_shaping:
            return float(raw_reward)
        strip = self.current_strip
        shaped = float(raw_reward) * strip.reward_multiplier
        if success:
            shaped += 10.0 * (1.0 + strip.difficulty)
            if episode_length < 1000:
                efficiency = 1.0 - (episode_length / 1000.0)
                shaped += 5.0 * efficiency * strip.difficulty
        else:
            shaped += -5.0 * (1.0 + strip.difficulty * 0.5)
        return shaped

    def get_curriculum_state(self) -> Dict:
        return {
            "current_strip_id": self.current_strip_id,
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "strips": [
                {
                    "strip_id": s.strip_id,
                    "difficulty": s.difficulty,
                    "episodes_attempted": s.episodes_attempted,
                    "episodes_succeeded": s.episodes_succeeded,
                    "success_rate": s.success_rate(),
                    "total_reward": s.total_reward,
                    "best_reward": s.best_reward,
                }
                for s in self.strips
            ],
            "advancement_history": self.advancement_history,
            "regression_history": self.regression_history,
        }

    def save_checkpoint(self, filename: str = "curriculum_state.json") -> Path:
        path = self.checkpoint_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.get_curriculum_state(), f, indent=2)
        logger.info("Curriculum checkpoint saved: %s", path)
        return path

    def load_checkpoint(self, filename: str = "curriculum_state.json") -> None:
        path = self.checkpoint_dir / filename
        if not path.exists():
            logger.warning("Curriculum checkpoint not found: %s", path)
            return
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        self.current_strip_id = int(state.get("current_strip_id", 0))
        self.episode_count = int(state.get("episode_count", 0))
        self.total_steps = int(state.get("total_steps", 0))
        self.advancement_history = list(state.get("advancement_history", []))
        self.regression_history = list(state.get("regression_history", []))
        for strip_data in state.get("strips", []):
            sid = int(strip_data.get("strip_id", 0))
            if sid < 0 or sid >= len(self.strips):
                continue
            strip = self.strips[sid]
            strip.episodes_attempted = int(strip_data.get("episodes_attempted", 0))
            strip.episodes_succeeded = int(strip_data.get("episodes_succeeded", 0))
            strip.total_reward = float(strip_data.get("total_reward", 0.0))
            strip.best_reward = float(strip_data.get("best_reward", strip.best_reward))
        logger.info(
            "Curriculum checkpoint loaded: strip %s, episode %s",
            self.current_strip_id,
            self.episode_count,
        )
