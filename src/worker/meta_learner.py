"""Game-agnostic UI meta-learning for menu navigation.

This module learns *vision-only* UI navigation patterns across episodes by
recording UI click actions on specific visual "scene hashes" and reusing them
when the same (or very similar) screen reappears.

Design goals:
- Game-agnostic: no hardcoded game logic, button coordinates, or labels.
- Deterministic: no random behavior (follow probability is deterministic).
- Lightweight: store only compact hashes + quantized click targets.
"""

from __future__ import annotations

import os
import time
import zlib
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _parse_u64_hex(scene_hash: Optional[str]) -> Optional[int]:
    if not scene_hash:
        return None
    try:
        return int(str(scene_hash).strip(), 16) & ((1 << 64) - 1)
    except Exception:
        return None


def _hamming_sim_u64(a: int, b: int) -> float:
    # Similarity in [0,1], based on 64-bit Hamming distance.
    ham = int(a ^ b).bit_count()
    return float(1.0 - (float(ham) / 64.0))


@dataclass(frozen=True)
class MetaLearnerConfig:
    enabled: bool = False
    # Only use suggestions before gameplay starts (safer default for training).
    pre_gameplay_only: bool = True

    # Memory bounds.
    max_sequences: int = 64
    max_steps_per_episode: int = 256
    max_scenes: int = 4096

    # Matching and target encoding.
    min_similarity: float = 0.88
    bin_grid: int = 32  # quantize click targets into bin_grid x bin_grid bins
    max_candidate_dist: float = 0.25  # max L2 distance (in normalized coords) to snap to UI candidates

    # Applying suggestions.
    follow_prob: float = 0.80  # deterministic
    scene_cooldown_s: float = 0.40
    seed: int = 0

    @classmethod
    def from_env(cls) -> "MetaLearnerConfig":
        def _i(key: str, default: int) -> int:
            try:
                return int(str(os.environ.get(key, str(default)) or str(default)).strip())
            except Exception:
                return int(default)

        def _f(key: str, default: float) -> float:
            try:
                return float(str(os.environ.get(key, str(default)) or str(default)).strip())
            except Exception:
                return float(default)

        enabled = _truthy(os.environ.get("METABONK_META_LEARNING", "0"))
        pre_gameplay_only = _truthy(os.environ.get("METABONK_META_LEARNER_PRE_GAMEPLAY_ONLY", "1"))
        max_sequences = max(0, _i("METABONK_META_LEARNER_MAX_SEQUENCES", cls.max_sequences))
        max_steps_per_episode = max(0, _i("METABONK_META_LEARNER_MAX_STEPS", cls.max_steps_per_episode))
        max_scenes = max(0, _i("METABONK_META_LEARNER_MAX_SCENES", cls.max_scenes))
        min_similarity = _clamp01(_f("METABONK_META_LEARNER_MIN_SIMILARITY", cls.min_similarity))
        bin_grid = max(8, _i("METABONK_META_LEARNER_BIN_GRID", cls.bin_grid))
        max_candidate_dist = _clamp01(_f("METABONK_META_LEARNER_MAX_CANDIDATE_DIST", cls.max_candidate_dist))
        follow_prob = _clamp01(_f("METABONK_META_LEARNER_FOLLOW_PROB", cls.follow_prob))
        scene_cooldown_s = max(0.0, _f("METABONK_META_LEARNER_SCENE_COOLDOWN_S", cls.scene_cooldown_s))
        seed = _i("METABONK_META_LEARNER_SEED", cls.seed)
        return cls(
            enabled=bool(enabled),
            pre_gameplay_only=bool(pre_gameplay_only),
            max_sequences=int(max_sequences),
            max_steps_per_episode=int(max_steps_per_episode),
            max_scenes=int(max_scenes),
            min_similarity=float(min_similarity),
            bin_grid=int(bin_grid),
            max_candidate_dist=float(max_candidate_dist),
            follow_prob=float(follow_prob),
            scene_cooldown_s=float(scene_cooldown_s),
            seed=int(seed),
        )


@dataclass(frozen=True)
class MetaSuggestion:
    x: float  # normalized (0..1)
    y: float  # normalized (0..1)
    similarity: float
    matched_scene_hash: str
    reason: str


@dataclass(frozen=True)
class MetaClickSuggestion:
    index: int
    similarity: float
    matched_scene_hash: str
    reason: str


class UINavigationMetaLearner:
    """Learn and reuse UI click actions keyed by visual scene fingerprints."""

    def __init__(self, cfg: Optional[MetaLearnerConfig] = None) -> None:
        self.cfg = cfg or MetaLearnerConfig.from_env()

        # Episode-local trace of UI steps.
        self._episode_steps: List[dict] = []

        # Successful sequences (bounded, most recent kept).
        self._successful_sequences: Deque[dict] = deque(maxlen=max(1, int(self.cfg.max_sequences)) if self.cfg.max_sequences > 0 else 1)

        # Scene memory: scene_u64 -> {bin -> count}, plus a small LRU for scene cooldown.
        self._scene_bins: "OrderedDict[int, Dict[Tuple[int, int], int]]" = OrderedDict()
        self._scene_last_apply_ts: "OrderedDict[int, float]" = OrderedDict()

        # Counters/metrics.
        self.success_sequences: int = 0
        self.success_steps: int = 0
        self.suggestions_considered: int = 0
        self.suggestions_applied: int = 0
        self.last_suggestion_similarity: float = 0.0
        self.last_suggestion_reason: str = ""

    def reset_episode(self) -> None:
        self._episode_steps = []

    def record_step(
        self,
        *,
        now: float,
        scene_hash: Optional[str],
        gameplay_started: bool,
        state_type: str,
        click_xy_norm: Optional[Tuple[float, float]],
        action_source: Optional[str] = None,
    ) -> None:
        """Record a UI step for the current episode (best-effort)."""
        if click_xy_norm is None:
            return
        st = str(state_type or "").strip().lower()
        if st != "menu_ui" and bool(gameplay_started):
            # Only capture UI steps (pre-gameplay by default).
            return
        if self.cfg.pre_gameplay_only and bool(gameplay_started):
            return

        h = _parse_u64_hex(scene_hash)
        if h is None:
            return

        if self.cfg.max_steps_per_episode > 0 and len(self._episode_steps) >= int(self.cfg.max_steps_per_episode):
            return

        x, y = click_xy_norm
        x = _clamp01(float(x))
        y = _clamp01(float(y))
        self._episode_steps.append(
            {
                "ts": float(now),
                "scene_hash": str(scene_hash),
                "scene_u64": int(h),
                "state_type": str(st or "uncertain"),
                "action": {"type": "click", "x": float(x), "y": float(y)},
                "action_source": str(action_source or ""),
            }
        )

    def record_episode_end(self, *, now: float, reached_gameplay: bool) -> bool:
        """On episode end, persist the episode steps if it reached gameplay."""
        if bool(reached_gameplay) and self._episode_steps:
            steps = list(self._episode_steps)
            self._successful_sequences.append(
                {
                    "ts": float(now),
                    "length": int(len(steps)),
                    "steps": steps,
                }
            )
            self.success_sequences += 1
            self.success_steps += int(len(steps))

            for s in steps:
                try:
                    h = int(s.get("scene_u64"))
                except Exception:
                    continue
                act = s.get("action") or {}
                if not isinstance(act, dict) or act.get("type") != "click":
                    continue
                try:
                    x = _clamp01(float(act.get("x", 0.0)))
                    y = _clamp01(float(act.get("y", 0.0)))
                except Exception:
                    continue
                self._remember_scene_action(scene_u64=int(h), x=float(x), y=float(y))

            self.reset_episode()
            return True

        self.reset_episode()
        return False

    def _remember_scene_action(self, *, scene_u64: int, x: float, y: float) -> None:
        # LRU insert/update.
        if scene_u64 in self._scene_bins:
            try:
                bins = self._scene_bins.pop(scene_u64)
            except Exception:
                bins = {}
        else:
            bins = {}
        self._scene_bins[scene_u64] = bins

        # Bound unique scenes.
        if self.cfg.max_scenes > 0:
            while len(self._scene_bins) > int(self.cfg.max_scenes):
                try:
                    self._scene_bins.popitem(last=False)
                except Exception:
                    self._scene_bins.clear()
                    break

        # Quantize click target.
        g = int(self.cfg.bin_grid)
        bx = max(0, min(g - 1, int(float(x) * float(g))))
        by = max(0, min(g - 1, int(float(y) * float(g))))
        key = (int(bx), int(by))
        bins[key] = int(bins.get(key, 0) or 0) + 1

    def _scene_cooldown_active(self, *, scene_u64: int, now: float) -> bool:
        if float(self.cfg.scene_cooldown_s) <= 0.0:
            return False
        try:
            last = float(self._scene_last_apply_ts.get(int(scene_u64), 0.0) or 0.0)
        except Exception:
            last = 0.0
        if last <= 0.0:
            return False
        return (float(now) - float(last)) < float(self.cfg.scene_cooldown_s)

    def _touch_scene_apply(self, *, scene_u64: int, now: float) -> None:
        if scene_u64 in self._scene_last_apply_ts:
            try:
                del self._scene_last_apply_ts[scene_u64]
            except Exception:
                pass
        self._scene_last_apply_ts[int(scene_u64)] = float(now)
        while len(self._scene_last_apply_ts) > 2048:
            try:
                self._scene_last_apply_ts.popitem(last=False)
            except Exception:
                self._scene_last_apply_ts.clear()
                break

    def _best_bin_xy(self, bins: Dict[Tuple[int, int], int]) -> Optional[Tuple[float, float]]:
        if not bins:
            return None
        best = None
        best_c = -1
        for k, c in bins.items():
            try:
                cc = int(c)
            except Exception:
                cc = 0
            if cc > best_c:
                best_c = cc
                best = k
        if best is None:
            return None
        bx, by = best
        g = int(self.cfg.bin_grid)
        x = (float(int(bx)) + 0.5) / float(max(1, g))
        y = (float(int(by)) + 0.5) / float(max(1, g))
        return (_clamp01(x), _clamp01(y))

    def suggest_action_from_history(self, *, scene_hash: Optional[str]) -> Optional[MetaSuggestion]:
        """Return a click suggestion (normalized) if a similar scene was seen before."""
        scene_u64 = _parse_u64_hex(scene_hash)
        if scene_u64 is None:
            return None
        if not self._scene_bins:
            return None

        best_scene = None
        best_sim = -1.0
        # Iterate over known scenes (bounded).
        for s_u64 in self._scene_bins.keys():
            sim = _hamming_sim_u64(int(scene_u64), int(s_u64))
            if sim > best_sim:
                best_sim = float(sim)
                best_scene = int(s_u64)
        if best_scene is None:
            return None
        if float(best_sim) < float(self.cfg.min_similarity):
            return None

        try:
            bins = self._scene_bins.get(int(best_scene)) or {}
        except Exception:
            bins = {}
        xy = self._best_bin_xy(bins)
        if xy is None:
            return None
        x, y = xy
        return MetaSuggestion(
            x=float(x),
            y=float(y),
            similarity=float(best_sim),
            matched_scene_hash=f"{int(best_scene):016x}",
            reason="history_match",
        )

    def suggest_click_index(
        self,
        *,
        now: float,
        scene_hash: Optional[str],
        ui_elements: Sequence[Sequence[float]],
        valid_indices: Sequence[int],
    ) -> Optional[MetaClickSuggestion]:
        """Suggest a discrete UI click index by snapping history to the current candidate set."""
        sug = self.suggest_action_from_history(scene_hash=scene_hash)
        if sug is None:
            return None

        # Cooldown is keyed on the matched scene (stable across minor hash noise).
        matched_u64 = _parse_u64_hex(sug.matched_scene_hash)
        if matched_u64 is not None and self._scene_cooldown_active(scene_u64=int(matched_u64), now=float(now)):
            return None

        best_i = None
        best_d = None
        for i in valid_indices:
            if i < 0 or i >= len(ui_elements):
                continue
            row = ui_elements[i]
            if not (isinstance(row, (list, tuple)) and len(row) >= 2):
                continue
            try:
                cx = float(row[0])
                cy = float(row[1])
            except Exception:
                continue
            dx = cx - float(sug.x)
            dy = cy - float(sug.y)
            d = float(dx * dx + dy * dy)
            if best_d is None or d < best_d:
                best_d = d
                best_i = int(i)

        if best_i is None or best_d is None:
            return None
        # Reject if snapping distance is too large (avoid bad clicks on coarse grids).
        if float(best_d) > float(self.cfg.max_candidate_dist) * float(self.cfg.max_candidate_dist):
            return None

        return MetaClickSuggestion(
            index=int(best_i),
            similarity=float(sug.similarity),
            matched_scene_hash=str(sug.matched_scene_hash),
            reason=str(sug.reason),
        )

    def should_follow_suggestion(
        self,
        *,
        instance_id: str,
        episode_idx: int,
        step: int,
        scene_hash: str,
        suggested_index: int,
    ) -> bool:
        """Deterministic decision to follow a suggestion based on follow_prob."""
        p = float(self.cfg.follow_prob)
        if p <= 0.0:
            return False
        if p >= 1.0:
            return True
        # Use a stable 32-bit hash as a deterministic RNG.
        key = f"{int(self.cfg.seed)}:{instance_id}:{int(episode_idx)}:{int(step)}:{scene_hash}:{int(suggested_index)}"
        h = int(zlib.adler32(key.encode("utf-8")) & 0xFFFFFFFF)
        u = float(h) / float(2**32)
        return u < p

    def note_suggestion_applied(self, *, similarity: float, reason: str) -> None:
        self.suggestions_applied += 1
        self.last_suggestion_similarity = float(similarity)
        self.last_suggestion_reason = str(reason or "")

    def note_suggestion_considered(self) -> None:
        self.suggestions_considered += 1

    def mark_scene_applied(self, *, matched_scene_hash: str, now: float) -> None:
        """Mark a matched scene as having an applied suggestion (for cooldown)."""
        u = _parse_u64_hex(matched_scene_hash)
        if u is None:
            return
        self._touch_scene_apply(scene_u64=int(u), now=float(now))

    def analyze_patterns(self) -> Dict[str, object]:
        """Return lightweight aggregate patterns from successful sequences."""
        lengths = [int(seq.get("length", 0) or 0) for seq in self._successful_sequences]
        avg_len = float(sum(lengths) / max(1, len(lengths))) if lengths else 0.0
        return {
            "success_sequences": int(self.success_sequences),
            "avg_ui_steps": float(avg_len),
            "unique_scenes": int(len(self._scene_bins)),
        }

    def metrics(self) -> Dict[str, float | str | None]:
        return {
            "meta_learning_enabled": 1.0 if bool(self.cfg.enabled) else 0.0,
            "meta_success_sequences": float(self.success_sequences),
            "meta_success_steps": float(self.success_steps),
            "meta_unique_scenes": float(len(self._scene_bins)),
            "meta_suggestions_considered": float(self.suggestions_considered),
            "meta_suggestions_applied": float(self.suggestions_applied),
            "meta_last_suggestion_similarity": float(self.last_suggestion_similarity),
            "meta_last_suggestion_reason": str(self.last_suggestion_reason),
            # Back-compat alias used by older monitoring scripts.
            "meta_learner_sequences_learned": float(self.success_sequences),
        }


__all__ = [
    "MetaLearnerConfig",
    "MetaSuggestion",
    "MetaClickSuggestion",
    "UINavigationMetaLearner",
]
