"""Go-Explore: State Archiving for Hard Exploration.

Implements Go-Explore algorithm for TAS-style glitch discovery:
- Cell-based state archiving  
- Probabilistic selection of interesting states
- Return-to-state via save states
- Robustification via imitation learning

References:
- First Return Then Explore (Go-Explore paper)
- Adversarial Environment Design specification
"""

from __future__ import annotations

import hashlib
import heapq
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class StateCell:
    """A discretized state cell in the Go-Explore archive."""
    
    cell_key: Tuple  # (x_grid, y_grid, z_grid, v_mag_bin, audio_bin)
    
    # Best trajectory to reach this cell
    trajectory: List[np.ndarray] = field(default_factory=list)  # Actions
    save_state: Optional[bytes] = None  # Serialized game state
    
    # Metrics
    times_visited: int = 0
    times_selected: int = 0
    discovery_step: int = 0
    
    # Quality metrics
    max_velocity: float = 0.0
    max_height: float = 0.0
    oob_distance: float = 0.0  # Distance from NavMesh
    audio_entropy: float = 0.0
    
    # Glitch score
    glitch_score: float = 0.0
    
    @property
    def novelty_weight(self) -> float:
        """Weight for selection - prefer under-explored cells."""
        return 1.0 / (self.times_selected + 1)
    
    @property
    def interest_weight(self) -> float:
        """Weight based on glitch potential."""
        return (
            0.3 * min(self.max_velocity / 100.0, 1.0) +
            0.2 * min(self.max_height / 50.0, 1.0) +
            0.3 * min(self.oob_distance / 10.0, 1.0) +
            0.2 * min(self.audio_entropy, 1.0)
        )


@dataclass
class GoExploreConfig:
    """Configuration for Go-Explore."""
    
    # Grid discretization
    position_bin_size: float = 2.0  # Meters
    velocity_bins: List[float] = field(default_factory=lambda: [0, 5, 20, 50, 100, 500])
    audio_entropy_bins: List[float] = field(default_factory=lambda: [0, 0.3, 0.6, 0.9])
    
    # Exploration
    explore_steps: int = 100  # Steps per exploration phase
    archive_selection_temp: float = 0.5  # Softmax temperature
    
    # Robustification
    robustify_threshold: int = 10  # Min visits before robustification
    
    # Archive limits
    max_archive_size: int = 100000


class CellMapper:
    """Maps continuous states to discrete cell keys."""
    
    def __init__(self, cfg: GoExploreConfig):
        self.cfg = cfg
    
    def state_to_cell(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        audio_entropy: float,
    ) -> Tuple:
        """Map continuous state to cell key."""
        # Position grid
        x_bin = int(position[0] / self.cfg.position_bin_size)
        y_bin = int(position[1] / self.cfg.position_bin_size)
        z_bin = int(position[2] / self.cfg.position_bin_size)
        
        # Velocity magnitude bin
        v_mag = np.linalg.norm(velocity)
        v_bin = 0
        for i, threshold in enumerate(self.cfg.velocity_bins):
            if v_mag >= threshold:
                v_bin = i
        
        # Audio entropy bin
        a_bin = 0
        for i, threshold in enumerate(self.cfg.audio_entropy_bins):
            if audio_entropy >= threshold:
                a_bin = i
        
        return (x_bin, y_bin, z_bin, v_bin, a_bin)


class GoExploreArchive:
    """The state archive for Go-Explore."""
    
    def __init__(self, cfg: Optional[GoExploreConfig] = None):
        self.cfg = cfg or GoExploreConfig()
        self.mapper = CellMapper(self.cfg)
        
        # Archive storage
        self.cells: Dict[Tuple, StateCell] = {}
        
        # Statistics
        self.total_steps = 0
        self.total_cells_discovered = 0
        
        # Priority queue for selection (max-heap via negation)
        self._selection_heap: List[Tuple[float, Tuple]] = []
    
    def add_or_update(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        audio_entropy: float,
        trajectory: List[np.ndarray],
        save_state: bytes,
        oob_distance: float = 0.0,
    ) -> Tuple[bool, StateCell]:
        """Add or update a cell. Returns (is_new, cell)."""
        cell_key = self.mapper.state_to_cell(position, velocity, audio_entropy)
        
        is_new = cell_key not in self.cells
        
        if is_new:
            # New cell discovered
            cell = StateCell(
                cell_key=cell_key,
                trajectory=trajectory,
                save_state=save_state,
                discovery_step=self.total_steps,
            )
            self.cells[cell_key] = cell
            self.total_cells_discovered += 1
        else:
            cell = self.cells[cell_key]
        
        # Update metrics
        cell.times_visited += 1
        cell.max_velocity = max(cell.max_velocity, np.linalg.norm(velocity))
        cell.max_height = max(cell.max_height, position[1])
        cell.oob_distance = max(cell.oob_distance, oob_distance)
        cell.audio_entropy = max(cell.audio_entropy, audio_entropy)
        
        # Update trajectory if shorter
        if is_new or len(trajectory) < len(cell.trajectory):
            cell.trajectory = trajectory
            cell.save_state = save_state
        
        # Compute glitch score
        cell.glitch_score = self._compute_glitch_score(cell)
        
        # Update selection heap
        priority = -(cell.novelty_weight + cell.interest_weight)
        heapq.heappush(self._selection_heap, (priority, cell_key))
        
        return is_new, cell
    
    def _compute_glitch_score(self, cell: StateCell) -> float:
        """Compute glitch potential score."""
        score = 0.0
        
        # High velocity = potential zip
        if cell.max_velocity > 50:
            score += 0.3
        if cell.max_velocity > 100:
            score += 0.3
        
        # OOB = potential clip
        if cell.oob_distance > 1:
            score += 0.2
        if cell.oob_distance > 5:
            score += 0.3
        
        # High audio entropy = physics explosion
        if cell.audio_entropy > 0.8:
            score += 0.2
        
        return min(score, 1.0)
    
    def select_cell(self) -> Optional[StateCell]:
        """Select a cell for exploration using weighted sampling."""
        if not self.cells:
            return None
        
        # Compute weights
        cells = list(self.cells.values())
        weights = np.array([
            c.novelty_weight + c.interest_weight
            for c in cells
        ])
        
        # Softmax with temperature
        weights = np.exp(weights / self.cfg.archive_selection_temp)
        probs = weights / weights.sum()
        
        # Sample
        idx = np.random.choice(len(cells), p=probs)
        selected = cells[idx]
        selected.times_selected += 1
        
        return selected
    
    def get_glitch_candidates(self, top_k: int = 10) -> List[StateCell]:
        """Get top glitch candidates."""
        sorted_cells = sorted(
            self.cells.values(),
            key=lambda c: c.glitch_score,
            reverse=True,
        )
        return sorted_cells[:top_k]
    
    def save(self, path: Path):
        """Save archive to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'cells': self.cells,
                'total_steps': self.total_steps,
                'total_cells_discovered': self.total_cells_discovered,
            }, f)
    
    def load(self, path: Path):
        """Load archive from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.cells = data['cells']
            self.total_steps = data['total_steps']
            self.total_cells_discovered = data['total_cells_discovered']


class GoExploreAgent:
    """Go-Explore agent for glitch discovery."""
    
    def __init__(
        self,
        cfg: Optional[GoExploreConfig] = None,
        action_space_size: int = 6,
    ):
        self.cfg = cfg or GoExploreConfig()
        self.archive = GoExploreArchive(self.cfg)
        self.action_space_size = action_space_size
        
        # Current exploration state
        self.current_trajectory: List[np.ndarray] = []
        self.current_start_cell: Optional[StateCell] = None
        self.explore_step = 0
        
        # Statistics
        self.total_episodes = 0
        self.glitches_found: List[StateCell] = []
    
    def start_exploration(self, save_state_fn: Callable[[], bytes]) -> Optional[bytes]:
        """Start a new exploration phase.
        
        Returns: Save state to restore to (None if starting fresh).
        """
        self.explore_step = 0
        self.current_trajectory = []
        
        # Select a cell to explore from
        cell = self.archive.select_cell()
        
        if cell is not None:
            self.current_start_cell = cell
            return cell.save_state
        else:
            self.current_start_cell = None
            return None
    
    def get_action(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        audio_entropy: float,
        policy_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """Get exploration action."""
        self.explore_step += 1
        
        # Random exploration or policy-guided
        if policy_fn is not None and random.random() > 0.3:
            action = policy_fn(position, velocity)
        else:
            # Random action
            action = np.random.randn(self.action_space_size) * 0.5
        
        # Occasionally inject extreme inputs (TAS-style)
        if random.random() < 0.1:
            action = np.random.choice([-1, 1], size=self.action_space_size)
        
        self.current_trajectory.append(action)
        
        return action
    
    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        audio_entropy: float,
        save_state_fn: Callable[[], bytes],
        oob_distance: float = 0.0,
    ) -> Dict[str, Any]:
        """Process a step, update archive."""
        self.archive.total_steps += 1
        
        # Add to archive
        is_new, cell = self.archive.add_or_update(
            position=position,
            velocity=velocity,
            audio_entropy=audio_entropy,
            trajectory=self.current_trajectory.copy(),
            save_state=save_state_fn(),
            oob_distance=oob_distance,
        )
        
        # Check for glitch
        glitch_detected = False
        if cell.glitch_score > 0.5:
            if cell not in self.glitches_found:
                self.glitches_found.append(cell)
                glitch_detected = True
        
        # Check if exploration phase complete
        should_reset = self.explore_step >= self.cfg.explore_steps
        
        result = {
            "is_new_cell": is_new,
            "cell_key": cell.cell_key,
            "glitch_score": cell.glitch_score,
            "glitch_detected": glitch_detected,
            "should_reset": should_reset,
            "archive_size": len(self.archive.cells),
        }
        
        if should_reset:
            self.total_episodes += 1
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        return {
            "total_steps": self.archive.total_steps,
            "total_cells": len(self.archive.cells),
            "total_episodes": self.total_episodes,
            "glitches_found": len(self.glitches_found),
            "top_velocity": max(
                (c.max_velocity for c in self.archive.cells.values()),
                default=0,
            ),
            "max_oob": max(
                (c.oob_distance for c in self.archive.cells.values()),
                default=0,
            ),
        }
