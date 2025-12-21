"""Kinetic Leaderboards and Ghost Ecosystem.

Real-time comparison against historic runs:
- Racing bar chart visualization
- 1000 concurrent ghost positions
- Pace differential calculations
- Graveyard zones (death heatmaps)

References:
- Racing Bar Chart visualization
- Trackmania/Super Meat Boy ghosts
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GhostRun:
    """A ghost of a past run."""
    
    run_id: int
    agent_name: str
    
    # Map seed - ghosts are only shown on matching seeds to prevent clipping
    map_seed: str  # e.g., "seed_12345" or level hash
    
    # Final stats
    final_score: float
    survival_time: float
    
    # Position data (sampled at fixed intervals)
    # positions[i] = position at timestamp i * sample_rate
    positions: np.ndarray  # Shape: [T, 3]
    sample_rate: float = 0.1  # 10 Hz
    
    # Classification
    is_top_10: bool = False
    is_world_record: bool = False
    
    @property
    def color(self) -> Tuple[int, int, int, int]:
        """Get ghost color."""
        if self.is_world_record:
            return (255, 215, 0, 200)  # Gold
        elif self.is_top_10:
            return (255, 100, 100, 180)  # Red
        else:
            return (200, 200, 200, 100)  # Gray, semi-transparent
    
    def get_position_at(self, timestamp: float) -> Optional[np.ndarray]:
        """Get position at a specific timestamp."""
        idx = int(timestamp / self.sample_rate)
        if 0 <= idx < len(self.positions):
            return self.positions[idx]
        return None


@dataclass
class LeaderboardEntry:
    """An entry in the kinetic leaderboard."""
    
    run_id: int
    agent_name: str
    score: float
    survival_time: float
    
    # For animation
    current_position: int = 0
    target_position: int = 0
    animation_progress: float = 1.0
    
    # Visual
    is_current: bool = False


@dataclass
class GraveyardZone:
    """A zone where many agents have died."""
    
    center: np.ndarray  # [x, y, z]
    radius: float
    death_count: int
    danger_level: float  # 0-1
    
    @property
    def color(self) -> Tuple[int, int, int, int]:
        """Get zone color based on danger."""
        alpha = int(self.danger_level * 150)
        return (255, 50, 50, alpha)


class GhostEcosystem:
    """Manages the ghost visualization system.
    
    IMPORTANT: Ghosts are map-seed-specific to prevent visual clipping.
    Procedural maps have different geometry per seed, so ghosts from
    one seed would clip through walls/floors on a different seed.
    """
    
    def __init__(self, max_ghosts: int = 1000, max_ghosts_per_seed: int = 100):
        self.max_ghosts = max_ghosts
        self.max_ghosts_per_seed = max_ghosts_per_seed
        
        # Ghost storage - keyed by map seed
        self.ghosts_by_seed: Dict[str, List[GhostRun]] = {}
        self.ghosts: List[GhostRun] = []  # Flat list for global rankings
        
        # Current active seed
        self.active_seed: Optional[str] = None
        
        # Top 10 (for current seed)
        self.top_10: List[GhostRun] = []
        self.world_record: Optional[GhostRun] = None
        
        # Global records (across all seeds)
        self.global_top_10: List[GhostRun] = []
        self.global_world_record: Optional[GhostRun] = None
        
        # Death positions - also keyed by seed
        self.death_positions_by_seed: Dict[str, List[np.ndarray]] = {}
        self.graveyard_zones: List[GraveyardZone] = []
    
    def set_active_seed(self, map_seed: str):
        """Set the currently active map seed.
        
        Only ghosts matching this seed will be displayed to prevent
        visual clipping on procedurally generated maps.
        """
        self.active_seed = map_seed
        
        # Initialize storage for this seed if needed
        if map_seed not in self.ghosts_by_seed:
            self.ghosts_by_seed[map_seed] = []
            self.death_positions_by_seed[map_seed] = []
        
        # Update top 10 for this seed
        self._update_top_10()
        self._update_graveyard_zones()
    
    def add_ghost(
        self,
        run_id: int,
        agent_name: str,
        map_seed: str,
        positions: np.ndarray,
        final_score: float,
        survival_time: float,
        death_position: Optional[np.ndarray] = None,
    ):
        """Add a ghost from a completed run.
        
        Args:
            map_seed: The procedural map seed. Ghosts are only shown
                     on runs with matching seeds to prevent clipping.
        """
        ghost = GhostRun(
            run_id=run_id,
            agent_name=agent_name,
            map_seed=map_seed,
            final_score=final_score,
            survival_time=survival_time,
            positions=positions,
        )
        
        # Add to global list
        self.ghosts.append(ghost)
        
        # Add to seed-specific list
        if map_seed not in self.ghosts_by_seed:
            self.ghosts_by_seed[map_seed] = []
        self.ghosts_by_seed[map_seed].append(ghost)
        
        # Track death position by seed
        if death_position is not None:
            if map_seed not in self.death_positions_by_seed:
                self.death_positions_by_seed[map_seed] = []
            self.death_positions_by_seed[map_seed].append(death_position)
            
            if map_seed == self.active_seed:
                self._update_graveyard_zones()
        
        # Trim global list
        if len(self.ghosts) > self.max_ghosts:
            removed = self.ghosts.pop(0)
            # Also remove from seed list
            seed_list = self.ghosts_by_seed.get(removed.map_seed, [])
            if removed in seed_list:
                seed_list.remove(removed)
        
        # Trim per-seed list
        seed_list = self.ghosts_by_seed[map_seed]
        if len(seed_list) > self.max_ghosts_per_seed:
            seed_list.pop(0)
        
        # Update top 10
        self._update_top_10()
        self._update_global_top_10()
    
    def _update_top_10(self):
        """Update top 10 ranking for current seed only."""
        if self.active_seed is None:
            self.top_10 = []
            self.world_record = None
            return
        
        # Get ghosts for current seed only
        seed_ghosts = self.ghosts_by_seed.get(self.active_seed, [])
        
        sorted_ghosts = sorted(
            seed_ghosts,
            key=lambda g: g.survival_time,
            reverse=True,
        )
        
        # Reset flags for this seed's ghosts
        for g in seed_ghosts:
            g.is_top_10 = False
            g.is_world_record = False
        
        self.top_10 = sorted_ghosts[:10]
        
        for g in self.top_10:
            g.is_top_10 = True
        
        if self.top_10:
            self.world_record = self.top_10[0]
            self.world_record.is_world_record = True
    
    def _update_global_top_10(self):
        """Update global top 10 across all seeds (for stats only)."""
        sorted_ghosts = sorted(
            self.ghosts,
            key=lambda g: g.survival_time,
            reverse=True,
        )
        
        self.global_top_10 = sorted_ghosts[:10]
        self.global_world_record = self.global_top_10[0] if self.global_top_10 else None
    
    def _update_graveyard_zones(self):
        """Cluster death positions into graveyard zones for current seed."""
        if self.active_seed is None:
            self.graveyard_zones = []
            return
        
        death_positions = self.death_positions_by_seed.get(self.active_seed, [])
        
        if len(death_positions) < 10:
            self.graveyard_zones = []
            return
        
        # Simple clustering by grid
        grid_size = 5.0
        clusters: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
        
        for pos in death_positions[-1000:]:  # Last 1000 deaths for this seed
            key = (
                int(pos[0] / grid_size),
                int(pos[1] / grid_size),
                int(pos[2] / grid_size),
            )
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(pos)
        
        # Convert to zones
        self.graveyard_zones = []
        max_deaths = max(len(v) for v in clusters.values())
        
        for key, positions in clusters.items():
            if len(positions) >= 5:  # Minimum deaths to create zone
                center = np.mean(positions, axis=0)
                zone = GraveyardZone(
                    center=center,
                    radius=grid_size,
                    death_count=len(positions),
                    danger_level=len(positions) / max_deaths,
                )
                self.graveyard_zones.append(zone)
    
    def get_ghost_positions_at(
        self,
        timestamp: float,
        include_herd: bool = True,
    ) -> Dict[str, Any]:
        """Get ghost positions at a timestamp for the current map seed only.
        
        Only returns ghosts matching the active_seed to prevent visual
        clipping on procedurally generated maps.
        """
        result = {
            "map_seed": self.active_seed,
            "top_10": [],
            "world_record": None,
            "herd": [],
            "alive_count": 0,
            "dead_count": 0,
        }
        
        if self.active_seed is None:
            return result
        
        # World record
        if self.world_record:
            pos = self.world_record.get_position_at(timestamp)
            if pos is not None:
                result["world_record"] = {
                    "run_id": self.world_record.run_id,
                    "position": pos.tolist(),
                    "name": self.world_record.agent_name,
                }
        
        # Top 10
        for ghost in self.top_10:
            pos = ghost.get_position_at(timestamp)
            status = "alive" if pos is not None else "dead"
            result["top_10"].append({
                "run_id": ghost.run_id,
                "position": pos.tolist() if pos is not None else None,
                "name": ghost.agent_name,
                "status": status,
            })
            if pos is not None:
                result["alive_count"] += 1
            else:
                result["dead_count"] += 1
        
        # Herd (remaining ghosts for this seed only)
        if include_herd:
            seed_ghosts = self.ghosts_by_seed.get(self.active_seed, [])
            for ghost in seed_ghosts:
                if ghost.is_top_10:
                    continue
                pos = ghost.get_position_at(timestamp)
                if pos is not None:
                    result["herd"].append(pos.tolist())
                    result["alive_count"] += 1
                else:
                    result["dead_count"] += 1
        
        return result
    
    def get_seed_stats(self) -> Dict[str, Any]:
        """Get statistics about ghosts per seed."""
        return {
            "active_seed": self.active_seed,
            "total_seeds": len(self.ghosts_by_seed),
            "ghosts_per_seed": {
                seed: len(ghosts)
                for seed, ghosts in self.ghosts_by_seed.items()
            },
            "active_seed_ghosts": len(self.ghosts_by_seed.get(self.active_seed, [])),
            "total_ghosts": len(self.ghosts),
        }
    
    def get_survival_probability(
        self,
        position: np.ndarray,
        timestamp: float,
    ) -> float:
        """Calculate survival probability based on graveyard zones for current seed."""
        # Check if in any graveyard zone (already seed-filtered)
        for zone in self.graveyard_zones:
            dist = np.linalg.norm(position - zone.center)
            if dist < zone.radius:
                # In a dangerous zone
                return 1.0 - zone.danger_level
        
        # Count how many ghosts survived to this point (for current seed only)
        seed_ghosts = self.ghosts_by_seed.get(self.active_seed, [])
        alive = sum(1 for g in seed_ghosts if g.survival_time >= timestamp)
        total = len(seed_ghosts)
        
        if total == 0:
            return 1.0
        
        return alive / total


class KineticLeaderboard:
    """Real-time racing bar chart leaderboard."""
    
    def __init__(self, max_entries: int = 10):
        self.max_entries = max_entries
        
        # Entries
        self.entries: Dict[int, LeaderboardEntry] = {}
        self.sorted_entries: List[LeaderboardEntry] = []
        
        # Current agent
        self.current_run_id: Optional[int] = None
        self.current_score: float = 0.0
    
    def set_entries(self, runs: List[Dict[str, Any]]):
        """Set leaderboard from historical runs."""
        self.entries = {}
        
        for run in runs:
            entry = LeaderboardEntry(
                run_id=run["run_id"],
                agent_name=run["agent_name"],
                score=run["score"],
                survival_time=run["survival_time"],
            )
            self.entries[run["run_id"]] = entry
        
        self._sort()
    
    def update_current(
        self,
        run_id: int,
        agent_name: str,
        score: float,
        survival_time: float,
    ):
        """Update current run's position."""
        self.current_run_id = run_id
        self.current_score = score
        
        # Create or update current entry
        if run_id not in self.entries:
            entry = LeaderboardEntry(
                run_id=run_id,
                agent_name=agent_name,
                score=score,
                survival_time=survival_time,
                is_current=True,
            )
            self.entries[run_id] = entry
        else:
            entry = self.entries[run_id]
            entry.score = score
            entry.survival_time = survival_time
            entry.is_current = True
        
        # Update positions
        old_positions = {e.run_id: i for i, e in enumerate(self.sorted_entries)}
        self._sort()
        
        # Set animation targets
        for i, entry in enumerate(self.sorted_entries):
            entry.current_position = old_positions.get(entry.run_id, i)
            entry.target_position = i
            if entry.current_position != entry.target_position:
                entry.animation_progress = 0.0
    
    def _sort(self):
        """Sort entries by score."""
        self.sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.score,
            reverse=True,
        )[:self.max_entries]
    
    def tick(self, dt: float):
        """Animate positions."""
        for entry in self.sorted_entries:
            if entry.animation_progress < 1.0:
                entry.animation_progress = min(1.0, entry.animation_progress + dt * 3)
    
    def get_visual_positions(self) -> List[Dict[str, Any]]:
        """Get entries with interpolated visual positions."""
        result = []
        
        for entry in self.sorted_entries:
            # Interpolate position
            t = self._ease_out_cubic(entry.animation_progress)
            visual_pos = entry.current_position + t * (entry.target_position - entry.current_position)
            
            result.append({
                "run_id": entry.run_id,
                "agent_name": entry.agent_name,
                "score": entry.score,
                "visual_position": visual_pos,
                "is_current": entry.is_current,
                "is_overtaking": entry.current_position > entry.target_position,
            })
        
        return result
    
    def _ease_out_cubic(self, t: float) -> float:
        """Easing function for smooth animation."""
        return 1 - (1 - t) ** 3
    
    def get_pace_differential(self) -> Dict[str, Any]:
        """Get pace differential vs #1."""
        if not self.sorted_entries:
            return {"differential": 0, "ahead": False}
        
        top = self.sorted_entries[0]
        current = self.entries.get(self.current_run_id)
        
        if current is None:
            return {"differential": 0, "ahead": False}
        
        diff = current.score - top.score
        
        return {
            "differential": diff,
            "ahead": diff > 0,
            "vs_name": top.agent_name,
            "color": "#00FF88" if diff >= 0 else "#FF4444",
        }


class ChokeMeter:
    """Survival probability visualization."""
    
    def __init__(self, ghost_ecosystem: GhostEcosystem):
        self.ghosts = ghost_ecosystem
        
        # Current state
        self.probability: float = 1.0
        self.in_red_zone: bool = False
        self.clutch_triggered: bool = False
        
        # Thresholds
        self.red_zone_threshold: float = 0.10  # 10%
        self.clutch_threshold: float = 0.01   # 1%
        
        # History
        self.probability_history: List[float] = []
    
    def update(self, position: np.ndarray, timestamp: float):
        """Update survival probability."""
        self.probability = self.ghosts.get_survival_probability(position, timestamp)
        
        self.probability_history.append(self.probability)
        if len(self.probability_history) > 300:  # 5 seconds at 60 Hz
            self.probability_history.pop(0)
        
        # Check zones
        was_in_red = self.in_red_zone
        self.in_red_zone = self.probability < self.red_zone_threshold
        
        # Clutch detection
        if was_in_red and self.probability < self.clutch_threshold:
            self.clutch_triggered = True
    
    def survived_clutch(self) -> bool:
        """Check if agent survived a clutch moment."""
        if self.clutch_triggered and self.probability > self.red_zone_threshold:
            self.clutch_triggered = False
            return True
        return False
    
    def get_visual_data(self) -> Dict[str, Any]:
        """Get data for rendering the choke meter."""
        return {
            "probability": self.probability,
            "in_red_zone": self.in_red_zone,
            "clutch_active": self.clutch_triggered,
            "zone": "DANGER" if self.in_red_zone else "SAFE",
            "color": self._get_color(),
            "history": self.probability_history[-60:],  # Last second
        }
    
    def _get_color(self) -> str:
        """Get color based on probability."""
        if self.probability < 0.01:
            return "#FF0000"  # Bright red
        elif self.probability < 0.10:
            return "#FF4444"  # Red
        elif self.probability < 0.30:
            return "#FF8800"  # Orange
        elif self.probability < 0.50:
            return "#FFFF00"  # Yellow
        else:
            return "#00FF88"  # Green
