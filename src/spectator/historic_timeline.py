"""Historic Timeline: The "World Line" of Metabonk.

Tracks every run from inception, providing temporal context:
- Run segments with color-coded results
- System-determined milestones (firsts, records)
- Viewer-set milestones (community pins)
- Era markers for system updates

References:
- Summoning Salt progression graphs
- The "Road" metaphor for continuous timeline
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class RunResult(Enum):
    """Result of a run."""
    IN_PROGRESS = "in_progress"
    DEATH = "death"
    VICTORY = "victory"
    CRASH = "crash"
    TIMEOUT = "timeout"


class MilestoneType(Enum):
    """Type of milestone on the timeline."""
    # System milestones
    FIRST_STAGE = "first_stage"
    FIRST_BOSS_KILL = "first_boss_kill"
    NEW_RECORD = "new_record"
    LEGENDARY_DROP = "legendary_drop"
    OVERCRIT_FIRST = "overcrit_first"
    MAX_DAMAGE = "max_damage"
    
    # Community milestones
    COMMUNITY_PIN = "community_pin"
    CLUTCH_MOMENT = "clutch_moment"
    NAMED_AGENT = "named_agent"
    BOUNTY_HIT = "bounty_hit"


@dataclass
class Milestone:
    """A pinned moment on the timeline."""
    
    milestone_id: str
    milestone_type: MilestoneType
    
    # Timing
    run_id: int
    timestamp_in_run: float  # Seconds into the run
    absolute_timestamp: float  # Unix timestamp
    
    # Content
    title: str
    description: str
    
    # Visual
    icon: str = "📌"
    color: str = "#FFD700"  # Gold
    
    # Attribution
    pinned_by: str = "SYSTEM"  # "SYSTEM" or viewer username
    
    # Data
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSegment:
    """A single run on the timeline."""
    
    run_id: int
    
    # Timing
    start_time: float  # Unix timestamp
    end_time: Optional[float] = None
    duration: float = 0.0
    
    # Result
    result: RunResult = RunResult.IN_PROGRESS
    
    # Stats
    final_score: float = 0.0
    max_stage: int = 1
    max_survival_time: float = 0.0
    max_damage: float = 0.0
    
    # Identity
    agent_name: Optional[str] = None  # Viewer-assigned name
    sponsor: Optional[str] = None  # Viewer who sponsored
    
    # Milestones within this run
    milestones: List[Milestone] = field(default_factory=list)
    
    # Visual
    @property
    def color(self) -> str:
        """Get color based on result."""
        colors = {
            RunResult.IN_PROGRESS: "#00FF88",
            RunResult.DEATH: "#FF4444",
            RunResult.VICTORY: "#FFD700",
            RunResult.CRASH: "#888888",
            RunResult.TIMEOUT: "#AA88FF",
        }
        return colors.get(self.result, "#FFFFFF")


@dataclass
class Era:
    """A period of the timeline (season/patch)."""
    
    era_id: str
    name: str  # e.g., "The Great Nerf", "Season 2"
    
    start_run: int
    end_run: Optional[int] = None
    
    description: str = ""
    
    # Stats
    total_runs: int = 0
    best_score: float = 0.0
    avg_survival: float = 0.0


@dataclass
class TimelineConfig:
    """Configuration for the historic timeline."""
    
    # Visual
    visible_runs: int = 50  # Runs visible at once
    compression_threshold: int = 100  # After this, compress old runs
    
    # Milestones
    auto_pin_threshold: float = 0.01  # Top 1% events
    
    # Community
    min_votes_for_pin: int = 5
    vote_duration_s: float = 30.0
    
    # Persistence
    save_interval_s: float = 60.0


class HistoricTimeline:
    """The World Line - complete history of all runs."""
    
    def __init__(self, cfg: Optional[TimelineConfig] = None):
        self.cfg = cfg or TimelineConfig()
        
        # Core data
        self.runs: List[RunSegment] = []
        self.milestones: List[Milestone] = []
        self.eras: List[Era] = []
        
        # Current state
        self.current_run: Optional[RunSegment] = None
        self.world_record_time: float = 0.0
        self.world_record_run: int = -1
        
        # Records for auto-pinning
        self.records = {
            "max_score": 0.0,
            "max_stage": 0,
            "max_damage": 0.0,
            "max_survival": 0.0,
            "first_stage_2": None,
            "first_boss_kill": None,
            "first_overcrit": None,
        }
        
        # Community
        self.pending_votes: List[Dict] = []
        self.named_agents: Dict[int, str] = {}
    
    # ═══════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════
    
    def start_run(self, agent_name: Optional[str] = None) -> RunSegment:
        """Start a new run."""
        run_id = len(self.runs)
        
        run = RunSegment(
            run_id=run_id,
            start_time=time.time(),
            agent_name=agent_name or f"Agent_{run_id:04d}",
        )
        
        self.runs.append(run)
        self.current_run = run
        
        return run
    
    def end_run(
        self,
        result: RunResult,
        final_score: float,
        max_stage: int,
        survival_time: float,
        max_damage: float,
    ):
        """End the current run."""
        if self.current_run is None:
            return
        
        run = self.current_run
        run.end_time = time.time()
        run.duration = run.end_time - run.start_time
        run.result = result
        run.final_score = final_score
        run.max_stage = max_stage
        run.max_survival_time = survival_time
        run.max_damage = max_damage
        
        # Check for records
        self._check_records(run)
        
        # Update world record
        if survival_time > self.world_record_time:
            self.world_record_time = survival_time
            self.world_record_run = run.run_id
            
            self.add_milestone(
                run.run_id,
                survival_time,
                MilestoneType.NEW_RECORD,
                "🏆 NEW WORLD RECORD",
                f"Survived {survival_time:.1f}s",
                color="#FFD700",
            )
        
        self.current_run = None
    
    def _check_records(self, run: RunSegment):
        """Check if run set any records."""
        # First Stage 2
        if run.max_stage >= 2 and self.records["first_stage_2"] is None:
            self.records["first_stage_2"] = run.run_id
            self.add_milestone(
                run.run_id, run.max_survival_time,
                MilestoneType.FIRST_STAGE,
                "🏛️ FIRST STAGE 2",
                "The Library unlocked!",
            )
        
        # Max damage
        if run.max_damage > self.records["max_damage"]:
            self.records["max_damage"] = run.max_damage
            self.add_milestone(
                run.run_id, run.max_survival_time,
                MilestoneType.MAX_DAMAGE,
                f"💥 {run.max_damage:.0f} DAMAGE",
                "New damage record!",
                color="#FF4444",
            )
    
    # ═══════════════════════════════════════════════════════════
    # MILESTONES
    # ═══════════════════════════════════════════════════════════
    
    def add_milestone(
        self,
        run_id: int,
        timestamp_in_run: float,
        milestone_type: MilestoneType,
        title: str,
        description: str,
        icon: str = "📌",
        color: str = "#FFD700",
        pinned_by: str = "SYSTEM",
        data: Optional[Dict] = None,
    ) -> Milestone:
        """Add a milestone to the timeline."""
        milestone_id = hashlib.md5(
            f"{run_id}:{timestamp_in_run}:{title}".encode()
        ).hexdigest()[:8]
        
        milestone = Milestone(
            milestone_id=milestone_id,
            milestone_type=milestone_type,
            run_id=run_id,
            timestamp_in_run=timestamp_in_run,
            absolute_timestamp=time.time(),
            title=title,
            description=description,
            icon=icon,
            color=color,
            pinned_by=pinned_by,
            data=data or {},
        )
        
        self.milestones.append(milestone)
        
        # Also add to run
        if 0 <= run_id < len(self.runs):
            self.runs[run_id].milestones.append(milestone)
        
        return milestone
    
    def start_community_vote(
        self,
        run_id: int,
        timestamp: float,
        proposed_title: str,
    ):
        """Start a community vote for a pin."""
        vote = {
            "run_id": run_id,
            "timestamp": timestamp,
            "title": proposed_title,
            "votes_yes": 0,
            "votes_no": 0,
            "started": time.time(),
            "voters": set(),
        }
        self.pending_votes.append(vote)
    
    def cast_vote(self, voter: str, vote_yes: bool) -> bool:
        """Cast a vote on the current pending vote."""
        if not self.pending_votes:
            return False
        
        vote = self.pending_votes[-1]
        
        if voter in vote["voters"]:
            return False  # Already voted
        
        vote["voters"].add(voter)
        if vote_yes:
            vote["votes_yes"] += 1
        else:
            vote["votes_no"] += 1
        
        # Check if vote passes
        if vote["votes_yes"] >= self.cfg.min_votes_for_pin:
            self.add_milestone(
                vote["run_id"],
                vote["timestamp"],
                MilestoneType.COMMUNITY_PIN,
                vote["title"],
                f"Pinned by chat ({vote['votes_yes']} votes)",
                icon="📍",
                color="#00FFFF",
                pinned_by="COMMUNITY",
            )
            self.pending_votes.pop()
            return True
        
        return False
    
    def name_agent(self, run_id: int, name: str, sponsor: str):
        """Assign a name to an agent (viewer sponsorship)."""
        if 0 <= run_id < len(self.runs):
            self.runs[run_id].agent_name = name
            self.runs[run_id].sponsor = sponsor
            self.named_agents[run_id] = name
            
            self.add_milestone(
                run_id, 0,
                MilestoneType.NAMED_AGENT,
                f"👤 {name}",
                f"Sponsored by {sponsor}",
                pinned_by=sponsor,
            )
    
    # ═══════════════════════════════════════════════════════════
    # ERAS
    # ═══════════════════════════════════════════════════════════
    
    def start_era(self, name: str, description: str = ""):
        """Start a new era (patch/season)."""
        if self.eras:
            self.eras[-1].end_run = len(self.runs) - 1
        
        era = Era(
            era_id=f"era_{len(self.eras)}",
            name=name,
            start_run=len(self.runs),
            description=description,
        )
        self.eras.append(era)
    
    # ═══════════════════════════════════════════════════════════
    # VISUALIZATION DATA
    # ═══════════════════════════════════════════════════════════
    
    def get_visible_runs(self) -> List[RunSegment]:
        """Get runs visible in the current view."""
        return self.runs[-self.cfg.visible_runs:]
    
    def get_timeline_data(self) -> Dict[str, Any]:
        """Get data for rendering the timeline UI."""
        visible = self.get_visible_runs()
        
        return {
            "total_runs": len(self.runs),
            "current_run": self.current_run.run_id if self.current_run else None,
            "world_record": {
                "run_id": self.world_record_run,
                "time": self.world_record_time,
            },
            "visible_runs": [
                {
                    "run_id": r.run_id,
                    "duration": r.duration,
                    "result": r.result.value,
                    "color": r.color,
                    "agent_name": r.agent_name,
                    "milestones": len(r.milestones),
                }
                for r in visible
            ],
            "eras": [
                {
                    "name": e.name,
                    "start_run": e.start_run,
                    "end_run": e.end_run,
                }
                for e in self.eras
            ],
            "recent_milestones": [
                {
                    "id": m.milestone_id,
                    "type": m.milestone_type.value,
                    "title": m.title,
                    "icon": m.icon,
                    "color": m.color,
                }
                for m in self.milestones[-10:]
            ],
        }
    
    def get_pace_vs_record(self, current_time: float) -> Dict[str, Any]:
        """Get pace comparison vs world record."""
        if self.world_record_run < 0:
            return {"differential": 0, "on_pace": True}
        
        # In a real implementation, we'd have timestamped data
        # For now, simple linear comparison
        differential = current_time - self.world_record_time
        
        return {
            "differential": differential,
            "on_pace": differential >= 0,
            "color": "#00FF88" if differential >= 0 else "#FF4444",
        }
    
    # ═══════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════
    
    def save(self, path: Path):
        """Save timeline to disk."""
        data = {
            "runs": [
                {
                    "run_id": r.run_id,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "duration": r.duration,
                    "result": r.result.value,
                    "final_score": r.final_score,
                    "max_stage": r.max_stage,
                    "max_survival_time": r.max_survival_time,
                    "max_damage": r.max_damage,
                    "agent_name": r.agent_name,
                    "sponsor": r.sponsor,
                }
                for r in self.runs
            ],
            "milestones": [
                {
                    "milestone_id": m.milestone_id,
                    "milestone_type": m.milestone_type.value,
                    "run_id": m.run_id,
                    "timestamp_in_run": m.timestamp_in_run,
                    "title": m.title,
                    "description": m.description,
                    "pinned_by": m.pinned_by,
                }
                for m in self.milestones
            ],
            "records": self.records,
            "world_record_time": self.world_record_time,
            "world_record_run": self.world_record_run,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path):
        """Load timeline from disk."""
        with open(path) as f:
            data = json.load(f)
        
        self.runs = [
            RunSegment(
                run_id=r["run_id"],
                start_time=r["start_time"],
                end_time=r.get("end_time"),
                duration=r.get("duration", 0),
                result=RunResult(r["result"]),
                final_score=r.get("final_score", 0),
                max_stage=r.get("max_stage", 1),
                max_survival_time=r.get("max_survival_time", 0),
                max_damage=r.get("max_damage", 0),
                agent_name=r.get("agent_name"),
                sponsor=r.get("sponsor"),
            )
            for r in data["runs"]
        ]
        
        self.world_record_time = data.get("world_record_time", 0)
        self.world_record_run = data.get("world_record_run", -1)
        self.records = data.get("records", self.records)
