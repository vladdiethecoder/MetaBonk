"""Community lore: pins, naming rights, and bounties.

Designed for stream overlays:
- Stores viewer "pins" on the timeline (funny glitches, clutch moments, etc.).
- Supports "agent adoption" via paid naming (Bonk Bucks).
- Supports bounties that pay out when any agent crosses a milestone.

Persistence is best-effort JSON (so restarts don't wipe community history).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> float:
    return time.time()


def _clamp_len(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


@dataclass
class Pin:
    pin_id: str
    ts: float
    instance_id: Optional[str] = None
    run_id: Optional[str] = None
    episode_idx: Optional[int] = None
    episode_t: Optional[float] = None
    title: str = ""
    note: str = ""
    created_by: Optional[str] = None
    created_by_id: Optional[str] = None
    kind: str = "community"  # community|system|admin
    # UI metadata (purely cosmetic; ok to omit).
    icon: Optional[str] = None
    color: Optional[str] = None
    glow: bool = False
    tag: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bounty:
    bounty_id: str
    created_ts: float
    title: str
    # Condition type:
    # - "survive_s": trigger when episode_t >= threshold_s
    # - "score": trigger when score >= threshold_score
    kind: str
    threshold: float
    # Pot economics
    pot_total: int = 0
    contributors: Dict[str, int] = field(default_factory=dict)  # user_id -> amount
    # Resolution
    active: bool = True
    claimed_ts: Optional[float] = None
    claimed_by_instance: Optional[str] = None
    claimed_run_id: Optional[str] = None
    claimed_episode_idx: Optional[int] = None
    payouts: List[Dict[str, Any]] = field(default_factory=list)  # {user_id, amount}


@dataclass
class LoreStore:
    path: Path
    pins: List[Pin] = field(default_factory=list)
    bounties: List[Bounty] = field(default_factory=list)
    # Paid/adopted names
    agent_names: Dict[str, str] = field(default_factory=dict)  # instance_id -> display_name
    run_names: Dict[str, str] = field(default_factory=dict)  # run_id -> display_name

    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text())
            self.pins = [Pin(**p) for p in (data.get("pins") or []) if isinstance(p, dict)]
            self.bounties = [Bounty(**b) for b in (data.get("bounties") or []) if isinstance(b, dict)]
            self.agent_names = {str(k): str(v) for k, v in (data.get("agent_names") or {}).items()}
            self.run_names = {str(k): str(v) for k, v in (data.get("run_names") or {}).items()}
        except Exception:
            return

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ts": _now(),
                "pins": [asdict(p) for p in self.pins[-2000:]],
                "bounties": [asdict(b) for b in self.bounties[-500:]],
                "agent_names": dict(self.agent_names),
                "run_names": dict(self.run_names),
            }
            self.path.write_text(json.dumps(payload, indent=2))
        except Exception:
            return

    def add_pin(
        self,
        *,
        ts: float,
        title: str,
        note: str = "",
        instance_id: Optional[str] = None,
        run_id: Optional[str] = None,
        episode_idx: Optional[int] = None,
        episode_t: Optional[float] = None,
        created_by: Optional[str] = None,
        created_by_id: Optional[str] = None,
        kind: str = "community",
        icon: Optional[str] = None,
        color: Optional[str] = None,
        glow: bool = False,
        tag: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Pin:
        pid = f"pin-{int(ts*1000)}-{len(self.pins)+1}"
        p = Pin(
            pin_id=pid,
            ts=float(ts),
            instance_id=instance_id,
            run_id=run_id,
            episode_idx=episode_idx,
            episode_t=episode_t,
            title=_clamp_len(title, 80),
            note=_clamp_len(note, 200),
            created_by=_clamp_len(created_by or "", 40) or None,
            created_by_id=_clamp_len(created_by_id or "", 80) or None,
            kind=str(kind or "community"),
            icon=_clamp_len(icon or "", 32) or None,
            color=_clamp_len(color or "", 24) or None,
            glow=bool(glow),
            tag=_clamp_len(tag or "", 48) or None,
            data=dict(data or {}),
        )
        self.pins.append(p)
        return p

    def create_bounty(
        self,
        *,
        title: str,
        kind: str,
        threshold: float,
        created_by_id: Optional[str] = None,
    ) -> Bounty:
        now = _now()
        bid = f"bty-{int(now*1000)}-{len(self.bounties)+1}"
        b = Bounty(
            bounty_id=bid,
            created_ts=float(now),
            title=_clamp_len(title, 80),
            kind=str(kind),
            threshold=float(threshold),
        )
        # For attribution we can optionally add as a synthetic "contributor" with 0 (not used in payouts).
        if created_by_id:
            b.contributors.setdefault(str(created_by_id), 0)
        self.bounties.append(b)
        return b

    def set_agent_name(self, instance_id: str, display_name: str) -> str:
        nm = _clamp_len(display_name, 28)
        self.agent_names[str(instance_id)] = nm
        return nm

    def set_run_name(self, run_id: str, display_name: str) -> str:
        nm = _clamp_len(display_name, 32)
        self.run_names[str(run_id)] = nm
        return nm


def default_lore_store() -> LoreStore:
    path = Path(os.environ.get("METABONK_LORE_DB", "temp/lore_store.json"))
    st = LoreStore(path=path)
    st.load()
    return st
