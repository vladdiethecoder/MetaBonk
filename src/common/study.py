"""Study (A/B test) specifications and helpers.

This is a lightweight, repo-native alternative to Hydra for local experiment
management. A study is a set of variants executed in isolated runs, each with:
  - unique METABONK_RUN_ID (run directory + metrics key)
  - optional env/CLI overrides
  - optional certification requirements
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .schemas import MBBaseModel


_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def slugify(value: str) -> str:
    v = str(value or "").strip()
    v = _SLUG_RE.sub("-", v).strip("-").lower()
    return v or "variant"


class StudyVariant(MBBaseModel):
    name: str
    description: Optional[str] = None
    # Env var overrides for this variant.
    env: Dict[str, str] = {}
    # Additional CLI args appended to `./start ...` for this variant.
    extra_args: List[str] = []
    # How long to keep the stack running for measurement (seconds).
    duration_s: float = 60.0
    # How long to wait for orchestrator + workers to become ready (seconds).
    ready_timeout_s: float = 120.0
    # Certification requirements (best-effort; enforced post-run).
    require_fps: bool = True
    require_vision: bool = True
    require_input_audit: bool = True

    @property
    def slug(self) -> str:
        return slugify(self.name)


class StudySpec(MBBaseModel):
    study_id: str
    title: str = "MetaBonk Study"
    experiment_id: str = "exp-omega"
    notes: Optional[str] = None
    base_env: Dict[str, str] = {}
    base_args: List[str] = []
    variants: List[StudyVariant]

    def validate_unique_variants(self) -> None:
        seen = set()
        for v in self.variants:
            s = v.slug
            if s in seen:
                raise ValueError(f"Duplicate variant slug: {s!r} (names must be unique after slugify)")
            seen.add(s)


def load_study(path: Path) -> StudySpec:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    raw: Any = yaml.safe_load(p.read_text())  # type: ignore[no-untyped-call]
    if not isinstance(raw, dict):
        raise ValueError("study YAML must be a mapping")
    spec = StudySpec(**raw)
    spec.validate_unique_variants()
    return spec


def write_study_template(path: Path) -> None:
    """Write a minimal example study file."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmpl = {
        "study_id": "gold-smoke",
        "title": "Gold Smoke A/B",
        "experiment_id": "exp-omega",
        "base_env": {
            "METABONK_SYNTHETIC_EYE": "1",
            "METABONK_SYNTHETIC_EYE_PASSTHROUGH": "0",
            "METABONK_VISION_AUDIT": "1",
            "METABONK_INPUT_AUDIT": "1",
        },
        "base_args": ["--mode", "train", "--workers", "1", "--no-ui", "--no-go2rtc"],
        "variants": [
            {
                "name": "baseline",
                "description": "Default gold smoke settings",
                "env": {},
                "extra_args": [],
                "duration_s": 45,
            }
        ],
    }
    p.write_text(yaml.safe_dump(tmpl, sort_keys=False))  # type: ignore[no-untyped-call]

