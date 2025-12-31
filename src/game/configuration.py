"""
Game Configuration Module.

Ensures game instances start with consistent, optimal settings:
- Lowest graphics quality
- Maximum FOV and camera distance
- Minimal latency (vsync off, 60 FPS cap)

This is applied out-of-process (editing the game's settings JSON) so all workers
share the same baseline configuration.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GameConfiguration:
    """
    Manages game configuration for MetaBonk instances.

    Default target is the Megabonk Steam appid (3405340). The settings path can be
    overridden with `METABONK_GAME_CONFIG_PATH`.
    """

    DEFAULT_APPID = 3405340

    # Common Steam userdata roots on Linux.
    STEAM_USERDATA_ROOTS: List[Path] = [
        Path(os.environ.get("MEGABONK_STEAM_USERDATA", "") or "").expanduser(),
        Path("~/.local/share/Steam/userdata").expanduser(),
        Path("~/.steam/steam/userdata").expanduser(),
        Path("~/.steam/root/userdata").expanduser(),
    ]

    # Megabonk stores its settings under the Proton prefix in a nested JSON structure.
    # This is the path that actually affects worker instances launched via MEGABONK_CMD_TEMPLATE.
    PROTON_PREFIX_CONFIG_REL = (
        Path("pfx")
        / "drive_c"
        / "users"
        / "steamuser"
        / "AppData"
        / "LocalLow"
        / "Ved"
        / "Megabonk"
        / "Saves"
        / "LocalDir"
        / "config.json"
    )

    # Best-effort "optimal" settings for the real on-disk config.json (per Proton prefix).
    # Types should match what the game writes (ints/floats).
    OPTIMAL_PROTON_SETTINGS: Dict[str, Dict[str, Any]] = {
        "cfVideoSettings": {
            "vsync": 0,
            "fps_limit": 60.0,
            # Quality toggles (keep anti_aliasing as-is; it's already 0 on fresh installs)
            "grass_quality": 0,
            "shadow_quality": 0,
            "shadow_resolution": 0,
            "soft_particles": 0,
            "bloom": 0,
            "motion_blur": 0,
            "ambient_occlusion": 0,
        },
        # Reduce motion and latency for vision-based agents.
        "cfGameSettings": {
            "camera_shake": 0,
            "input_delay": 0.0,
        },
    }

    # Gamescope defaults used for launches (separate from the game config format).
    OPTIMAL_GAMESCOPE: Dict[str, Any] = {
        "resolution_width": 1280,
        "resolution_height": 720,
        "fps_limit": 60,
    }

    @staticmethod
    def _repo_root() -> Path:
        # configuration.py lives at <repo>/src/game/configuration.py
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _compatdata_root(cls) -> Path:
        return cls._repo_root() / "temp" / "compatdata"

    @classmethod
    def _instance_compatdata_root(cls, worker_id: str) -> Path:
        return cls._compatdata_root() / str(worker_id)

    @classmethod
    def _default_proton_config_path(cls, worker_id: str) -> Path:
        return cls._instance_compatdata_root(worker_id) / cls.PROTON_PREFIX_CONFIG_REL

    @classmethod
    def _find_existing_proton_config_path(cls, worker_id: str) -> Optional[Path]:
        """Find an existing Proton prefix config.json for this worker id."""
        inst_root = cls._instance_compatdata_root(worker_id)
        candidate = cls._default_proton_config_path(worker_id)
        try:
            if candidate.exists():
                return candidate
        except Exception:
            pass
        users_dir = inst_root / "pfx" / "drive_c" / "users"
        try:
            if not users_dir.exists():
                return None
        except Exception:
            return None
        # Fallback: search for any user profile that contains the Megabonk config.
        try:
            for p in users_dir.glob("*/AppData/LocalLow/**/Megabonk/Saves/LocalDir/config.json"):
                if p.is_file():
                    return p
        except Exception:
            return None
        return None

    @classmethod
    def _load_template_proton_config(cls) -> Optional[Dict[str, Any]]:
        """Load a template config.json from any existing worker prefix (best-effort)."""
        root = cls._compatdata_root()
        try:
            if not root.exists():
                return None
        except Exception:
            return None
        try:
            for p in root.glob("*/pfx/drive_c/users/*/AppData/LocalLow/**/Megabonk/Saves/LocalDir/config.json"):
                try:
                    if not p.is_file():
                        continue
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "cfVideoSettings" in data:
                        return data
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def __init__(self, worker_id: str, *, appid: int | None = None) -> None:
        self.worker_id = str(worker_id)
        self.appid = int(appid) if appid is not None else int(self.DEFAULT_APPID)

        self.config_path: Optional[Path] = None
        override = str(os.environ.get("METABONK_GAME_CONFIG_PATH", "") or "").strip()
        if override:
            self.config_path = Path(override).expanduser()
            return

        # Prefer the per-worker Proton prefix (used by the actual worker game instances).
        proton_existing = self._find_existing_proton_config_path(self.worker_id)
        if proton_existing is not None:
            self.config_path = proton_existing
            return
        proton_default = self._default_proton_config_path(self.worker_id)
        try:
            if proton_default.parent.exists() or proton_default.parents[5].exists():  # temp/compatdata/<id>/pfx
                self.config_path = proton_default
                return
        except Exception:
            pass

        # Legacy: try Steam userdata layouts (may not affect per-instance Proton prefixes).
        steam_user_id = self._find_steam_user_id(self.appid)
        if steam_user_id is None:
            logger.warning("%s: could not locate Steam userdata directory", self.worker_id)
            return

        self.config_path = self._default_settings_path(steam_user_id, self.appid)

    @classmethod
    def _find_steam_user_id(cls, appid: int) -> Optional[int]:
        """Pick a Steam user id directory from userdata roots."""
        candidates: List[Path] = []
        for root in cls.STEAM_USERDATA_ROOTS:
            if not root or str(root) in (".", "/"):
                continue
            try:
                if root.exists():
                    candidates.append(root)
            except Exception:
                continue
        for root in candidates:
            try:
                ids = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
            except Exception:
                continue
            # Prefer a user that already has this appid directory.
            for p in ids:
                try:
                    if (p / str(appid)).exists():
                        return int(p.name)
                except Exception:
                    continue
            if ids:
                try:
                    return int(ids[0].name)
                except Exception:
                    continue
        return None

    @staticmethod
    def _default_settings_path(steam_user_id: int, appid: int) -> Path:
        """
        Resolve an on-disk settings.json path.

        Megabonk can store config either under:
          - userdata/<steamid>/<appid>/remote/settings.json (common Steam Cloud layout), or
          - userdata/<steamid>/remote/<appid>/config/settings.json (PRD layout).

        If nothing exists yet, we pick a deterministic path and create parent dirs on write.
        """
        roots = [
            p
            for p in GameConfiguration.STEAM_USERDATA_ROOTS
            if p and str(p) not in (".", "/") and p.exists()
        ]
        userdata = roots[0] if roots else Path("~/.local/share/Steam/userdata").expanduser()
        base = userdata / str(int(steam_user_id))

        # Candidate layouts (first existing wins).
        candidates = [
            base / str(int(appid)) / "remote" / "settings.json",
            base / str(int(appid)) / "remote" / "config" / "settings.json",
            base / "remote" / str(int(appid)) / "config" / "settings.json",
            base / str(int(appid)) / "local" / "settings.json",
            base / str(int(appid)) / "local" / "config" / "settings.json",
        ]
        for c in candidates:
            try:
                if c.exists():
                    return c
            except Exception:
                continue
        return candidates[0]

    @staticmethod
    def _is_proton_config(data: Dict[str, Any]) -> bool:
        return isinstance(data.get("cfVideoSettings"), dict) or isinstance(data.get("cfGameSettings"), dict)

    def _build_fresh_proton_config(self) -> Dict[str, Any]:
        template = self._load_template_proton_config()
        if isinstance(template, dict):
            return dict(template)
        # Minimal skeleton; the game will usually overwrite/extend as needed.
        return {
            "settingsVersion": 1,
            "hasSelectedLanguage": True,
            "cfGameSettings": {},
            "cfVideoSettings": {},
            "cfAudioSettings": {},
            "cfControlSettings": {},
            "cfVisualsSettings": {},
            "otherSettings": {},
            "preferences": {},
        }

    def _apply_nested_settings(
        self,
        current: Dict[str, Any],
        updates: Dict[str, Dict[str, Any]],
        *,
        force: bool,
    ) -> Tuple[Dict[str, Any], bool]:
        updated: Dict[str, Any] = dict(current)
        changed = False
        for section, section_updates in updates.items():
            existing = updated.get(section)
            if not isinstance(existing, dict):
                existing = {}
            next_section = dict(existing)
            if force:
                for k, v in section_updates.items():
                    if next_section.get(k) != v:
                        next_section[k] = v
                        changed = True
            else:
                for k, v in section_updates.items():
                    if k not in next_section:
                        next_section[k] = v
                        changed = True
            updated[section] = next_section
        return updated, changed

    def apply_optimal_settings(self, *, force: bool = False) -> bool:
        """
        Apply optimal settings to game configuration.

        Args:
            force: Overwrite existing keys with OPTIMAL_SETTINGS.
        """
        if self.config_path is None:
            logger.error("%s: cannot apply settings - config path not found", self.worker_id)
            return False

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("%s: cannot create settings dir (%s): %s", self.worker_id, self.config_path.parent, e)
            return False

        current_settings: Dict[str, Any] = {}
        if self.config_path.exists():
            try:
                current_settings = json.loads(self.config_path.read_text(encoding="utf-8"))
                if not isinstance(current_settings, dict):
                    current_settings = {}
            except Exception:
                current_settings = {}
        else:
            # When the prefix exists but the game hasn't written config.json yet,
            # seed it from a template so we can reliably toggle perf-critical knobs.
            current_settings = self._build_fresh_proton_config()

        if self._is_proton_config(current_settings):
            updated, changed = self._apply_nested_settings(
                current_settings, self.OPTIMAL_PROTON_SETTINGS, force=force
            )
        else:
            # Legacy fallback: flat dict settings (PRD placeholder).
            updated = dict(current_settings)
            changed = False
            for section, section_updates in self.OPTIMAL_PROTON_SETTINGS.items():
                for k, v in section_updates.items():
                    flat_key = f"{section}.{k}"
                    if force:
                        if updated.get(flat_key) != v:
                            updated[flat_key] = v
                            changed = True
                    else:
                        if flat_key not in updated:
                            updated[flat_key] = v
                            changed = True

        if not changed and self.config_path.exists():
            logger.info("%s: settings already optimal (%s)", self.worker_id, self.config_path)
            return True

        tmp = self.config_path.with_suffix(self.config_path.suffix + f".tmp.{int(time.time())}")
        try:
            tmp.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(self.config_path)
        except Exception as e:
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            logger.error("%s: failed to write settings (%s): %s", self.worker_id, self.config_path, e)
            return False

        logger.info("%s: applied optimal game settings -> %s", self.worker_id, self.config_path)
        return True

    def validate_settings(self) -> Dict[str, bool]:
        """Validate that current settings match OPTIMAL_SETTINGS."""
        if self.config_path is None or not self.config_path.exists():
            return {}
        try:
            current = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(current, dict):
            return {}
        if self._is_proton_config(current):
            out: Dict[str, bool] = {}
            for section, section_updates in self.OPTIMAL_PROTON_SETTINGS.items():
                cur_section = current.get(section)
                if not isinstance(cur_section, dict):
                    cur_section = {}
                for k, v in section_updates.items():
                    out[f"{section}.{k}"] = cur_section.get(k) == v
            return out
        # Legacy fallback: flatten expected keys as "<section>.<key>".
        expected: Dict[str, Any] = {}
        for section, section_updates in self.OPTIMAL_PROTON_SETTINGS.items():
            for k, v in section_updates.items():
                expected[f"{section}.{k}"] = v
        return {k: (current.get(k) == v) for k, v in expected.items()}

    def get_gamescope_args(self) -> List[str]:
        """Recommended Gamescope args for resolution/FPS control."""
        w = int(self.OPTIMAL_GAMESCOPE.get("resolution_width", 1280))
        h = int(self.OPTIMAL_GAMESCOPE.get("resolution_height", 720))
        fps = int(self.OPTIMAL_GAMESCOPE.get("fps_limit", 60))
        return ["-w", str(w), "-h", str(h), "-r", str(fps), "--force-windows-fullscreen"]


def configure_all_workers(num_workers: int, *, instance_prefix: str = "omega", appid: int | None = None) -> None:
    """Configure N worker instances with optimal settings (best-effort)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        n = int(num_workers)
    except Exception:
        n = 0
    if n <= 0:
        return
    logger.info("Configuring %d workers with optimal game settings...", n)
    for i in range(n):
        worker_id = f"{instance_prefix}-{i}"
        cfg = GameConfiguration(worker_id, appid=appid)
        ok = cfg.apply_optimal_settings(force=True)
        if not ok:
            logger.warning("⚠️  %s: configuration failed", worker_id)
            continue
        mismatches = [k for k, v in (cfg.validate_settings() or {}).items() if not v]
        if mismatches:
            logger.warning("⚠️  %s: settings mismatched (%d keys)", worker_id, len(mismatches))
        else:
            logger.info("✅ %s: configuration applied", worker_id)
    logger.info("Configuration complete")


def _main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Apply optimal Megabonk settings for MetaBonk workers.")
    ap.add_argument("workers", nargs="?", default="12", help="Number of workers (default: 12)")
    ap.add_argument("--instance-prefix", default=os.environ.get("METABONK_INSTANCE_PREFIX", "omega"))
    ap.add_argument(
        "--appid",
        type=int,
        default=int(os.environ.get("MEGABONK_APPID", str(GameConfiguration.DEFAULT_APPID))),
    )
    args = ap.parse_args()

    configure_all_workers(int(args.workers), instance_prefix=str(args.instance_prefix), appid=int(args.appid))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
