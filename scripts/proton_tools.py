#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _steam_common_root() -> Path:
    custom = os.environ.get("MEGABONK_STEAM_COMMON") or os.environ.get("STEAM_COMMON_DIR") or ""
    if custom:
        return Path(custom).expanduser()
    # Default to the game dir's library if provided.
    game_dir = os.environ.get("MEGABONK_GAME_DIR") or ""
    if game_dir:
        gd = Path(game_dir).expanduser().resolve()
        return gd.parent.parent
    # Best-effort fallback: common Steam library path.
    return Path.home() / ".steam" / "steam" / "steamapps" / "common"


def _list_tools(common_root: Path) -> list[str]:
    if not common_root.exists():
        return []
    return sorted([p.name for p in common_root.iterdir() if p.is_dir() and "Proton" in p.name])


def _has_tool(common_root: Path, name: str) -> bool:
    p = common_root / name / "proton"
    return p.exists()


def _open_steam_install(appid: str) -> None:
    subprocess.check_call(["xdg-open", f"steam://install/{appid}"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Proton tools for MetaBonk triage.")
    parser.add_argument("--list", action="store_true", help="List detected Proton tools.")
    parser.add_argument("--install-experimental", action="store_true", help="Open Steam to install Proton Experimental.")
    parser.add_argument("--install-protonup-qt", action="store_true", help="Install ProtonUp-Qt via Flatpak (Flathub).")
    parser.add_argument("--common-root", default="", help="Override Steam common directory.")
    args = parser.parse_args()

    common_root = Path(args.common_root).expanduser() if args.common_root else _steam_common_root()
    if args.list:
        tools = _list_tools(common_root)
        if not tools:
            print(f"[proton_tools] no Proton tools detected in {common_root}")
        else:
            print(f"[proton_tools] tools in {common_root}:")
            for t in tools:
                print(f"  - {t}")
        return 0

    if args.install_experimental:
        if _has_tool(common_root, "Proton Experimental"):
            print("[proton_tools] Proton Experimental already installed.")
            return 0
        print("[proton_tools] Opening Steam install for Proton Experimental...")
        _open_steam_install("1493710")
        return 0

    if args.install_protonup_qt:
        print("[proton_tools] Installing ProtonUp-Qt via Flatpak...")
        subprocess.check_call(["flatpak", "install", "-y", "flathub", "net.davidotek.pupgui2"])
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
