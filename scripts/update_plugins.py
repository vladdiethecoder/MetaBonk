#!/usr/bin/env python3
"""Build + install BepInEx IL2CPP plugins into your Megabonk install.

This is a convenience wrapper for:
  - Building the plugins in this repo against your *real* UnityEngine modules
  - Copying the resulting DLLs into `BepInEx/plugins/`

Usage:
  python scripts/update_plugins.py --game-dir "/path/to/steamapps/common/Megabonk"

Notes:
  - Requires `dotnet` on PATH.
  - You can also set `MEGABONK_GAME_DIR` instead of `--game-dir`.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from tempfile import TemporaryDirectory
from pathlib import Path


def _find_game_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if (p / "Megabonk.exe").exists():
            return p
        raise SystemExit(f"--game-dir missing Megabonk.exe: {p}")

    env = os.environ.get("MEGABONK_GAME_DIR") or ""
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "Megabonk.exe").exists():
            return p

    user = os.environ.get("USER") or ""
    candidates: list[Path] = []
    if user:
        candidates += [Path(p) for p in Path(f"/run/media/{user}").glob("*/SteamLibrary/steamapps/common/Megabonk")]
        candidates += [Path(p) for p in Path(f"/run/media/{user}").glob("*/steamapps/common/Megabonk")]
    candidates.append(Path("~/.local/share/Steam/steamapps/common/Megabonk").expanduser())

    for c in candidates:
        try:
            if (c / "Megabonk.exe").exists():
                return c.resolve()
        except Exception:
            continue
    raise SystemExit("Could not auto-detect Megabonk install; pass --game-dir or set MEGABONK_GAME_DIR.")


def _dotnet() -> str:
    d = shutil.which("dotnet")
    if not d:
        raise SystemExit("dotnet not found on PATH")
    return d


def _build(csproj: Path, *, bepinex_core: Path, unity_managed: Path) -> Path:
    dotnet = _dotnet()
    cmd = [
        dotnet,
        "build",
        str(csproj),
        "-c",
        "Release",
        f"/p:BepInExCoreDir={str(bepinex_core)}",
        f"/p:UnityManagedDir={str(unity_managed)}",
    ]
    subprocess.check_call(cmd)
    # Most projects here target net6.0.
    out_dir = csproj.parent / "bin" / "Release" / "net6.0"
    if not out_dir.exists():
        raise SystemExit(f"Build output not found: {out_dir}")
    dlls = list(out_dir.glob("*.dll"))
    if not dlls:
        raise SystemExit(f"No DLLs produced in: {out_dir}")
    # Prefer the project assembly name if present.
    preferred = out_dir / (csproj.stem + ".dll")
    if preferred.exists():
        return preferred
    return dlls[0]


def _install(dll_path: Path, *, plugins_dir: Path, folder: str) -> None:
    dst_dir = plugins_dir if not folder else (plugins_dir / folder)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (f"{dll_path.stem}.dll" if not folder else dll_path.name)
    shutil.copy2(dll_path, dst)
    print(f"[plugins] installed {dll_path.name} -> {dst}")


def _clean_bepinex_cache(game_dir: Path) -> None:
    for rel in ("BepInEx/interop", "BepInEx/cache", "BepInEx/dummy", "BepInEx/DumpedAssemblies"):
        target = game_dir / rel
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            print(f"[bepinex] removed {target}")


def _backup_bepinex(game_dir: Path) -> None:
    bep = game_dir / "BepInEx"
    if bep.exists():
        ts = int(time.time())
        dst = game_dir / f"BepInEx.backup.{ts}"
        shutil.move(str(bep), str(dst))
        print(f"[bepinex] backed up {bep} -> {dst}")
    for fname in ("winhttp.dll", "doorstop_config.ini"):
        src = game_dir / fname
        if src.exists():
            ts = int(time.time())
            dst = game_dir / f"{fname}.bak.{ts}"
            shutil.move(str(src), str(dst))
            print(f"[bepinex] backed up {src.name} -> {dst}")


def _download(url: str, dst: Path) -> None:
    print(f"[bepinex] downloading {url}")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    dst.write_bytes(data)


def _resolve_bepinex_url(build: str, arch: str) -> str:
    page_url = "https://builds.bepinex.dev/projects/bepinex_be"
    with urllib.request.urlopen(page_url) as resp:
        html = resp.read().decode("utf-8", "replace")
    links = re.findall(r'href=\"([^\"]+)\"', html)
    prefix = f"/projects/bepinex_be/{build}/"
    candidates = [
        link
        for link in links
        if link.startswith(prefix)
        and "BepInEx-Unity.IL2CPP" in link
        and f"-{arch}-" in link
        and link.endswith(".zip")
    ]
    if not candidates:
        raise SystemExit(f"Could not find BepInEx IL2CPP {arch} build {build} on {page_url}")
    return urllib.parse.urljoin(page_url, candidates[0])


def _extract_bepinex(zip_path: Path, game_dir: Path) -> None:
    with TemporaryDirectory() as td:
        tmp = Path(td)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        root = tmp
        children = [p for p in root.iterdir()]
        if len(children) == 1 and children[0].is_dir():
            cand = children[0]
            if (cand / "BepInEx").exists():
                root = cand
        if not (root / "BepInEx").exists():
            raise SystemExit(f"Downloaded archive does not contain BepInEx/: {zip_path}")
        for child in root.iterdir():
            dest = game_dir / child.name
            if child.is_dir():
                shutil.copytree(child, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(child, dest)
        print(f"[bepinex] installed from {zip_path} into {game_dir}")


def _install_bepinex(*, game_dir: Path, build: str, url: str | None, zip_path: str | None) -> None:
    if zip_path:
        zp = Path(zip_path).expanduser().resolve()
        if not zp.exists():
            raise SystemExit(f"BepInEx zip not found: {zp}")
        _backup_bepinex(game_dir)
        _extract_bepinex(zp, game_dir)
        return

    urls: list[str] = []
    if url:
        urls.append(url)
    if build:
        try:
            urls.append(_resolve_bepinex_url(build, "win-x64"))
        except Exception as exc:
            print(f"[bepinex] failed to resolve win-x64 build {build}: {exc}")
        try:
            urls.append(_resolve_bepinex_url(build, "win-x86"))
        except Exception as exc:
            print(f"[bepinex] failed to resolve win-x86 build {build}: {exc}")
    if not urls:
        raise SystemExit("No BepInEx source provided. Pass --bepinex-build, --bepinex-url, or --bepinex-zip.")

    with TemporaryDirectory() as td:
        tmp = Path(td)
        last_err: Exception | None = None
        for candidate in urls:
            try:
                target = tmp / Path(candidate).name
                _download(candidate, target)
                _backup_bepinex(game_dir)
                _extract_bepinex(target, game_dir)
                return
            except Exception as exc:
                last_err = exc
                print(f"[bepinex] failed to fetch {candidate}: {exc}")
        raise SystemExit(f"Failed to install BepInEx from candidates: {urls}\nlast error: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build + install Megabonk BepInEx plugins from this repo")
    ap.add_argument("--game-dir", default=os.environ.get("MEGABONK_GAME_DIR", ""))
    ap.add_argument("--no-install", action="store_true", help="Only build; do not copy into the game folder")
    ap.add_argument("--install-bepinex", action="store_true", help="Download/install a pinned BepInEx build before building plugins")
    ap.add_argument("--bepinex-build", default=os.environ.get("METABONK_BEPINEX_BUILD", ""))
    ap.add_argument("--bepinex-url", default=os.environ.get("METABONK_BEPINEX_URL", ""))
    ap.add_argument("--bepinex-zip", default=os.environ.get("METABONK_BEPINEX_ZIP", ""))
    ap.add_argument("--clean-bepinex-cache", action="store_true", help="Remove BepInEx cache/interop/dummy directories")
    ap.add_argument(
        "--with-metabonk-plugin",
        action="store_true",
        help="Also build/install MetabonkPlugin (requires Unity.MLAgents.dll or other required refs to exist).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    game_dir = _find_game_dir(args.game_dir or None)
    if args.install_bepinex or args.bepinex_build or args.bepinex_url or args.bepinex_zip:
        _install_bepinex(
            game_dir=game_dir,
            build=str(args.bepinex_build or "").strip(),
            url=str(args.bepinex_url or "").strip() or None,
            zip_path=str(args.bepinex_zip or "").strip() or None,
        )
    if args.clean_bepinex_cache:
        _clean_bepinex_cache(game_dir)
    bepinex_core = game_dir / "BepInEx" / "core"
    # IL2CPP builds do not ship `*_Data/Managed`; BepInEx IL2CPP generates proxy
    # assemblies in `BepInEx/interop` which include UnityEngine modules.
    unity_managed = game_dir / "Megabonk_Data" / "Managed"
    if not unity_managed.exists():
        unity_managed = game_dir / "BepInEx" / "interop"
    plugins_dir = game_dir / "BepInEx" / "plugins"

    if not bepinex_core.exists():
        raise SystemExit(f"BepInEx core dir not found: {bepinex_core}")
    if not unity_managed.exists():
        raise SystemExit(f"Unity Managed dir not found (IL2CPP): {unity_managed}")
    if not plugins_dir.exists():
        raise SystemExit(f"BepInEx plugins dir not found: {plugins_dir}")

    # Install BonkLink into the plugins root as `BonkLink.dll` to avoid BepInEx
    # recursively loading extra DLLs from subdirectories (a common source of
    # duplicate plugin warnings / crashes).
    projects: list[tuple[Path, str]] = [(repo_root / "plugins" / "BonkLink" / "BonkLink.csproj", "")]
    metabonk_csproj = repo_root / "unity_plugin" / "MetabonkPlugin" / "MetabonkPlugin.csproj"
    if args.with_metabonk_plugin:
        # MetabonkPlugin currently references ML-Agents types; only build it when the
        # required proxy assembly exists in the chosen managed/interop dir.
        if (unity_managed / "Unity.MLAgents.dll").exists():
            projects.append((metabonk_csproj, "MetabonkPlugin"))
        else:
            raise SystemExit(
                f"Unity.MLAgents.dll not found in {unity_managed}. "
                "Re-run without --with-metabonk-plugin, or provide the needed assemblies."
            )

    for csproj, folder in projects:
        if not csproj.exists():
            raise SystemExit(f"Missing project: {csproj}")
        print(f"[plugins] building {csproj}")
        dll = _build(csproj, bepinex_core=bepinex_core, unity_managed=unity_managed)
        if not args.no_install:
            if folder == "":
                legacy_dir = plugins_dir / "BonkLink"
                if legacy_dir.exists():
                    try:
                        disabled = game_dir / "BepInEx" / "disabled_plugins"
                        disabled.mkdir(parents=True, exist_ok=True)
                        ts = int(time.time())
                        dst = disabled / f"BonkLink_dir_{ts}"
                        shutil.move(str(legacy_dir), str(dst))
                        print(f"[plugins] moved legacy BonkLink dir -> {dst}")
                    except Exception:
                        print(f"[plugins] warning: legacy BonkLink dir exists (may cause duplicate loads): {legacy_dir}")
            _install(dll, plugins_dir=plugins_dir, folder=folder)

    print("[plugins] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
