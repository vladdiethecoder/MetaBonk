#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/MegaBonk"
  exit 1
fi

GAME_DIR="$(cd "$1" && pwd)"
if [[ ! -d "$GAME_DIR/BepInEx" ]]; then
  echo "error: $GAME_DIR does not look like a MegaBonk install (missing BepInEx/)"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/dist/research_plugin"
mkdir -p "$OUT_DIR"

echo "Building ResearchPlugin against $GAME_DIR ..."
dotnet build "$ROOT/mods/ResearchPlugin.csproj" -c Release -p:MEGABONK_DIR="$GAME_DIR" -o "$OUT_DIR"

DLL="$OUT_DIR/ResearchPlugin.dll"
if [[ ! -f "$DLL" ]]; then
  echo "error: build did not produce ResearchPlugin.dll"
  exit 1
fi

echo "Installing to $GAME_DIR/BepInEx/plugins/ResearchPlugin.dll"
mkdir -p "$GAME_DIR/BepInEx/plugins"
cp -f "$DLL" "$GAME_DIR/BepInEx/plugins/ResearchPlugin.dll"

echo "Done."

