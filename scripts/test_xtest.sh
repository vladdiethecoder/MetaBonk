#!/usr/bin/env bash
set -euo pipefail

echo "═══════════════════════════════"
echo "MetaBonk XTEST Extension Test"
echo "═══════════════════════════════"
echo

if [[ -z "${DISPLAY:-}" ]]; then
  echo "✗ FAIL: DISPLAY is not set"
  echo "Hint: run inside the worker environment (gamescope/Xwayland or Smithay Eye compositor)."
  exit 1
fi

echo "DISPLAY=${DISPLAY}"
echo

if ! command -v xdpyinfo >/dev/null 2>&1; then
  echo "✗ FAIL: xdpyinfo not found"
  echo "Install (Fedora): sudo dnf install -y xorg-x11-utils"
  exit 1
fi

echo "Test 1: xdpyinfo -ext XTEST"
if xdpyinfo -display "${DISPLAY}" -ext XTEST 2>/dev/null | grep -q "XTEST"; then
  echo "✓ PASS: XTEST extension found"
else
  echo "✗ FAIL: XTEST extension NOT found on DISPLAY=${DISPLAY}"
  echo
  echo "xdpyinfo output (tail):"
  xdpyinfo -display "${DISPLAY}" -ext XTEST 2>&1 | tail -n 50 || true
  echo
  echo "Hint: for gamescope, try setting:"
  echo "  export GAMESCOPE_XWAYLAND_ARGS='+extension XTEST'"
  exit 1
fi

echo
echo "Test 2: xdotool (optional)"
if command -v xdotool >/dev/null 2>&1; then
  xdotool version >/dev/null 2>&1 && echo "✓ PASS: xdotool works ($(xdotool version))" || {
    echo "✗ FAIL: xdotool failed on DISPLAY=${DISPLAY}"
    exit 1
  }
else
  echo "⚠ SKIP: xdotool not installed"
fi

echo
echo "Test 3: Python libxdo backend (ctypes)"
python3 - <<'PY'
import os

disp = os.environ.get("DISPLAY") or ""
if not disp:
    raise SystemExit("DISPLAY missing")

from src.input.libxdo_backend import LibXDoBackend

b = LibXDoBackend(display=disp)
try:
    print("✓ PASS: LibXDoBackend connected successfully")
finally:
    b.close()
PY

echo
echo "═══════════════════════════════"
echo "✓ All XTEST checks passed"
echo "═══════════════════════════════"
