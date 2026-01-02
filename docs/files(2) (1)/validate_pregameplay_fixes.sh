#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper kept under docs/. Prefer running:
#   ./scripts/validate_pregameplay_fixes.sh

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${ROOT}/scripts/validate_pregameplay_fixes.sh" "$@"
