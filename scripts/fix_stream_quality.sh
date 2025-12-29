#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

exec bash "${REPO_ROOT}/docs/stream/fix_stream_quality.sh" "$@"

