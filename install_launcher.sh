#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

chmod +x launch launch.py

echo "✓ Installed: ./launch"
echo "✓ Installed: ./launch.py"
echo
echo "Quick start:"
echo "  ./launch"
