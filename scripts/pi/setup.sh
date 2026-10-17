#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ "$(uname -m)" != "aarch64" ]; then
    echo "This needs 64-bit Raspberry Pi OS (aarch64); found $(uname -m)." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen
echo "Ready. Start the coordinator with: scripts/pi/coordinator.sh configs/pilot.yaml"
