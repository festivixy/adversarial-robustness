#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv, the Python project manager ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen
if [ $# -ge 1 ]; then
    exec uv run robustness-gap lab --coordinator "$1"
fi
exec uv run robustness-gap lab --setup
