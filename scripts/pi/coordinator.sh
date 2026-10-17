#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
config="${1:-configs/pilot.yaml}"
shift || true
exec uv run robustness-gap server --config "$config" "$@"
