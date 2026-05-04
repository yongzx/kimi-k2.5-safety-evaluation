#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${BENCHMARK_CONFIG:-${REPROD_CONFIG:-configs/kimi_k25_paper_reprod.yaml}}"

if command -v codex-env >/dev/null 2>&1; then
  codex-env -- uv run python scripts/run_all.py \
    --config "$CONFIG_PATH" \
    --only psychosisbench \
    "$@"
else
  uv run python scripts/run_all.py \
    --config "$CONFIG_PATH" \
    --only psychosisbench \
    "$@"
fi
