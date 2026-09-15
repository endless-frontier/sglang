#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/qwen38_flash_next_pd_tp8_2p2d.json}"
RUN_ID="${2:-qwen38-pd-dry-run}"
OUT="${3:-rendered/${RUN_ID}.json}"
PYTHONPATH=. exec python3 -m deployment.qwen38_flash_next_h200.cli render --config "$CONFIG" --run-id "$RUN_ID" --output "$OUT"
