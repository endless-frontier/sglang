#!/usr/bin/env bash
set -Eeuo pipefail

export RUST_LOG="${QWEN38_ROUTER_RUST_LOG:-info}"
MAX_CONCURRENT_REQUESTS="${QWEN38_PD_MAX_CONCURRENT_REQUESTS:-4}"

exec python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://10.10.1.156:41000 8998 \
    --prefill http://10.10.1.157:41000 8998 \
    --decode http://10.10.1.154:42000 \
    --decode http://10.10.1.155:42000 \
    --prefill-policy round_robin \
    --decode-policy round_robin \
    --worker-startup-timeout-secs 1800 \
    --request-timeout-secs 7200 \
    --shutdown-grace-period-secs 300 \
    --max-concurrent-requests "${MAX_CONCURRENT_REQUESTS}" \
    --queue-size 0 \
    --host 0.0.0.0 \
    --port 40000
