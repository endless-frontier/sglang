#!/usr/bin/env bash
set -Eeuo pipefail

# 把下面四个 <...> 占位符换成实际节点内网 IP（端口需与 worker 一致）后再运行。

export RUST_LOG="${QWEN38_ROUTER_RUST_LOG:-info}"
MAX_CONCURRENT_REQUESTS="${QWEN38_PD_MAX_CONCURRENT_REQUESTS:-200}"
QUEUE_SIZE="${QWEN38_PD_QUEUE_SIZE:-200}"
QUEUE_TIMEOUT_SECS="${QWEN38_PD_QUEUE_TIMEOUT_SECS:-600}"

exec python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://<PREFILL-1-IP>:41000 8998 \
    --prefill http://<PREFILL-2-IP>:41000 8998 \
    --decode http://<DECODE-1-IP>:42000 \
    --decode http://<DECODE-2-IP>:42000 \
    --prefill-policy round_robin \
    --decode-policy round_robin \
    --worker-startup-timeout-secs 1800 \
    --request-timeout-secs 7200 \
    --shutdown-grace-period-secs 300 \
    --max-concurrent-requests "${MAX_CONCURRENT_REQUESTS}" \
    --queue-size "${QUEUE_SIZE}" \
    --queue-timeout-secs "${QUEUE_TIMEOUT_SECS}" \
    --host 0.0.0.0 \
    --port 40000
