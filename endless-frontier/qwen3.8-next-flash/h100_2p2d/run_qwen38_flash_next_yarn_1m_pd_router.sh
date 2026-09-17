#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# Qwen3.8-Flash-Next 1M —— H100 2P2D Router
#   对外: 0.0.0.0:40000
#
# 用法（必须给四个 worker 的内网 IP，逗号分隔；按你的集群填）：
#   export QWEN38_PD_PREFILL_IPS=<PREFILL1_IP>,<PREFILL2_IP>
#   export QWEN38_PD_DECODE_IPS=<DECODE1_IP>,<DECODE2_IP>
#   nohup setsid bash run_qwen38_flash_next_yarn_1m_pd_router.sh > /tmp/qwen38_router.log 2>&1 &
# ============================================================================

export RUST_LOG="${QWEN38_ROUTER_RUST_LOG:-info}"

if [[ -z "${QWEN38_PD_PREFILL_IPS:-}" || -z "${QWEN38_PD_DECODE_IPS:-}" ]]; then
    cat >&2 <<'EOF'
ERROR: 需要四个 worker 的内网 IP。示例：
  export QWEN38_PD_PREFILL_IPS=10.0.0.11,10.0.0.12   # prefill-1,prefill-2
  export QWEN38_PD_DECODE_IPS=10.0.0.13,10.0.0.14    # decode-1,decode-2
  bash run_qwen38_flash_next_yarn_1m_pd_router.sh
EOF
    exit 2
fi

PREFILL_IPS="${QWEN38_PD_PREFILL_IPS:-}"
DECODE_IPS="${QWEN38_PD_DECODE_IPS:-}"
PREFILL_PORT="${QWEN38_PD_PREFILL_PORT:-41000}"
DECODE_PORT="${QWEN38_PD_DECODE_PORT:-42000}"
BOOTSTRAP_PORT="${QWEN38_PD_BOOTSTRAP_PORT:-8998}"
ROUTER_HOST="${QWEN38_PD_ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${QWEN38_PD_ROUTER_PORT:-40000}"

# 令牌桶：refill 速率 = max-concurrent-requests；queue-size=0 时超限直接 429
MAX_CONCURRENT_REQUESTS="${QWEN38_PD_MAX_CONCURRENT_REQUESTS:-200}"
QUEUE_SIZE="${QWEN38_PD_QUEUE_SIZE:-200}"
QUEUE_TIMEOUT_SECS="${QWEN38_PD_QUEUE_TIMEOUT_SECS:-600}"

args=(
    --pd-disaggregation
    --prefill-policy round_robin
    --decode-policy round_robin
    --worker-startup-timeout-secs 1800
    --request-timeout-secs 7200
    --shutdown-grace-period-secs 300
    --health-check-interval-secs 10
    --health-failure-threshold 3
    --health-success-threshold 2
    --health-check-timeout-secs 5
    --max-concurrent-requests "${MAX_CONCURRENT_REQUESTS}"
    --queue-size "${QUEUE_SIZE}"
    --queue-timeout-secs "${QUEUE_TIMEOUT_SECS}"
    --host "${ROUTER_HOST}"
    --port "${ROUTER_PORT}"
)

IFS=',' read -r -a prefills <<< "${PREFILL_IPS}"
for ip in "${prefills[@]}"; do
    args+=(--prefill "http://${ip}:${PREFILL_PORT}" "${BOOTSTRAP_PORT}")
done

IFS=',' read -r -a decodes <<< "${DECODE_IPS}"
for ip in "${decodes[@]}"; do
    args+=(--decode "http://${ip}:${DECODE_PORT}")
done

printf '[pd-router] 启动：prefill=%s  decode=%s  listen=%s:%s\n' \
    "${PREFILL_IPS}" "${DECODE_IPS}" "${ROUTER_HOST}" "${ROUTER_PORT}" >&2

exec python3 -m sglang_router.launch_router "${args[@]}"
