#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# Qwen3.8-Flash-Next 1M —— H100 4P4D Router（4 prefill + 4 decode）
#   对外: 0.0.0.0:40000
#
# 用法（必须给 8 个 worker 的内网 IP，逗号分隔；按你的集群填）：
#   export QWEN38_PD_PREFILL_IPS=<PREFILL1_IP>,<PREFILL2_IP>,<PREFILL3_IP>,<PREFILL4_IP>
#   export QWEN38_PD_DECODE_IPS=<DECODE1_IP>,<DECODE2_IP>,<DECODE3_IP>,<DECODE4_IP>
#   nohup setsid bash run_qwen38_flash_next_yarn_1m_pd_router.sh > /tmp/qwen38_router.log 2>&1 &
# ============================================================================

export RUST_LOG="${QWEN38_ROUTER_RUST_LOG:-info}"

if [[ -z "${QWEN38_PD_PREFILL_IPS:-}" || -z "${QWEN38_PD_DECODE_IPS:-}" ]]; then
    cat >&2 <<'EOF'
ERROR: 需要 8 个 worker 的内网 IP（4 prefill + 4 decode）。示例：
  export QWEN38_PD_PREFILL_IPS=10.0.0.11,10.0.0.12,10.0.0.15,10.0.0.16   # prefill-1..4
  export QWEN38_PD_DECODE_IPS=10.0.0.13,10.0.0.14,10.0.0.17,10.0.0.18     # decode-1..4
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
# 8 个 worker（4P4D）x 96 max-running-requests = 384
#
# 路由策略选择（实测见 README「路由策略」）：
#   cache_aware（默认）按前缀亲和选 prefill，重复前缀命中该机 L1 radix / L2 HiCache：
#                300k TTFT 14.1s → 1.43s，800k TTFT 99.8s → 4.88s。
#                代价：4 路"全新前缀"并发时会给同一台 prefill 排两条（冷并发 wall 37.9s vs 25.5s）。
#   round_robin  严格轮转，冷并发负载最匀（4x300k 并发 wall 25.5s / 47.0k tok/s），
#                但同一前缀第二次必落别的机器，L1/L2 全 miss。
MAX_CONCURRENT_REQUESTS="${QWEN38_PD_MAX_CONCURRENT_REQUESTS:-384}"
QUEUE_SIZE="${QWEN38_PD_QUEUE_SIZE:-384}"
QUEUE_TIMEOUT_SECS="${QWEN38_PD_QUEUE_TIMEOUT_SECS:-600}"

PREFILL_POLICY="${QWEN38_PD_PREFILL_POLICY:-cache_aware}"
DECODE_POLICY="${QWEN38_PD_DECODE_POLICY:-round_robin}"

args=(
    --pd-disaggregation
    --prefill-policy "${PREFILL_POLICY}"
    --decode-policy "${DECODE_POLICY}"
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
