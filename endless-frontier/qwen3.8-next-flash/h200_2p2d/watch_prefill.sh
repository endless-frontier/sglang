#!/usr/bin/env bash
set -Eeuo pipefail

# Run one copy on each Prefill host. It restarts only the local Prefill
# process, so a failure on one host does not disturb the other Prefill host.
WORKER_SCRIPT="${QWEN38_WORKER_SCRIPT:-/mnt/data/xinyu/sglang-qwen38-upstream-1789383617/endless-frontier/qwen3.8-next-flash/h200_2p2d/run_qwen38_flash_next_yarn_1m_pd_worker.sh}"
LOCAL_IP="${QWEN38_LOCAL_IP:?Set QWEN38_LOCAL_IP to this host private IP}"
API_PORT="${QWEN38_PREFILL_API_PORT:-41000}"
BOOTSTRAP_PORT="${QWEN38_PREFILL_BOOTSTRAP_PORT:-8998}"
INTERVAL="${QWEN38_WATCH_INTERVAL_SECS:-10}"
FAILURES="${QWEN38_WATCH_FAILURE_THRESHOLD:-3}"
STARTUP_GRACE="${QWEN38_WATCH_STARTUP_GRACE_SECS:-1800}"
RESTART_GRACE="${QWEN38_WATCH_RESTART_GRACE_SECS:-1800}"
LOG_FILE="${QWEN38_PREFILL_LOG:-/tmp/qwen38_prefill.log}"
PID_FILE="${QWEN38_PREFILL_PID_FILE:-/tmp/qwen38_prefill.pid}"
LOCK_FILE="${QWEN38_PREFILL_WATCH_LOCK:-/tmp/qwen38_prefill_watchdog.lock}"

exec 9>"${LOCK_FILE}"
flock -n 9 || { echo "watchdog already running" >&2; exit 1; }

log() { echo "[$(date -Is)] $*"; }

worker_pid() {
    [[ -s "${PID_FILE}" ]] && { read -r pid <"${PID_FILE}"; [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null && { echo "${pid}"; return; }; }
    return 1
}

stop_worker() {
    local pid="${1:-}"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
    # The worker is launched with setsid, so its process group contains all
    # TP ranks. Never use pkill by command line: it can kill the watchdog.
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    for _ in {1..30}; do
        kill -0 "${pid}" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
}

start_worker() {
    mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${PID_FILE}")"
    : >"${PID_FILE}"
    nohup setsid bash "${WORKER_SCRIPT}" prefill "${LOCAL_IP}" "${API_PORT}" "${BOOTSTRAP_PORT}" \
        >>"${LOG_FILE}" 2>&1 &
    echo "$!" >"${PID_FILE}"
    local new_pid
    new_pid="$(<"${PID_FILE}")"
    log "started Prefill worker pid=${new_pid} api=${LOCAL_IP}:${API_PORT}"
}

health_ok() {
    curl -fsS --connect-timeout 2 --max-time 5 "http://${LOCAL_IP}:${API_PORT}/health" >/dev/null
}

trap 'log "watchdog stopped"; exit 0' INT TERM

log "watching Prefill ${LOCAL_IP}:${API_PORT}; interval=${INTERVAL}s threshold=${FAILURES}"
need_startup_grace=0
if ! worker_pid >/dev/null; then
    # Do not start a second worker when this watchdog is attached after the
    # worker was started manually and its PID file is missing.
    if health_ok; then
        log "existing Prefill is healthy (no PID file); monitoring it"
    else
        start_worker
        need_startup_grace=1
    fi
fi
if (( need_startup_grace )); then
    deadline=$((SECONDS + STARTUP_GRACE))
    while (( SECONDS < deadline )); do
        health_ok && { log "Prefill became healthy"; break; }
        sleep "${INTERVAL}"
    done
fi

bad=0
while true; do
    if health_ok; then
        bad=0
    else
        bad=$((bad + 1))
        log "health check failed (${bad}/${FAILURES})"
    fi
    if (( bad >= FAILURES )); then
        old_pid="$(worker_pid 2>/dev/null || true)"
        log "restarting unhealthy Prefill pid=${old_pid:-unknown}"
        stop_worker "${old_pid}"
        start_worker
        bad=0
        deadline=$((SECONDS + RESTART_GRACE))
        while (( SECONDS < deadline )); do
            health_ok && { log "Prefill recovered"; break; }
            sleep "${INTERVAL}"
        done
    fi
    sleep "${INTERVAL}"
done
