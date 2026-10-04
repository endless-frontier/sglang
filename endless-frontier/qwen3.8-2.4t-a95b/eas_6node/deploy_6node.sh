#!/usr/bin/env bash
# The container start command for this model, standalone so it can be read and tested.
# EAS runs this as `script` in the service definition; run it by hand only for debugging.
#
#   deploy_qwen38_24t_a95b_6node.sh
#
# The platform injects, per instance of a distributed service:
#   RANK_ID        instance number inside the unit (0..size-1)
#   COMM_IFNAME    the NIC reserved for inter-node traffic (net0 with RDMA, else eth1)
#   RANK_IP        that NIC's IP
#   MASTER_ADDRESS rank 0's IP  -> the rendezvous address for the engine
set -uo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/data/wangruisi/models/Qwen3.8-2.4T-A95B-FP8}
MACHINES=${MACHINES:-6}
LOG=${LOG:-/mnt/data/wangruisi/eas/qwen38-24t-$(hostname).log}

mkdir -p "$(dirname "$LOG")" /tmp/eas-home /tmp/eas-tmp /tmp/triton-cache /tmp/torchinductor /tmp/xdg-cache
export HOME=/tmp/eas-home TMPDIR=/tmp/eas-tmp TRITON_CACHE_DIR=/tmp/triton-cache \
       TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor XDG_CACHE_HOME=/tmp/xdg-cache
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
export NCCL_SOCKET_IFNAME=${COMM_IFNAME:-net0} GLOO_SOCKET_IFNAME=${COMM_IFNAME:-net0}
export PYTHONPATH=/opt/sglang/python PYTHONUNBUFFERED=1

{
  echo "=== start $(date -Is) host=$(hostname)"
  echo "--- injected contract ---"
  for v in RANK_ID COMM_IFNAME RANK_IP MASTER_ADDRESS; do eval "echo $v=\${$v:-MISSING}"; done
  if [ -z "${RANK_ID:-}" ] || [ -z "${MASTER_ADDRESS:-}" ]; then
    echo "CONTRACT-MISSING: refusing to load 2.45 TB without a rank and a rendezvous address"
    echo "--- environment (filtered) ---"; env | sort | grep -iE "rank|master|comm|world|node|nccl" || true
    exit 42
  fi
  # The flag set is the SGLang cookbook's verified cell for this model, with the pipeline
  # dimension carrying our machine count (the cookbook's own cell uses four 141 GB nodes).
  exec python3 -m sglang.launch_server \
    --model-path "$MODEL_DIR" \
    --served-model-name qwen3.8-2.4t-a95b \
    --nnodes "$MACHINES" --node-rank "${RANK_ID:-0}" \
    --dist-init-addr "${MASTER_ADDRESS}:20000" \
    --tp-size 8 --pp-size "$MACHINES" --dist-timeout 1800 \
    --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer \
    --mamba-full-memory-ratio 0.95 --mamba-ssm-dtype bfloat16 \
    --max-prefill-tokens 8192 --page-size 64 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --max-running-requests 48 --host 0.0.0.0 --port 8000 --trust-remote-code
} 2>&1 | tee -a "$LOG"
