#!/usr/bin/env bash
# 本模型容器的启动命令，单独放出来方便阅读与本地调试。
# EAS 会把本文件内容作为服务定义里的 script 执行；手工运行仅用于排障。
#
#   deploy_6node.sh
#
# 平台会为分布式单元的每个实例注入：
#   RANK_ID        实例编号（0..size-1）
#   COMM_IFNAME    组网网卡（开 RDMA 为 net0，否则 eth1）
#   RANK_IP        该网卡 IP
#   MASTER_ADDRESS 0 号实例 IP -> 引擎的 rendezvous 地址

MODEL_DIR=${MODEL_DIR:-/mnt/data/<你的目录>/models/Qwen3.8-2.4T-A95B-FP8}
MACHINES=${MACHINES:-6}
LOG=${LOG:-/mnt/data/<你的目录>/eas/qwen38-24t-$(hostname).log}

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
    echo "CONTRACT-MISSING: 没有 rank 与 rendezvous 地址，拒绝加载 2.45 TB 权重"
    echo "--- environment (filtered) ---"; env | sort | grep -iE "rank|master|comm|world|node|nccl" || true
    exit 42
  fi
  # 参数取自 SGLang 官方 cookbook 对本模型的验证配置，把流水线维度的 4 台改成我们的机器数。
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
