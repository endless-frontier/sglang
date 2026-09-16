#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <prefill|decode> <local-ip> <api-port> [bootstrap-port]" >&2
    exit 2
fi

ROLE="$1"
LOCAL_IP="$2"
API_PORT="$3"
BOOTSTRAP_PORT="${4:-8998}"

if [[ "${ROLE}" != "prefill" && "${ROLE}" != "decode" ]]; then
    echo "Invalid PD role: ${ROLE}" >&2
    exit 2
fi

MODEL_PATH="${QWEN38_MODEL_PATH:-/mnt/data/public_models/Qwen3.8-Flash-Next}"
SGLANG_SOURCE="${QWEN38_SGLANG_SOURCE:-/mnt/data/xinyu/sglang-qwen38-upstream-1789383617}"
CONFIG_PATH="${MODEL_PATH}/config.json"
BACKUP_PATH="${MODEL_PATH}/config.json.native.bak"

# All workers must see byte-for-byte equivalent YaRN settings. The unique
# temporary filename makes this safe when the four nodes start together.
if [[ ! -f "${BACKUP_PATH}" ]]; then
    cp -p "${CONFIG_PATH}" "${BACKUP_PATH}"
fi

python3 - "${CONFIG_PATH}" <<'PY'
import json
import os
import socket
import sys

config_path = sys.argv[1]
temp_path = f"{config_path}.{socket.gethostname()}.{os.getpid()}.tmp"
with open(config_path, "r", encoding="utf-8") as file:
    config = json.load(file)

config.setdefault("text_config", {})["rope_parameters"] = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
}
config["max_position_embeddings"] = 262144

with open(temp_path, "w", encoding="utf-8") as file:
    json.dump(config, file, ensure_ascii=False, indent=2)
    file.write("\n")
os.replace(temp_path, config_path)
PY

export CUDA_HOME="${QWEN38_CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${QWEN38_CUDACXX:-${CUDA_HOME}/bin/nvcc}"
export PATH="${CUDA_HOME}/bin${PATH:+:$PATH}"
export PYTHONPATH="$SGLANG_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MOE_CONFIG_DIR="${QWEN38_MOE_CONFIG_DIR:-/mnt/data/xinyu/moe_configs}"
export SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED="${QWEN38_FUSE_SWIGLU_INTERLEAVED:-1}"

# The cluster is RoCE v2. Index 3 is the routable IPv4 GID on all eight HCAs.
export MC_GID_INDEX="${QWEN38_MC_GID_INDEX:-3}"
export MC_TCP_ENABLE_CONNECTION_POOL="${QWEN38_MC_TCP_ENABLE_CONNECTION_POOL:-true}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${QWEN38_PD_BOOTSTRAP_TIMEOUT:-1800}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${QWEN38_PD_WAITING_TIMEOUT:-1800}"
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE="${QWEN38_PD_THREAD_POOL_SIZE:-12}"
export SGLANG_DISAGGREGATION_QUEUE_SIZE="${QWEN38_PD_QUEUE_SIZE:-4}"

# SGLang 0.5.18 only supports decode DCP>1 for MLA/hybrid-MLA pools. Qwen3.8
# uses a hybrid GDN + GQA pool, so PD must use DCP1 on both sides.
DCP_SIZE="${QWEN38_PD_DCP_SIZE:-1}"
MAX_RUNNING_REQUESTS="${QWEN38_PD_MAX_RUNNING_REQUESTS:-96}"
CUDA_GRAPH_MAX_BS_DECODE="${QWEN38_PD_CUDA_GRAPH_MAX_BS_DECODE:-32}"
EP_SIZE="${QWEN38_EP_SIZE:-}"
OPTIMIZED="${QWEN38_OPTIMIZED:-1}"
if [[ "${OPTIMIZED}" == "1" ]]; then
    # TP8 uses all eight H200s on every host. Set QWEN38_TP_SIZE=4 only when
    # intentionally reproducing the official single-node TP4 recipe.
    TP_SIZE="${QWEN38_TP_SIZE:-8}"
    MEM_FRACTION_STATIC="${QWEN38_MEM_FRACTION_STATIC:-0.85}"
    MAX_TOTAL_TOKENS="${QWEN38_PD_MAX_TOTAL_TOKENS:-6000000}"
else
    TP_SIZE="${QWEN38_TP_SIZE:-8}"
    MEM_FRACTION_STATIC="${QWEN38_MEM_FRACTION_STATIC:-0.90}"
    MAX_TOTAL_TOKENS="${QWEN38_PD_MAX_TOTAL_TOKENS:-1200000}"
fi

# Match every GPU to its nearest 200 Gb/s HCA (verified with nvidia-smi topo).
IB_DEVICE_MAP='{"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3","4":"mlx5_4","5":"mlx5_5","6":"mlx5_6","7":"mlx5_7"}'

args=(
    --model-path "${MODEL_PATH}"
    --tp "${TP_SIZE}"
    --dcp-size "${DCP_SIZE}"
    --dcp-comm-backend ag_rs
    --nnodes 1
    --node-rank 0
    --trust-remote-code
    --host "${LOCAL_IP}"
    --port "${API_PORT}"
    --attention-backend fa3
    --kv-cache-dtype bfloat16
    --context-length "${QWEN38_CONTEXT_LENGTH:-1048576}"
    --max-total-tokens "${MAX_TOTAL_TOKENS}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --chunked-prefill-size "${QWEN38_CHUNKED_PREFILL_SIZE:-8192}"
    --page-size 64
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --num-continuous-decode-steps 1
    --scheduler-recv-interval 1
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --enable-fused-moe-sum-all-reduce
    --moe-runner-backend auto
    --speculative-moe-runner-backend auto
    --mamba-backend triton
    --mamba-radix-cache-strategy extra_buffer
    --linear-attn-prefill-backend "${QWEN38_LINEAR_ATTN_PREFILL_BACKEND:-triton}"
    --linear-attn-decode-backend flashinfer
    --linear-attn-verify-backend triton
    --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
    --flashinfer-allreduce-fusion-backend auto
    --disaggregation-mode "${ROLE}"
    --disaggregation-transfer-backend mooncake
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
    --disaggregation-ib-device "${IB_DEVICE_MAP}"
)

if [[ -n "${EP_SIZE}" ]]; then
    args+=(--ep-size "${EP_SIZE}")
fi

# Official H200 Qwen3.8 recipe: TP4, FlashInfer GDN, BF16 recurrent state,
# NEXTN/MTP and decode CUDA graphs. Keep the conservative TP8/no-MTP profile
# as the default for compatibility, and enable this profile explicitly while
# benchmarking production throughput.
if [[ "${OPTIMIZED}" == "1" ]]; then
    args+=(
        --mamba-ssm-dtype bfloat16
        --linear-attn-prefill-backend flashinfer
        --linear-attn-decode-backend flashinfer
        --linear-attn-verify-backend triton
        --tokenizer-worker-num 6
    )
    # Qwen3.8 QSA draft-prefill currently triggers CUDA illegal-address
    # failures in the MTP/EAGLE path. Keep NEXTN only on decode workers;
    # prefill remains on the stable non-speculative path.
    if [[ "${ROLE}" == "decode" ]]; then
        args+=(
            --speculative-algorithm NEXTN
            --speculative-num-steps 3
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 4
        )
    fi
fi

exec python3 -m sglang.launch_server "${args[@]}"
