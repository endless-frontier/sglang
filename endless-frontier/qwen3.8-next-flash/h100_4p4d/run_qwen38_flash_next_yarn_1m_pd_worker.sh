#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# Qwen3.8-Flash-Next 1M —— H100 4P4D worker（每机 8 卡 TP8；8 台 = 4 prefill + 4 decode）
#
# 拓扑（与本目录 README 一致，IP 换成你集群的内网地址）：
#   prefill-1  <PREFILL1_IP>  prefill  41000  bootstrap 8998
#   prefill-2  <PREFILL2_IP>  prefill  41000  bootstrap 8998
#   prefill-3  <PREFILL3_IP>  prefill  41000  bootstrap 8998
#   prefill-4  <PREFILL4_IP>  prefill  41000  bootstrap 8998
#   decode-1   <DECODE1_IP>   decode   42000  bootstrap 8998
#   decode-2   <DECODE2_IP>   decode   42000  bootstrap 8998
#   decode-3   <DECODE3_IP>   decode   42000  bootstrap 8998
#   decode-4   <DECODE4_IP>   decode   42000  bootstrap 8998
#   router: 在任一节点 40000
#
# 用法：
#   bash run_qwen38_flash_next_yarn_1m_pd_worker.sh prefill <PREFILL1_IP> 41000 8998
#   bash run_qwen38_flash_next_yarn_1m_pd_worker.sh decode  <DECODE1_IP>  42000 8998
#   bash run_qwen38_flash_next_yarn_1m_pd_worker.sh prefill <PREFILL1_IP> 41000 8998 --check-only
#
# 关键点（踩坑记录见 ../DEPLOYMENT_PRACTICE.md）：
#   * 源码树默认 /mnt/data/xinyuzhu/sglang（dev 分支，已 merge main：含 QSA gather clamp 修复）；
#     用 QWEN38_SGLANG_SOURCE 指到自己的源码树；
#   * config.json 就地检测：已是 1M（YaRN factor=4.0）-> 不动；不是 -> 备份 .native.bak 后原子改写；
#   * NEXTN 只在 decode 开：QSA draft-prefill 在超长上下文有 CUDA 非法地址缺陷；
#   * attention backend 不指定时（auto）：Prefill 选 fa3，Decode（NEXTN + page_size 64）选 flashinfer
#     —— 所以镜像必须带 flashinfer >= 0.6.18（本 H100 镜像满足）；排障可用
#     QWEN38_ATTENTION_BACKEND=fa3|flashinfer 强制两端统一；
#   * KV 传输默认 mooncake + RoCE v2（MC_GID_INDEX=3，mlx5_0..7 一卡一 HCA）；
#     RDMA 不通时可用 QWEN38_PD_TRANSFER_BACKEND=mooncake_tcp 走 TCP；
#   * --max-total-tokens 默认不传（由显存 profile 自动算），需要封顶再用 QWEN38_PD_MAX_TOTAL_TOKENS。
#   * 显存调优（2026-09-17 实测，TTFT/TPS 不变）：
#     - mem-fraction-static 默认 0.93（原 0.85）：prefill 1.47M -> 1.74M、decode 2.25M -> 2.72M token；
#     - prefill 默认 --max-mamba-cache-size 640（原为自动 sizing=2771 槽/18.98GB，用量 <1%）
#       -> prefill KV 池 2.94M token；用 QWEN38_PD_MAX_MAMBA_CACHE_SIZE=auto 恢复自动。
#     细节见 README「KV cache 容量调优」。
#   * HiCache（L2 host 内存池）默认：**prefill 开 ratio 1.5**（host 池 +4.40M token/rank，
#     每台约 490GB 内存），decode 不开；QWEN38_PD_HICACHE_RATIO=off 关闭。见 README「HiCache」。
#   * /health 的 detokenizer 检查窗口 20s -> 300s（SGLANG_HEALTH_CHECK_TIMEOUT=300，
#     QWEN38_PD_HEALTH_CHECK_TIMEOUT 可覆盖）：detokenizer 抖动时 /health 不再秒级 503，
#     详见 README「看门狗 / 为什么不能只用 pgrep」。
# ============================================================================

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <prefill|decode> <local-ip> <api-port> [bootstrap-port] [--check-only]" >&2
    exit 2
fi

ROLE="$1"
LOCAL_IP="$2"
API_PORT="$3"
BOOTSTRAP_PORT="${4:-8998}"
CHECK_ONLY=0
for arg in "$@"; do
    [[ "$arg" == "--check-only" ]] && CHECK_ONLY=1
done

if [[ "${ROLE}" != "prefill" && "${ROLE}" != "decode" ]]; then
    echo "Invalid PD role: ${ROLE} (expect prefill|decode)" >&2
    exit 2
fi

# 本次交付模型：qwen38_flash_bio_0915_4ep iter_0001285（qwen4_exp，rope 原生 default
# -> 首次启动会就地改写为 YaRN 1M，原生配置备份为 config.json.native.bak）
MODEL_PATH="${QWEN38_MODEL_PATH:-/mnt/data/yuzhucai/dlc_outputs/qwen38_flash_bio_0915_4ep/iter_0001285/hf}"
SGLANG_SOURCE="${QWEN38_SGLANG_SOURCE:-/mnt/data/xinyuzhu/sglang}"
TP_SIZE="${QWEN38_TP_SIZE:-8}"
CONTEXT_LENGTH="${QWEN38_CONTEXT_LENGTH:-1048576}"
MEM_FRACTION_STATIC="${QWEN38_MEM_FRACTION_STATIC:-0.93}"
MAX_RUNNING_REQUESTS="${QWEN38_MAX_RUNNING_REQUESTS:-96}"
CHUNKED_PREFILL_SIZE="${QWEN38_CHUNKED_PREFILL_SIZE:-8192}"
PAGE_SIZE="${QWEN38_PAGE_SIZE:-64}"
CUDA_GRAPH_MAX_BS_DECODE="${QWEN38_CUDA_GRAPH_MAX_BS_DECODE:-32}"
TOKENIZER_WORKER_NUM="${QWEN38_TOKENIZER_WORKER_NUM:-6}"
SPECULATIVE="${QWEN38_SPECULATIVE:-1}"          # 仅 decode 生效
ATTENTION_BACKEND="${QWEN38_ATTENTION_BACKEND:-auto}"
TRANSFER_BACKEND="${QWEN38_PD_TRANSFER_BACKEND:-mooncake}"
MAX_TOTAL_TOKENS="${QWEN38_PD_MAX_TOTAL_TOKENS:-}"
MAX_MAMBA_CACHE_SIZE="${QWEN38_PD_MAX_MAMBA_CACHE_SIZE:-}"
# HiCache（L2 host 内存池）——当前交付方案的默认值：
#   prefill 默认 ratio 1.5（host KV 池 4,403,456 token/rank = 54.11GB/rank，另加 7.07GB/rank
#   的 host Mamba state，每台约 490GB 内存）；decode 默认不开。
#   实测：冷启动 TTFT / decode TPS 与不开 HiCache 一致；L2 命中把 300k 重复 prompt 从 14s 降到 1.6s。
#   关闭：QWEN38_PD_HICACHE_RATIO=off（也认 none/disable/0）；改值：=2.0 等
#   （1TB 节点上限 ≈2.2，超了启动会报 Not enough host memory available）。
HICACHE_RATIO="${QWEN38_PD_HICACHE_RATIO:-}"
case "${HICACHE_RATIO}" in
    off|none|disable|disabled|no|0) HICACHE_RATIO="" ;;
    "") HICACHE_RATIO="$( [[ "${ROLE}" == "prefill" ]] && printf '1.5' || printf '' )" ;;
esac
HICACHE_SIZE_GB="${QWEN38_PD_HICACHE_SIZE:-}"         # 例：64 -> 每 rank 固定 64GB host 池（覆盖 ratio）
HICACHE_WRITE_POLICY="${QWEN38_PD_HICACHE_WRITE_POLICY:-write_through}"
HICACHE_MEM_LAYOUT="${QWEN38_PD_HICACHE_MEM_LAYOUT:-page_first}"
HICACHE_STORAGE_BACKEND="${QWEN38_PD_HICACHE_STORAGE_BACKEND:-}"   # 例：file（L3 磁盘层）
HICACHE_STORAGE_DIR="${QWEN38_PD_HICACHE_STORAGE_DIR:-}"
HICACHE_STORAGE_CONFIG="${QWEN38_PD_HICACHE_STORAGE_CONFIG:-}"     # 例：{"max_size":"128G","min_free_space":"50G"}
# Mamba/SSM state 池：prefill 端若交给自动 sizing（--mamba-full-memory-ratio 0.9），
# 实测会膨胀到 18.98GB / 2771 槽（conv 0.71 + ssm 18.27），而日志里 mamba usage < 1%，
# 把 KV 池挤到只剩 ~20GB；封顶 640 槽（~4.4GB）把腾出的显存给 KV：
# prefill 池 1.73M -> 2.94M token（+69%），并发/性能不变。
# 640//ratio(=5) = 128 ≥ max_running_requests(96)，不会压低并发；=auto 恢复自动 sizing。
if [[ "${MAX_MAMBA_CACHE_SIZE}" == "auto" ]]; then
    MAX_MAMBA_CACHE_SIZE=""
elif [[ -z "${MAX_MAMBA_CACHE_SIZE}" && "${ROLE}" == "prefill" ]]; then
    MAX_MAMBA_CACHE_SIZE=640
fi
SERVED_MODEL_NAME="${QWEN38_SERVED_MODEL_NAME:-qwen38-flash-next-1m}"
IB_DEVICE_MAP="${QWEN38_PD_IB_DEVICE_MAP:-{\"0\":\"mlx5_0\",\"1\":\"mlx5_1\",\"2\":\"mlx5_2\",\"3\":\"mlx5_3\",\"4\":\"mlx5_4\",\"5\":\"mlx5_5\",\"6\":\"mlx5_6\",\"7\":\"mlx5_7\"}}"

die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf '\033[33mWARN\033[0m: %s\n' "$*" >&2; }
note() { printf '[pd-%s] %s\n' "$ROLE" "$*"; }

# ---------------------------------------------------------------------------
# 0. 源码树 / 模型目录检查
# ---------------------------------------------------------------------------
[[ -f "${SGLANG_SOURCE}/python/sglang/srt/models/qwen4_exp.py" ]] || \
    die "源码树缺少 qwen4_exp 实现：${SGLANG_SOURCE}（用 QWEN38_SGLANG_SOURCE 指定）"
[[ -d "$MODEL_PATH" ]] || die "模型目录不存在：$MODEL_PATH"
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || \
    die "缺少 $MODEL_PATH/model.safetensors.index.json"
[[ -f "$MODEL_PATH/tokenizer.json" ]] || die "缺少 $MODEL_PATH/tokenizer.json"
[[ -f "$MODEL_PATH/config.json" ]] || die "缺少 $MODEL_PATH/config.json（YaRN 配置）"

# ---------------------------------------------------------------------------
# 1. YaRN 配置：就地检测
#    已是 1M（rope_type=yarn & factor=4.0）-> 原样不动
#    不是 -> 备份 config.json.native.bak 后原子改写（四台机器同时启动也安全）
# ---------------------------------------------------------------------------
python3 - "$MODEL_PATH" <<'PY' || die "生成 YaRN 配置失败"
import json
import os
import shutil
import sys

model_path = sys.argv[1]
config_path = os.path.join(model_path, "config.json")

with open(config_path, encoding="utf-8") as fh:
    config = json.load(fh)

text_config = config.get("text_config")
if config.get("model_type") != "qwen4_exp" or not isinstance(text_config, dict):
    raise SystemExit(f"模型不是 qwen4_exp 架构：model_type={config.get('model_type')!r}")

native = text_config.get("max_position_embeddings")
if native != 262144:
    raise SystemExit(f"原生 max_position_embeddings={native!r}，期望 262144")

target_rope = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
}

def is_1m(rope):
    try:
        return (
            (rope or {}).get("rope_type") == "yarn"
            and float((rope or {}).get("factor", 0) or 0) == 4.0
            and int((rope or {}).get("original_max_position_embeddings", 0) or 0) == int(native)
        )
    except (TypeError, ValueError):
        return False

if is_1m(text_config.get("rope_parameters")) and config.get("max_position_embeddings") == native:
    print("[yarn] 已是 1M 配置，保持不动：%s" % config_path, file=sys.stderr)
    raise SystemExit(0)

backup_path = os.path.join(model_path, "config.json.native.bak")
if not os.path.exists(backup_path):
    if not os.access(model_path, os.W_OK) or not os.access(config_path, os.W_OK):
        raise SystemExit("模型目录不可写，无法就地改写 YaRN 配置：%s" % model_path)
    shutil.copy2(config_path, backup_path)
    print("[yarn] 原生配置已备份：%s" % backup_path, file=sys.stderr)
else:
    print("[yarn] 原生备份已存在，保持不动：%s" % backup_path, file=sys.stderr)

text_config["rope_parameters"] = target_rope
config["max_position_embeddings"] = native

tmp_path = "%s.tmp.%d" % (config_path, os.getpid())
with open(tmp_path, "w", encoding="utf-8") as fh:
    json.dump(config, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
os.replace(tmp_path, config_path)
print("[yarn] 已改写为 1M 配置：%s" % config_path, file=sys.stderr)
PY

# ---------------------------------------------------------------------------
# 1.5 权重分片完整性 + YaRN 生效性（不加载权重，秒级）
# ---------------------------------------------------------------------------
python3 - "$MODEL_PATH" "$CONTEXT_LENGTH" <<'PY' || die "模型配置/权重校验失败"
import json
import os
import sys

model_dir, requested = sys.argv[1], int(sys.argv[2])

with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
    config = json.load(fh)

if config.get("model_type") != "qwen4_exp":
    raise SystemExit("unexpected model_type=%r" % config.get("model_type"))

rope = (config.get("text_config") or {}).get("rope_parameters") or {}
if rope.get("rope_type") != "yarn" or float(rope.get("factor", 0) or 0) != 4.0:
    raise SystemExit("YaRN 未生效：%r" % rope)

native = (config.get("text_config") or {}).get("max_position_embeddings")
if int(rope.get("original_max_position_embeddings", 0) or 0) != int(native):
    raise SystemExit("original_max_position_embeddings 与原生 max_position_embeddings 不一致")

if requested > 1048576:
    raise SystemExit("本脚本最多支持 1048576 上下文（YaRN factor 4 x 262144）")

with open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8") as fh:
    weight_map = json.load(fh)["weight_map"]

shards = sorted(set(weight_map.values()))
missing = [s for s in shards if not os.path.exists(os.path.join(model_dir, s))]
if missing:
    listed = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
    raise SystemExit("权重分片缺失 %d/%d：%s" % (len(missing), len(shards), listed))

print("模型校验通过：qwen4_exp / 原生 %s / YaRN factor=%s / 上下文 %d / 分片 %d 齐全"
      % (native, rope["factor"], requested, len(shards)))
PY

# ---------------------------------------------------------------------------
# 2. GPU / HCA 预检 + 环境
# ---------------------------------------------------------------------------
GPU_LIST="${QWEN38_CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TP_SIZE - 1)))}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

CUDA_HOME="${QWEN38_CUDA_HOME:-/usr/local/cuda}"
if [[ -d "$CUDA_HOME" ]]; then
    export CUDA_HOME CUDACXX="${QWEN38_CUDACXX:-${CUDA_HOME}/bin/nvcc}"
    export PATH="${CUDA_HOME}/bin${PATH:+:$PATH}"
else
    warn "未找到 ${CUDA_HOME}，沿用镜像现有 CUDA 环境"
fi

export PYTHONPATH="$SGLANG_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED="${QWEN38_FUSE_SWIGLU_INTERLEAVED:-1}"
# /health 的 detokenizer 心跳窗口：SGLang 默认 20s（服务端 HEALTH_CHECK_TIMEOUT，
# http_server.py:193 读的就是这个环境变量）——detokenizer 稍有抖动 /health 就返 503，
# 会被 Router / 看门狗当成硬失败。这里放宽到 300s（5 分钟）。
export SGLANG_HEALTH_CHECK_TIMEOUT="${QWEN38_PD_HEALTH_CHECK_TIMEOUT:-300}"

# PD / Mooncake
export MC_GID_INDEX="${QWEN38_MC_GID_INDEX:-3}"          # RoCE v2 IPv4 GID（mlx5_i 的 eth_i）
export MC_TCP_ENABLE_CONNECTION_POOL="${QWEN38_MC_TCP_ENABLE_CONNECTION_POOL:-true}"
export MC_MS_AUTO_DISC="${QWEN38_MC_MS_AUTO_DISC:-1}"
[[ -d /mnt/data/xinyu/moe_configs ]] && export SGLANG_MOE_CONFIG_DIR="${QWEN38_MOE_CONFIG_DIR:-/mnt/data/xinyu/moe_configs}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${QWEN38_PD_BOOTSTRAP_TIMEOUT:-1800}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${QWEN38_PD_WAITING_TIMEOUT:-1800}"
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE="${QWEN38_PD_THREAD_POOL_SIZE:-12}"
export SGLANG_DISAGGREGATION_QUEUE_SIZE="${QWEN38_PD_QUEUE_SIZE:-4}"

note "模型目录    : $MODEL_PATH"
note "源码树      : $SGLANG_SOURCE"
if [[ -d "$SGLANG_SOURCE/.git" ]]; then
    note "源码树版本  : $(git -C "$SGLANG_SOURCE" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')@$(git -C "$SGLANG_SOURCE" rev-parse --short HEAD 2>/dev/null || echo '?')"
fi
note "角色/地址   : $ROLE @ ${LOCAL_IP}:${API_PORT}（bootstrap ${BOOTSTRAP_PORT}）"
note "TP=$TP_SIZE context=$CONTEXT_LENGTH mem-fraction-static=$MEM_FRACTION_STATIC max-running-requests=$MAX_RUNNING_REQUESTS"
note "max-mamba-cache-size=${MAX_MAMBA_CACHE_SIZE:-auto}"
note "hicache: ratio=${HICACHE_RATIO:-off} size=${HICACHE_SIZE_GB:-off}GB storage=${HICACHE_STORAGE_BACKEND:-none}"
note "attention-backend=$ATTENTION_BACKEND transfer-backend=$TRANSFER_BACKEND gid-index=$MC_GID_INDEX"
note "health-check-timeout=${SGLANG_HEALTH_CHECK_TIMEOUT}s（detokenizer 心跳窗口；SGLang 默认 20s）"
note "NEXTN: $([[ "$ROLE" == decode && "$SPECULATIVE" == 1 ]] && echo on || echo off)"

# ---------------------------------------------------------------------------
# 3. 启动参数
# ---------------------------------------------------------------------------
args=(
    --model-path "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tp "${TP_SIZE}"
    --dcp-size 1
    --dcp-comm-backend ag_rs
    --nnodes 1
    --node-rank 0
    --trust-remote-code
    --host "${LOCAL_IP}"
    --port "${API_PORT}"
    --kv-cache-dtype bfloat16
    --context-length "${CONTEXT_LENGTH}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --page-size "${PAGE_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --num-continuous-decode-steps 1
    --scheduler-recv-interval 1
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --enable-fused-moe-sum-all-reduce
    --moe-runner-backend auto
    --speculative-moe-runner-backend auto
    --mamba-backend triton
    --mamba-ssm-dtype bfloat16
    --mamba-radix-cache-strategy extra_buffer
    --linear-attn-prefill-backend "${QWEN38_LINEAR_ATTN_PREFILL_BACKEND:-flashinfer}"
    --linear-attn-decode-backend "${QWEN38_LINEAR_ATTN_DECODE_BACKEND:-flashinfer}"
    --linear-attn-verify-backend "${QWEN38_LINEAR_ATTN_VERIFY_BACKEND:-triton}"
    --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
    --flashinfer-allreduce-fusion-backend auto
    --tokenizer-worker-num "${TOKENIZER_WORKER_NUM}"
    --disaggregation-mode "${ROLE}"
    --disaggregation-transfer-backend "${TRANSFER_BACKEND}"
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
    --disaggregation-ib-device "${IB_DEVICE_MAP}"
)

# attention backend：auto = 不传，交给 SGLang 自选（本模型 H100 上 = flashinfer）
if [[ "$ATTENTION_BACKEND" != "auto" ]]; then
    args+=(--attention-backend "${ATTENTION_BACKEND}")
fi

if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    args+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi

if [[ -n "${MAX_MAMBA_CACHE_SIZE}" ]]; then
    args+=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
fi

# HiCache：L2 host 内存层（--hicache-ratio / --hicache-size），可选 L3 存储层
if [[ -n "${HICACHE_RATIO}" || -n "${HICACHE_SIZE_GB}" || -n "${HICACHE_STORAGE_BACKEND}" ]]; then
    args+=(--enable-hierarchical-cache --hicache-write-policy "${HICACHE_WRITE_POLICY}")
    args+=(--hicache-mem-layout "${HICACHE_MEM_LAYOUT}")
    if [[ -n "${HICACHE_RATIO}" ]]; then
        args+=(--hicache-ratio "${HICACHE_RATIO}")
    fi
    if [[ -n "${HICACHE_SIZE_GB}" ]]; then
        args+=(--hicache-size "${HICACHE_SIZE_GB}")
    fi
    if [[ -n "${HICACHE_STORAGE_BACKEND}" ]]; then
        [[ -n "${HICACHE_STORAGE_DIR}" ]] && export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${HICACHE_STORAGE_DIR}"
        args+=(--hicache-storage-backend "${HICACHE_STORAGE_BACKEND}")
        [[ -n "${HICACHE_STORAGE_CONFIG}" ]] && args+=(--hicache-storage-backend-extra-config "${HICACHE_STORAGE_CONFIG}")
    fi
fi

# NEXTN/MTP：只在 decode 打开（QSA draft-prefill 长上下文有 CUDA 非法地址缺陷）
if [[ "$ROLE" == "decode" && "$SPECULATIVE" == "1" ]]; then
    args+=(
        --speculative-algorithm NEXTN
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
    )
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
    printf '校验通过（%s）。启动命令：\n  ' "$ROLE"
    printf '%q ' python3 -m sglang.launch_server "${args[@]}"
    printf '\n'
    exit 0
fi

cat >&2 <<EOF
[deploy] 启动 SGLang ${ROLE} worker（TP${TP_SIZE}, 1M, NEXTN=$([[ "$ROLE" == decode && "$SPECULATIVE" == 1 ]] && echo on || echo off)）
[deploy] 就绪判断：curl -s http://${LOCAL_IP}:${API_PORT}/health   # 200
[deploy] 日志：/tmp/qwen38_${ROLE}.log
EOF

exec python3 -m sglang.launch_server "${args[@]}"
