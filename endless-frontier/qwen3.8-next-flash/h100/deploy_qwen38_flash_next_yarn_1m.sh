#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# Qwen3.8-Flash-Next —— 单机 8 卡 H100 / TP8 / BF16 / YaRN 1M 部署脚本
#
# 使用镜像（已验证）：
#   dptech-sh-pai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/dptech-namespace/sglang:sglang-0-5-18-qwen38-next-flash-h100-1m
#   该镜像 = CUDA 13.0 devel + PyTorch 2.13.0+cu130 + SGLang 0.5.18 +
#   sglang-kernel 0.4.7 + flashinfer-python/cubin/jit-cache 0.6.18
#
# 本脚本做四件事：
#   1) 探测 Qwen3.8 兼容源码树（默认 /mnt/data/xinyuzhu/sglang，dev 分支）并打印分支/commit；
#   2) 检查模型目录里的 config.json 是不是 1M（YaRN factor=4.0）配置：
#      是 -> 原样不动；不是 -> 先备份 config.json.native.bak，再原子改写；
#   3) 校验权重分片完整性、mtp.*（NEXTN 前提）、YaRN 生效性、
#      并预测 SGLang 实际会选哪个 attention backend（见 §5，和 flashinfer 版本断言强相关）；
#   4) 固定 CUDA_HOME/CUDACXX/PATH 并启动 SGLang（TP8、NEXTN、1M 上下文）。
#
# 用法：
#   bash deploy_qwen38_flash_next_yarn_1m.sh --check-only   # 只校验 + 打印最终命令
#   nohup setsid bash deploy_qwen38_flash_next_yarn_1m.sh \
#       > /tmp/qwen38_single.log 2>&1 &                     # 后台启动
#   curl -s http://127.0.0.1:40000/health                   # 200 即就绪
#
# 详细说明见同目录 README.md，环境基线/踩坑见 ../DEPLOYMENT_PRACTICE.md
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

MODEL_PATH="${QWEN38_MODEL_PATH:-/mnt/data/public_models/Qwen3.8-Flash-Next}"
DEFAULT_SGLANG_SOURCE="/mnt/data/xinyuzhu/sglang"
SGLANG_SOURCE="${QWEN38_SGLANG_SOURCE:-}"
SERVED_MODEL_NAME="${QWEN38_SERVED_MODEL_NAME:-qwen38-flash-next-1m}"
HOST="${QWEN38_HOST:-0.0.0.0}"
PORT="${QWEN38_PORT:-40000}"
GPU_LIST="${QWEN38_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TP_SIZE="${QWEN38_TP_SIZE:-8}"
CONTEXT_LENGTH="${QWEN38_CONTEXT_LENGTH:-1048576}"
MEM_FRACTION_STATIC="${QWEN38_MEM_FRACTION_STATIC:-0.90}"
CUDA_GRAPH_MAX_BS_DECODE="${QWEN38_CUDA_GRAPH_MAX_BS_DECODE:-32}"
MAX_RUNNING_REQUESTS="${QWEN38_MAX_RUNNING_REQUESTS:-96}"
SPECULATIVE="${QWEN38_SPECULATIVE:-1}"
# auto = 不传 --attention-backend，由 SGLang 自选（H100 镜像里会落到 flashinfer，
# 因为 QSA 压缩注意力把 page_size 固定为 64，fa3 的自动选择条件不满足，见 §5）。
# flashinfer 低于 0.6.18 的镜像请设 QWEN38_ATTENTION_BACKEND=fa3（H200 PD 线上就是这么跑的）。
ATTENTION_BACKEND="${QWEN38_ATTENTION_BACKEND:-auto}"
CHECK_ONLY="${QWEN38_CHECK_ONLY:-0}"
SKIP_CHECKS="${QWEN38_SKIP_CHECKS:-0}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf '\033[33mWARN\033[0m: %s\n' "$*" >&2; }
note() { printf '[deploy] %s\n' "$*"; }

usage() {
    cat <<'EOF'
用法：
  deploy_qwen38_flash_next_yarn_1m.sh                前台启动服务
  deploy_qwen38_flash_next_yarn_1m.sh --check-only   只做校验并打印启动命令
  deploy_qwen38_flash_next_yarn_1m.sh --help

常用环境变量：
  QWEN38_MODEL_PATH              默认 /mnt/data/public_models/Qwen3.8-Flash-Next
  QWEN38_SGLANG_SOURCE           覆盖源码树（默认 /mnt/data/xinyuzhu/sglang，dev 分支）
  QWEN38_ATTENTION_BACKEND       auto（默认）| fa3 | flashinfer | triton
                                 auto：由 SGLang 自选（本模型 = flashinfer，需 flashinfer>=0.6.18）
                                 fa3 ：显式指定，可绕开 flashinfer>=0.6.18 的启动断言
  QWEN38_SERVED_MODEL_NAME       对外模型名，默认 qwen38-flash-next-1m
  QWEN38_CONTEXT_LENGTH          单请求上下文，默认 1048576（上限）
  QWEN38_MEM_FRACTION_STATIC     0.90，OOM 时降到 0.85
  QWEN38_CUDA_GRAPH_MAX_BS_DECODE 32，显存紧张时降到 16
  QWEN38_MAX_RUNNING_REQUESTS    96（与 PD 方案一致）
  QWEN38_SPECULATIVE             1=开 NEXTN（默认），0=关
  QWEN38_PORT / QWEN38_HOST      40000 / 0.0.0.0
  QWEN38_API_KEY                 设置后启用 --api-key
  QWEN38_SKIP_CHECKS             1=跳过 torch/GPU/sglang 运行时校验
  QWEN38_CHECK_ONLY              1 等价于 --check-only

说明：
  --model-path 就是 QWEN38_MODEL_PATH 这个原始模型目录本身，不使用 overlay/软链。
  YaRN 只保留一份 config.json：已是 1M（YaRN factor=4.0）版本 -> 原样使用，不动文件；
  不是 -> 先备份成 config.json.native.bak，再原子改写。
EOF
}

case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --check-only) CHECK_ONLY=1; shift ;;
esac
[[ "$#" -eq 0 ]] || die "不支持的参数：$*（只接受 --check-only / --help）"

for legacy_var in QWEN38_CONFIG_MODE QWEN38_OVERLAY_DIR; do
    if [[ -n "${!legacy_var:-}" ]]; then
        warn "${legacy_var} 已废弃并被忽略：本脚本直接使用原始模型目录，不使用 overlay/软链"
    fi
done

# ---------------------------------------------------------------------------
# 0. 源码树探测（需要含 qwen4_exp 实现）
# ---------------------------------------------------------------------------
sglang_source_ok() { [[ -f "$1/python/sglang/srt/models/qwen4_exp.py" ]]; }

if [[ -n "$SGLANG_SOURCE" ]]; then
    sglang_source_ok "$SGLANG_SOURCE" || \
        die "QWEN38_SGLANG_SOURCE 下缺少 qwen4_exp 实现：$SGLANG_SOURCE"
else
    for candidate in \
        "$DEFAULT_SGLANG_SOURCE" \
        /mnt/data/xinyu/sglang-qwen38 \
        /mnt/data/xinyu/sglang-qwen38-upstream-1789383617 \
        "${SCRIPT_DIR}/../../sglang"
    do
        if sglang_source_ok "$candidate"; then
            SGLANG_SOURCE="$(cd -- "$candidate" && pwd -P)"
            break
        fi
    done
    if [[ -z "$SGLANG_SOURCE" ]]; then
        warn "没找到含 qwen4_exp 的源码树，将直接使用镜像内置 sglang；" \
             "若启动报 ModuleNotFoundError: qwen4_exp，请设置 QWEN38_SGLANG_SOURCE"
    fi
fi

# ---------------------------------------------------------------------------
# 1. 模型目录与关键文件
# ---------------------------------------------------------------------------
[[ -d "$MODEL_PATH" ]] || die "模型目录不存在：$MODEL_PATH"
[[ -f "$MODEL_PATH/config.json" ]] || die "缺少 $MODEL_PATH/config.json"
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || \
    die "缺少 $MODEL_PATH/model.safetensors.index.json"
[[ -f "$MODEL_PATH/tokenizer.json" ]] || die "缺少 $MODEL_PATH/tokenizer.json"

# ---------------------------------------------------------------------------
# 2. YaRN 配置：先检测原始模型目录的 config.json 是不是 1M（YaRN factor=4.0）版本
#    已是 1M -> 原样保留，一个字节都不动
#    不是   -> 先备份成 config.json.native.bak，再原子改写为 1M 配置
#    注意 text_config.max_position_embeddings 保持原生 262144，只改 rope_parameters
# ---------------------------------------------------------------------------
RUNTIME_MODEL_PATH="$MODEL_PATH"

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
    raise SystemExit(
        f"模型不是 qwen4_exp 架构：model_type={config.get('model_type')!r}"
    )

native = text_config.get("max_position_embeddings")
if native != 262144:
    raise SystemExit(
        f"原生 max_position_embeddings={native!r}，期望 262144；"
        "请确认这是 Qwen3.8-Flash-Next（qwen4_exp）完整 HF 产物"
    )

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
    print("[yarn] 已是 1M（YaRN factor=4.0）配置，保持不动：%s" % config_path, file=sys.stderr)
    raise SystemExit(0)

backup_path = os.path.join(model_path, "config.json.native.bak")
if not os.path.exists(backup_path):
    if not os.access(model_path, os.W_OK) or not os.access(config_path, os.W_OK):
        raise SystemExit(
            "模型目录不可写，无法就地改写 YaRN 配置：%s\n"
            "请给该目录写权限，或把模型拷到可写目录后用 QWEN38_MODEL_PATH 指过去。" % model_path
        )
    shutil.copy2(config_path, backup_path)
    print("[yarn] 原生配置已备份：%s" % backup_path, file=sys.stderr)
else:
    print("[yarn] 原生备份已存在，保持不动：%s" % backup_path, file=sys.stderr)

# 当前不是 1M 配置：备份后改写（Transformers 5 还会校验顶层 max_position_embeddings）
text_config["rope_parameters"] = target_rope
config["max_position_embeddings"] = native

tmp_path = "%s.tmp.%d" % (config_path, os.getpid())
try:
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp_path, config_path)   # 原子替换，容器中断不会留下坏 JSON
except OSError as exc:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    raise SystemExit("就地改写 %s 失败：%s" % (config_path, exc))

print("[yarn] 已改写为 1M 配置：%s（原生备份 %s）" % (config_path, backup_path), file=sys.stderr)
PY

RUNTIME_CONFIG="${RUNTIME_MODEL_PATH}/config.json"
note "模型目录      : $MODEL_PATH"
if [[ -n "$SGLANG_SOURCE" ]]; then
    note "源码树        : $SGLANG_SOURCE"
    if [[ -d "$SGLANG_SOURCE/.git" ]]; then
        source_branch="$(git -C "$SGLANG_SOURCE" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
        source_rev="$(git -C "$SGLANG_SOURCE" rev-parse --short HEAD 2>/dev/null || echo '?')"
        note "源码树版本    : ${source_branch}@${source_rev}"
        [[ "$source_branch" == "dev" ]] || \
            warn "源码树不在 dev 分支（当前 ${source_branch}）；本配方与 endless-frontier 文档以 dev 为准"
    fi
fi

# ---------------------------------------------------------------------------
# 3. 配置生效性 + 权重分片完整性 + NEXTN 前提检查（不加载权重，很快）
# ---------------------------------------------------------------------------
python3 - "$RUNTIME_CONFIG" "$RUNTIME_MODEL_PATH" "$CONTEXT_LENGTH" "$SPECULATIVE" <<'PY' || die "模型配置/权重校验失败"
import json
import os
import sys

config_path, model_dir, requested, speculative = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
)

with open(config_path, encoding="utf-8") as fh:
    config = json.load(fh)

if config.get("model_type") != "qwen4_exp":
    raise SystemExit(f"unexpected model_type={config.get('model_type')!r}")

rope = (config.get("text_config") or {}).get("rope_parameters") or {}
if rope.get("rope_type") != "yarn" or float(rope.get("factor", 0)) != 4.0:
    raise SystemExit(f"YaRN 未生效：{rope!r}")

native = (config.get("text_config") or {}).get("max_position_embeddings")
if int(rope.get("original_max_position_embeddings", 0)) != int(native):
    raise SystemExit(
        f"original_max_position_embeddings={rope.get('original_max_position_embeddings')} "
        f"与原生 max_position_embeddings={native} 不一致"
    )

if requested > 1048576:
    raise SystemExit("本脚本最多支持 1048576 上下文（YaRN factor 4 × 262144）")

index_path = os.path.join(model_dir, "model.safetensors.index.json")
with open(index_path, encoding="utf-8") as fh:
    weight_map = json.load(fh)["weight_map"]

shards = sorted(set(weight_map.values()))
missing = [s for s in shards if not os.path.exists(os.path.join(model_dir, s))]
if missing:
    listed = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
    raise SystemExit(f"权重分片缺失 {len(missing)}/{len(shards)}：{listed}")

has_mtp = any(name.startswith("mtp.") for name in weight_map)
if speculative and not has_mtp:
    raise SystemExit("开启 NEXTN 但 checkpoint 里没有 mtp.* 权重；请设 QWEN38_SPECULATIVE=0")

print(
    "模型校验通过：qwen4_exp / 原生上下文 %s / YaRN factor=%s / 请求上下文 %d / "
    "分片 %d 个齐全 / mtp=%s" % (native, rope["factor"], requested, len(shards), has_mtp)
)
PY

# ---------------------------------------------------------------------------
# 4. 运行时环境（CUDA 固定到系统 toolkit，避免 pip 版 nvcc 抢 PATH）
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
CUDA_HOME="${QWEN38_CUDA_HOME:-/usr/local/cuda}"
if [[ -d "$CUDA_HOME" ]]; then
    export CUDA_HOME CUDACXX="${QWEN38_CUDACXX:-${CUDA_HOME}/bin/nvcc}"
    export PATH="${CUDA_HOME}/bin${PATH:+:$PATH}"
else
    warn "未找到 ${CUDA_HOME}，沿用镜像现有 CUDA 环境（TileLang JIT 可能需要 nvcc）"
fi
if [[ -n "$SGLANG_SOURCE" ]]; then
    export PYTHONPATH="$SGLANG_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# ---------------------------------------------------------------------------
# 5. 运行时校验：GPU / SGLang 来源与版本 / sglang-kernel / flashinfer
#    以及「SGLang 实际会选哪个 attention backend」的预测——这决定 engine.py 里
#    flashinfer_python >= 0.6.18 的启动断言会不会被触发（详见 README「版本边界」）
# ---------------------------------------------------------------------------
if [[ "$SKIP_CHECKS" != "1" ]]; then
    EXPECTED_GPUS="$(printf '%s' "$GPU_LIST" | tr ',' '\n' | grep -c '[^[:space:]]')"
    SGLANG_SOURCE="$SGLANG_SOURCE" EXPECTED_GPUS="$EXPECTED_GPUS" \
    ATTENTION_BACKEND="$ATTENTION_BACKEND" SPECULATIVE="$SPECULATIVE" \
    RUNTIME_CONFIG="$RUNTIME_CONFIG" python3 - <<'PY' || \
        die "运行时环境校验失败（确认无误时可用 QWEN38_SKIP_CHECKS=1 跳过）"
import importlib
import json
import os
import re
import sys


def num3(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version or "")
    return tuple(int(part) for part in match.groups()) if match else None


def dist_version(*names):
    import importlib.metadata as metadata

    for name in names:
        try:
            return metadata.version(name)
        except Exception:  # noqa: BLE001
            continue
    return None


expected = int(os.environ["EXPECTED_GPUS"])
source = os.environ.get("SGLANG_SOURCE", "")
requested_backend = os.environ.get("ATTENTION_BACKEND", "auto")
speculative = os.environ.get("SPECULATIVE", "1") == "1"

try:
    import torch
except Exception as exc:  # noqa: BLE001
    raise SystemExit("无法 import torch：%s" % exc)

if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用（torch=%s, torch.cuda.is_available()=False）" % torch.__version__)
gpus = torch.cuda.device_count()
if gpus != expected:
    raise SystemExit("期望 %d 张可见 GPU，实际 %d 张；检查 CUDA_VISIBLE_DEVICES" % (expected, gpus))
caps = {torch.cuda.get_device_capability(i) for i in range(gpus)}
if caps != {(9, 0)}:
    print("[check] 警告：GPU 算力为 %s，本配方按 H100/H200(sm90) 调参" % sorted(caps), file=sys.stderr)

import sglang  # noqa: E402

version = getattr(sglang, "__version__", "unknown")
parsed = num3(version)
if parsed is None:
    print("[check] 警告：sglang 版本 %r 无法解析，跳过版本比较" % version, file=sys.stderr)
elif parsed < (0, 5, 18):
    raise SystemExit("需要 SGLang >= 0.5.18（Qwen4-Exp 支持），当前 %s" % version)

if source:
    loaded = os.path.realpath(sglang.__file__)
    wanted = os.path.realpath(os.path.join(source, "python"))
    if not loaded.startswith(wanted + os.sep):
        raise SystemExit(
            "实际加载的 sglang 位于 %s，不在 PYTHONPATH 指定的 %s 下；"
            "DLC launcher 会覆盖 PYTHONPATH（见 DEPLOYMENT_PRACTICE §2.3）" % (loaded, wanted)
        )

importlib.import_module("sglang.srt.models.qwen4_exp")

kernel_version = dist_version("sglang-kernel")
kernel_parsed = num3(kernel_version) if kernel_version else None
if kernel_parsed is None:
    print("[check] 警告：无法确定 sglang-kernel 版本，要求 >= 0.4.7", file=sys.stderr)
elif kernel_parsed < (0, 4, 7):
    raise SystemExit(
        "sglang-kernel=%s 过旧（Decode 的 PLE conv state 会失败）；"
        "请 pip install -U 'sglang-kernel==0.4.7'" % kernel_version
    )

flashinfer_version = dist_version("flashinfer_python", "flashinfer-python")

# --- 预测 SGLang 会选哪个 attention backend（复刻 get_default_attn_backend）---
with open(os.environ["RUNTIME_CONFIG"], encoding="utf-8") as fh:
    hf_config = json.load(fh)
text_config = hf_config.get("text_config") or {}
# QSA 压缩注意力（text_config.indexer_n_heads）会把 page_size 固定成 64
has_qsa = text_config.get("indexer_n_heads") is not None
page_size = 64 if has_qsa else 1
eagle_topk = 1 if speculative else None
hopper_fa3_default = (
    any(cap[0] == 9 for cap in caps)
    and tuple(map(int, (torch.version.cuda or "0.0").split(".")[:2])) >= (12, 3)
)
if requested_backend and requested_backend != "auto":
    predicted_backend = requested_backend
elif hopper_fa3_default and (eagle_topk is None or (eagle_topk == 1 and page_size in (1, None))):
    predicted_backend = "fa3"
else:
    predicted_backend = "flashinfer"

skip_kernel_check = (
    os.environ.get("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "").strip().lower() in ("true", "1")
)
fi_parsed = num3(flashinfer_version) if flashinfer_version else None
hint = (
    "两种修法：\n"
    "  A. 用配套镜像（含 flashinfer 0.6.18）：\n"
    "     dptech-sh-pai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/dptech-namespace/"
    "sglang:sglang-0-5-18-qwen38-next-flash-h100-1m\n"
    "     或 pip install -U --no-deps flashinfer_python==0.6.18 "
    "flashinfer-cubin==0.6.18 flashinfer-jit-cache==0.6.18\n"
    "  B. 显式换后端：QWEN38_ATTENTION_BACKEND=fa3（生产 PD 就是这么跑的）\n"
    "  逃生门：SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 会同时跳过 flashinfer 和 "
    "sglang-kernel 两个断言，仅排障时用。"
)
if "flashinfer" in predicted_backend and not skip_kernel_check:
    if fi_parsed is not None and fi_parsed < (0, 6, 18):
        raise SystemExit(
            "attention backend 解析为 %s，而 engine.py 对该后端要求 flashinfer_python>=0.6.18，"
            "当前是 %s。\n%s" % (predicted_backend, flashinfer_version, hint)
        )
elif "flashinfer" in predicted_backend and skip_kernel_check:
    print(
        "[check] 提示：SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1，"
        "跳过 flashinfer>=0.6.18 断言（flashinfer-python=%s）" % (flashinfer_version or "unknown"),
        file=sys.stderr,
    )

for extra_name in ("flashinfer-cubin", "flashinfer-jit-cache"):
    extra_version = dist_version(extra_name)
    if extra_version and num3(extra_version) != fi_parsed:
        print(
            "[check] 警告：%s=%s 与 flashinfer-python=%s 版本不一致，建议一并对齐"
            % (extra_name, extra_version, flashinfer_version),
            file=sys.stderr,
        )

print(
    "运行时校验通过：sglang=%s（%s）/ torch=%s / GPU=%d×%s / "
    "sglang-kernel=%s / flashinfer-python=%s / 预测 attention backend=%s"
    % (version, os.path.dirname(os.path.dirname(os.path.abspath(sglang.__file__))),
       torch.__version__, gpus, sorted(caps), kernel_version, flashinfer_version,
       predicted_backend)
)
PY
fi

# ---------------------------------------------------------------------------
# 6. 启动参数
# ---------------------------------------------------------------------------
ARGS=(
    --model-path "$RUNTIME_MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --tp "$TP_SIZE"
    --nnodes 1
    --node-rank 0
    --trust-remote-code
    --host "$HOST"
    --port "$PORT"
    --context-length "$CONTEXT_LENGTH"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE"
    --chunked-prefill-size 8192
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --linear-attn-prefill-backend flashinfer
    --linear-attn-decode-backend flashinfer
    --linear-attn-verify-backend triton
    --mamba-ssm-dtype bfloat16
    --mamba-radix-cache-strategy extra_buffer
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
)

# QWEN38_ATTENTION_BACKEND=auto 时不传该参数，交给 SGLang 自选；
# 显式指定（fa3/flashinfer/triton）时下传，可绕开 flashinfer>=0.6.18 断言。
if [[ "$ATTENTION_BACKEND" != "auto" ]]; then
    ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi

if [[ "$SPECULATIVE" == "1" ]]; then
    ARGS+=(
        --speculative-algorithm NEXTN
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
    )
fi

if [[ -n "${QWEN38_API_KEY:-}" ]]; then
    ARGS+=(--api-key "$QWEN38_API_KEY")
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
    printf '校验通过。启动命令：\n  '
    printf '%q ' python3 -m sglang.launch_server "${ARGS[@]}"
    printf '\n'
    exit 0
fi

cat >&2 <<EOF
[deploy] 启动 SGLang（TP${TP_SIZE}, context=${CONTEXT_LENGTH}, NEXTN=${SPECULATIVE}, attention-backend=${ATTENTION_BACKEND}）
[deploy] 就绪判断：curl -s http://127.0.0.1:${PORT}/health   # 返回 200
[deploy] 试一下：  curl -s -X POST http://127.0.0.1:${PORT}/v1/chat/completions \\
[deploy]            -H 'Content-Type: application/json' \\
[deploy]            -d '{"model":"${SERVED_MODEL_NAME}","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
[deploy] 长上下文 >262k 的 prefill 存在 QSA CUDA 崩溃缺陷（DEPLOYMENT_PRACTICE §6），
[deploy] 排障时先用 QWEN38_SPECULATIVE=0 关掉 NEXTN 再复现。
EOF

exec python3 -m sglang.launch_server "${ARGS[@]}"
