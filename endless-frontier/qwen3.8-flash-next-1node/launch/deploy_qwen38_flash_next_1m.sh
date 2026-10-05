#!/usr/bin/env bash
# Qwen3.8-Flash-Next —— 单机 8 卡 H100 级 / TP8 / BF16 / 1M（YaRN 随检查点自带）
#
# 与团队脚本（endless-frontier/qwen3.8-next-flash/h100/）的三点区别，都是刻意的：
#
#   1) 源码来自镜像自身（/opt/sglang/python），不指向任何共享 CPFS 源码树；
#      共享树只作为对照，不是运行依赖。
#   2) 不改写任何共享模型文件。该检查点自己的 text_config 已带 YaRN（factor=4.0），
#      因此只需允许超过"推导上下文"，不需要像团队脚本那样把 config.json 就地重写
#      （他们在共享目录里留下 config.json.native.bak，正是他们自己实践笔记里说的
#      陈旧文件句柄风险的来源）。
#   3) 参数写在这里，可复核；团队脚本的其余校验（分片完整性、mtp、backend 预测）
#      在我们这次运行里由作业日志单独记录。
#
# 用法：
#   bash deploy_qwen38_flash_next_1m.sh            # 前台启动
#   QWEN38_CHECK_ONLY=1 bash deploy_qwen38_flash_next_1m.sh   # 只打印将执行的命令
set -Eeuo pipefail

MODEL_PATH="${QWEN38_MODEL_PATH:-/mnt/data/public_data/public_model/Qwen3.8/Qwen3.8-Flash-Next-1M}"
SGLANG_SOURCE="${QWEN38_SGLANG_SOURCE:-/opt/sglang/python}"
SERVED_MODEL_NAME="${QWEN38_SERVED_MODEL_NAME:-qwen38-flash-next-1m}"
HOST="${QWEN38_HOST:-0.0.0.0}"
PORT="${QWEN38_PORT:-40000}"
TP_SIZE="${QWEN38_TP_SIZE:-8}"
CONTEXT_LENGTH="${QWEN38_CONTEXT_LENGTH:-1048576}"
MEM_FRACTION_STATIC="${QWEN38_MEM_FRACTION_STATIC:-0.90}"
CUDA_GRAPH_MAX_BS_DECODE="${QWEN38_CUDA_GRAPH_MAX_BS_DECODE:-32}"
MAX_RUNNING_REQUESTS="${QWEN38_MAX_RUNNING_REQUESTS:-96}"
SPECULATIVE="${QWEN38_SPECULATIVE:-1}"
CHECK_ONLY="${QWEN38_CHECK_ONLY:-0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SGLANG_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

# 该检查点的 YaRN 配置在 text_config 里，SGLang 推导出的上下文是原生 262144，
# 不放行就无法请求 1024k —— 团队脚本同样设置了这一条。
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tp-size "$TP_SIZE"
  --trust-remote-code
  --context-length "$CONTEXT_LENGTH"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
)

if [[ "$SPECULATIVE" == "1" ]]; then
  ARGS+=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'ERROR: 模型目录不存在：%s\n' "$MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$SGLANG_SOURCE/sglang" ]]; then
  printf 'WARN: 源码目录里没有 sglang 包：%s\n' "$SGLANG_SOURCE" >&2
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  printf '模型：%s\n' "$MODEL_PATH"
  printf '源码：%s\n' "$SGLANG_SOURCE"
  printf '将执行：\n'
  printf 'python3 -m sglang.launch_server'
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

exec python3 -m sglang.launch_server "${ARGS[@]}"
