#!/usr/bin/env bash
# GLM-5.3-Flash 单机 8 卡启动脚本（H100/H200 级别，sm90）
#
# 用到的参数来自 SGLang 官方 cookbook（docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx）：
# 该模型是 320B 总参 / 18B 激活的 FP8 MoE，混合注意力（MLA + DSA + KDA）+ mHC + MTP。
# H100/H200 上官方推荐的组合是「BF16 KV cache + TileLang DSA」；FP8 KV + TRT-LLM DSA 只适用于 Blackwell。
#
# 用法：
#   bash deploy_glm53_flash_1m.sh --check-only     # 只做校验并打印将要执行的命令
#   bash deploy_glm53_flash_1m.sh                  # 前台启动（建议 nohup/setsid 托管）
#
# 环境变量（全部可选）：
#   GLM53_MODEL_PATH        模型目录，默认 /mnt/data/public_data/public_model/GLM5.3/GLM-5.3-Flash
#   GLM53_SGLANG_SOURCE     源码树根目录（需含 python/sglang/srt/models/glm5_next.py）
#                           镜像自带 SGLang 早于 GLM-5.3，若不指定则必须依赖镜像内已覆盖过的源码树
#   GLM53_SERVED_NAME       对外模型名，默认 glm-5.3-flash
#   GLM53_HOST / GLM53_PORT 监听地址，默认 0.0.0.0 / 8000
#   GLM53_TP_SIZE           TP，默认 8
#   GLM53_MEM_FRACTION      mem-fraction-static，默认 0.78（显存紧张时降到 0.70）
#   GLM53_SPECULATIVE       1=开 MTP（默认，5-1-6），0=High Throughput（不带 --speculative-*）
#   GLM53_MAX_RUNNING_REQUESTS 默认 64
#   GLM53_ATTENTION_BACKEND 默认 dsa（TileLang DSA）；只有排障时才考虑换成别的
#   GLM53_API_KEY           设置后启用 --api-key

set -euo pipefail

MODEL_PATH="${GLM53_MODEL_PATH:-/mnt/data/public_data/public_model/GLM5.3/GLM-5.3-Flash}"
SGLANG_SOURCE="${GLM53_SGLANG_SOURCE:-}"
SERVED_NAME="${GLM53_SERVED_NAME:-glm-5.3-flash}"
HOST="${GLM53_HOST:-0.0.0.0}"
PORT="${GLM53_PORT:-8000}"
TP_SIZE="${GLM53_TP_SIZE:-8}"
MEM_FRACTION="${GLM53_MEM_FRACTION:-0.78}"
SPECULATIVE="${GLM53_SPECULATIVE:-1}"
MAX_RUNNING_REQUESTS="${GLM53_MAX_RUNNING_REQUESTS:-64}"
ATTENTION_BACKEND="${GLM53_ATTENTION_BACKEND:-dsa}"
API_KEY="${GLM53_API_KEY:-}"
CHECK_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=1 ;;
    -h|--help)
      sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done

die() { echo "错误：$*" >&2; exit 1; }
note() { echo "[glm53] $*"; }

[[ -d "$MODEL_PATH" ]] || die "模型目录不存在：$MODEL_PATH"
[[ -f "$MODEL_PATH/config.json" ]] || die "缺少 $MODEL_PATH/config.json"
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || die "缺少 $MODEL_PATH/model.safetensors.index.json"

# 源码树：GLM-5.3 需要 glm5_next 实现，镜像自带的 SGLang 0.5.18 没有它。
if [[ -n "$SGLANG_SOURCE" ]]; then
  [[ -f "$SGLANG_SOURCE/python/sglang/srt/models/glm5_next.py" ]] || \
    die "GLM53_SGLANG_SOURCE 下缺少 glm5_next 实现：$SGLANG_SOURCE"
fi

# 运行时校验：sglang 实际解析到哪个源码树，决定了模型能否加载。
RUNTIME_TREE="$(python3 - <<'PY' 2>/dev/null || true
import sglang
print(sglang.__file__)
PY
)"
note "模型目录      : $MODEL_PATH"
note "运行时 SGLang : ${RUNTIME_TREE:-未知}"

python3 - "$MODEL_PATH" "$SPECULATIVE" <<'PY' || die "模型配置校验失败"
import json, os, sys
model_dir, speculative = sys.argv[1], sys.argv[2] == "1"
cfg = json.load(open(os.path.join(model_dir, "config.json")))
arch = (cfg.get("architectures") or [""])[0]
tc = cfg.get("text_config") or {}
if arch != "Glm5NextForConditionalGeneration":
    sys.exit(f"架构不是 Glm5NextForConditionalGeneration：{arch}")
if not os.path.exists(os.path.join(model_dir, "model.safetensors.index.json")):
    sys.exit("缺少分片索引")
shards = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
mtp = int(tc.get("num_nextn_predict_layers") or 0)
print(f"模型校验通过：{arch} / layers={tc.get('num_hidden_layers')} / "
      f"max_position_embeddings={cfg.get('max_position_embeddings')} / 分片 {len(shards)} 个 / mtp={bool(mtp)}")
if speculative and not mtp:
    sys.exit("要求开启 MTP，但检查点没有 num_nextn_predict_layers")
PY

CMD=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host "$HOST" --port "$PORT"
  --tp-size "$TP_SIZE"
  --attention-backend "$ATTENTION_BACKEND"
  --dsa-prefill-backend tilelang
  --dsa-decode-backend tilelang
  --linear-attn-backend triton
  --kv-cache-dtype bfloat16          # H100/H200：BF16 KV + TileLang DSA 是官方推荐组合
  --quantization fp8                 # 检查点是 FP8 e4m3
  --moe-runner-backend triton
  --mem-fraction-static "$MEM_FRACTION"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --chunked-prefill-size 8192
  --max-prefill-tokens 8192
  --disable-prefill-cuda-graph
  --reasoning-parser glm45
  --tool-call-parser glm47
  --trust-remote-code
)

if [[ "$SPECULATIVE" == "1" ]]; then
  CMD+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 5
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 6
  )
fi

[[ -n "$API_KEY" ]] && CMD+=(--api-key "$API_KEY")

if [[ "$CHECK_ONLY" == "1" ]]; then
  note "校验通过。启动命令："
  printf '  %q' "${CMD[@]}"; echo
  exit 0
fi

# 注意：本检查点默认开启 thinking，回答在 content，思考过程在 reasoning_content；
# 调用方要给 max_tokens 留出思考的空间，否则会以 finish_reason=length 结束。
if [[ -n "$SGLANG_SOURCE" ]]; then
  export PYTHONPATH="$SGLANG_SOURCE/python:${PYTHONPATH:-}"
fi

note "启动中：${CMD[0]} ${CMD[1]} ... （模型权重约 306 GB，8 卡加载通常 4~6 分钟）"
exec "${CMD[@]}"
