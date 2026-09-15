#!/usr/bin/env bash
set -Eeuo pipefail

# Qwen3.8-Flash-Next, single node / 8 GPUs / BF16 / YaRN 1M.
# The production platform supplies the SGLang 0.5.18 image and mounts /mnt/data.

MODEL_PATH="${QWEN38_MODEL_PATH:-/mnt/data/public_models/Qwen3.8-Flash-Next}"
SGLANG_SOURCE="${QWEN38_SGLANG_SOURCE:-/mnt/data/xinyu/sglang-qwen38}"
HOST="${QWEN38_HOST:-0.0.0.0}"
PORT="${QWEN38_PORT:-40000}"
GPU_LIST="${QWEN38_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONTEXT_LENGTH="${QWEN38_CONTEXT_LENGTH:-1048576}"
MEM_FRACTION_STATIC="${QWEN38_MEM_FRACTION_STATIC:-0.90}"
CUDA_GRAPH_MAX_BS_DECODE="${QWEN38_CUDA_GRAPH_MAX_BS_DECODE:-32}"
CHECK_ONLY="${QWEN38_CHECK_ONLY:-0}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  deploy_qwen38_flash_next_yarn_1m.sh          start the server in foreground
  deploy_qwen38_flash_next_yarn_1m.sh --check-only
                                               validate and print launch command

Environment overrides:
  QWEN38_MEM_FRACTION_STATIC=0.90|0.85  lower only if startup/runtime OOMs
  QWEN38_CUDA_GRAPH_MAX_BS_DECODE=16|32  reduce graph capture memory if needed
  QWEN38_CONTEXT_LENGTH=1048576          requested single-request context

The image and /mnt/data mount are configured by the production platform.
EOF
}

case "${1:-}" in
    --help|-h)
        usage
        exit 0
        ;;
    --check-only)
        CHECK_ONLY=1
        shift
        ;;
esac
[[ "$#" -eq 0 ]] || die "unexpected argument(s); use --check-only or --help"

CONFIG_PATH="${MODEL_PATH}/config.json"
BACKUP_PATH="${MODEL_PATH}/config.json.native.bak"

[[ -d "$MODEL_PATH" ]] || die "model directory not found: $MODEL_PATH"
[[ -f "$CONFIG_PATH" ]] || die "missing model config: $CONFIG_PATH"
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || \
    die "missing model index: $MODEL_PATH/model.safetensors.index.json"
[[ -f "$MODEL_PATH/tokenizer.json" ]] || die "missing tokenizer: $MODEL_PATH/tokenizer.json"
[[ -f "$SGLANG_SOURCE/python/sglang/srt/models/qwen4_exp.py" ]] || \
    die "Qwen3.8-compatible SGLang source is missing: $SGLANG_SOURCE"

# Keep a byte-for-byte native copy so the change can be reverted without
# touching any weight or tokenizer files.
if [[ ! -f "$BACKUP_PATH" ]]; then
    cp -p "$CONFIG_PATH" "$BACKUP_PATH"
fi

# Transformers/SGLang read YaRN from text_config.rope_parameters.  Replace
# the object atomically so a container interruption cannot leave invalid JSON.
python3 - "$CONFIG_PATH" <<'PY'
import json
import os
import sys

path = sys.argv[1]
tmp = path + ".yarn.tmp"
with open(path, encoding="utf-8") as f:
    config = json.load(f)

text_config = config.setdefault("text_config", {})
native = text_config.get("max_position_embeddings")
if native != 262144:
    raise SystemExit(
        f"unexpected native max_position_embeddings={native!r}; expected 262144"
    )
# Transformers 5 validates the outer multimodal config as well as its nested
# text config.  The checkpoint only stores this field under text_config, so
# mirror the native value at the top level for that validator.
config["max_position_embeddings"] = native
text_config["rope_parameters"] = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
}
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")
os.replace(tmp, path)
PY

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTHONPATH="$SGLANG_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

if [[ "${QWEN38_SKIP_CHECKS:-0}" != "1" ]]; then
    python3 - "$CONFIG_PATH" "$CONTEXT_LENGTH" <<'PY'
import json
import sys

path, requested = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as f:
    config = json.load(f)
if config.get("model_type") != "qwen4_exp":
    raise SystemExit(f"unexpected model_type={config.get('model_type')!r}")
text_config = config.get("text_config") or {}
rope = text_config.get("rope_parameters") or {}
if rope.get("rope_type") != "yarn" or float(rope.get("factor", 0)) != 4.0:
    raise SystemExit(f"YaRN configuration was not applied: {rope!r}")
if requested > 1048576:
    raise SystemExit("this script supports at most 1048576 context tokens")
print(
    "model config: qwen4_exp, native context=262144, "
    f"YaRN factor={rope['factor']}, requested context={requested}"
)
PY

    python3 - <<'PY'
import importlib
import torch
import sglang

if not sglang.__version__.startswith("0.5.18"):
    raise SystemExit(f"expected SGLang 0.5.18, found {sglang.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
count = torch.cuda.device_count()
if count != 8:
    raise SystemExit(f"expected 8 visible GPUs, found {count}")
importlib.import_module("sglang.srt.models.qwen4_exp")
print(
    "runtime: sglang=" + sglang.__version__
    + ", visible_gpus=" + str(count)
    + ", capability=" + str([torch.cuda.get_device_capability(i) for i in range(count)])
)
PY
fi

ARGS=(
    --model-path "$MODEL_PATH"
    --tp 8
    --nnodes 1
    --node-rank 0
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE"
    --chunked-prefill-size 8192
    --linear-attn-prefill-backend flashinfer
    --linear-attn-decode-backend flashinfer
    --linear-attn-verify-backend triton
    --mamba-ssm-dtype bfloat16
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --max-running-requests 96
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --trust-remote-code
    --host "$HOST"
    --port "$PORT"
    --context-length "$CONTEXT_LENGTH"
)

if [[ -n "${QWEN38_API_KEY:-}" ]]; then
    ARGS+=(--api-key "$QWEN38_API_KEY")
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
    printf 'Validation passed. Launch command:\n  '
    printf '%q ' python3 -m sglang.launch_server "${ARGS[@]}"
    printf '\n'
    exit 0
fi

exec python3 -m sglang.launch_server "${ARGS[@]}"
