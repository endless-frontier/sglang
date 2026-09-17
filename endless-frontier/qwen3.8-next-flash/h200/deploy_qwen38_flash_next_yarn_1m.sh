#!/usr/bin/env bash
set -Eeuo pipefail

# Qwen3.8-Flash-Next, single node / 8 GPUs / BF16 / YaRN 1M.
#
# Image used today (H200):
#   pai-ai-prod-acr-registry.cn-shanghai.cr.aliyuncs.com/acr_namespace/scimaster:sglang-0-5-18-cuda13-qwen38-next-pd
#   CUDA 13.0 devel + PyTorch cu130 + SGLang 0.5.18 + sglang-kernel 0.4.7 +
#   flashinfer 0.6.17
#
# Attention backend: this image ships flashinfer 0.6.17, while engine.py
# asserts flashinfer_python >= 0.6.18 whenever flashinfer is the attention
# backend.  For Qwen4-Exp the implicit choice IS flashinfer (QSA pins
# page_size=64, so the Hopper fa3 auto-selection does not apply), so we pass
# --attention-backend fa3 explicitly - exactly what the production PD workers
# do.  QWEN38_ATTENTION_BACKEND=auto restores SGLang's own choice;
# =flashinfer is correct once the image ships >= 0.6.18.
# Full trace: ../DEPLOYMENT_PRACTICE.md section 2.4.

MODEL_PATH="${QWEN38_MODEL_PATH:-/mnt/data/public_models/Qwen3.8-Flash-Next}"
SGLANG_SOURCE="${QWEN38_SGLANG_SOURCE:-/mnt/data/xinyu/sglang-qwen38}"
HOST="${QWEN38_HOST:-0.0.0.0}"
PORT="${QWEN38_PORT:-40000}"
GPU_LIST="${QWEN38_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONTEXT_LENGTH="${QWEN38_CONTEXT_LENGTH:-1048576}"
MEM_FRACTION_STATIC="${QWEN38_MEM_FRACTION_STATIC:-0.90}"
CUDA_GRAPH_MAX_BS_DECODE="${QWEN38_CUDA_GRAPH_MAX_BS_DECODE:-32}"
ATTENTION_BACKEND="${QWEN38_ATTENTION_BACKEND:-fa3}"
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
  QWEN38_ATTENTION_BACKEND=fa3|flashinfer|auto
                                         attention backend (default fa3; the
                                         PD image ships flashinfer 0.6.17 and
                                         engine.py wants >= 0.6.18 when
                                         flashinfer is selected)

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

[[ -d "$MODEL_PATH" ]] || die "model directory not found: $MODEL_PATH"
[[ -f "$CONFIG_PATH" ]] || die "missing model config: $CONFIG_PATH"
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || \
    die "missing model index: $MODEL_PATH/model.safetensors.index.json"
[[ -f "$MODEL_PATH/tokenizer.json" ]] || die "missing tokenizer: $MODEL_PATH/tokenizer.json"
[[ -f "$SGLANG_SOURCE/python/sglang/srt/models/qwen4_exp.py" ]] || \
    die "Qwen3.8-compatible SGLang source is missing: $SGLANG_SOURCE"

# YaRN lives in text_config.rope_parameters.  Keep exactly one config file:
# if the directory already carries the 1M config (YaRN factor 4.0) it is left
# untouched; otherwise the native file is copied to config.json.native.bak and
# the config is rewritten atomically (a container interruption cannot leave
# invalid JSON).  Rewriting unconditionally is what makes concurrent readers
# on a NAS fail with ESTALE, so the detection matters.
python3 - "$CONFIG_PATH" <<'PY'
import json
import os
import shutil
import sys

path = sys.argv[1]
model_dir = os.path.dirname(path)
with open(path, encoding="utf-8") as f:
    config = json.load(f)

text_config = config.setdefault("text_config", {})
native = text_config.get("max_position_embeddings")
if native != 262144:
    raise SystemExit(
        f"unexpected native max_position_embeddings={native!r}; expected 262144"
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
            and int((rope or {}).get("original_max_position_embeddings", 0) or 0)
            == int(native)
        )
    except (TypeError, ValueError):
        return False


if is_1m(text_config.get("rope_parameters")) and config.get("max_position_embeddings") == native:
    print("yarn: already the 1M config, left untouched: " + path)
    raise SystemExit(0)

backup_path = os.path.join(model_dir, "config.json.native.bak")
if not os.path.exists(backup_path):
    shutil.copy2(path, backup_path)
    print("yarn: native config backed up to " + backup_path)

# Transformers 5 validates the outer multimodal config as well as its nested
# text config.  The checkpoint only stores this field under text_config, so
# mirror the native value at the top level for that validator.
config["max_position_embeddings"] = native
text_config["rope_parameters"] = target_rope

tmp = path + ".yarn.tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")
os.replace(tmp, path)
print("yarn: rewrote " + path + " to the 1M config")
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

    CONFIG_PATH="$CONFIG_PATH" ATTENTION_BACKEND="$ATTENTION_BACKEND" python3 - <<'PY'
import importlib
import importlib.metadata as metadata
import json
import os
import re
import torch
import sglang


def num3(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version or "")
    return tuple(int(part) for part in match.groups()) if match else None


if not sglang.__version__.startswith("0.5.18"):
    raise SystemExit(f"expected SGLang 0.5.18, found {sglang.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
count = torch.cuda.device_count()
if count != 8:
    raise SystemExit(f"expected 8 visible GPUs, found {count}")
importlib.import_module("sglang.srt.models.qwen4_exp")

# Predict the attention backend SGLang will use: engine.py requires
# flashinfer_python >= 0.6.18 only when flashinfer is that backend.
requested_backend = os.environ.get("ATTENTION_BACKEND", "auto")
with open(os.environ["CONFIG_PATH"], encoding="utf-8") as fh:
    text_config = json.load(fh).get("text_config") or {}
has_qsa = text_config.get("indexer_n_heads") is not None
page_size = 64 if has_qsa else 1
caps = [torch.cuda.get_device_capability(i) for i in range(count)]
hopper = (
    any(cap[0] == 9 for cap in caps)
    and tuple(map(int, (torch.version.cuda or "0.0").split(".")[:2])) >= (12, 3)
)
if requested_backend != "auto":
    backend = requested_backend
elif hopper and page_size in (1, None):   # NEXTN is always on here (eagle topk 1)
    backend = "fa3"
else:
    backend = "flashinfer"

flashinfer_version = None
for name in ("flashinfer_python", "flashinfer-python"):
    try:
        flashinfer_version = metadata.version(name)
        break
    except Exception:
        continue
skip_assert = os.environ.get(
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", ""
).strip().lower() in ("true", "1")
if "flashinfer" in backend and not skip_assert:
    parsed = num3(flashinfer_version)
    if parsed is not None and parsed < (0, 6, 18):
        raise SystemExit(
            "attention backend resolves to %s, but engine.py requires "
            "flashinfer_python>=0.6.18 for that backend (image has %s).\n"
            "Fix: QWEN38_ATTENTION_BACKEND=fa3 (what production PD does), or "
            "upgrade flashinfer_python/-cubin/-jit-cache to 0.6.18, or set "
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 as an escape hatch."
            % (backend, flashinfer_version)
        )
print(
    "runtime: sglang=" + sglang.__version__
    + ", visible_gpus=" + str(count)
    + ", capability=" + str(caps)
    + ", attention_backend=" + backend
    + ", flashinfer=" + str(flashinfer_version)
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

# "auto" = let SGLang pick (only safe when flashinfer >= 0.6.18 is installed).
if [[ "$ATTENTION_BACKEND" != "auto" ]]; then
    ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi

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
