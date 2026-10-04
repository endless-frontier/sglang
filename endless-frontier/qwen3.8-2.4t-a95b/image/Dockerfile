# Qwen3.8-2.4T-A95B serving image
#
# Derived 2026-10-03 from the proven GLM-5.3-flash image recipe (ef-pai-serving/glm-5.3-flash/Dockerfile),
# which is the authority for *why* each piece is here — base choice, CUDA devel toolkit, the two
# libcuda symlinks, python3-dev, and uv-instead-of-pip. The differences are exactly two:
#
#   1. the SGLang pin: sgl-project/sglang **v0.5.21** (2026-10-02) instead of the team's fork at
#      d6221be. Both implement this model's architecture, `Qwen3_5MoeForCausalLM` (upstream since at
#      least v0.5.18), so this is a deliberate pin choice — v0.5.21 is four release trains newer than
#      the team image's 0.5.18 and carries later fixes, including the 2026-10-02 EAGLE3/DFLASH
#      aux-layer fix. It also declares the same torch (2.13.0) and flashinfer (0.6.18) our existing
#      image already carries, so the rest of the recipe transfers without change.
#   2. the self-check at the end: it asserts this model's module and entry class are present, since a
#      pin older than the architecture is the one failure mode that produces a silently wrong image.
#
# The model is served from the platform (EAS), not built in: this image carries the runtime only.

FROM public.ecr.aws/ubuntu/ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.12.5
ARG SGLANG_REPO=sgl-project/sglang
ARG SGLANG_COMMIT=v0.5.21

# --- operating system and CUDA toolkit (same reasoning as the GLM image) -----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git python3 python3-dev python3-pip python3-venv build-essential \
        libibverbs-dev librdmacm-dev rdma-core libnuma-dev \
        ffmpeg libsm6 libxext6 \
    && curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb \
    && apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-13-0 cuda-compat-13-0 \
    && ln -sf /usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
    && ln -sf /usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64 \
    LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs \
    PYTHONUNBUFFERED=1

# --- uv (HTTP/2; pip's HTTP/1.1 path is ~200x slower in this environment) ----------------------
RUN curl -fsSL -o /tmp/uv.tgz \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    && tar -xzf /tmp/uv.tgz -C /usr/local/bin --strip-components=1 \
        uv-x86_64-unknown-linux-gnu/uv \
    && rm -f /tmp/uv.tgz && uv --version

# --- Python stack, same pins the validated GLM run carried -------------------------------------
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

RUN uv pip install --system --break-system-packages --no-cache \
        --find-links https://mirrors.aliyun.com/pytorch-wheels/cu130/ \
        torch==2.13.0+cu130 torchvision==0.28.0+cu130 torchaudio==2.11.0+cu130

# --- the SGLang source, pinned to the release we chose -----------------------------------------
RUN curl -fsSL -o /tmp/sglang.tar.gz \
        "https://codeload.github.com/${SGLANG_REPO}/tar.gz/${SGLANG_COMMIT}" \
    && mkdir -p /opt/sglang \
    && tar -xzf /tmp/sglang.tar.gz -C /opt/sglang --strip-components=1 \
    && rm -f /tmp/sglang.tar.gz

# --- its declared runtime dependencies (sglang's own pyproject is the authority) ---------------
RUN python3 -c "import tomllib; deps = tomllib.load(open('/opt/sglang/python/pyproject.toml','rb'))['project']['dependencies']; open('/tmp/requirements.txt','w').write(chr(10).join(deps))" \
    && echo "declared dependencies: $(grep -c . /tmp/requirements.txt)" \
    && grep -E '^(torch|flashinfer|sglang-kernel)' /tmp/requirements.txt

RUN cd /tmp && uv pip install --system --break-system-packages --no-cache \
        --find-links https://mirrors.aliyun.com/pytorch-wheels/cu130/ \
        -r /tmp/requirements.txt

ENV PYTHONPATH=/opt/sglang/python
WORKDIR /opt

# --- the image proves itself -------------------------------------------------------------------
# The architecture check is the load-bearing one for this model: a pin without Qwen3_5MoeForCausalLM
# fails at load time, not build time, unless it is asserted here. The toolchain checks are the two
# whose absence cost earlier jobs (Triton's runtime C build needs Python.h and -l:libcuda.so.1).
RUN python3 -c "import sglang, torch, pathlib; \
print('sglang', sglang.__version__); \
print('torch', torch.__version__); \
p = pathlib.Path(sglang.__file__).parent / 'srt/models/qwen3_5_text.py'; \
assert p.exists(), 'qwen3_5_text.py missing: this pin predates the model'; \
src = p.read_text(); \
assert 'class Qwen3_5MoeForCausalLM' in src and 'EntryClass = [Qwen3_5MoeForCausalLM' in src, \
    'Qwen3_5MoeForCausalLM not registered in this pin'; \
print('arch check: Qwen3_5MoeForCausalLM present and registered')" \
 && nvcc --version | tail -1 \
 && PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
 && printf '#include <Python.h>\nint main(void){return 0;}\n' > /tmp/headers.c \
 && gcc -fPIC -c /tmp/headers.c -o /tmp/headers.o -I"$PYINC" \
 && echo "toolchain check: Python.h resolves at $PYINC" \
 && printf 'int main(void){return 0;}\n' > /tmp/linkcheck.c \
 && gcc /tmp/linkcheck.c -l:libcuda.so.1 -o /tmp/linkcheck \
 && echo 'toolchain check: -l:libcuda.so.1 resolves' \
 && rm -f /tmp/headers.c /tmp/headers.o /tmp/linkcheck.c /tmp/linkcheck
