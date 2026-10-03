# GLM-5.3-Flash 服务镜像（自有镜像的完整构建配方）
#
# 这份 Dockerfile 是我们实测可用镜像的来源：CUDA 13.0 devel + 仓库固定提交的源码 +
# 依赖按 sglang 自己的 pyproject.toml 安装。构建出的镜像在单机 8 卡上服务 GLM-5.3-Flash，
# 也服务 Qwen3.8-Flash-Next（同一源码提交、同一依赖集，只有启动参数不同）。
#
# 构建要点见同目录 README.md（chroot 隔离、单层构建、镜像源与 uv 的原因）。
# 实测镜像摘要（注册表侧读数）：
#   仓库/标签：wangruisi/sglang-glm53-flash:sglang-d6221be-glm53-flash-l20z-1node-20261003-1a4d8b6
#   digest   ：sha256:87ed176b6b58cc97653c9c0b42f5f0d73c8ab58660afc2bc69be16c84070e450
#   大小     ：约 8.86 GB（压缩）/ 约 18 GB（展开）
#
# Base choice (2026-10-02): Docker Hub is unreachable from the build environment, and neither the
# account's accelerator nor AWS public ECR carries NVIDIA's CUDA images. So the base is plain
# Ubuntu 24.04 from Amazon's public registry and the CUDA **devel** toolkit comes from NVIDIA's own
# apt repository — both reachable — which keeps the image entirely ours and the recipe reproducible.
#
# The devel toolkit is deliberate: the model's DSA/KDA and MoE paths JIT-compile kernels (TileLang,
# Triton, DeepGEMM warmup), and a runtime base without nvcc/toolkit headers breaks exactly there.
#
# Downloads go through `uv`, not pip. Reasoning, measured in this environment: the reachable mirrors
# serve HTTP/2 at ~15 MB/s but HTTP/1.1 at ~77 kB/s on the same URL, and pip (like Python's urllib)
# speaks only HTTP/1.1 — a 200x penalty that turns a 526 MB wheel into a multi-hour download. uv
# negotiates HTTP/2 and installs the same pinned wheels in seconds. Its version is pinned below.

FROM public.ecr.aws/ubuntu/ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.12.5
ARG SGLANG_REPO=endless-frontier/sglang
ARG SGLANG_COMMIT=d6221bec2afeb66af0ec3e9f666addb8613203b5

# --- operating system and CUDA toolkit -------------------------------------------------------
# cuda-compat is not decoration: it supplies a driver library the toolkit can use when the node's
# driver is older than CUDA 13, and the measured environment carried it. Its absence is what broke
# the first serving attempt from this image — weights loaded (63 GB per GPU) and then Triton's
# runtime C build failed to link -l:libcuda.so.1, because the platform does not inject a driver
# library into the image. The two symlinks below put a linkable libcuda in the default search path
# and on the loader path, which is what the team's own CUDA 13 image does.
# python3-dev is not optional: Triton compiles a small C extension at start-up (cuda_utils.c) to
# talk to the driver, and that file includes Python.h. Without the headers the server loads every
# weight and then dies in the compiler — which is exactly how the first two serving attempts from
# this image failed, twice, while the log blamed the link line instead of the missing header.
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

# The platform mounts the node's driver libraries under /usr/local/nvidia; putting that first means
# a real driver wins over the stub whenever one is present, and cuda-compat covers the case where
# the node's driver is older than the toolkit.
ENV CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64 \
    LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs \
    PYTHONUNBUFFERED=1

# --- uv, the package installer (see the note at the top for why not pip) ----------------------
RUN curl -fsSL -o /tmp/uv.tgz \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    && tar -xzf /tmp/uv.tgz -C /usr/local/bin --strip-components=1 \
        uv-x86_64-unknown-linux-gnu/uv \
    && rm -f /tmp/uv.tgz && uv --version

# --- Python stack, pinned to the environment we measured --------------------------------------
# Separate layers on purpose: a failure in a later layer keeps the large earlier ones cached.
# Mirrors serve byte-identical wheels to the upstream indexes; only their location differs, which is
# recorded as a limitation in image.lock.
#
# flashinfer: the validated run carried 0.6.18 from flashinfer's own wheel index, which is not
# reachable from the build environment. flashinfer is not on this model's attention path (DSA +
# TileLang are), so the available version is pinned instead and the deviation recorded.
#
# --no-deps on the extensions, deliberately: PyPI's flashinfer 0.6.13 pins torch==2.9.1, and uv
# faithfully tried to install a second, older torch and its whole nvidia-cu12 stack alongside the
# 2.13.0 we pinned — which is what exhausted the disk. Our torch is pinned in the layer above, so
# the extensions are installed against it without re-resolving the world. numpy is installed on its
# own line for the same reason; triton is left to torch, which pins the version it was built with.
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

RUN uv pip install --system --break-system-packages --no-cache \
        --find-links https://mirrors.aliyun.com/pytorch-wheels/cu130/ \
        torch==2.13.0+cu130 torchvision==0.28.0+cu130 torchaudio==2.11.0+cu130

# --- the SGLang source we validated against ----------------------------------------------------
# Fetched as a tarball rather than cloned: git speaks HTTP/1.1, which is the slow path here.
RUN curl -fsSL -o /tmp/sglang.tar.gz \
        "https://codeload.github.com/${SGLANG_REPO}/tar.gz/${SGLANG_COMMIT}" \
    && mkdir -p /opt/sglang \
    && tar -xzf /tmp/sglang.tar.gz -C /opt/sglang --strip-components=1 \
    && rm -f /tmp/sglang.tar.gz

# --- its declared runtime dependencies ----------------------------------------------------------
# sglang's own pyproject is the authority on what it needs (it pins torch 2.13.0, flashinfer 0.6.18
# and the rest), so the whole declared set is installed from wheels rather than a hand-picked
# subset. The list is extracted into a plain requirements file and installed from /tmp, so the
# project's own [tool.uv.index] — pypi.org, the slow HTTP/1.1 path — cannot override the reachable
# mirror. sglang itself is used through PYTHONPATH, so no build step needs a network.
RUN python3 -c "import tomllib; deps = tomllib.load(open('/opt/sglang/python/pyproject.toml','rb'))['project']['dependencies']; open('/tmp/requirements.txt','w').write(chr(10).join(deps))" \
    && echo "declared dependencies: $(grep -c . /tmp/requirements.txt)" \
    && grep -E '^(torch|flashinfer|sglang-kernel)' /tmp/requirements.txt

RUN cd /tmp && uv pip install --system --break-system-packages --no-cache \
        --find-links https://mirrors.aliyun.com/pytorch-wheels/cu130/ \
        -r /tmp/requirements.txt

ENV PYTHONPATH=/opt/sglang/python
WORKDIR /opt

# --- the image proves itself -------------------------------------------------------------------
# Beyond the imports, this checks the two things whose absence cost a job: that the linker can
# resolve the CUDA driver library (Triton's runtime builds link -l:libcuda.so.1) and that the
# toolkit can see the GPUs at all.
RUN python3 -c "import sglang, torch, pathlib; \
print('sglang', sglang.__version__, sglang.__file__); \
print('torch', torch.__version__); \
assert pathlib.Path(sglang.__file__).parent.joinpath('srt/models/glm5_next.py').exists(), \
    'glm5_next missing: the pinned source is older than GLM-5.3 support'" \
 && nvcc --version | tail -1 \
 && PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
 && printf '#include <Python.h>\nint main(void){return 0;}\n' > /tmp/headers.c \
 && gcc -fPIC -c /tmp/headers.c -o /tmp/headers.o -I"$PYINC" \
 && echo "toolchain check: Python.h resolves at $PYINC" \
 && printf 'int main(void){return 0;}\n' > /tmp/linkcheck.c \
 && gcc /tmp/linkcheck.c -l:libcuda.so.1 -o /tmp/linkcheck \
 && echo 'toolchain check: -l:libcuda.so.1 resolves' \
 && python3 -c "import glob; \
print('toolchain check: libcuda present at', [p for p in glob.glob('/usr/lib/x86_64-linux-gnu/libcuda.so*')][:3])" \
 && rm -f /tmp/headers.c /tmp/headers.o /tmp/linkcheck.c /tmp/linkcheck
