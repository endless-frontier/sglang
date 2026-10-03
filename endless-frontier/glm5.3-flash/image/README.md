# 自有镜像：构建与服务

团队既有镜像需要外挂一份含 `glm5_next` 的源码树才能服务 GLM-5.3-Flash。这份配方把它去掉：
**源码与依赖都进镜像**，作业或服务里只设 `PYTHONPATH=/opt/sglang/python` 即可。

已实测：该镜像在单机 8 卡上服务 GLM-5.3-Flash（作业内 0.3 s / 33.4 s），
也能服务 Qwen3.8-Flash-Next（0.7 s / 1.9 s），并已作为 EAS 在线服务对外提供（见 `../eas/`）。

| 项 | 值 |
|---|---|
| 仓库/标签 | `wangruisi/sglang-glm53-flash:sglang-d6221be-glm53-flash-l20z-1node-20261003-1a4d8b6` |
| digest | `sha256:87ed176b6b58cc97653c9c0b42f5f0d73c8ab58660afc2bc69be16c84070e450` |
| 大小 | 约 8.86 GB（压缩）/ 约 18 GB（展开） |
| 基础 | Ubuntu 24.04（公有 registry）+ NVIDIA 官方 apt 源装的 CUDA 13.0 **devel** |
| 源码 | 本仓库提交 `d6221bec2afeb66af0ec3e9f666addb8613203b5`，放在 `/opt/sglang/python` |
| 依赖 | 按 `python/pyproject.toml` 声明的清单安装（torch 2.13.0+cu130、flashinfer-python 0.6.18、sglang-kernel 0.4.7、tilelang 0.1.12、transformers 5.12.1 等） |
| 自检 | 构建最后一步导入 sglang/torch、断言 `glm5_next` 存在、编译含 `Python.h` 的文件、链接 `-l:libcuda.so.1` |

## 构建

```bash
# 在任一有 podman、磁盘 ≥90 GB 的实例上（无需任何凭据，Dockerfile 自己拉源码与依赖）
podman system prune -af
mkdir -p /root/build/glm53 && cp Dockerfile /root/build/glm53/
cd /root/build/glm53
podman build --isolation=chroot --layers=false --tag glm53-flash:local .
```

约十五分钟，大头是下载 torch。

## 这些参数为什么必须这么写（每条都踩过）

**`--isolation=chroot`**：沙箱化的实例跑不了 podman 默认运行时，第一个 RUN 就会
`mount /proc to '/proc': Operation not permitted`；chroot 隔离不建嵌套容器，RUN 正常执行。

**`--layers=false`**：默认每步都靠复制容器来提交，95 GB 的盘会在写层时耗尽
（`no space left on device`）。单层构建避开它；构建前先 `prune`。

**镜像源 + 不用 pip**：同一 URL 上可达的镜像站 HTTP/2 约 15 MB/s、HTTP/1.1 只有约 77 kB/s，
而 pip 与 Python 的 urllib 只会 HTTP/1.1 —— 526 MB 的 torch 轮子要下几小时。改用 **uv**
（会说 HTTP/2）后是分钟级；源码也用 codeload tarball 而不是 `git clone`（git 同样是 HTTP/1.1）。

**基础镜像为什么要自己拼**：构建环境连不上 Docker Hub，账号的镜像加速器既没有 `nvidia/cuda`
也没有 `ubuntu`，NVIDIA 的镜像又不在公有 ECR 上。于是用 Ubuntu 24.04 + NVIDIA 官方 apt 源装
CUDA。必须是 **devel**：TileLang/Triton/DeepGEMM 这些内核在启动时 JIT 编译，runtime 基础镜像会在这里崩。

**必须装 `python3-dev`**：Triton 启动时会现场编译一个 C 扩展（`cuda_utils.c`，包含 `Python.h`）。
缺头文件时的报错长这样 —— 因为那行命令以 `-l:libcuda.so.1` 结尾，极易误判成缺 CUDA 驱动：

```text
gcc ... cuda_utils.c ... -l:libcuda.so.1 ... returned non-zero exit status 1
cuda_utils.c:9:10: fatal error: Python.h: No such file or directory
```

镜像自检里加了一条“编译含 `Python.h` 的文件 + 链接 `-l:libcuda.so.1`”，这类问题在构建期就暴露，
不会再占用一张 8 卡节点。

**依赖用 sglang 自己声明的清单**：Dockerfile 从 `pyproject.toml` 提取 `dependencies` 安装，
不要手写子集。手工列的时候漏掉了 flashinfer 的 pin，解析器会安静地装进第二份更老的 torch
（`flashinfer-python` 0.6.13 依赖 `torch==2.9.1`），把磁盘吃掉。

## 关于 `cuda-compat`

镜像里装了 `cuda-compat-13-0` 并把 CUDA stub 软链进系统库目录——这是 DLC 路线（驱动挂在
`/usr/local/nvidia/...`）的前向兼容需要。但**在 EAS 上它会反过来遮住平台注入的新驱动**，
所以 EAS 的服务脚本必须重设 `LD_LIBRARY_PATH`，见 `../eas/README.md` 第 2 条。
