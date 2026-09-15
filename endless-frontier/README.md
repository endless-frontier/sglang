# endless-frontier

本仓库以 `sgl-project/sglang` 的 `main` 为基线，在 `dev` 分支维护 Qwen3.8 Flash Next 的部署配方。模型权重和运行日志不提交到 Git。

## 方案选择

| 目录 | 资源 | 入口 | 适用场景 |
|---|---:|---|---|
| `qwen3.8-next-flash/aliyun_h200_2p2d` | 阿里云 DLC，4 节点 × 8 GPU | `deployment... cli.py` | 生产用 2 Prefill + 2 Decode，TP8 |
| `qwen3.8-next-flash/h200_2p2d` | 手工 SSH，4 节点 × 8 GPU | 两个 shell 脚本 | 已有固定内网节点、需要快速试验 |
| `qwen3.8-next-flash/h200` | 单节点 8 GPU | `deploy_...sh` | 单机验证或低并发 |

三套方案都将模型原生 262,144 上下文通过 YaRN factor 4 扩展到 1,048,576，并启用 `--reasoning-parser qwen3` 与 `--tool-call-parser qwen3_coder`。

## 通用环境

推荐使用包含 CUDA 13.0 **devel**（含 `nvcc` 和 toolkit headers）的 SGLang 0.5.18 镜像；runtime 镜像缺少编译扩展所需的头文件。Python、PyTorch cu130、`sglang-kernel==0.4.7`、FlashInfer、Triton 版本应与镜像锁定。所有节点必须使用同一镜像和同一份 SGLang 源码。模型目录应为 `/mnt/data/public_models/Qwen3.8-Flash-Next`，并包含 `config.json`、权重 index、分片和 tokenizer。跨节点 PD 需要 IB/RDMA、NVIDIA IBGDA/GDRCopy 及 Mooncake 可用。

不要把 AccessKey、运行 receipt、`.runtime/`、模型文件或节点 IP 提交到仓库。
