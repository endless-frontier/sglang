# 手工 SSH：H200 2P + 2D（TP8）

四台机器各使用 8 GPU：两台运行 Prefill、两台运行 Decode。worker 脚本会写入 YaRN factor 4 配置并启动 SGLang；Router 脚本将请求转发到两类 worker。

## 环境

四台主机需使用相同的 CUDA 13.0 devel 镜像/环境，且包含 CUDA toolkit headers、PyTorch cu130、SGLang 0.5.18、`sglang-kernel==0.4.7`、FlashInfer、Triton。设置：

```bash
export QWEN38_SGLANG_SOURCE=/mnt/data/xinyu/sglang-qwen38-upstream-1789383617
export QWEN38_MODEL_PATH=/mnt/data/public_models/Qwen3.8-Flash-Next
```

模型目录在所有机器必须一致可读。确认 IB/RDMA、Mooncake、RoCE GID（默认 `MC_GID_INDEX=3`）可用，并放通 worker API、bootstrap 和分布式端口。

## 启动 worker

在两台 Prefill 主机分别执行（第三个参数为 API 端口，第四个为 bootstrap 端口）：

```bash
bash run_qwen38_flash_next_yarn_1m_pd_worker.sh prefill <PREFILL_IP> 41000 8998
```

在两台 Decode 主机分别执行：

```bash
bash run_qwen38_flash_next_yarn_1m_pd_worker.sh decode <DECODE_IP> 42000 8998
```

脚本默认 TP8、PP1、上下文 1,048,576、`max_total_tokens=6000000`（设置 `QWEN38_OPTIMIZED=0` 可切换到更保守的 1,200,000），并包含 `--reasoning-parser qwen3` 和 `--tool-call-parser qwen3_coder`。为避开 Qwen3.8 QSA 的 draft-prefill CUDA 非法地址问题，NEXTN/EAGLE speculative 参数只在 Decode worker 上启用，Prefill worker 使用普通路径。

## 启动 Router

编辑 `run_qwen38_flash_next_yarn_1m_pd_router.sh` 中的四个 URL，将示例 IP 换成实际内网 IP；端口必须与 worker 一致。然后在任一可访问四台机器的节点运行：

```bash
bash run_qwen38_flash_next_yarn_1m_pd_router.sh
```

Router 默认监听 `0.0.0.0:40000`。先访问 `/health`，再向 `http://<router-ip>:40000/v1/chat/completions` 发送 OpenAI 兼容请求。停止时结束 worker 和 Router 进程，并恢复模型目录中的 `config.json.native.bak`（脚本会自动保留原始配置）。

## 排查

检查 `curl http://<ip>:<port>/health`、`ss -ltnp`、`nvidia-smi` 和 `ibdev2netdev`。跨节点连接失败通常是安全组/端口或 RoCE GID；编译报错通常是 nvcc 与 headers 版本不一致，应统一 devel 镜像。
