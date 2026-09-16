# 单机 H200：YaRN 1M（TP8）

在一台 8×H200 主机上以 TP8 启动 Qwen3.8 Flash Next，适合验证模型、YaRN 配置、
LoRA/微调权重和 OpenAI 兼容 API。没有 PD 转发，服务直接监听 `0.0.0.0:40000`。

## 要求

- CUDA 13.0 devel 镜像（含 `nvcc`/headers）、PyTorch cu130、SGLang 0.5.18、
  `sglang-kernel==0.4.7`、FlashInfer、Triton；`CUDA_HOME=/usr/local/cuda`。
- Qwen3.8 兼容 SGLang 源码（默认 `/mnt/data/xinyu/sglang-qwen38`，可用
  `QWEN38_SGLANG_SOURCE` 覆盖）。
- 模型目录（默认 `/mnt/data/public_models/Qwen3.8-Flash-Next`，可用
  `QWEN38_MODEL_PATH` 覆盖）：需包含 `config.json`、`model.safetensors.index.json`、
  权重分片和 `tokenizer.json`，且 `model_type=qwen4_exp`、
  原生 `max_position_embeddings=262144`。

## 启动

```bash
# 只校验环境并打印最终命令，不加载模型
bash deploy_qwen38_flash_next_yarn_1m.sh --check-only

# 正式启动（前台运行，生产容器里可作为 entrypoint）
bash deploy_qwen38_flash_next_yarn_1m.sh
```

脚本会：

1. 备份原生配置到 `config.json.native.bak`（只做一次），再原子写入
   `text_config.rope_parameters`（YaRN factor 4、`original_max_position_embeddings=262144`、
   `rope_theta=10000000`）并在顶层镜像 `max_position_embeddings`；
2. 校验 `model_type=qwen4_exp`、YaRN 已生效、SGLang 版本 0.5.18、8 张可见 GPU；
3. 以 TP8 启动 SGLang：FlashInfer GDN、BF16 mamba state、NEXTN speculative
   decoding、`--max-running-requests 96`、`--reasoning-parser qwen3`、
   `--tool-call-parser qwen3_coder`，并设置 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`。

启动完成后：

```bash
curl -s http://127.0.0.1:40000/health
curl -s -X POST http://<host>:40000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

## 常用环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `QWEN38_MODEL_PATH` | `/mnt/data/public_models/Qwen3.8-Flash-Next` | 也可指向微调 checkpoint（如内部 bio 权重），只要保持 `qwen4_exp` 架构与原生 262144 |
| `QWEN38_SGLANG_SOURCE` | `/mnt/data/xinyu/sglang-qwen38` | Qwen3.8 兼容源码树 |
| `QWEN38_CONTEXT_LENGTH` | `1048576` | 单请求上下文上限，脚本上限即 1,048,576 |
| `QWEN38_MEM_FRACTION_STATIC` | `0.90` | OOM 时降到 0.85 |
| `QWEN38_CUDA_GRAPH_MAX_BS_DECODE` | `32` | 显存紧张时降到 16 |
| `QWEN38_PORT` / `QWEN38_HOST` | `40000` / `0.0.0.0` | 监听地址 |
| `QWEN38_API_KEY` | 空 | 设置后启用 `--api-key` |
| `QWEN38_CHECK_ONLY` | `0` | 置 1 等价于 `--check-only` |

## 容量与性能（实测）

- 单请求输入 + 输出上限 = `--context-length` = 1,048,576 token；预留 20k 输出时
  输入约 1,028,576。
- `--max-running-requests 96`（脚本默认）；`mem-fraction-static` 0.85–0.90 之间
  按显存余量选择。
- 作为参考：单机 TP8 下 1k 输入 / 128 输出约 350+ token/s；PD 方案里
  300k 输入 / 10k 输出的端到端解码约 440 token/s（见 `../h200_2p2d/README.md`）。

## 退出与恢复

结束进程后如果想还原原生 262k 配置，用备份覆盖即可：

```bash
cp /mnt/data/public_models/Qwen3.8-Flash-Next/config.json.native.bak \
   /mnt/data/public_models/Qwen3.8-Flash-Next/config.json
```

不要提交模型目录、编译缓存或 `config.json.native.bak`。
