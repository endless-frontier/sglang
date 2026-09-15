# 单机 H200：YaRN 1M

在一台 8×H200 主机上以 TP8 启动 Qwen3.8 Flash Next，适合验证模型、YaRN 配置和 API。单机没有 PD 转发，服务默认监听 `0.0.0.0:40000`。

## 要求

使用 CUDA 13.0 devel 镜像（含 nvcc/headers）、PyTorch cu130、SGLang 0.5.18、`sglang-kernel==0.4.7`、FlashInfer 和 Triton。将模型放在 `/mnt/data/public_models/Qwen3.8-Flash-Next`，将 Qwen3.8 兼容 SGLang 源码放在 `/mnt/data/xinyu/sglang-qwen38`；可用 `QWEN38_MODEL_PATH`、`QWEN38_SGLANG_SOURCE` 覆盖。主机必须能看到 8 张 GPU。

## 启动

```bash
bash deploy_qwen38_flash_next_yarn_1m.sh --check-only
bash deploy_qwen38_flash_next_yarn_1m.sh
```

脚本会原子更新 `text_config.rope_parameters`（YaRN factor 4、原生长度 262,144、请求长度上限 1,048,576），启用 TP8、FlashInfer GDN、NEXTN speculative decoding、`--reasoning-parser qwen3` 和 `--tool-call-parser qwen3_coder`。默认静态显存比例 0.90、Decode 最大并发 96；OOM 时可设置 `QWEN38_MEM_FRACTION_STATIC=0.85` 或降低 CUDA graph batch。

启动后检查 `curl http://127.0.0.1:40000/health`，请求地址为 `http://<host>:40000/v1/chat/completions`。退出进程后可用 `config.json.native.bak` 恢复原生模型配置；不要提交该模型目录或缓存。
