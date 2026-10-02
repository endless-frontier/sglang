# GLM-5.3-Flash 部署配方（单机 8 卡）

GLM-5.3-Flash 在**单机 8 卡 H100 级别（sm90）**上的可复现启动配方。已在该硬件上实测：
服务正常起、短请求与 16k 长请求都能正常返回，实测数字见 [`RESULTS.md`](RESULTS.md)。

## 模型要点

| 属性 | 说明 |
|---|---|
| 架构 | 320B 总参 / 18B 激活的 MoE，混合注意力（MLA + DSA + KDA）+ mHC 残差 + MTP |
| 精度 | 权重 FP8（e4m3）；H100/H200 上 KV cache 用 BF16 |
| 上下文 | 262144 原生，检查点声明 `max_position_embeddings=1048576`（1M） |
| 生成默认 | `temperature=1.0`、`top_p=0.95`、**默认开启 thinking** |
| 权重体积 | 62 个分片，约 306 GB |

参数依据来自 SGLang 官方 cookbook：`docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx`。
**不要照搬 Qwen3.8 那套参数**：两者注意力后端不同，`--attention-backend fa3` 在这个模型上会崩（见下文）。

## 已验证环境

| 组件 | 值 |
|---|---|
| 节点 | 8 × `L20Z`（H100 级别，80 GiB/卡，计算能力 (9,0)=sm90） |
| 镜像 | 含 SGLang 0.5.18 / torch 2.13.0+cu130 / sglang-kernel 0.4.7 / flashinfer 0.6.18 的 CUDA 13.0 镜像 |
| 源码树 | `dev` 分支的仓库源码，需含 `python/sglang/srt/models/glm5_next.py` |
| 模型目录 | `/mnt/data/public_data/public_model/GLM5.3/GLM-5.3-Flash` |

**镜像自带的 SGLang 早于 GLM-5.3，没有 `glm5_next`**：直接用镜像内源码会在启动时报
`KeyError: 'glm5_next'` → `ValueError: The checkpoint ... has model type glm5_next but Transformers does not recognize this architecture`。
必须让进程用我们自己的源码树：设置 `GLM53_SGLANG_SOURCE`（脚本会把它放进 `PYTHONPATH`）；
若镜像里的 SGLang 是 editable 安装（`/sgl-workspace/sglang`），仅设 `PYTHONPATH` 可能仍解析到镜像树，
此时把源码覆盖进去即可（`cp -a <tree>/python/sglang/. /sgl-workspace/sglang/python/sglang/`）。

## 启动

```bash
export GLM53_MODEL_PATH=/mnt/data/public_data/public_model/GLM5.3/GLM-5.3-Flash
export GLM53_SGLANG_SOURCE=/mnt/data/<your-account>/sglang        # 含 glm5_next 的源码树根目录
export GLM53_PORT=8000

bash h100/deploy_glm53_flash_1m.sh --check-only                   # 1) 校验 + 打印启动命令
nohup setsid bash h100/deploy_glm53_flash_1m.sh > /tmp/glm53.log 2>&1 &   # 2) 启动
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" = 200 ]; do sleep 10; done
```

关键参数（脚本里都有注释）：

- `--attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang`
  `--linear-attn-backend triton`：该模型的 DSA/KDA 混合注意力必须走这套；H100/H200 上是官方推荐组合。
- `--kv-cache-dtype bfloat16` + `--quantization fp8`：FP8 权重 + BF16 KV；FP8 KV 只和 Blackwell 的 TRT-LLM DSA 搭配。
- `--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`：
  用检查点自带的 MTP 头做投机解码（官方 Low Latency 配方）。要更高吞吐可设 `GLM53_SPECULATIVE=0`。
- `--mem-fraction-static 0.78`、`--max-running-requests 64`、`--disable-prefill-cuda-graph`：显存与图捕获的稳妥起点。
- `--reasoning-parser glm45 --tool-call-parser glm47`：思考过程进 `reasoning_content`，工具调用进 `tool_calls`。

## 三个坑（都踩过）

1. **镜像自带 SGLang 太老**（见上）：必须用带 `glm5_next` 的源码树。
2. **`--attention-backend fa3` 会崩**：62 个分片全部加载完之后，在 decode CUDA-graph capture 阶段报
   `RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 64, 1, 0]` →
   `Exception: Capture cuda graph failed`。此时 SGLang 打印的建议（调小 `--mem-fraction-static` /
   `--cuda-graph-max-bs-decode`）是误导，真正原因是后端选错。用 DSA 后端即可。
3. **DLC 的 `UserCommand` 会被 `/bin/sh` 解析**：多行、带引号的命令会被拆坏（表现为
   `Syntax error: ";" unexpected` 或 `unexpected end of file`）。可靠写法是**单行、不含任何引号**：

   ```text
   echo <base64 脚本> | base64 -d > /tmp/driver.py; python3 -u /tmp/driver.py
   ```

   把逻辑写在脚本里，顺便还能在容器内每 15 秒采样显存/GPU 利用率并打印进度，
   异常能第一时间看到。

## 交付前验证清单

1. `curl /health` 返回 200；`curl /v1/models` 里出现对外模型名与 `max_model_len`。
2. 一个短请求 + 一个长（≥16k）请求都能返回，且 `finish_reason=stop`（注意给 thinking 留 token 预算）。
3. 容器内 `nvidia-smi` 显示每卡显存占用与权重规模相符（FP8 约 38 GB/卡 + KV）。
4. 日志里 `Load weight end`、`The server is fired up and ready to roll!` 都出现。
5. 提交/启动过程不落地任何 AccessKey、receipt、真实集群 ID（参见仓库根目录的仓库卫生约定）。

## 待办

- **我们自己的镜像**：目前依赖团队既有镜像，正在构建每模型一份的自有镜像（CUDA 13.0 devel + 仓库源码 + 固定版本依赖）。
- **对外暴露**：EAS 转发（同 VPC）尚未接；当前只有集群内可达的 `8000`。
- **PD 分离 / 多机**：单机稳定后再评估，参考同仓库 Qwen3.8 的 2P2D 配方。
