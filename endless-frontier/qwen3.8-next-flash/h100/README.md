# 单机 H100：YaRN 1M（TP8）

在一台 8×H100 主机上以 TP8 启动 Qwen3.8 Flash Next（`qwen4_exp`），把原生
262,144 上下文通过 YaRN factor 4 扩展到 **1,048,576（1M）**，对外提供 OpenAI
兼容 API。没有 PD 转发，服务直接监听 `0.0.0.0:40000`。

配套脚本：`deploy_qwen38_flash_next_yarn_1m.sh`（本目录）。

## 使用镜像（已验证）

```text
dptech-sh-pai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/dptech-namespace/sglang:sglang-0-5-18-qwen38-next-flash-h100-1m
```

镜像内容：CUDA 13.0 devel（含 `nvcc` + toolkit headers）+ PyTorch 2.13.0+cu130 +
SGLang 0.5.18 + `sglang-kernel==0.4.7` + `flashinfer-python/-cubin/-jit-cache`
**0.6.18**（三者必须同版本；flashinfer 低于 0.6.18 会踩 §5 的启动断言）。

## 0. 快速开始（Agent 照抄这一段即可）

```bash
# 前提：8 张 H100 空闲；/mnt/data 已挂载（模型与源码都在上面）
export QWEN38_MODEL_PATH=/mnt/data/yuzhucai/dlc_outputs/qwen38_flash_bio_0915/iter_0000963/hf
export SOURCE=/mnt/data/xinyuzhu/sglang
export SCRIPT=$SOURCE/endless-frontier/qwen3.8-next-flash/h100/deploy_qwen38_flash_next_yarn_1m.sh

# 1) 只校验、不加载权重（约 20~30 s，会打印最终启动命令和预测的 attention backend）
bash "$SCRIPT" --check-only

# 2) 正式启动（冷启动约 6 分钟：读 336GB 权重 + 抓 CUDA graph；缓存热时约 1 分钟）
nohup setsid bash "$SCRIPT" > /tmp/qwen38_single.log 2>&1 &

# 3) 等就绪（health 返回 200）
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:40000/health)" = "200" ]; do sleep 10; done
echo ready

# 4) 冒烟测试
curl -s -X POST http://127.0.0.1:40000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-1m","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

> `QWEN38_MODEL_PATH` 必须显式设置：集群上没有默认的
> `/mnt/data/public_models/Qwen3.8-Flash-Next`。当前已验证的内部权重是
> `/mnt/data/yuzhucai/dlc_outputs/qwen38_flash_bio_0915/iter_0000963/hf`，
> 任何 `qwen4_exp` 架构、原生 `max_position_embeddings=262144` 的 HF 产物都可以。

## 1. 环境要求

| 项 | 要求 |
|---|---|
| GPU | 8×H100（sm90），单机 TP8 |
| 镜像 | 见上方 H100 镜像（CUDA 13.0 devel，必须带 `nvcc`/headers） |
| 挂载 | `/mnt/data`（模型 + 源码 + 本仓库） |
| 源码 | Qwen3.8 兼容 SGLang（默认 `/mnt/data/xinyuzhu/sglang`，dev 分支，含 `qwen4_exp`） |
| 模型 | `qwen4_exp` + 原生 262144 + `mtp.*` 权重（NEXTN 需要）+ `tokenizer.json` |
| 端口 | `40000`（可用 `QWEN38_PORT` 覆盖） |

脚本会用 `PYTHONPATH` 把源码树顶到镜像内置 SGLang 前面，并在校验阶段确认
**实际加载的 sglang 真的来自源码树**（DLC launcher 会覆盖 `PYTHONPATH`，
见 `../DEPLOYMENT_PRACTICE.md` §2.3）。

## 2. 脚本做了什么

1. **源码树探测**：`QWEN38_SGLANG_SOURCE` > `/mnt/data/xinyuzhu/sglang` >
   `/mnt/data/xinyu/sglang-qwen38*`，并打印 `分支@commit`（非 dev 分支会告警）。
2. **YaRN 配置（只保留一份 config）**：检查模型目录的 `config.json`
   是否为 1M 版本（`rope_type=yarn` + `factor=4.0` +
   `original_max_position_embeddings=262144`）。
   - 是 → 原样使用，**一个字节都不动**；
   - 不是 → 先备份 `config.json.native.bak`，再原子改写。
3. **模型校验**：`model_type=qwen4_exp`、YaRN 已生效、73 个权重分片齐全、
   `mtp.*` 存在（`QWEN38_SPECULATIVE=0` 时跳过）。
4. **运行时校验 + 启动**：固定 `CUDA_HOME/CUDACXX/PATH`；校验 GPU 数量/算力、
   sglang 版本与来源、`sglang-kernel>=0.4.7`、flashinfer 版本，并**预测 SGLang
   实际会选哪个 attention backend**（见 §5）；最后以 TP8 + NEXTN 启动。

## 3. 常用环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `QWEN38_MODEL_PATH` | `/mnt/data/public_models/Qwen3.8-Flash-Next`（本集群不存在，必设） | HF 模型目录；也可指向微调产物 |
| `QWEN38_SGLANG_SOURCE` | `/mnt/data/xinyuzhu/sglang` | Qwen3.8 兼容源码树 |
| `QWEN38_ATTENTION_BACKEND` | `auto` | `auto` = 由 SGLang 自选（本模型 → `flashinfer`，需 ≥0.6.18）；`fa3`/`flashinfer`/`triton` = 显式指定，`fa3` 可绕开 flashinfer 版本断言 |
| `QWEN38_SERVED_MODEL_NAME` | `qwen38-flash-next-1m` | 对外模型名 |
| `QWEN38_CONTEXT_LENGTH` | `1048576` | 单请求上下文上限（脚本上限 1,048,576） |
| `QWEN38_MEM_FRACTION_STATIC` | `0.90` | 显存紧张 / OOM 时降到 `0.85` |
| `QWEN38_CUDA_GRAPH_MAX_BS_DECODE` | `32` | 显存紧张时降到 `16` |
| `QWEN38_MAX_RUNNING_REQUESTS` | `96` | 与 PD 方案一致 |
| `QWEN38_SPECULATIVE` | `1` | `0` 关闭 NEXTN（长上下文排障用，见 §6） |
| `QWEN38_PORT` / `QWEN38_HOST` | `40000` / `0.0.0.0` | 监听地址 |
| `QWEN38_API_KEY` | 空 | 设置后启用 `--api-key` |
| `QWEN38_SKIP_CHECKS` | `0` | `1` 跳过运行时校验（排障用） |
| `QWEN38_CHECK_ONLY` | `0` | `1` 等价于 `--check-only` |

## 4. 实测容量与性能（8×H100，BF16，TP8）

| 指标 | 实测值 |
|---|---|
| `context_len` | 1,048,576（单请求输入+输出上限） |
| `max_total_num_tokens` | 1,856,704 |
| `max_running_requests` | 96 |
| `max_mamba_cache_size` | 1,410 |
| 每卡权重 | 31.25 GB（target）+ 0.98 GB（MTP） |
| 每卡显存占用 | 约 77~80 GB / 81 GB（`mem-fraction-static=0.90`） |
| 冷启动耗时 | 约 6 分钟（首次读 336GB 权重；NAS 缓存热时约 1 分钟） |
| 短请求 | 首 token 约 0.9 s（1k 以内输入） |
| 128k 输入 prefill | 约 4.5 s（≈28k token/s） |
| 并发 | 8 并发短请求全通过；4 并发 32k 输入全通过 |

启动完成后的日志里应能看到：

```text
max_total_num_tokens=1856704, ... context_len=1048576, available_gpu_mem=...
```

## 5. 版本边界：attention backend 与 `flashinfer>=0.6.18` 断言（重要）

**结论先说**：本模型（QSA 压缩注意力 + NEXTN）在 H100 上不指定
`--attention-backend` 时，SGLang 会自动选 **flashinfer**；而 SGLang 对
flashinfer 注意力后端有 `flashinfer_python>=0.6.18` 的启动断言。所以要么用带
0.6.18 的镜像（上面那个 H100 镜像就是），要么显式 `QWEN38_ATTENTION_BACKEND=fa3`。

代码路径（`dev@7ab7009c2`）：

1. `python/sglang/srt/entrypoints/engine.py:1719-1732`：只有
   `attention_backends_of(...)` 里出现 `flashinfer`（或 `dsa_topk_backend=flashinfer`）
   时才 `assert_pkg_version("flashinfer_python", "0.6.18")`，且整体受
   `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK` 控制。
2. `python/sglang/srt/arg_groups/model_override_base.py:164` `attention_backends_of()`
   只读 `--attention-backend` / `--prefill-attention-backend` /
   `--decode-attention-backend`；**`--linear-attn-*-backend flashinfer` 不算**，
   所以 GDN 线性注意力用 0.6.17 是允许的。
3. 不指定 `--attention-backend` 时走 `arg_groups/overrides.py:1157`
   `_attention_backend_default()` → `get_default_attn_backend()`
   （`model_override_base.py:295`）：Hopper 上**只有**
   `is_no_spec_infer_or_topk_one()` 成立才选 `fa3`（第 329 行）。
4. `is_no_spec_infer_or_topk_one()`（`utils/common.py:3699`）要求
   `page_size in (1, None)`；而 Qwen4-Exp 的 QSA 压缩注意力在
   `arg_groups/model_overrides/qwen4_exp.py:84` 把 `page_size` 固定成 **64**
   （日志会打印 `Setting page size to 64 for compressed QSA ...`）→ 条件不成立
   → 落到 else 分支 → 默认 **flashinfer**（日志：
   `Attention backend not specified. Use flashinfer backend by default.`）。

因此三种处理方式：

| 方案 | 做法 | 适用 |
|---|---|---|
| A（推荐） | 用带 flashinfer 0.6.18 的镜像；或 `pip install -U --no-deps flashinfer_python==0.6.18 flashinfer-cubin==0.6.18 flashinfer-jit-cache==0.6.18` | 本 H100 镜像就是这条路 |
| B | `QWEN38_ATTENTION_BACKEND=fa3`（生产 PD 也是 `fa3` + `--page-size 64` + NEXTN） | 镜像里 flashinfer < 0.6.18 |
| C | `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` | 仅排障；会同时跳过 `sglang-kernel` 断言 |

脚本会在 `--check-only` / 启动前把预测的后端和版本打出来，命中问题直接给出
A/B/C 三种修法。

## 6. 排障

| 现象 | 处理 |
|---|---|
| `flashinfer_python>=0.6.18` 断言失败 | §5：用 0.6.18 镜像，或 `QWEN38_ATTENTION_BACKEND=fa3` |
| 启动报 `ModuleNotFoundError: sglang.srt...qwen4_exp` | `PYTHONPATH` 被 launcher 覆盖；设 `QWEN38_SGLANG_SOURCE` 指到源码树 |
| 启动 OOM | `QWEN38_MEM_FRACTION_STATIC=0.85`，必要时 `QWEN38_CUDA_GRAPH_MAX_BS_DECODE=16` |
| 单请求 >262k prefill 时 CUDA illegal access | 已知缺陷（`../DEPLOYMENT_PRACTICE.md` §6）：先用 `QWEN38_SPECULATIVE=0` 关 NEXTN 复现/规避，并限制超长并发 |
| `OSError: [Errno 116] Stale file handle`（读 config.json） | NAS 上有人对该文件做了 rename-over（例如另一个进程正在改写）；确认没有并发改同目录后重启即可 |
| 权重分片缺失 | 校验阶段会列出缺哪些；补齐后重跑 |

## 7. 还原原生 262k 配置

```bash
cp <MODEL_PATH>/config.json.native.bak <MODEL_PATH>/config.json
```

## 8. 交付前验证清单

1. `bash deploy_qwen38_flash_next_yarn_1m.sh --check-only` 全绿，且
   `预测 attention backend` 与实际一致。
2. `/health` 返回 200；日志出现 `max_total_num_tokens=...` 且 `context_len=1048576`。
3. 一个短请求 + 一个 128k 左右输入的长请求都能正常返回。
4. 日志里没有 `CUDA error`、`illegal memory access`、`Stale file handle`。
5. 没有把 AccessKey、`.runtime/`、真实节点 IP 提交进仓库。
