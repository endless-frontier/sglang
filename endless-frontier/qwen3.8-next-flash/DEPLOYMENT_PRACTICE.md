# Qwen3.8 Flash Next 部署实践（各方案共用）

这份文档记录各套配方（`h100`、`h100_2p2d`、`h200`、`h200_2p2d`、
`aliyun_h200_2p2d`）共用的环境基线、参数含义和踩坑经验。单个方案的启动命令见各自
目录下的 README：

| 目录 | 场景 |
|---|---|
| `h100/` | **H100 单机 8 卡**，已配套镜像，最快路径起服务 |
| `h100_2p2d/` | **H100 四机 2P2D**（手工 SSH，含压测脚本与参考性能） |
| `h200/` | H200 单机 8 卡 |
| `h200_2p2d/` | H200 四机 2P2D（手工 SSH） |
| `aliyun_h200_2p2d/` | 阿里云 EAS/DLC 版 2P2D |

## 1. 环境基线

| 组件 | 要求 | 备注 |
|---|---|---|
| CUDA | 13.0 devel（`nvcc` + headers） | runtime 镜像无法 JIT 编译 QSA/TileLang kernel |
| PyTorch | 2.13.0+cu130 | 与 CUDA 13.0 匹配 |
| SGLang | 0.5.18（官方 `main` 基线） | 需要含 `qwen4_exp` 配置/模型的版本 |
| sglang-kernel | **0.4.7** | 0.4.6.post1 会在 Decode 的 PLE conv state 传输上失败 |
| flashinfer | ≥0.6.18（当 flashinfer 被选作 attention backend 时） | `flashinfer_python`/`-cubin`/`-jit-cache` 三者必须同版本，见 §2.4 |
| 镜像（H100 单机 / H100 2P2D） | `dptech-sh-pai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/dptech-namespace/sglang:sglang-0-5-18-qwen38-next-flash-h100-1m` | 内置 flashinfer 0.6.18、mooncake 0.3.12、sglang-router 0.3.2；`h100/` 与 `h100_2p2d/` 验证过 |
| 镜像（H200 / PD） | `pai-ai-prod-acr-registry.cn-shanghai.cr.aliyuncs.com/acr_namespace/scimaster:sglang-0-5-18-cuda13-qwen38-next-pd` | 内置 flashinfer 0.6.17，必须显式 `--attention-backend fa3`，见 §2.4 |
| 模型 | `/mnt/data/public_models/Qwen3.8-Flash-Next` | `model_type=qwen4_exp`，原生 262,144 |
| 源码 | `/mnt/data/xinyuzhu/sglang`（`dev` 分支，**推荐**）；老的 H200 树 `/mnt/data/xinyu/sglang-qwen38-upstream-1789383617` | 通过 `PYTHONPATH` 优先于镜像内置 SGLang；dev 已含 §6 的 QSA 修复 |
| 传输 | Mooncake + RoCE v2（`MC_GID_INDEX=3`）+ IBGDA | PD 分离必需 |
| Parser | `--reasoning-parser qwen3`、`--tool-call-parser qwen3_coder` | 三套方案都要带 |

## 2. 四个最容易踩的环境坑

### 2.1 nvcc 与 CUDA headers 版本不一致

镜像里同时存在系统 CUDA 13.0（`/usr/local/cuda`）和 pip 包 `nvidia/cu13` 的
`nvcc 13.3` 时，TileLang 会自动选中 pip 的 nvcc，编译报：

```text
RuntimeError: CUDA compiler and CUDA toolkit headers are incompatible
```

修法：显式固定到系统 CUDA。

```bash
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH
```

更稳的做法是直接换 CUDA 13.0 devel 镜像，不要装 `nvidia-cu13` 的 pip 包。

### 2.2 `sglang-kernel` 版本不一致

官方 Qwen4-Exp 的 PD 修复要求 `sglang-kernel>=0.4.7`。集群里混用
0.4.6.post1/0.4.7 时，Decode worker 会报：

```text
AssertionError: self.conv_state is not None
  at sglang/srt/mem_cache/ple_state_pool.py
```

```bash
pip install -U 'sglang-kernel==0.4.7'
```

四台机器要一致；升级后逐个重启 worker。

### 2.3 PYTHONPATH 被启动器覆盖

本地启动脚本用 `export PYTHONPATH=...` 没问题，但 DLC 的 launcher 会重写环境变量，
子进程仍会加载镜像内置的 `/s-lworkspace/sglang`，日志表现为：

```text
ModuleNotFoundError: No module named 'sglang.srt.configs.qwen4_exp'
```

DLC 版 `worker.py` 已把源码路径注入子进程env（`QWEN38_SGLANG_SOURCE`），
不要改回只依赖 shell 导出。

### 2.4 attention backend 与 `flashinfer_python >= 0.6.18` 断言

**现象**：镜像里是 flashinfer 0.6.17，启动直接失败：

```text
flashinfer_python is installed with version 0.6.17, which is less than the
minimum required version 0.6.18. Please uninstall the old version and reinstall
the latest version by following the instructions at
https://docs.flashinfer.ai/installation.html.
```

前置日志里通常还有这一行（说明自动选中的不是 fa3）：

```text
Attention backend not specified. Use flashinfer backend by default.
```

**因果链**（`dev@7ab7009c2`）：

1. `python/sglang/srt/entrypoints/engine.py:1719-1732`：只有当
   `attention_backends_of(...)` 的结果里出现 `flashinfer`（或
   `dsa_topk_backend=flashinfer`）时，才断言 `flashinfer_python>=0.6.18`。
2. `arg_groups/model_override_base.py:164`：`attention_backends_of()` 只看
   `--attention-backend` / `--prefill-attention-backend` /
   `--decode-attention-backend`——**`--linear-attn-*-backend flashinfer` 不算**，
   所以 GDN 线性注意力用 0.6.17 是被允许的（线上就是这么跑的）。
3. 不指定 `--attention-backend` 时：`arg_groups/overrides.py:1157`
   `_attention_backend_default()` → `get_default_attn_backend()`
   （`model_override_base.py:295`）。Hopper 上**只有**
   `is_no_spec_infer_or_topk_one()` 为真才选 `fa3`（`:329`）。
4. `utils/common.py:3699` `is_no_spec_infer_or_topk_one()` 要求
   `page_size in (1, None)`；而 Qwen4-Exp 的 QSA 压缩注意力在
   `arg_groups/model_overrides/qwen4_exp.py:84` 把 `page_size` 固定为 **64**
   → 条件不成立 → 落到 else 分支 → 默认 **flashinfer**。
5. 生产 PD 之所以没事：`h200_2p2d/run_qwen38_flash_next_yarn_1m_pd_worker.sh:107`
   显式写了 `--attention-backend fa3`（配合 `--page-size 64` + NEXTN）。
   **单机脚本如果没写，就会踩这个断言。**

**三种修法**：

| 方案 | 做法 | 说明 |
|---|---|---|
| A | 用带 flashinfer 0.6.18 的镜像（如 H100 镜像），或 `pip install -U --no-deps flashinfer_python==0.6.18 flashinfer-cubin==0.6.18 flashinfer-jit-cache==0.6.18` | 保持 SGLang 自选后端；三件套必须同版本 |
| B | 显式 `--attention-backend fa3`（H200 单机脚本默认就是这样，`QWEN38_ATTENTION_BACKEND=fa3`） | 与线上 PD 一致，H100/H200（sm90）可用 |
| C | `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` | 仅排障：它会跳过 `_KERNEL_VERSION_CHECK_PACKAGES` 里的 **flashinfer 与 sglang-kernel 两个**断言 |

**两个脚本都已经内置了这条判断**：

- `h100/deploy_qwen38_flash_next_yarn_1m.sh`：默认 `QWEN38_ATTENTION_BACKEND=auto`
  （交给 SGLang 自选，配套镜像里是 flashinfer 0.6.18），启动前**预测**实际后端；
  如果预测结果是 flashinfer 而版本 < 0.6.18，直接给出 A/B/C 三种修法并退出。
- `h200/deploy_qwen38_flash_next_yarn_1m.sh`：默认 `fa3`（绕开断言），
  同样会预测后端并做版本判断。

注意：fa3 只在 Hopper（sm90）可用。如果机器是 A100/sm80，`--attention-backend fa3`
不可用，只能走 A 或 C。

## 3. 模型与 YaRN

- 脚本**先检测**目标目录里的 `config.json` 是不是 1M（YaRN factor 4.0）版本：
  已经是就不动文件；不是才备份成 `config.json.native.bak` 并原子改写。这样同一个
  模型目录被多个容器/NAS 客户端并发启动时，不会因为反复 rename-over 触发
  `OSError: [Errno 116] Stale file handle`。YaRN 写入内容：

```json
"rope_parameters": {
  "rope_type": "yarn", "factor": 4.0, "rope_theta": 10000000,
  "original_max_position_embeddings": 262144,
  "mrope_interleaved": true, "mrope_section": [11, 11, 10],
  "partial_rotary_factor": 0.25
}
```

- 同时设置 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`，否则 SGLang 会按
  “派生上下文 262144 < 请求 1048576” 拒绝或给出告警。
- 顶层 `max_position_embeddings` 也要写（Transformers 5 会校验外层多模态配置）。
- 四台/四个 Pod 的配置必须字节级一致，否则 PD 两端 KV 语义不同。

## 4. PD 分离参数

- 每节点 TP8、PP1、DCP1（Qwen3.8 是 GDN + GQA 混合池，DCP>1 不支持）。
- `--disaggregation-transfer-backend mooncake`，bootstrap 端口 8998。
- NEXTN/EAGLE 只在 Decode 打开：QSA 的 draft-prefill 在长上下文会崩（见 §6）。
- `--max-running-requests 96` 两端一致；不传时 SGLang 会推导（实测 2929）并被
  mamba state cache 截断到 1181，显存占用大起大落，不利于稳定。
- `--max-total-tokens 6000000` 只是上限：Decode 实测能到 6,000,000，
  Prefill 受 profile 限制约 3,700,000。

## 5. Router 限流与 429

Router（`sglang_router`/`smg`）的限流是令牌桶：refill 速率 =
`--max-concurrent-requests`（未显式设 `rate_limit_tokens_per_second` 时），
`--queue-size 0` 表示不排队、直接 429：

```text
WARN smg::middleware src/middleware.rs:610:
No tokens available and queuing is disabled, returning 429
```

早期脚本沿用了 `--max-concurrent-requests 4 --queue-size 0`，导致“10 个并发几乎全
429”。当前三套方案统一使用：

```text
--max-concurrent-requests 200  --queue-size 200  --queue-timeout-secs 600
```

排查 429 时先看 Router 日志的这行，再确认是否 worker 侧已经崩了（两者表现不同：
worker 崩溃表现为 502/超时/健康检查失败）。

## 6. 长上下文 Prefill 的 QSA CUDA 崩溃（dev 已修复）

- 触发（修复前）：单请求 prefill 超过约 262k token。
- 日志：

```text
CUDA error: an illegal memory access was encountered
qwen4_exp.py -> qsa_indexer.py -> get_prefill_mqa_inputs -> sequence_lengths.tolist()
User-specified context_length (1048576) is greater than the derived context_length (262144)
```

- 根因与修复（upstream `main`）：
  - `d72e59508`（reland of #38346 / #39446）：QSA compressed-K gather 越界，
    `python/sglang/srt/layers/attention/qsa/qsa_indexer.py:324` 加上
    `group_locs = group_locs.clamp_max(source_keys.shape[0] - 1)`；
  - `2c0a70960`：MTP 复用旧 CUDA stream，不再无限创建 stream。
- 当前状态：`dev` 已于 2026-09-17 merge upstream `main`（merge commit `07d41b2ae`），
  上面两个提交都在 dev 上。H100 2P2D（`h100_2p2d/`）用该树实测
  **300k 与 800k 输入 + 10k 输出均无 CUDA 报错**（见 `h100_2p2d/RESULTS.md`）。
- 仍需注意：老的 H200 源码树（`/mnt/data/xinyu/sglang-qwen38-upstream-*`，基于 fork
  的旧 commit）**没有**这两个修复，长上下文仍会崩；建议 H200 那套也切到
  `/mnt/data/xinyuzhu/sglang` 的 dev 分支，切换前保留 §7 的 watchdog 与限流。
- 影响（未修复时）：Worker 进程仍在、`/metrics` 可访问，但推理线程已死；
  `/health` 随后超时。两个 Prefill 都崩则服务整体不可用。

## 7. 日志、进程与重启

- 日志：`/tmp/qwen38_prefill.log`、`/tmp/qwen38_decode.log`、`/tmp/qwen38_router.log`、
  `/tmp/qwen38_prefill_watchdog.log`（机器重建即丢失，重要结论请落库到本仓库文档）。
- PID：`/tmp/qwen38_prefill.pid`（watchdog 依赖它做重启；手动启停务必同步）。
- 重启顺序：停 watchdog → 停 worker（`kill -TERM -- -<pid>`）→ 起 worker →
  等 `/health` 200 → 起 watchdog。Decode 不受影响。
- 时区：节点 `TZ=Asia/Shanghai`，但部分组件按 UTC 打日志，跨组件对齐时间需要换算。

## 8. 交付前的验证清单

1. `curl /health` 四个 worker 全 200，Router `/workers` 全 `is_healthy=true`
   （H100 2P2D 建议再跑一次 300k 输入 / 10k 输出，见 `h100_2p2d/RESULTS.md`）。
2. 一个短请求 + 一个长请求（例如 300k 输入 / 128 输出）都能正常返回。
3. Router 日志无 `No tokens available`；`/metrics` 能看到 NEXTN 接受率（Decode）。
4. 确认 watchdog 在两台 Prefill 上运行，且 PID 文件指向真实 worker 进程。
5. 确认没有把 AccessKey、receipt、`.runtime/`、`rendered/`、真实节点 IP 提交进仓库。
