# Qwen3.8 Flash Next 部署实践（三套方案共用）

这份文档记录三套配方（`h200`、`h200_2p2d`、`aliyun_h200_2p2d`）共用的环境基线、
参数含义和踩坑经验。单个方案的启动命令见各自目录下的 README。

## 1. 环境基线

| 组件 | 要求 | 备注 |
|---|---|---|
| CUDA | 13.0 devel（`nvcc` + headers） | runtime 镜像无法 JIT 编译 QSA/TileLang kernel |
| PyTorch | 2.13.0+cu130 | 与 CUDA 13.0 匹配 |
| SGLang | 0.5.18（官方 `main` 基线） | 需要含 `qwen4_exp` 配置/模型的版本 |
| sglang-kernel | **0.4.7** | 0.4.6.post1 会在 Decode 的 PLE conv state 传输上失败 |
| 模型 | `/mnt/data/public_models/Qwen3.8-Flash-Next` | `model_type=qwen4_exp`，原生 262,144 |
| 源码 | `/mnt/data/xinyu/sglang-qwen38-upstream-1789383617` | 通过 `PYTHONPATH` 优先于镜像内置 SGLang |
| 传输 | Mooncake + RoCE v2（`MC_GID_INDEX=3`）+ IBGDA | PD 分离必需 |
| Parser | `--reasoning-parser qwen3`、`--tool-call-parser qwen3_coder` | 三套方案都要带 |

## 2. 三个最容易踩的环境坑

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

## 3. 模型与 YaRN

- 脚本会备份原生 `config.json` 为 `config.json.native.bak`，再原子写入 YaRN：

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

## 6. 已知缺陷：长上下文 Prefill 的 QSA CUDA 崩溃

- 触发：单请求 prefill 超过约 262k token。
- 日志：

```text
CUDA error: an illegal memory access was encountered
qwen4_exp.py -> qsa_indexer.py -> get_prefill_mqa_inputs -> sequence_lengths.tolist()
User-specified context_length (1048576) is greater than the derived context_length (262144)
```

- 影响：Worker 进程仍在、`/metrics` 可访问，但推理线程已死；`/health` 随后超时。
  两个 Prefill 都崩则服务整体不可用。
- 现有缓解：`watch_prefill.sh` 做健康检查 + 自动重启；Router 会自动摘除/恢复节点。
- 尚未根治：需要上游修复 QSA/indexer 对 YaRN 长上下文（>262k）的处理；
  在修复前建议限制超长请求的并发并保留 watchdog。

## 7. 日志、进程与重启

- 日志：`/tmp/qwen38_prefill.log`、`/tmp/qwen38_decode.log`、`/tmp/qwen38_router.log`、
  `/tmp/qwen38_prefill_watchdog.log`（机器重建即丢失，重要结论请落库到本仓库文档）。
- PID：`/tmp/qwen38_prefill.pid`（watchdog 依赖它做重启；手动启停务必同步）。
- 重启顺序：停 watchdog → 停 worker（`kill -TERM -- -<pid>`）→ 起 worker →
  等 `/health` 200 → 起 watchdog。Decode 不受影响。
- 时区：节点 `TZ=Asia/Shanghai`，但部分组件按 UTC 打日志，跨组件对齐时间需要换算。

## 8. 交付前的验证清单

1. `curl /health` 四个 worker 全 200，Router `/workers` 全 `is_healthy=true`。
2. 一个短请求 + 一个长请求（例如 300k 输入 / 128 输出）都能正常返回。
3. Router 日志无 `No tokens available`；`/metrics` 能看到 NEXTN 接受率（Decode）。
4. 确认 watchdog 在两台 Prefill 上运行，且 PID 文件指向真实 worker 进程。
5. 确认没有把 AccessKey、receipt、`.runtime/`、`rendered/`、真实节点 IP 提交进仓库。
