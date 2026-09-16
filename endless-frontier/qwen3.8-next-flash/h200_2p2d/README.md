# 手工 SSH：H200 2P + 2D（TP8，YaRN 1M）

四台 8 卡 H200 主机组成 PD 分离集群：两台 Prefill、两台 Decode，每个实例 TP8，
YaRN factor 4 把原生 262,144 上下文扩展到 1,048,576；一台 Router 负责把
OpenAI 兼容请求分发到 Prefill/Decode 组合。

## 拓扑与端口

| 角色 | 数量 | 监听地址 | 说明 |
|---|---:|---|---|
| Prefill worker | 2 | `<prefill-ip>:41000`，bootstrap `8998` | 只做 prefill，KV 经 Mooncake 传 Decode |
| Decode worker | 2 | `<decode-ip>:42000` | 只做 decode，启用 NEXTN/MTP |
| Router | 1 | `0.0.0.0:40000` | `--pd-disaggregation`，prefill/decode 各自 round robin |

客户端只需要访问 Router：`http://<router-ip>:40000/v1/chat/completions`。

## 环境要求

四个节点必须使用同一镜像、同一份 SGLang 源码、同一模型目录：

- CUDA 13.0 **devel** 镜像（含 `nvcc` 与 toolkit headers）、PyTorch cu130、
  SGLang 0.5.18、`sglang-kernel==0.4.7`、FlashInfer、Triton。
  **不要**让 pip 包 `nvidia/cu13` 的 `nvcc 13.3` 抢到 `PATH`，见
  `../DEPLOYMENT_PRACTICE.md`。
- Qwen3.8 兼容源码：`/mnt/data/xinyu/sglang-qwen38-upstream-1789383617`
  （官方 `main` + Qwen4-Exp PD 修复）。
- 模型：`/mnt/data/public_models/Qwen3.8-Flash-Next`（`config.json`、权重分片、
  tokenizer 齐全）。
- 跨节点：IB/RDMA + NVIDIA IBGDA/GDRCopy、Mooncake、RoCE v2（默认 `MC_GID_INDEX=3`）。
- 安全组放通 worker API（41000/42000）、bootstrap（8998）、Router（40000）
  以及分布式端口。

```bash
export QWEN38_SGLANG_SOURCE=/mnt/data/xinyu/sglang-qwen38-upstream-1789383617
export QWEN38_MODEL_PATH=/mnt/data/public_models/Qwen3.8-Flash-Next
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
```

worker 脚本首次启动会把模型 `config.json` 备份为 `config.json.native.bak`，
再原子写入 YaRN 配置（factor 4、`original_max_position_embeddings=262144`）。

## 启动顺序

1）两台 Prefill 主机各执行一次（参数：角色、本机内网 IP、API 端口、bootstrap 端口）：

```bash
nohup setsid bash run_qwen38_flash_next_yarn_1m_pd_worker.sh \
    prefill <PREFILL_IP> 41000 8998 >> /tmp/qwen38_prefill.log 2>&1 &
```

2）两台 Decode 主机各执行一次：

```bash
nohup setsid bash run_qwen38_flash_next_yarn_1m_pd_worker.sh \
    decode <DECODE_IP> 42000 8998 >> /tmp/qwen38_decode.log 2>&1 &
```

3）在任一能访问四台机器的节点启动 Router（先把脚本里的四个 URL 改成实际 IP）：

```bash
bash run_qwen38_flash_next_yarn_1m_pd_router.sh
```

4）验证：

```bash
curl -s http://<router-ip>:40000/health
curl -s http://<router-ip>:40000/workers          # 4 个 worker 必须 is_healthy=true
curl -s -X POST http://<router-ip>:40000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

模型加载约 30 秒（权重在页缓存时）到数分钟；`/health` 返回 200 才算就绪。

## 关键参数（worker）

脚本默认 TP8、PP1、`--context-length 1048576`、`--max-total-tokens 6000000`、
`--mem-fraction-static 0.85`、`--max-running-requests 96`、`--chunked-prefill-size 8192`、
`--page-size 64`、`--cuda-graph-max-bs-decode 32`、`--tokenizer-worker-num 6`，
并启用 FlashInfer GDN、BF16 mamba state、decode CUDA graph。

几个容易踩的点：

- **`--max-running-requests` 默认 96（Prefill 与 Decode 相同）**，用
  `QWEN38_PD_MAX_RUNNING_REQUESTS` 覆盖。Prefill 不设该参数时，SGLang 会按 KV pool
  推导（实测约 2929），随后被 mamba state cache 截断到 1181；显式 96 可避免这种大起大落。
- **NEXTN/EAGLE 只在 Decode 打开**：Qwen3.8 QSA 的 draft-prefill 在长上下文会触发
  CUDA 非法地址（见下“已知问题”），Prefill 走普通路径。
- `--reasoning-parser qwen3` 与 `--tool-call-parser qwen3_coder` 必须保留。
- `--mamba-radix-cache-strategy extra_buffer` 是 Qwen4-Exp PLE 状态所需的。

## 关键参数（Router 与 429）

Router 默认：

```text
--max-concurrent-requests 200    # 可用 QWEN38_PD_MAX_CONCURRENT_REQUESTS 覆盖
--queue-size 200                 # 可用 QWEN38_PD_QUEUE_SIZE 覆盖
--queue-timeout-secs 600         # 可用 QWEN38_PD_QUEUE_TIMEOUT_SECS 覆盖
```

这三个值很关键。Router 的限流中间件是令牌桶，refill 速率等于
`max-concurrent-requests`；当 `--queue-size 0` 时没有令牌立即返回 429，日志为：

```text
WARN smg::middleware src/middleware.rs:610:
No tokens available and queuing is disabled, returning 429
```

历史配置 `--max-concurrent-requests 4 --queue-size 0`（早期脚本默认值）会让 10 个并发
请求中的大部分直接失败——本仓库已改成 200/200/600；若要用更小的并发，请同时给出
非零队列，否则就会出现“10 并发全 429”。注意队列只影响排队，实际并行度仍受 worker 的
`--max-running-requests` 限制。

## 运维：Prefill watchdog

`watch_prefill.sh` 在每台 Prefill 主机各跑一份，每 10 秒检查本机
`<QWEN38_LOCAL_IP>:41000/health`，连续 3 次失败后用 PID 文件 + 进程组信号重启本机
Prefill（不碰 Decode，也不用宽泛的 `pkill`）。启动/重启后有 30 分钟保护窗口，避免
模型加载期间被二次重启。

```bash
export QWEN38_LOCAL_IP=<本机内网IP>
export QWEN38_WORKER_SCRIPT=$PWD/run_qwen38_flash_next_yarn_1m_pd_worker.sh
nohup bash watch_prefill.sh >> /tmp/qwen38_prefill_watchdog.log 2>&1 &
```

watcher 通过 `/tmp/qwen38_prefill.pid` 记录 worker PID。**手动重启 worker 时务必同步写
这个 PID 文件**（`nohup setsid ... & echo $! > /tmp/qwen38_prefill.pid`），否则 watchdog
会把进程当成“无主”，故障时可能拉起第二个进程导致端口冲突。

Router 会周期性重新探测静态注册的 worker：worker 恢复后自动回到 healthy，一般
2–5 分钟（默认 60 秒间隔、失败 3 次摘除、成功 2 次恢复）。想更快可以加：

```text
--health-check-interval-secs 10 --health-failure-threshold 3 \
--health-success-threshold 2 --health-check-timeout-secs 5
```

## 变更 / 重启运行手册

改 worker 参数（例如并发上限）时的标准步骤：

```bash
# 1. 停 watchdog（否则它会立刻把进程拉回来）
kill <watchdog-pid>

# 2. 停 worker（进程组，等于 watchdog 的停法）
kill -TERM -- -$(cat /tmp/qwen38_prefill.pid)

# 3. 用新脚本启动并写回 PID 文件
cd /root && nohup setsid bash <worker.sh> prefill <PREFILL_IP> 41000 8998 \
    >> /tmp/qwen38_prefill.log 2>&1 &
echo $! > /tmp/qwen38_prefill.pid

# 4. 等 /health 200 后把 watchdog 拉起来
export QWEN38_LOCAL_IP=<本机内网IP>
nohup bash <watch_prefill.sh> >> /tmp/qwen38_prefill_watchdog.log 2>&1 &
```

只在 Prefill 参数变化时才需要重启 Prefill；Decode 不受影响。改动多台机器时逐台执行。

## 实测性能（TP8，1M YaRN）

| 场景 | TTFT | 有效 Decode |
|---|---:|---:|
| 1,024 输入 / 128 输出 | 约 1.4 s | 约 358 token/s |
| 100,000 输入 / 128 输出 | 约 8.6 s | 约 359 token/s |
| 300,000 输入 / 128 输出 | 约 17.3 s | 约 254 token/s |
| 300,000 输入 / 10,000 输出 | 18.7 s | 约 441 token/s（服务端）/ 约 374 token/s（客户端流式） |

300k/10k 那次 NEXTN 接受率 99.92%、平均接受长度 3.997。20k 输出未跑完整程；
按 440 token/s 估算约 45 秒纯解码时间（不含 TTFT）。

## 容量上限

- 单请求输入 + 输出 ≤ `--context-length` = **1,048,576** token；预留 20k 输出时输入上限
  约 1,028,576。
- Decode worker KV pool：每台 6,000,000 token（两台名义合计 12M）。
- Prefill worker KV pool：profiled 值约 3,698,688 token（足够承载在途 prefill，
  KV 随后转给 Decode）。`--max-total-tokens` 是上限而非保证，实际取决于显存 profile。

## 已知问题

**长上下文 Prefill 触发 QSA CUDA 非法地址（未根治）**

- 现象：超过约 262k token 的 prefill 请求会让 Prefill worker 报
  `CUDA error: an illegal memory access was encountered`，堆栈为
  `qwen4_exp.py -> qsa_indexer.py -> get_prefill_mqa_inputs`；之后 detokenizer
  心跳超时、`/health` 超时，但进程仍在。
- 线索：日志同时出现
  `User-specified context_length (1048576) is greater than the derived context_length (262144)`，
  说明 YaRN 已开启，但 QSA/indexer 的部分内部元数据仍按原生 262k 处理。
- 影响：单个 Prefill 节点崩溃后 Router 会摘除它，另一个节点接过流量；
  两个都崩则服务不可用。
- 缓解：`watch_prefill.sh` 自动重启（约 30 秒检测 + 约 90 秒加载）；
  生产上建议限制超长请求的并发，并关注 SGLang 上游对 QSA + YaRN 的修复。
- 排查时不要只看 `/health`：崩溃后进程还在、`/metrics` 还能访问，但推理线程已死。

**时区**

节点 `TZ=Asia/Shanghai`，但部分组件（Scheduler watchdog、NCCL）按 UTC 打日志，
对比 Router/Prefill 日志时间时要先确认时区，否则会得出错误的因果顺序。

## 排查清单

- worker 不健康：`curl http://<ip>:<port>/health`、`tail /tmp/qwen38_*.log`、
  `nvidia-smi`、`ibdev2netdev`。
- 启动即退出：先看 `sglang-kernel` 版本（必须 0.4.7）与 CUDA headers/nvcc 是否匹配。
- 跨节点失败：安全组端口、RoCE GID（`MC_GID_INDEX`）、IB 设备映射。
- 客户端 429：检查 Router 的 `--max-concurrent-requests` / `--queue-size`。
- 客户端 502/超时：确认 Prefill 是否已崩（见上），以及是否经由 EAS/网关转发。
