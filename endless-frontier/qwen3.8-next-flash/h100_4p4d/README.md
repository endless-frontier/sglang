# PD 分离 H100 4P4D：YaRN 1M（TP8 × 8 机）

八台 **8×H100(80G)** 主机组成 PD 分离集群：**四台 Prefill + 四台 Decode**，
每个实例 TP8；YaRN factor 4 把原生 262,144 上下文扩到 **1,048,576（1M）**；
一台 Router 把 OpenAI 兼容请求分发给 Prefill/Decode 组合，
Prefill 的 KV 经 **Mooncake + RoCE v2** 直传 Decode。

本目录是 [`../h100_2p2d/`](../h100_2p2d/) 的横向扩展版：**worker 脚本、镜像、调参、
显存/容量结论完全一致**，只是 prefill 与 decode 各从 2 台变 4 台，
Router 从 `round_robin` 换成 **`cache_aware`**（原因见「路由策略」）。

已在本集群实测通过（8 台 8 卡 H100，300k / 800k 长上下文压测无崩溃），
参考性能见 [`RESULTS.md`](RESULTS.md)。

## 拓扑与端口

| 角色 | 数量 | 监听地址 | 说明 |
|---|---:|---|---|
| Prefill worker | 4 | `<PREFILLn_IP>:41000`，bootstrap `8998` | 只做 prefill，KV 经 Mooncake 传 Decode |
| Decode worker | 4 | `<DECODEn_IP>:42000` | 只做 decode，启用 NEXTN/MTP |
| Router | 1 | `0.0.0.0:40000` | `--pd-disaggregation`，prefill 默认 `cache_aware`、decode `round_robin` |

客户端只访问 Router：`http://<router-ip>:40000/v1/chat/completions`。
（Router 是纯 CPU 的 Rust 进程，可以放在任一节点上，也可以单独放一台机器。）

## 使用镜像（已验证）

```text
dptech-sh-pai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/dptech-namespace/sglang:sglang-0-5-18-qwen38-next-flash-h100-1m
```

镜像内容（实测 `importlib.metadata.version`）：CUDA 13.0 devel + **PyTorch 2.13.0+cu130** +
**SGLang 0.5.18** + `sglang-kernel==0.4.7` + `flashinfer-python / -cubin / -jit-cache`
**0.6.18** + `mooncake-transfer-engine-cuda13 0.3.12` + `sglang-router 0.3.2` +
`transformers 5.12.1`。

> ⚠️ **不要用 H200 那套 PD 镜像**（`...scimaster:sglang-0-5-18-cuda13-qwen38-next-pd`）：
> 它的 flashinfer 是 0.6.17，而本配方在 Decode 上会落到 flashinfer 注意力后端，
> 启动时会被 `flashinfer_python>=0.6.18` 断言拦住（原因见「attention backend」一节）。

## 环境要求

八个节点必须**同镜像、同源码树、同模型目录**：

| 项 | 要求 |
|---|---|
| GPU | 8 ×（8×H100 sm90），每实例单机 TP8 |
| 内存 | 每台 ≥ 768 GB（prefill 开 HiCache ratio 1.5 后约用 620 GB） |
| 镜像 | 见上方 H100 镜像，八台完全一致 |
| 源码 | Qwen3.8 兼容 SGLang（默认 `/mnt/data/xinyuzhu/sglang`，**dev 分支**） |
| 模型 | `qwen4_exp`、原生 `max_position_embeddings=262144`、带 `mtp.*`（NEXTN 需要） |
| 挂载 | 八台都挂 `/mnt/data`（模型 + 源码） |
| 网络 | 八台内网互通（可以跨 /24）；每台 8 张 RoCE v2 HCA（`mlx5_0..7`），可用 GID index = **3** |
| 端口 | 41000 / 42000（worker）、8998（bootstrap）、40000（Router）需互相放通 |

`/mnt/data/xinyuzhu/sglang` 这份源码树必须包含 Qwen4-Exp 的 PD 支持与 QSA 修复
（`git -C <SOURCE> log --oneline -1` 应包含 upstream main 的 merge）。

## 0. 快速开始（Agent 照抄这一段）

```bash
# ---- 变量 ----
export SOURCE=/mnt/data/xinyuzhu/sglang
export DIR=$SOURCE/endless-frontier/qwen3.8-next-flash/h100_4p4d
export QWEN38_MODEL_PATH=/mnt/data/yuzhucai/dlc_outputs/qwen38_flash_bio_0915_4ep/iter_0001285/hf
export PREFILL1=<prefill-1 IP> PREFILL2=<prefill-2 IP> PREFILL3=<prefill-3 IP> PREFILL4=<prefill-4 IP>
export DECODE1=<decode-1 IP>   DECODE2=<decode-2 IP>   DECODE3=<decode-3 IP>   DECODE4=<decode-4 IP>
export ROUTER=<router 所在节点 IP>          # 建议放 decode-1

# ---- 1) 先只校验（每台各跑一次自己的角色；会把 config.json 检测/改写成 1M，秒级）----
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh prefill $PREFILL1 41000 8998 --check-only   # prefill-1
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh prefill $PREFILL2 41000 8998 --check-only   # prefill-2
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh prefill $PREFILL3 41000 8998 --check-only   # prefill-3
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh prefill $PREFILL4 41000 8998 --check-only   # prefill-4
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh decode  $DECODE1  42000 8998 --check-only   # decode-1
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh decode  $DECODE2  42000 8998 --check-only   # decode-2
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh decode  $DECODE3  42000 8998 --check-only   # decode-3
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh decode  $DECODE4  42000 8998 --check-only   # decode-4

# ---- 2) 启动四台 Prefill（每台各自执行；可并行）----
nohup setsid bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh prefill <本机IP> 41000 8998 \
    > /tmp/qwen38_prefill.log 2>&1 &
# prefill 默认开 HiCache ratio 1.5（额外占 ~490GB host 内存/台；不需要就加 QWEN38_PD_HICACHE_RATIO=off）
# 等 /health 200（冷启动约 5~6 分钟：336GB 权重 + CUDA graph）
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://<本机IP>:41000/health)" = 200 ]; do sleep 10; done

# ---- 3) 启动四台 Decode（每台各自执行；建议等 Prefill 起来后再起，错开 NAS 读盘）----
nohup setsid bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh decode <本机IP> 42000 8998 \
    > /tmp/qwen38_decode.log 2>&1 &
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://<本机IP>:42000/health)" = 200 ]; do sleep 10; done

# ---- 4) Router（任一节点，建议 decode-1）----
export QWEN38_PD_PREFILL_IPS=$PREFILL1,$PREFILL2,$PREFILL3,$PREFILL4
export QWEN38_PD_DECODE_IPS=$DECODE1,$DECODE2,$DECODE3,$DECODE4
nohup setsid bash $DIR/run_qwen38_flash_next_yarn_1m_pd_router.sh > /tmp/qwen38_router.log 2>&1 &

# ---- 5) 验证 ----
curl -s http://$ROUTER:40000/health                       # 200
curl -s http://$ROUTER:40000/workers | python3 -m json.tool    # 8 个 worker 全 is_healthy=true
curl -s -X POST http://$ROUTER:40000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-1m","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

四台 Prefill 可以并行启动；Decode 建议在 Prefill 起来之后再起（错开 NAS 读盘，
336GB × 8 台同时读会明显变慢）。Router 要等 worker 就绪后再起，
也可以用 `--worker-startup-timeout-secs 1800`（脚本已带）先起 Router 等 worker。

## 关键参数（worker）

与 2P2D 完全一致（同一份 worker 脚本）：

| 参数 | 值 | 说明 |
|---|---|---|
| `--tp` | 8 | 每机 8 卡 |
| `--context-length` | 1048576 | YaRN 4 × 262144 |
| `--max-running-requests` | **96** | Prefill / Decode 相同（`QWEN38_MAX_RUNNING_REQUESTS` 覆盖） |
| `--mem-fraction-static` | **0.93** | 0.85→0.93 使 KV 池 +18%（prefill）/ +21%（decode） |
| `--page-size` | 64 | QSA 压缩注意力强制（`model_overrides/qwen4_exp.py:84`） |
| `--chunked-prefill-size` | 8192 | |
| `--cuda-graph-max-bs-decode` | 32 | 显存紧张时降到 16 |
| `--mamba-radix-cache-strategy` | `extra_buffer` | Qwen4-Exp PLE/GDN state 需要 |
| `--max-mamba-cache-size` | prefill **640** / decode 不传 | 不封顶会自动涨到 2771 槽 / 18.98 GB（用量 <1%）；封顶后 KV 池 1.74M→2.94M token（`QWEN38_PD_MAX_MAMBA_CACHE_SIZE` 覆盖，`=auto` 恢复自动） |
| `--enable-hierarchical-cache` | **prefill 默认开**（`--hicache-ratio 1.5`）/ decode 不开 | L2 host 内存池，见「HiCache」 |
| `--linear-attn-*-backend` | flashinfer / flashinfer / triton | GDN 线性注意力后端 |
| NEXTN | **只在 Decode 开** | Prefill 保持非投机路径（QSA draft-prefill 在长上下文有历史崩溃） |
| `--max-total-tokens` | 不传（按显存 profile 自动算） | 需要封顶用 `QWEN38_PD_MAX_TOTAL_TOKENS` |
| `--disaggregation-transfer-backend` | `mooncake` | RDMA 不通时用 `mooncake_tcp` |
| `--disaggregation-ib-device` | `{"0":"mlx5_0",...,"7":"mlx5_7"}` | 一卡一 HCA，`nvidia-smi topo -m` 里 GPUi↔NICi 为 PIX |
| `MC_GID_INDEX` | 3 | RoCE v2 IPv4 GID（`/sys/class/infiniband/mlx5_i/ports/1/gids/3`） |

### attention backend：Prefill 与 Decode **不一样**（正常现象）

| 角色 | 实际 backend | 为什么 |
|---|---|---|
| Prefill | **fa3** | 没有投机 → `speculative_eagle_topk is None` → Hopper + CUDA≥12.3 直接选 fa3 |
| Decode | **flashinfer** | 开了 NEXTN（`--speculative-eagle-topk 1`）+ QSA 把 `page_size` 固定成 64 → `is_no_spec_infer_or_topk_one()` 要求 `page_size in (1, None)`，条件不成立 → 落到 flashinfer 分支 |

所以 H100 这套**必须用 flashinfer ≥ 0.6.18 的镜像**（Decode 会走到
`entrypoints/engine.py:1727` 的 `assert_pkg_version("flashinfer_python", "0.6.18")`）。

## 容量（实测，脚本默认配置）

| 指标 | Prefill（每台） | Decode（每台） |
|---|---:|---:|
| KV pool `max_total_num_tokens` | **2,935,616** | **2,716,224** |
| 角色合计（×4 台） | **11,742,464** ≈ 11.74 M | **10,864,896** ≈ 10.86 M |
| `max_running_requests` | 96 | 96 |
| 并发上限（Router 口径） | 4 × 96 = 384 | 4 × 96 = 384 |
| 单请求输入 + 输出上限 | 1,048,576（`--context-length`） | 同左 |

> 每台 worker 内部是 TP8，8 个 rank 报的是同一份逻辑 KV 池，**不要乘 8**。
> 「角色合计」只是 4 个独立 cache pool 的理论相加，实际请求由 Router 分散到不同
> worker，不是一个物理连续的池子。

## 路由策略（4P4D 相比 2P2D 唯一的行为差异）

Router 的 prefill 策略从 `round_robin` 改成 **`cache_aware`**：按前缀亲和选 prefill，
让重复前缀命中**同一台**的 L1 radix cache / L2 HiCache。
2P2D 时只有两台 prefill，round_robin 命中率本来就只有 1/2，问题不突出；
到 4 台就是 1/4，前缀复用基本失效。

实测（同一批 salted prompt，`bench_pd.py`）：

| 用例 | round_robin | cache_aware |
|---|---:|---:|
| 300k 冷启动 TTFT | 16.19 s | 14.09 s |
| 300k 同前缀第二次 TTFT | 14.40 s（≈没命中） | **1.43 s / 1.51 s** |
| 800k 冷启动 TTFT | 99.48 s | 99.77 s |
| 800k 同前缀第二次 TTFT | — | **4.88 s** |

**代价**：4 路「全新前缀」同时到达时，cache_aware 会把其中两条排到同一台 prefill
（冷启动没有缓存可亲和，回落到负载估计，而估计是滞后的）：

| 4 × (300k 输入 / 2k 输出) 并发 | wall | 各请求 TTFT |
|---|---:|---|
| `cache_aware`（默认） | 37.9 s | 26.9 / 26.9 / 14.7 / 14.6 s |
| `round_robin` | **25.5 s** | 14.2 / 14.0 / 14.0 / 14.0 s |

选型建议：

* **有前缀复用**（多轮对话、固定 system prompt 的 RAG、同一篇长文反复追问）→ `cache_aware`（默认）。
* **纯吞吐、每次都是全新长 prompt**（批量评测、离线打分）→ 切回 `round_robin`：
  `QWEN38_PD_PREFILL_POLICY=round_robin bash run_qwen38_flash_next_yarn_1m_pd_router.sh`

## HiCache（L2：host DRAM 二级缓存，**prefill 默认开**）

Prefill 端再挂一层 host DRAM 缓存：GPU KV 池当 L1、host 内存池当 L2，
兜住「长 prompt 反复出现、但 GPU 池装不下」的复用场景。**prefill 默认 ratio 1.5**、
decode 不开；`QWEN38_PD_HICACHE_RATIO=off` 关闭，改值用 `=2.0` 等。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `QWEN38_PD_HICACHE_RATIO` | prefill **1.5** / decode 关 | host KV 池 = ratio × GPU KV 池（**每 rank 口径，不乘 TP**）；`off`/`none`/`disable`/`0` 关闭 |
| `QWEN38_PD_HICACHE_SIZE` | 关 | 直接给每 rank 的 host 池 GB 数，覆盖 ratio |
| `QWEN38_PD_HICACHE_WRITE_POLICY` | `write_through` | 另可选 `write_through_selective` / `write_back` |
| `QWEN38_PD_HICACHE_MEM_LAYOUT` | `page_first` | |
| `QWEN38_PD_HICACHE_STORAGE_BACKEND` + `_DIR` + `_CONFIG` | 关 | 再加 L3 磁盘层，如 `file` + `{"max_size":"128G","min_free_space":"50G"}`（`max_size` 是**每 rank** 配额，节点磁盘占用 = 8×） |

### 容量与 host 内存开销（ratio = 1.5，每台 Prefill）

| 项 | 每 rank | 每台（8 rank；token 容量不乘 8） |
|---|---:|---:|
| GPU KV 池（L1，不变） | 2,935,616 token / 33.6 GB | 2.94 M token |
| host KV 池（L2） | 4,403,456 token / 54.11 GB | 4.40 M token / **433 GB** |
| host Mamba state（HiCache 附带） | — / 7.07 GB | **57 GB** |
| 合计 host 内存 | 61.2 GB | **~490 GB**（实测 `free`：used 618 GB / total 1000 GB） |

- 单台 Prefill 理论可复用 **2.94 M（L1）+ 4.40 M（L2）= 7.34 M token**；单请求上限仍是 1 M。
- 4 台 Prefill 的 L2 合计 **17.6 M token / 1.73 TB** host 内存。
- host 预算是 `(available − 10 GiB) / ranks_per_host`，每 0.1 ratio ≈ 3.4 GB/rank
  → 1 TB 节点最多 **ratio ≈ 2.2**，超了启动直接报 `Not enough host memory available`。
- 冷启动开销 ≈ 0（不吃显存、不吃算力），只有 L2 miss 时才多一次 host→GPU 拷贝。

## 看门狗（自动重启，跑在 Router 那台机器上）

`watch_worker.py` 常驻监控 4 台 prefill + 4 台 decode（外加本机 Router），
异常时自动重启对应节点。**默认就跑在 Router 所在机器（decode-2）**。

### 为什么不能只用 `pgrep`

这套部署真正遇到的故障是**进程僵死**：`sglang.launch_server` 父进程还好好活着，
但 detokenizer 心跳停了 / scheduler 卡死 ——

```text
Health check failed. Server couldn't get a response from detokenizer for last 20 seconds.
Scheduler watchdog timeout (self.watchdog_timeout=300, self.soft=False)
Subprocess scheduler_0 (pid=54464) crashed with exit code -6.
```

`/health` 返 503（20 秒才回），进程却还在。**光 `pgrep` 查不出来，只有
「/health 连续 N 次非 200」能查出来。** 所以判定同时用两个信号：

| 信号 | 判定 | 动作 |
|---|---|---|
| SSH 上去 `pgrep sglang.launch_server` 查不到 | `process-gone` | 重启 |
| 进程在，但 `/health` 连续 3 次失败，且**首次失败之后一次活都没干** | `process-wedged` | 重启 |
| 进程在，`/health` 不正常，但首次失败之后还在干活 | 「忙」不是僵死 | **不重启**，计数清零 |
| SSH 都连不上 | `node-unreachable` | 只告警（也重启不了） |

「首次失败之后有没有干过活」= 读该节点 `/tmp/qwen38_<role>.log` 里最后一条
`Prefill batch` / `Decode batch` 的时间戳。**注意不能看日志最后一行** ——
僵死的服务会一直刷 `Health check failed`，看最后一行会以为它很活跃。

### 启动 / 停止 / 查看

```bash
# 在 Router 那台机器上（默认 decode-2 / 10.0.1.127）
cd /mnt/data/xinyuzhu/sglang/endless-frontier/qwen3.8-next-flash/h100_4p4d
nohup setsid python3 watch_worker.py > /tmp/qwen38_watchdog.out 2>&1 &

python3 watch_worker.py --status          # 看当前状态（JSON 摘要）
tail -f /tmp/qwen38_watchdog.log          # 看动作日志
python3 -c "import json;print(json.load(open('/tmp/qwen38_watchdog_status.json')))"

python3 watch_worker.py --once --dry-run  # 单次巡检，只判定不重启
kill $(cat /tmp/qwen38_watchdog.pid)      # 停止
python3 test_watch_worker.py              # 离线单测（打桩，不碰真机）
```

⚠️ **重启看门狗前先 `kill` 掉旧的**：单实例锁在 `/tmp/qwen38_watchdog.pid`，
重复启动会直接报错退出。

### 参数（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `QWEN38_WATCH_PREFILL_IPS` / `_DECODE_IPS` | 本集群 4+4 个 IP | 逗号分隔的节点列表 |
| `QWEN38_WATCH_INTERVAL` | 30 | 巡检间隔秒 |
| `QWEN38_WATCH_FAIL_THRESHOLD` | 3 | 连续失败几次算故障 |
| `QWEN38_WATCH_COOLDOWN` | **300** | 同一节点两次重启最小间隔（5 分钟） |
| `QWEN38_WATCH_STARTUP_TIMEOUT` | 900 | 重启后等它加载完的窗口（权重 336GB，约 5~6 分钟） |
| `QWEN38_WATCH_MAX_CONCURRENT` | 2 | 同时重启的节点数上限（错开 NAS 读盘） |
| `QWEN38_WATCH_HTTP_TIMEOUT` | 30 | 单次 `/health` 超时 |
| `QWEN38_WATCH_MIN_SOFT_DURATION` | 180 | 超时类失败还要持续这么久才动手 |
| `QWEN38_WATCH_DRY_RUN` | 0 | 1 = 只判定不重启 |
| `QWEN38_WATCH_MONITOR_ROUTER` | 1 | 0 = 不监控本机 Router |

### 两个必须知道的坑（都是实测踩出来的）

1. **别把「忙」当「僵」**。2026-09-17 现场：prefill-2 真挂了被重启，它的流量
   瞬间压到 prefill-3，prefill-3 的 HTTP 线程被大 chunk 堵住，`/health` 连续超时
   —— 早期版本（10s 超时 + 只看日志活跃度）把**健康的 prefill-3 也重启了**。
   现在的三重防护：`/health` 超时放宽到 30s；超时类失败要额外持续 180s；
   重启前复核「首次失败之后有没有干活」。
   判定「真故障 vs 被连累」的硬证据是**时间线**：prefill-2 首次失败在 19:37:46，
   prefill-3 首次失败在 **19:52:36**（正好是 prefill-2 摘掉之后 18 秒）。
2. **`/health` 返 503 和「超时」不是一回事**。503 是服务自己说「我不健康」
   （detokenizer 心跳断了），是硬失败；超时可能只是忙。所以两者阈值处理不同。

### 局限

- 看门狗自己**没有**被监控：它挂了就没人重启它。跑在同一台机器上的 Router 同理。
  要更稳可以两台机器各跑一份（`QWEN38_WATCH_MONITOR_ROUTER=0` 跑第二份，避免双方
  同时重启 Router）。
- 「node-unreachable」只告警不重启（SSH 都不通，重启也无从谈起）。
- 节点镜像是**没有 ssh 客户端**的（`/usr/bin/ssh` 不存在、apt 也装不了
  `openssh-client`），所以看门狗用 `paramiko`（已 `pip install`）直连内网
  `10.0.x.x:22`。换镜像时记得带上 paramiko，否则看门狗会 `ModuleNotFoundError`。

## 压测

`bench_pd.py` 用模型自带 tokenizer 构造指定长度的输入，流式统计 TTFT / 端到端 /
decode TPS（token 数取服务端 usage），默认 `ignore_eos=1` 强制生成到 `max_tokens`：

```bash
python3 $DIR/bench_pd.py --input-tokens 300000 --output-tokens 10000 \
    --url http://<router-ip>:40000 --salt s1 --label '300k/10k'
python3 $DIR/bench_pd.py --input-tokens 800000 --output-tokens 10000 \
    --url http://<router-ip>:40000 --salt s2 --label '800k/10k'
```

`--salt` 会给 prompt 加盐：**不同 salt = 全新前缀（冷启动）**，**同一 salt 重发 =
命中 L1/L2**。想做冷启动基线就每次换 salt，想验证前缀缓存就复用 salt。

实测结果见 [`RESULTS.md`](RESULTS.md)。

## 坑与开关

1. **`config.json` 就地检测**：worker 启动时先判断模型目录是不是已经是 1M 配置
   （`rope_type=yarn` + `factor=4.0` + `original_max_position_embeddings=262144`）：
   是 → 一个字节都不动；不是 → 先备份 `config.json.native.bak`，再**原子**改写。
   八台同时启动也安全（都写同样的内容），但**不要**在服务运行中反复改写该文件，
   NAS 上会出现 `OSError: [Errno 116] Stale file handle`。
2. **模型目录用原始路径，不用软链/overlay**：`--model-path` 直接指模型目录本身。
3. **`MC_GID_INDEX=3`**：本类机器每台 8 张 RoCE v2 HCA，`mlx5_i` 的 GID 3 是
   IPv4 GID（ndev=`ethi`）。换机型先 `cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/*`
   确认哪个 index 是 `RoCE v2`，再改 `QWEN38_MC_GID_INDEX`。
4. **Router 限流**：令牌桶 refill 速率 = `--max-concurrent-requests`；
   `--queue-size 0` 时超限直接 429。4P4D 脚本默认 384 / 384 / 600（= 8 worker × 96），
   要压更大并发改 `QWEN38_PD_MAX_CONCURRENT_REQUESTS` 等。
5. **重启 Router 要杀 `sglang::router`**：`pkill -f sglang_router` 杀不掉真正监听
   40000 的进程（进程名是 `sglang::router`）。用 `pkill -f 'sglang::router'`，
   否则新 Router 会因为 Prometheus 端口被占而启动失败（`Address already in use`）。
6. **NEXTN 只在 Decode**：Prefill 走非投机路径；上线前务必跑一次 >262k 的长 prefill 验证。
7. **重启 worker**：`kill -TERM -- -<pid>`（进程组）；脚本用 `nohup setsid` 启动，
   PID 记不住就用 `pgrep -f sglang.launch_server`。

## 排障

| 现象 | 处理 |
|---|---|
| Decode 启动报 `flashinfer_python>=0.6.18` | 用带 0.6.18 的 H100 镜像，或 `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`（仅排障） |
| `FileNotFoundError: model-000xx-of-00073.safetensors` | 模型目录没拷完 / 被别人改过；`--check-only` 会列出缺失分片 |
| `OSError ... Stale file handle` | NAS 上有人 rename-over 了正在读的文件；确认无人改写后重启 |
| worker `/health` 超时但进程还在、`/metrics` 有响应 | 推理线程已死（历史上是 QSA 长上下文崩溃）；看日志有没有 `illegal memory access`，重启该 worker |
| Router `/workers` 里 worker `is_healthy=false` | worker 没起来或端口未放通；先直连 worker `/health` |
| 新 Router 起不来，报 `FailedToCreateHTTPListener("Address already in use")` | 旧 Router 没杀干净，见「坑与开关」第 5 条 |
| 客户端 429 | Router 限流（见「坑与开关」第 4 条） |
| 客户端 502 / 超时 | 看是不是某个 worker 崩了（Router 会摘掉它，其余顶上） |

## 交付前验证清单

1. 八台 `--check-only` 全绿（模型分片齐全、YaRN 生效、源码树在 dev 分支）。
2. `/workers` 返回 8 个 `is_healthy=true`（4 prefill + 4 decode），`/health` 200。
3. 一个短请求 + 一个 300k 左右的长请求（10k 输出）都能正常返回。
4. 日志里没有 `CUDA error`、`illegal memory access`、`Stale file handle`。
5. 没有把 AccessKey、`.runtime/`、真实节点 IP 提交进仓库。
