# PD 分离 H100 2P2D：YaRN 1M（TP8 × 4 机）

四台 **8×H100(80G)** 主机组成 PD 分离集群：**两台 Prefill + 两台 Decode**，
每个实例 TP8；YaRN factor 4 把原生 262,144 上下文扩到 **1,048,576（1M）**；
一台 Router 把 OpenAI 兼容请求 round-robin 分发给 Prefill/Decode 组合，
Prefill 的 KV 经 **Mooncake + RoCE v2** 直传 Decode。

已在本集群实测通过（4 台 8 卡 H100，300k / 800k 长上下文压测无崩溃），
参考性能见 [`RESULTS.md`](RESULTS.md)。

## 拓扑与端口

| 角色 | 数量 | 监听地址 | 说明 |
|---|---:|---|---|
| Prefill worker | 2 | `<PREFILL_IP>:41000`，bootstrap `8998` | 只做 prefill，KV 经 Mooncake 传 Decode |
| Decode worker | 2 | `<DECODE_IP>:42000` | 只做 decode，启用 NEXTN/MTP |
| Router | 1 | `0.0.0.0:40000` | `--pd-disaggregation`，prefill/decode 各自 round robin |

客户端只访问 Router：`http://<router-ip>:40000/v1/chat/completions`。
（Router 是纯 CPU 的 Rust 进程，可以放在任一节点上，也可以单独放一台机器。）

## 使用镜像（已验证）

```text
dptech-sh-pai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/dptech-namespace/sglang:sglang-0-5-18-qwen38-next-flash-h100-1m
```

镜像内容：CUDA 13.0 devel + PyTorch 2.13.0+cu130 + SGLang 0.5.18 +
`sglang-kernel==0.4.7` + `flashinfer-python/-cubin/-jit-cache` **0.6.18** +
`mooncake-transfer-engine-cuda13 0.3.12` + `sglang-router 0.3.2`。

> ⚠️ **不要用 H200 那套 PD 镜像**（`...scimaster:sglang-0-5-18-cuda13-qwen38-next-pd`）：
> 它的 flashinfer 是 0.6.17，而本配方在 Decode 上会落到 flashinfer 注意力后端，
> 启动时会被 `flashinfer_python>=0.6.18` 断言拦住（原因见下面“attention backend”一节）。

## 环境要求

四个节点必须**同镜像、同源码树、同模型目录**：

| 项 | 要求 |
|---|---|
| GPU | 4 ×（8×H100 sm90），每实例单机 TP8 |
| 镜像 | 见上方 H100 镜像，四台完全一致 |
| 源码 | Qwen3.8 兼容 SGLang（默认 `/mnt/data/xinyuzhu/sglang`，**dev 分支**） |
| 模型 | `qwen4_exp`、原生 `max_position_embeddings=262144`、带 `mtp.*`（NEXTN 需要） |
| 挂载 | 四台都挂 `/mnt/data`（模型 + 源码） |
| 网络 | 四台内网互通；每台 8 张 RoCE v2 HCA（`mlx5_0..7`），可用 GID index = **3** |
| 端口 | 41000 / 42000（worker）、8998（bootstrap）、40000（Router）需互相放通 |

`/mnt/data/xinyuzhu/sglang` 这份源码树必须包含 Qwen4-Exp 的 PD 支持与 QSA 修复
（`git -C <SOURCE> log --oneline -1` 应包含 upstream main 的 merge）。

## 0. 快速开始（Agent 照抄这一段）

```bash
# ---- 变量 ----
export SOURCE=/mnt/data/xinyuzhu/sglang
export DIR=$SOURCE/endless-frontier/qwen3.8-next-flash/h100_2p2d
export QWEN38_MODEL_PATH=/mnt/data/public_data/public_model/Qwen3.8/Qwen3.8-Flash-Next-1M
export PREFILL1=<prefill-1 内网IP> PREFILL2=<prefill-2 内网IP>
export DECODE1=<decode-1 内网IP>   DECODE2=<decode-2 内网IP>

# ---- 1) 先只校验（每台各跑一次自己的角色；会把 config.json 检测/改写成 1M，秒级）----
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh prefill $PREFILL1 41000 8998 --check-only   # 在 prefill-1 上
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh prefill $PREFILL2 41000 8998 --check-only   # 在 prefill-2 上
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh decode  $DECODE1  42000 8998 --check-only   # 在 decode-1 上
bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh decode  $DECODE2  42000 8998 --check-only   # 在 decode-2 上

# ---- 2) 启动两台 Prefill（每台各自执行）----
nohup setsid bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh prefill <本机IP> 41000 8998 \
    > /tmp/qwen38_prefill.log 2>&1 &
# 等 /health 200（冷启动约 5~6 分钟：336GB 权重 + CUDA graph）
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://<本机IP>:41000/health)" = 200 ]; do sleep 10; done

# ---- 3) 启动两台 Decode（每台各自执行）----
nohup setsid bash $DIR/run_qwen38_flash_next_yarn_1m_pd_worker.sh decode <本机IP> 42000 8998 \
    > /tmp/qwen38_decode.log 2>&1 &
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://<本机IP>:42000/health)" = 200 ]; do sleep 10; done

# ---- 4) Router（任一节点，建议 decode-1）----
export QWEN38_PD_PREFILL_IPS=$PREFILL1,$PREFILL2
export QWEN38_PD_DECODE_IPS=$DECODE1,$DECODE2
nohup setsid bash $DIR/run_qwen38_flash_next_yarn_1m_pd_router.sh > /tmp/qwen38_router.log 2>&1 &

# ---- 5) 验证 ----
curl -s http://<router-ip>:40000/health                      # 200
curl -s http://<router-ip>:40000/workers | python3 -m json.tool   # 4 个 worker 全 is_healthy=true
curl -s -X POST http://<router-ip>:40000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-1m","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

两台 Prefill 可以并行启动；Decode 建议在 Prefill 起来之后再起（错开 NAS 读盘，
336GB × 4 台同时读会明显变慢）。Router 要等 worker 就绪后再起，
也可以用 `--worker-startup-timeout-secs 1800`（脚本已带）先起 Router 等 worker。

## 关键参数（worker）

| 参数 | 值 | 说明 |
|---|---|---|
| `--tp` | 8 | 每机 8 卡 |
| `--context-length` | 1048576 | YaRN 4 × 262144 |
| `--max-running-requests` | **96** | Prefill / Decode 相同（`QWEN38_MAX_RUNNING_REQUESTS` 覆盖） |
| `--mem-fraction-static` | 0.85 | H100 80G：权重 42 GiB/卡，剩 ~26 GiB 给 KV |
| `--page-size` | 64 | QSA 压缩注意力强制（`model_overrides/qwen4_exp.py:84`） |
| `--chunked-prefill-size` | 8192 | |
| `--cuda-graph-max-bs-decode` | 32 | 显存紧张时降到 16 |
| `--mamba-radix-cache-strategy` | `extra_buffer` | Qwen4-Exp PLE/GDN state 需要 |
| `--linear-attn-*-backend` | flashinfer / flashinfer / triton | GDN 线性注意力后端 |
| NEXTN | **只在 Decode 开** | Prefill 保持非投机路径（QSA draft-prefill 在长上下文有历史崩溃） |
| `--max-total-tokens` | 不传（按显存 profile 自动算） | 需要封顶用 `QWEN38_PD_MAX_TOTAL_TOKENS` |
| `--disaggregation-transfer-backend` | `mooncake` | RDMA 不通时用 `mooncake_tcp` |
| `--disaggregation-ib-device` | `{"0":"mlx5_0",...,"7":"mlx5_7"}` | 一卡一 HCA，`nvidia-smi topo -m` 里 GPUi↔NICi 为 PIX |
| `MC_GID_INDEX` | 3 | RoCE v2 IPv4 GID（`/sys/class/infiniband/mlx5_i/ports/1/gids/3`） |

### attention backend：Prefill 与 Decode **不一样**（正常现象）

| 角色 | 实际 backend | 为什么 |
|---|---|---|
| Prefill | **fa3** | 没有投机 → `speculative_eagle_topk is None` → Hopper + CUDA≥12.3 直接选 fa3（`model_override_base.py:323-329`） |
| Decode | **flashinfer** | 开了 NEXTN（`--speculative-eagle-topk 1`）+ QSA 把 `page_size` 固定成 64 → `is_no_spec_infer_or_topk_one()`（`utils/common.py:3590`）要求 `page_size in (1, None)`，条件不成立 → 落到 flashinfer 分支 |

所以：**H100 这套必须用 flashinfer ≥ 0.6.18 的镜像**（Decode 会走到
`entrypoints/engine.py:1727` 的 `assert_pkg_version("flashinfer_python", "0.6.18")`）。
想换成 fa3 就跑不了 NEXTN，想换镜像就得让 flashinfer 够新。用
`QWEN38_ATTENTION_BACKEND` 可以强制两端都用一个后端（排障用）。

## 容量（实测，`mem-fraction-static=0.85`）

| 指标 | Prefill（每台） | Decode（每台） |
|---|---:|---:|
| KV pool `max_total_num_tokens` | **1,467,328** | **2,246,336** |
| `max_running_requests` | 96 | 96 |
| 每卡显存占用 | 73.5–75.2 GB / 79.2 GB | 72.3–72.8 GB / 79.2 GB |
| 每卡权重 | 约 41.9 GiB（335.3 GiB / 8） | 同左 |

单请求输入 + 输出 ≤ `--context-length` = 1,048,576；800k 输入 + 10k 输出实测通过
（Decode 侧 full token usage 0.36）。

## 压测

`bench_pd.py` 用模型自带 tokenizer 构造指定长度的输入，流式统计 TTFT / 端到端 /
decode TPS（token 数取服务端 usage），默认 `ignore_eos=1` 强制生成到 `max_tokens`：

```bash
python3 $DIR/bench_pd.py --input-tokens 300000 --output-tokens 10000 \
    --url http://<router-ip>:40000 --label '300k/10k'
python3 $DIR/bench_pd.py --input-tokens 800000 --output-tokens 10000 \
    --url http://<router-ip>:40000 --label '800k/10k'
```

实测结果见 [`RESULTS.md`](RESULTS.md)（300k：TTFT 21.1 s / 417 tok/s；800k：TTFT 86.2 s / 375 tok/s）。

## 坑与开关

1. **`config.json` 就地检测**：worker 启动时先判断模型目录是不是已经是 1M 配置
   （`rope_type=yarn` + `factor=4.0` + `original_max_position_embeddings=262144`）：
   是 → 一个字节都不动；不是 → 先备份 `config.json.native.bak`，再**原子**改写。
   四台同时启动也安全（都写同样的内容），但**不要**在服务运行中反复改写该文件，
   NAS 上会出现 `OSError: [Errno 116] Stale file handle`。
2. **模型目录用原始路径，不用软链/overlay**：`--model-path` 直接指模型目录本身。
3. **`MC_GID_INDEX=3`**：本类机器每台 8 张 RoCE v2 HCA，`mlx5_i` 的 GID 3 是
   IPv4 GID（ndev=`ethi`）。如果换成别的机型，先 `cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/*`
   确认哪个 index 是 `RoCE v2`，再改 `QWEN38_MC_GID_INDEX`。
4. **Router 限流**：令牌桶 refill 速率 = `--max-concurrent-requests`；
   `--queue-size 0` 时超限直接 429。脚本默认 200 / 200 / 600（并发/队列/超时），
   要压更大并发改 `QWEN38_PD_MAX_CONCURRENT_REQUESTS` 等。
5. **NEXTN 只在 Decode**：Prefill 走非投机路径；上线前务必跑一次 >262k 的长 prefill 验证。
6. **重启 worker**：`kill -TERM -- -<pid>`（进程组）；脚本用 `nohup setsid` 启动，
   PID 记不住就用 `pgrep -f sglang.launch_server`。

## 排障

| 现象 | 处理 |
|---|---|
| Decode 启动报 `flashinfer_python>=0.6.18` | 用带 0.6.18 的 H100 镜像，或 `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`（仅排障） |
| `FileNotFoundError: model-000xx-of-00131.safetensors` | 模型目录没拷完 / 被别人改过；`bash <worker.sh> ... --check-only` 会列出缺失分片 |
| `OSError ... Stale file handle` | NAS 上有人 rename-over 了正在读的文件；确认无人改写后重启 |
| worker `/health` 超时但进程还在、`/metrics` 有响应 | 推理线程已死（历史上是 QSA 长上下文崩溃）；看日志有没有 `illegal memory access`，重启该 worker |
| Router `/workers` 里 worker `is_healthy=false` | worker 没起来或端口未放通；先直连 worker `/health` |
| 客户端 429 | Router 限流（见上第 4 条） |
| 客户端 502 / 超时 | 看是不是某个 worker 崩了（Router 会摘掉它，另一个顶上） |

## 交付前验证清单

1. 四台 `--check-only` 全绿（模型分片齐全、YaRN 生效、源码树在 dev 分支）。
2. `/workers` 返回 4 个 `is_healthy=true`，`/health` 200。
3. 一个短请求 + 一个 300k 左右的长请求（10k 输出）都能正常返回。
4. 日志里没有 `CUDA error`、`illegal memory access`、`Stale file handle`。
5. 没有把 AccessKey、`.runtime/`、真实节点 IP 提交进仓库。
