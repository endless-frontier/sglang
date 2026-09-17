# H100 2P2D 参考性能（2026-09-17 实测）

## 环境

| 项 | 值 |
|---|---|
| 硬件 | 4 ×（8×H100 80G，sm90，132 SM），NV18 卡间互联 |
| 镜像 | `dptech-sh-pai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/dptech-namespace/sglang:sglang-0-5-18-qwen38-next-flash-h100-1m`（flashinfer 0.6.18 / mooncake 0.3.12 / sglang-router 0.3.2） |
| 源码 | `/mnt/data/xinyuzhu/sglang` `dev@3d31adc95`（= dev + merge upstream main，含 QSA gather clamp 修复 `d72e59508`、MTP stream 复用 `2c0a70960`） |
| 模型 | `/mnt/data/public_data/public_model/Qwen3.8/Qwen3.8-Flash-Next-1M`（131 分片，335.3 GiB，YaRN factor=4 就地改写） |
| 拓扑 | 2 × Prefill(TP8, :41000) + 2 × Decode(TP8, :42000, NEXTN) + Router(:40000) |
| KV 传输 | Mooncake + RoCE v2（`MC_GID_INDEX=3`，8 × mlx5_i） |
| 服务参数 | `--max-running-requests 96`（两端）、`--context-length 1048576`、`--mem-fraction-static 0.93`、prefill `--max-mamba-cache-size 640`、`--page-size 64`、`--chunked-prefill-size 8192`、`--cuda-graph-max-bs-decode 32` |

## KV cache pool 容量与显存

| 角色 | `max_total_num_tokens` | attention backend | 每卡 KV / 余量 |
|---|---:|---|---|
| Prefill ×2 | **2,935,616** | fa3 | K+V 33.6 GB，余 3.98 GB / 79.2 GB |
| Decode ×2 | **2,716,224** | flashinfer（+ NEXTN/EAGLE） | K+V 31.1 GB（另 draft 2.6 GB），余 2.05 GB / 79.2 GB |

每 token 成本 ≈ **11.44 KB/卡**（K 与 V 各半），四台一致；单请求输入 + 输出
上限仍 = `--context-length` = 1,048,576（800k 输入 + 10k 输出实测通过）。

### 从 1.47M/2.25M 到 2.94M/2.72M 的两步调优（TTFT/TPS 实测不变）

| 配置 | Prefill 池 | Decode 池 |
|---|---:|---:|
| 基线：mamba 自动 sizing + `mem-fraction-static 0.85` | 1,467,328 | 2,246,336 |
| ① `--mem-fraction-static 0.85 → 0.93` | 1,735,232 | 2,716,224 |
| ② prefill `--max-mamba-cache-size 640`（最终配置 = ①+②） | **2,935,616** | 2,716,224 |

原理见 [`README.md`](README.md) 的「KV cache 容量调优」：hybrid SSM(GDN) 的 Mamba state
池在 prefill 上被自动 sizing 到 2771 槽 / 18.98 GB，而 `mamba usage` 常年 0.01。

## 压测结果

客户端：`bench_pd.py`，经 Router 单请求串行，`temperature=0`、`ignore_eos=1`
（强制生成到 10,000 token）、`reasoning_effort=low`。

| 用例 | prompt tokens（服务端） | completion | **TTFT** | 端到端 | **Decode TPS**（客户端） | NEXTN 接受长度 |
|---|---:|---:|---:|---:|---:|---:|
| 300k 输入 / 10k 输出（基线 0.85） | 296,883 | 10,000 | 21.09 s | 45.07 s | 416.9 tok/s | 3.95–4.00 |
| 300k 输入 / 10k 输出（最终配置） | 296,883 | 10,000 | **17.82 s** | **41.81 s** | **416.7 tok/s** | 4.00 |
| 800k 输入 / 10k 输出（基线 0.85） | 791,619 | 10,000 | 86.15 s | 112.82 s | 374.9 tok/s | 4.00 |
| 800k 输入 / 10k 输出（最终配置） | 791,619 | 10,000 | **86.26 s** | **113.63 s** | 365.4 tok/s | 4.00 |

- 前缀缓存（最终配置，640 mamba 槽）：同一个 300k prompt 二次请求 TTFT 4.87 s → **2.06 s**
  （冷启动 17.8 s），缩容后 radix 前缀缓存仍正常。
- Prefill 单 chunk（8192 token）实测 3.5k–18k token/s（随 chunk 位置波动）；
  端到端有效输入吞吐：300k ≈ 16.7k token/s，800k ≈ 9.2k token/s（含 KV 传输与首 token）。
- 两次长上下文压测**均未出现** QSA `CUDA error: an illegal memory access`
  （dev 已含 main 的 clamp 修复；修复前 300k+ prefill 必崩）。

## 原始输出

```text
[bench] 300k/10k:          prompt=296883 completion=10000 TTFT=21.086s e2e=45.072s decode=416.9 tok/s
[bench] 800k/10k:          prompt=791619 completion=10000 TTFT=86.151s e2e=112.819s decode=374.9 tok/s
[bench] 300k/10k mamba640: prompt=296883 completion=10000 TTFT=17.817s e2e=41.811s decode=416.7 tok/s
[bench] 800k/10k mamba640: prompt=791619 completion=10000 TTFT=86.263s e2e=113.628s decode=365.4 tok/s
[bench] prefix-probe1:     prompt=296883 completion=200    TTFT=4.866s
[bench] prefix-probe2:     prompt=296883 completion=200    TTFT=2.064s
```

## 复现命令

```bash
export DIR=/mnt/data/xinyuzhu/sglang/endless-frontier/qwen3.8-next-flash/h100_2p2d
# worker 脚本默认已是调优配置（mem-fraction 0.93 + prefill mamba 640），直接启动即为上表数值
python3 $DIR/bench_pd.py --input-tokens 300000 --output-tokens 10000 --url http://<router-ip>:40000 --label '300k/10k'
python3 $DIR/bench_pd.py --input-tokens 800000 --output-tokens 10000 --url http://<router-ip>:40000 --label '800k/10k'
```

说明：客户端构造的 token 数（300,000 / 800,000）与服务端 `prompt_tokens`
（296,883 / 791,619）略有差异，来自 decode→encode 往返；以服务端计数为准。

## HiCache L2（prefill `--hicache-ratio 1.5`）实测（2026-09-17 晚）

在 Prefill-1 上开 HiCache、Prefill-2 保持关闭，Router 临时只挂一台 Prefill 做隔离 A/B；
两侧用同一批带随机前缀盐（`--salt`）的 prompt，保证每次都真正冷启动。

> 交付默认：**prefill 开 ratio 1.5、decode 不开**（脚本默认值；`QWEN38_PD_HICACHE_RATIO=off` 关闭）。
> 下面是当初做取舍时的隔离 A/B 证据。

| 用例 | 无 HiCache（Prefill-2） | HiCache 1.5（Prefill-1） |
|---|---:|---:|
| 300k 冷启动 | TTFT 13.84 s / 416.9 tok/s | TTFT 13.95 s / 400.3 tok/s |
| 800k 冷启动 | TTFT 99.68 s / 351.1 tok/s | TTFT 99.41 s / 370.9 tok/s（复测 99.19 s） |
| 300k L1 命中（同 prompt 重发） | 1.24 s | 1.85 s |
| 300k L2 命中（12 × 300k 把 L1 挤爆后重发） | — | **1.64 s** |
| 800k L2 命中（同上） | — | 59.4 s |
| 节点 host 内存 | ~130 GB | ~620 GB（+490 GB） |

服务端日志证据（Prefill-1）：

```text
# 300k L1 命中
[15:37:50] Prefill batch, #new-token: 19,   #cached-token: 296960, ...
# 300k L2 命中（L1 已被 12×300k 挤掉）
[15:41:42] Prefill batch, #new-token: 19,   #cached-token: 296960, ...   input throughput 4.34 tok/s
# 800k L2 部分命中：540,672 token 从 host 恢复，另 243,013 重算
[15:41:53] Prefill batch, #new-token: 8192, #cached-token: 540672, #pending-token: 243013, ... 1515.25 tok/s
```

边界与注意：

- host 池每 rank 4,403,456 token / 54.11 GB（8 rank ≈ 433 GB），另有 7.07 GB/rank 的
  host Mamba state；节点 1 TB 内存最多支持 **ratio ≈ 2.2**（每 0.1 ratio ≈ 3.4 GB/rank）。
- 本次实验累计写入 host 池约 6.4 M token > 4.40 M 容量 → 800k 那次只恢复了 54 万 token，
  余下 24 万 token 落在长上下文位置重算（每 chunk 3.5–5k token/s），所以是 59 s 而不是几秒。
- 300k 的 L2 命中 1.64 s 与 L1 命中同级；冷启动两种配置无差别。
- `/flush_cache` 会连 host 池一起清空（`hiradix_cache.reset()`），别用它测 L2。

> **对上方「压测结果」的修正**：800k / 86.26 s 那行是 prompt 前缀已被前面的 300k 跑热
> （服务端 `#cached-token: 296832`）时测的；真正冷启动的 800k 现在实测 **99.4–99.7 s**
> （开不开 HiCache 一致）。300k 冷启动当时测到 17.82 s，本次隔离 A/B 为 13.84–13.95 s，
> 差异来自测量方式（单 Prefill 隔离、无并发干扰），以本次为准。

## 参考：H200 2P2D（同事那套 6.21M / 6.17M token 每 worker）

对比与归因见 [`README.md`](README.md)「KV cache 容量调优 → 和 H200 那套怎么比」。
一句话：权重（31.25 GB/卡）和 mamba 池两边一样，H200 141G 的显存残差是 H100 80G 的
3~4 倍，池子因此差 2 倍以上；对方没有额外开关，我们这边把 mamba 还给 KV 后已经贴到
H100 的上限。
