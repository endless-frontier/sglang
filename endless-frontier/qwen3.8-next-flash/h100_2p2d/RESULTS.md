# H100 2P2D 参考性能（2026-09-17 实测）

## 环境

| 项 | 值 |
|---|---|
| 硬件 | 4 ×（8×H100 80G，sm90，132 SM），NV18 卡间互联 |
| 镜像 | `dptech-sh-pai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/dptech-namespace/sglang:sglang-0-5-18-qwen38-next-flash-h100-1m`（flashinfer 0.6.18 / mooncake 0.3.12 / sglang-router 0.3.2） |
| 源码 | `/mnt/data/xinyuzhu/sglang` `dev@07d41b2ae`（= dev + merge upstream main，含 QSA gather clamp 修复 `d72e59508`、MTP stream 复用 `2c0a70960`） |
| 模型 | `/mnt/data/public_data/public_model/Qwen3.8/Qwen3.8-Flash-Next-1M`（131 分片，335.3 GiB，YaRN factor=4 就地改写） |
| 拓扑 | 2 × Prefill(TP8, :41000) + 2 × Decode(TP8, :42000, NEXTN) + Router(:40000) |
| KV 传输 | Mooncake + RoCE v2（`MC_GID_INDEX=3`，8 × mlx5_i） |
| 服务参数 | `--max-running-requests 96`（两端）、`--context-length 1048576`、`--mem-fraction-static 0.85`、`--page-size 64`、`--chunked-prefill-size 8192`、`--cuda-graph-max-bs-decode 32` |

## KV cache pool 容量与显存

| 角色 | `max_total_num_tokens` | attention backend | 每卡显存占用 |
|---|---:|---|---|
| Prefill ×2 | **1,467,328** | fa3 | 73.5–75.2 GB / 79.2 GB |
| Decode ×2 | **2,246,336** | flashinfer（+ NEXTN/EAGLE） | 72.3–72.8 GB / 79.2 GB |

（单请求输入+输出上限 = `--context-length` = 1,048,576；800k 输入时 Decode 侧
`full token usage` 0.36。）

## 压测结果

客户端：`bench_pd.py`，经 Router 单请求串行，`temperature=0`、`ignore_eos=1`
（强制生成到 10,000 token）、`reasoning_effort=low`。

| 用例 | prompt tokens（服务端） | completion | **TTFT** | 端到端 | **Decode TPS**（客户端） | 服务端 gen throughput | NEXTN 接受长度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300k 输入 / 10k 输出 | 296,883 | 10,000 | **21.09 s** | 45.07 s | **416.9 tok/s** | 427.1 tok/s | 3.95–4.00 |
| 800k 输入 / 10k 输出 | 791,619 | 10,000 | **86.15 s** | 112.82 s | **374.9 tok/s** | 381.7 tok/s | 4.00 |

- Prefill 单 chunk（8192 token）实测 3.5k–18k token/s（随 chunk 位置/内容波动）；
  端到端有效输入吞吐：300k ≈ 14.1k token/s，800k ≈ 9.2k token/s（含 KV 传输与首 token）。
- 8×H100 单机（同镜像、`mem-fraction-static=0.90`）作对照：`max_total_num_tokens`
  = 1,856,704；128k 输入 TTFT ≈ 4.5 s。
- 两次长上下文压测**均未出现** QSA `CUDA error: an illegal memory access`
  （dev 已含 main 的 clamp 修复；修复前 300k+ prefill 必崩）。

## 原始输出

```text
[bench] 300k/10k: prompt=296883(server)/300000(client) completion=10000 TTFT=21.086s e2e=45.072s decode=416.9 tok/s wall=221.9 tok/s
[bench] 800k/10k: prompt=791619(server)/800000(client) completion=10000 TTFT=86.151s e2e=112.819s decode=374.9 tok/s wall=88.6 tok/s
```

Decode worker 侧同一时刻的日志：

```text
[12:50:55 TP0] Decode batch, #running-req: 1, #full token: 801536, full token usage: 0.36,
               mamba num: 1, mamba usage: 0.01, accept len: 4.00, accept rate: 1.00,
               cuda graph: True, gen throughput (token/s): 381.74
```

## 复现命令

```bash
export DIR=/mnt/data/xinyuzhu/sglang/endless-frontier/qwen3.8-next-flash/h100_2p2d
python3 $DIR/bench_pd.py --input-tokens 300000 --output-tokens 10000 --url http://<router-ip>:40000 --label '300k/10k'
python3 $DIR/bench_pd.py --input-tokens 800000 --output-tokens 10000 --url http://<router-ip>:40000 --label '800k/10k'
```

说明：客户端构造的 token 数（300,000 / 800,000）与服务端 `prompt_tokens`
（296,883 / 791,619）略有差异，来自 decode→encode 往返；以服务端计数为准。
