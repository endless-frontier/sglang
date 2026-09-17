# H100 4P4D（TP8 × 8 机）实测结果

测试日期 2026-09-17；模型 `qwen38_flash_bio_0915_4ep/iter_0001285/hf`（qwen4_exp，YaRN 1M）；
镜像 `dptech-namespace/sglang:sglang-0-5-18-qwen38-next-flash-h100-1m`；
源码 `/mnt/data/xinyuzhu/sglang`（dev 分支）；客户端 `bench_pd.py`（流式，取服务端 usage）。

## 配置摘要

| 项 | 值 |
|---|---|
| 拓扑 | 4 prefill + 4 decode，每台 8 卡 TP8，Router 1 台 |
| `--mem-fraction-static` | 0.93 |
| `--max-running-requests` | 96（每台，P/D 相同） |
| prefill `--max-mamba-cache-size` | 640 |
| prefill HiCache | ratio 1.5（host 池 4,403,456 token/rank） |
| decode NEXTN | 开（num-steps 3 / topk 1 / draft 4） |
| Router prefill 策略 | `cache_aware`（另有 `round_robin` 对照） |

## 1. 单请求冷启动（新前缀，每次换 salt）

| 用例 | TTFT | 端到端 | decode TPS | 备注 |
|---|---:|---:|---:|---|
| 300k 输入 / 10k 输出 | **14.05 s** | 38.25 s | **413.1 tok/s** | cache_aware，最终配置 |
| 300k 输入 / 10k 输出 | 16.19 s | 40.71 s | 407.7 tok/s | round_robin，同配置（跨轮次抖动） |
| 800k 输入 / 10k 输出 | **98.98 s** | 126.69 s | **360.9 tok/s** | cache_aware，最终配置 |
| 800k 输入 / 10k 输出 | 99.48 s | 127.38 s | 358.4 tok/s | round_robin |

跨轮次抖动约 ±2 s（300k）/ ±1 s（800k），主要来自 NAS 读盘与
chunked-prefill 的排队位置，不是路由策略造成的差异。

> 说明：客户端按 300,000 / 800,000 token 构造 prompt，服务端 `prompt_tokens`
> 分别是 297,040 / 792,040（重复文本在中段被 BPE 合并）。TTFT 是首 token 时间，
> 端到端含全部 10,000 个输出 token。

## 2. 前缀复用（同一 salt 重发）

这是 4P4D 开 `cache_aware` 的核心收益：

| 用例 | 冷启动 TTFT | 同前缀第二次 TTFT | 加速比 |
|---|---:|---:|---:|
| 300k（cache_aware） | 14.09 s | **1.43 s**（复测 1.51 s） | 9.9× |
| 800k（cache_aware） | 99.77 s | **4.88 s** | 20.4× |
| 300k（round_robin） | 16.19 s | 14.40 s | 1.1×（≈没命中） |

- `cache_aware` 下 800k 也能**整段命中**：host 池 4.40 M token > 单条 792k，
  且 4 台 prefill 各管各的前缀，不会像 2P2D 那样被别的请求挤掉
  （2P2D 实测 800k 只能部分命中：59.4 s）。
- `round_robin` 下同一前缀第二次会落到另一台 prefill，L1/L2 全 miss，
  TTFT 与冷启动持平 —— 这就是 4P4D 默认改成 `cache_aware` 的原因。

## 3. 并发（4 路 × 300k 输入 / 2k 输出，同时发出）

| Router prefill 策略 | wall | 各请求 TTFT | 聚合吞吐（读+写） |
|---|---:|---|---:|
| `round_robin` | **25.47 s** | 14.19 / 14.04 / 14.02 / 14.00 s | **46,954 tok/s** |
| `cache_aware`（默认） | 37.91 s | 26.90 / 14.73 / 26.89 / 14.62 s | 31,553 tok/s |

`round_robin` 把 4 条全新前缀均匀摊到 4 台 prefill，4 条几乎同时开始 decode；
`cache_aware` 在「没有任何缓存可亲和」时会回落到滞后的负载估计，
把其中两条排到同一台，后一条要等前一条 prefill 完（26.9 ≈ 14 + 13 s）。

**结论**：前缀复用为主的在线服务用 `cache_aware`（默认）；
批量离线跑全新长 prompt 时切 `round_robin`（`QWEN38_PD_PREFILL_POLICY=round_robin`）。

单请求 decode TPS（4 路并发时各请求）：`round_robin` 382 / 365 / 340 / 346，
`cache_aware` 372 / 376 / 364 / 371 —— 并发下 decode TPS 基本不掉，
说明 4 台 decode 各跑各的、没有互相干扰。

## 4. 容量（每台 worker）

| 指标 | Prefill | Decode |
|---|---:|---:|
| `max_total_num_tokens`（GPU KV 池） | **2,935,616** | **2,716,224** |
| 角色合计（×4 台） | 11,742,464 ≈ **11.74 M** | 10,864,896 ≈ **10.86 M** |
| `page_size` | 64 | 64 |
| `max_running_requests` | 96 | 96 |
| `--context-length` | 1,048,576 | 1,048,576 |

Prefill 额外 L2（HiCache，ratio 1.5，每台）：

| 项 | 每 rank | 每台（token 容量不乘 8） |
|---|---:|---:|
| host KV 池 | 4,403,456 token / 54.11 GB | 4.40 M token / 433 GB |
| host Mamba state | — / 7.07 GB | 57 GB |
| 合计 host 内存 | 61.2 GB | ~490 GB |

4 台 Prefill 合计：GPU **11.74 M** + L2 host **17.6 M** token。
实测节点内存：prefill `used 618 GB / total 1000 GB`，decode `used 163 GB`。

单请求上限仍是 1 M（`--context-length`），200 个 1M 请求也无法同时驻留 ——
11.74 M / 10.86 M 是**整个角色池的 token 总量**，不是单请求额度。

## 5. 与 2P2D 对比（同机型、同镜像、同参数，模型不同）

| 项 | 2P2D（public 模型） | 4P4D（bio_4ep 模型） |
|---|---:|---:|
| 300k 冷 TTFT / decode TPS | 17.82 s / 416.7 | 14.05 s / 413.1 |
| 800k 冷 TTFT / decode TPS | 99.4 s / ~365 | 98.98 s / 360.9 |
| 300k 前缀复用 TTFT | 1.6 s（Router 只挂 1 台 prefill 时） | 1.43 s |
| 800k 前缀复用 TTFT | 59.4 s（部分命中） | **4.88 s（整段命中）** |
| GPU KV 池（每 worker） | 2,935,616 / 2,716,224 | 2,935,616 / 2,716,224 |
| 角色合计 KV | 5.87 M / 5.43 M | **11.74 M / 10.86 M** |
| 并发上限（Router 令牌桶） | 200 | **384**（= 8 worker × 96） |

单请求 TTFT/decode TPS 与 2P2D 基本持平（瓶颈是单条请求自己的 TP8 计算，
加机器不会让一条请求变快）；4P4D 的收益在**总容量 ×2、并发 ×2、前缀复用可用**。

## 6. 复现命令

```bash
DIR=/mnt/data/xinyuzhu/sglang/endless-frontier/qwen3.8-next-flash/h100_4p4d
TOK=<模型目录>          # 用 --tokenizer 指定，默认取 public 模型

# 冷启动：每次换 salt
python3 $DIR/bench_pd.py --url http://<router>:40000 --tokenizer $TOK \
    --input-tokens 300000 --output-tokens 10000 --salt cold-a
python3 $DIR/bench_pd.py --url http://<router>:40000 --tokenizer $TOK \
    --input-tokens 800000 --output-tokens 10000 --salt cold-b

# 前缀复用：复用同一个 salt
python3 $DIR/bench_pd.py --url http://<router>:40000 --tokenizer $TOK \
    --input-tokens 800000 --output-tokens 10000 --salt cold-b

# 4 路并发冷启动
OUT=$(mktemp -d); for i in 1 2 3 4; do
  python3 $DIR/bench_pd.py --url http://<router>:40000 --tokenizer $TOK \
      --input-tokens 300000 --output-tokens 2048 --salt "c$i" > $OUT/$i.json 2>/dev/null &
done; wait
```

> 首次请求会触发 JIT/CUDA graph 编译，TTFT 明显偏慢（4k 请求 6.3 s → 0.24 s）；
> 正式测数前先打 2~3 个短请求预热，或接受第一轮数字失真。

## 7. 稳定的结论 / 未验证项

已验证：

1. 8 worker（4P4D）全健康，300k / 800k 长上下文压测无 `CUDA error`、无
   `illegal memory access`（这是 2P2D 时期的历史问题，dev 分支已含 QSA clamp 修复）。
2. 冷启动 TTFT 与 router 策略无关；前缀复用只跟 `cache_aware` 有关。
3. 并发下 decode TPS 不掉。

未验证 / 需按业务确认：

- 更高并发（>4 路同时 300k）时的排队情况与 429 触发点（Router 限流 384）。
- 真机 GPU 型号显示为 `NVIDIA L20Z`（与前几批一致，实际按 H100 sm90 调参运行）。
- 前缀复用收益依赖「同一前缀稳定落到同一台 prefill」，即 `cache_aware` 的
  亲和命中率；热点前缀挤在一台时的倾斜程度未量化。
