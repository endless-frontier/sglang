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
# prefill 默认开 HiCache ratio 1.5（额外占 ~490GB host 内存/台；不需要就加 QWEN38_PD_HICACHE_RATIO=off）
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
| `--mem-fraction-static` | **0.93** | 0.85→0.93 使 KV 池 +18%（prefill）/ +21%（decode），见「KV cache 容量调优」 |
| `--page-size` | 64 | QSA 压缩注意力强制（`model_overrides/qwen4_exp.py:84`） |
| `--chunked-prefill-size` | 8192 | |
| `--cuda-graph-max-bs-decode` | 32 | 显存紧张时降到 16 |
| `--mamba-radix-cache-strategy` | `extra_buffer` | Qwen4-Exp PLE/GDN state 需要 |
| `--max-mamba-cache-size` | prefill **640** / decode 不传 | prefill 若不封顶会自动涨到 2771 槽 / 18.98 GB（用量 <1%）；封顶后 KV 池 1.74M→2.94M token（`QWEN38_PD_MAX_MAMBA_CACHE_SIZE` 覆盖，`=auto` 恢复自动） |
| `--enable-hierarchical-cache` | **prefill 默认开**（`--hicache-ratio 1.5`）/ decode 不开 | L2 host 内存池：+4.40M token/rank（54.11 GB/rank），每台 ~490 GB 内存，冷启动 TTFT 实测无损；`QWEN38_PD_HICACHE_RATIO=off` 关闭，见「HiCache」 |
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

## 容量（实测，脚本默认配置）

| 指标 | Prefill（每台） | Decode（每台） |
|---|---:|---:|
| KV pool `max_total_num_tokens` | **2,935,616** | **2,716,224** |
| 每卡 KV 池字节 | K+V 33.6 GB（K 16.8 + V 16.8） | K+V 31.1 GB（另 NEXTN draft KV 2.6 GB） |
| `max_running_requests` | 96 | 96 |
| 每卡启动后余量 | 3.98 GB / 79.2 GB | 2.05 GB / 79.2 GB |
| 每卡权重 | 约 31.25 GB（实测 `Load weight end`） | 同左 |
| 每 token 成本 | ≈ 11.44 KB/卡 | ≈ 11.44 KB/卡 |

单请求输入 + 输出 ≤ `--context-length` = 1,048,576；800k 输入 + 10k 输出实测通过。

## KV cache 容量调优（怎么把池子做大）

初始交付（`mem-fraction-static=0.85` + mamba 自动 sizing）每台只有
Prefill 1,467,328 / Decode 2,246,336 token。两步调优后 **Prefill 2,935,616 /
Decode 2,716,224 token**，TTFT / decode TPS 实测不变：

| 步骤 | 改什么 | Prefill 池 | Decode 池 | 代价 |
|---|---|---:|---:|---|
| ① | `--mem-fraction-static 0.85 → 0.93` | 1.47M → **1.74M** | 2.25M → **2.72M** | 启动后余量只剩 3.98 / 2.05 GB，**别再往上加** |
| ② | prefill 加 `--max-mamba-cache-size 640` | 1.74M → **2.94M** | 不变 | mamba 槽位 2771 → 640（够 96 并发 + 544 个前缀缓存态） |

**第 ② 步的原理**：Qwen4-Exp 是 hybrid SSM(GDN) 模型，Mamba/SSM state 池由
`--mamba-full-memory-ratio`（默认 0.9，即「剩余显存的一半」）自动 sizing
（`python/sglang/srt/mem_cache/kv_cache_configurator.py:_handle_max_mamba_cache`）。
同一份配置下：

| 角色 | `max_mamba_cache_size` | conv+ssm 占用 | 运行期 `mamba usage` |
|---|---:|---:|---:|
| Decode | 96（自动，PD decode 关了 radix cache） | 3.27 GB | 0.01 |
| Prefill | **2771（自动）** | **18.98 GB** | **0.01** |

也就是说 prefill 白占了 ~15 GB 显存。封顶 640 槽（≈4.4 GB）后腾出的显存全部进 KV 池，
池子从 1.74M 涨到 2.94M token（+69%）。

注意两点：
- `scheduler` 会用 `max_mamba_cache_size // ratio` 反推 `max_running_requests`
  （本配置 ratio=5 = base 3 + `extra_buffer`/overlap 2），`640//5 = 128 ≥ 96`，并发不受影响；
  若日志出现 `max_running_requests is capped to N by the mamba state cache`，就是值设小了。
- Mamba 槽位同时承担「前缀缓存的 state」，所以不能无限缩；实测 640 槽下同一个 300k
  prompt 二次请求 TTFT 4.87 s → **2.06 s**（冷启 17.8 s），前缀缓存正常。

### 还想更大？（按需评估）

| 手段 | 效果 | 说明 |
|---|---|---|
| `--kv-cache-dtype fp8_e4m3` | token 容量约 ×2 | QSA 后端支持 fp8 pool（pool 直接写 fp8、无 per-tensor scale，见 `layers/attention/qsa/sparse_attn.py:429`）；会改数值精度，两端要一起切 |
| `QWEN38_PD_HICACHE_RATIO` 提到 2.5（或用 `QWEN38_PD_HICACHE_SIZE` 直接给 GB） | host KV 池 4.40M → 7.3M token/rank | **默认已开 1.5**（见「HiCache」）；1 TB 节点上限 ≈2.2（host 预算 ≈105 GB/rank），2.5 要更大内存 |
| `--hicache-storage-backend file`（`QWEN38_PD_HICACHE_STORAGE_*`） | 再叠一层 L3 磁盘缓存 | H200 那套就是 file 550 GB；`max_size` 是**每 rank** 配额（节点磁盘占用 = 8×）。本机 `/mnt/data` CPFS 46 TB 可用 |
| `--cuda-graph-max-bs-decode 32 → 16` | +1~2 GB（<5% token） | 高并发 decode 略降，性价比低 |

### 和 H200 那套（6.21M / 6.17M token 每 worker）怎么比

同事的 2P2D：32 卡（每角色 2 worker × TP8）、H200 141G，报 prefill
`24,262 pages × page_size 256 = 6.21M` token/worker、decode `24,121 × 256 = 6.17M`，
并额外开了 prefill HiCache（ratio 2.5 / file 550 GB）。

用我们实测的**每 token 11.44 KB/卡**反推就清楚了：

| 项 | H100 80G（我们） | H200 141G（同事） |
|---|---|---|
| 显存 | 79.2 GB | 141 GB |
| 权重（TP8，实测） | 31.25 GB | 31.25 GB（**一样**） |
| Mamba state（不封顶时） | 18.98 GB | 18.98 GB（**一样**） |
| 7% 余量 | 5.5 GB | 9.9 GB |
| 剩下给 KV | ~21 GB → **~1.9M token** | ~79 GB → **~6.9M token** |

结论：
1. **差的不是参数，是显存档位**。权重/激活/mamba 这些固定项两边完全一样，
   H200 的「残差」是 H100 的 3~4 倍，池子自然差 2 倍以上；按上表算 H200 应得 ~6.9M，
   与同事报的 6.21M 基本吻合（差异来自他们 mem-fraction 略低）。
2. 他们的 `--page-size 256` 与我们的 64 **不影响容量**——QSA 压缩注意力在本仓库里
   强制 `page_size=64`（`arg_groups/model_overrides/qwen4_exp.py:84`），
   比较容量只看 `max_total_num_tokens`，不要比 page 数。
3. 他们的 550 GB HiCache 是 **host/文件二级缓存**，不是 GPU KV 池；收益体现在
   重复前缀命中，不改变单请求 1M 上限。我们这边也已默认开 L2（`--hicache-ratio 1.5`，
   暂没挂 L3）：冷启动无损，300k 重复前缀 14 s → 1.6 s（见「HiCache」）。
4. 我们做完 ①+② 后，prefill 单台 2.94M / decode 2.72M 已经是 H100 80G 上能挤出的量。

## HiCache（L2：host DRAM 二级缓存，**prefill 默认开**，含实测）

Prefill 端可以再挂一层 host DRAM 缓存：GPU KV 池当 L1、host 内存池当 L2，
用来兜住「长 prompt 反复出现、但 GPU 池装不下」的复用场景（Decode 端不需要）。
**prefill 默认开启（ratio 1.5）**、decode 不开；想改值/关掉只用一个环境变量：

```bash
# 两台 Prefill 都这样起
QWEN38_PD_HICACHE_RATIO=1.5 nohup setsid bash run_qwen38_flash_next_yarn_1m_pd_worker.sh \
    prefill <本机IP> 41000 8998 > /tmp/qwen38_prefill.log 2>&1 &
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `QWEN38_PD_HICACHE_RATIO` | prefill **1.5** / decode 关 | host KV 池 = ratio × GPU KV 池（**每 rank 口径，不乘 TP**）；`off`/`none`/`disable`/`0` 关闭 |
| `QWEN38_PD_HICACHE_SIZE` | 关 | 直接给每 rank 的 host 池 GB 数，覆盖 ratio |
| `QWEN38_PD_HICACHE_WRITE_POLICY` | `write_through` | 另可选 `write_through_selective` / `write_back` |
| `QWEN38_PD_HICACHE_MEM_LAYOUT` | `page_first` | |
| `QWEN38_PD_HICACHE_STORAGE_BACKEND` + `QWEN38_PD_HICACHE_STORAGE_DIR` + `QWEN38_PD_HICACHE_STORAGE_CONFIG` | 关 | 再加 L3 磁盘层，如 `file` + `{"max_size":"128G","min_free_space":"50G"}`（`max_size` 是**每 rank** 配额，节点磁盘占用 = 8×） |

### 容量与 host 内存开销（ratio = 1.5，每台 Prefill）

| 项 | 每 rank | 每台（8 rank；token 容量不乘 8） |
|---|---:|---:|
| GPU KV 池（L1，不变） | 2,935,616 token / 33.6 GB | 2.94 M token |
| host KV 池（L2） | 4,403,456 token / 54.11 GB | 4.40 M token / **433 GB** |
| host Mamba state（HiCache 附带） | — / 7.07 GB | **57 GB** |
| 合计 host 内存 | 61.2 GB | **~490 GB**（`free` 130 GB → 620 GB） |

- 启动日志会打印 `Allocating kv hierarchical KV host pool: 4403456 tokens, 54.11 GB host memory.`
- host 预算是 `(available − 10 GiB) / ranks_per_host ≈ 105 GB/rank`，每 0.1 ratio ≈ 3.4 GB/rank
  → 1 TB 节点最多 **ratio ≈ 2.2**，超了启动直接报 `Not enough host memory available`。
- 开了之后单台 Prefill 理论可复用 **2.94 M（L1）+ 4.40 M（L2）= 7.34 M token**；
  单请求上限仍是 1 M（`--context-length`）。

### 实测（同一批 salted prompt；Router 只挂一台 Prefill 做隔离 A/B）

| 用例 | 无 HiCache | HiCache 1.5 |
|---|---:|---:|
| 300k 冷启动 TTFT | 13.84 s | 13.95 s |
| 800k 冷启动 TTFT | 99.68 s | 99.41 s（复测 99.19 s） |
| 300k 二次命中（L1）TTFT | 1.24 s | 1.85 s |
| 300k 淘汰后命中（L2）TTFT | — | **1.64 s** |
| 800k 淘汰后命中（L2，部分）TTFT | — | 59.4 s（冷启动 99.4 s） |
| decode TPS（300k / 800k） | 416.9 / 351.1 | 400.3 / 370.9 |

1. **冷启动开销 ≈ 0**：300k/800k 冷 prefill 与不开 HiCache 打平（±0.2 s），decode TPS 不变
   —— 只多花 host 内存，不吃显存、不吃算力。
2. L2 命中显著：300k 重复 prompt 从 14 s → **1.6 s**（≈ L1 命中水平）。
3. 800k 只有**部分命中**：host 池 4.40 M token，本次实验累计写入 ~6.4 M → 被淘汰；
   服务端 `#cached-token: 540672`（共 791,877），剩余 24 万 token 在长上下文位置重算
   （每 chunk 3.5–5k token/s）→ 59.4 s。要覆盖 800k 级别复用，ratio 需 ≥2.5 或再挂 L3。
4. 关闭：`QWEN38_PD_HICACHE_RATIO=off` 后重启 Prefill；改比例 `=2.0` 等（1 TB 节点上限 ≈2.2）。

## 压测

`bench_pd.py` 用模型自带 tokenizer 构造指定长度的输入，流式统计 TTFT / 端到端 /
decode TPS（token 数取服务端 usage），默认 `ignore_eos=1` 强制生成到 `max_tokens`：

```bash
python3 $DIR/bench_pd.py --input-tokens 300000 --output-tokens 10000 \
    --url http://<router-ip>:40000 --label '300k/10k'
python3 $DIR/bench_pd.py --input-tokens 800000 --output-tokens 10000 \
    --url http://<router-ip>:40000 --label '800k/10k'
```

实测结果见 [`RESULTS.md`](RESULTS.md)（最终配置：300k TTFT 17.8 s / 417 tok/s；800k TTFT 86.3 s / 365 tok/s；
基线 0.85 时为 21.1 s / 417 与 86.2 s / 375 —— 说明调容量没有牺牲 TTFT/TPS）。

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
