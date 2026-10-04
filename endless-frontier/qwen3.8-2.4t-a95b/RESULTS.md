# 实测结果 — Qwen3.8-2.4T-A95B

2026-10-04 在 PAI 上实测。形态：**EAS 多机分布式推理，1 个副本跨 6 台 H100 级机器（48 卡）**，
权重 2.496 TB 的 FP8 检查点从共享存储直接读取，客户端经公网入口访问。

## 请求实测

| 请求 | 输入 tokens | 回答 | `finish_reason` | 耗时 |
| --- | --- | --- | --- | --- |
| `Reply with exactly: pong` | 57 | `pong` | stop | **6.2 s** |
| 填充 + `Reply with exactly one word: longpong` | 16,065 | `longpong` | stop | **32.3 s** |
| 两个暗号分别埋在开头与 ~12 万 token 处 | 240,100 | 两个暗号都按序答对 | stop | **88.5 s** |

最后一行才是长上下文的关键证据：信息必须**从开头和 12 万 token 深处被取回**，
如果服务层悄悄截断或滑窗丢弃前文，这一条必然失败。

## 启动耗时

| | |
| --- | --- |
| 实例起来 → 权重加载完 → 服务就绪 | 约 **18 分钟**（2.5 TB 读入 48 张卡） |
| 服务转为 `Running` | 重新部署后约 5 分钟 |

## 这个模型的脾气（上线前必须知道）

- **思考（thinking）永远开启，且算在输出 token 预算里。** `max_tokens: 64` 会返回
  空 `content` + `finish_reason: length` —— 预算全花在思考上了；至少留几百。
- **推理过程单独返回**在 `reasoning_content`，正式答案在 `content`。
- **`max_model_len` = 262,144**（检查点原生窗口）。官方的 ~1M 扩展配置本次未测。

## 身份与证据

| | |
| --- | --- |
| 镜像 | `sglang-main0e6d7eb-qwen38-24t-a95b-h100-eas-20261004`，digest `sha256:9478bff4fcb257418e7ee7a6e5eba572eddc58e4f9e501e81f10a93289ef1d35`（8.86 GB 压缩 / 18 GB 解压） |
| SGLang 源码 | `sgl-project/sglang` @ `0e6d7eba5f16ff3e35e58622a72bf0c36bafde63`（main，官方模型卡指向的 day‑0 构建） |
| 权重 | `Qwen/Qwen3.8-2.4T-A95B-FP8` revision `d2dc3565`，224 个文件、2.496 TB；213 个分片的 sha256 全部校验通过 |
| 并行形态日志 | `nnodes 6 · node_rank 0–5 · tp_size 8 · pp_size 6 · dist_init_addr :20000`，NCCL 走 RDMA（`NET/IB/…/GDRDMA`） |
| 平台注入变量 | 每个实例都拿到 `RANK_ID` 0–5、`COMM_IFNAME=net0`、`RANK_IP`、`MASTER_ADDRESS`=0 号实例 IP |

## 未测（不要在报告里写成已测）

并发与吞吐；超过 262,144 的窗口；专家并行（EP）在本硬件上的表现。
另外：**首次部署曾长时间停在 `Waiting`、入口返回 503**，原样重新部署即正常 `Running` ——
记录为偶发，未解释。
