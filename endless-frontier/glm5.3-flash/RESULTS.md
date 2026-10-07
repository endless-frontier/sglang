# GLM-5.3-Flash 实测结果

记录本配方在实测环境下的数字。日期、任务 ID 与镜像都写清楚，便于复核。

## 环境

- 节点：8 × `L20Z`（H100 级别，80 GiB/卡，sm90），Lingjun 资源
- 镜像：CUDA 13.0 + SGLang 0.5.18 + sglang-kernel 0.4.7 + flashinfer-python 0.6.18 + torch 2.13.0+cu130
- 源码树：仓库 `dev` 分支（含 `glm5_next`），放在共享 CPFS 上由 `PYTHONPATH` 注入
- 模型目录：`/mnt/data/public_data/public_model/GLM5.3/GLM-5.3-Flash`（62 分片，约 306 GB，FP8 e4m3）

## 启动与显存

| 指标 | 实测值 |
|---|---|
| 权重加载 | `Load weight end. elapsed=272.31 s`（62 分片，多线程加载） |
| 每卡显存（权重加载后） | `mem usage=37.98 GB`，`avail mem=39.56 GB`（TP8、FP8） |
| 每卡显存（图捕获后、空载） | 约 70 GB |
| 宿主内存占用 | 约 30 GiB |
| 从启动到 `/health` 200 | 约 8 分钟（加载 ~4.5 分钟 + DeepGEMM warmup + decode 图捕获） |
| 图捕获期间 GPU 利用率 | 0 ↔ 100% 交替；生成期间各 rank 30%~95% |

## 请求实测

| 请求 | 结果 | 延迟 | token |
|---|---|---|---|
| `Reply with exactly: pong` | `content='pong'`，`finish_reason=stop` | 0.3 s | 17 prompt / 35 completion（含 33 思考） |
| ~3.8k token 填充 + `reply with exactly: longpong` | `content='longpong'`，`finish_reason=stop` | 37.7 s | 3820 prompt / 47 completion（含 44 思考） |

同一批请求里也能看到该检查点的行为特征：答案在 `content`，思考过程在 `reasoning_content`
（上表两例分别为 130 / 206 字符）。因此**调用方必须为 thinking 预留 `max_tokens`**，
否则会以 `finish_reason=length` 结束而拿不到正文——首次冒烟测试就是这样，正文为空。

## 用自有镜像复测（2026-10-03）

自有镜像里已含源码与依赖（见 [`image/`](image/README.md)），作业不再注入任何共享源码树。

- 镜像：`wangruisi/sglang-glm53-flash:sglang-d6221be-glm53-flash-l20z-1node-20261003-1a4d8b6`
- 作业：`glmimg-20261003-045425`（单机 8 卡，8 卡上限与上面同一硬件）

| 请求 | 结果 | 延迟 | token |
|---|---|---|---|
| `Reply with exactly: pong` | `content='pong'`，`finish_reason=stop` | 0.3 s | 17/35（含 33 思考） |
| ~3.8k token 填充 + `longpong` | `content='longpong'`，`finish_reason=stop` | 33.4 s | 3820/51（含 48 思考） |

与上面用团队镜像 + 共享源码树的数字一致，说明自有镜像没有引入回退。

## 通过 EAS 公网入口实测（2026-10-03）

服务名 `ef_glm53_flash_1node`，8 卡，共用 Lingjun 额度，CPFS 直接挂到 `/mnt/data/`，
镜像与启动参数同上面一条。

| 请求 | 结果 | 延迟（含网关与网络） |
|---|---|---|
| `/health` | 200（需带服务访问令牌） | — |
| `Reply with exactly: pong` | `content='pong'`，`finish_reason=stop` | **1.2 s** |
| ~3.8k token 填充 + `longpong` | `content='longpong'`，`finish_reason=stop` | **34.8 s** |

短请求多出约 0.9 s 的网关与网络开销，长请求基本一致。EAS 路线上的四个坑（script 引用形式、
`cuda-compat` 遮住平台注入的驱动、健康检查窗口、镜像用公网域名）见 [`eas/README.md`](eas/README.md)。

## 长上下文实测（2026-10-03，单机 8 卡）

启动日志给出本配置下 KV 池的实际容量：**`max_total_num_tokens=855168`**（同一个镜像与参数，`context_len=1048576` 是声明值）。
也就是说：**模型声明 1M，单机这套配置实际可用约 855k token**，差约 18%。

| 请求长度 | 结果 | 说明 |
|---|---|---|
| ~16k | ✅ | 交付验收要求的那一档 |
| ~300k | ✅ | 服务端读数 `#full token: 300096`，prefill 吞吐约 21k token/s |
| ~700k | ✅ | |
| ~840k | ✅ | 接近池上限仍可服务 |
| ~950k | ❌ | `Input length (950020 tokens) exceeds the maximum allowed length (855162 tokens)` —— 干净拒绝，不是崩溃；重跑一次结果一致 |

对照组（同机同卡、同一天、同一个自有镜像，换 Qwen3.8-Flash-Next 的参数）：它的 KV 池是 **1,665,984** token，
**~950k 的单请求可以跑通**。所以“能不能到 1M”取决于配方（KV 池大小），不是模型的硬声明。

要把 GLM 推到真正的 1M，只有三条路：调高 `--mem-fraction-static`（0.78 时已出现过分配告警，需实测权衡）、
换更省 KV 的结构（FP8 KV 只配 Blackwell 的 TRT-LLM DSA）、或上多机 / PD。

## 失败尝试（保留记录，便于排障）

| 现象 | 原因 | 结论 |
|---|---|---|
| `KeyError: 'glm5_next'` → `ValueError: ... has model type glm5_next but Transformers does not recognize this architecture` | 镜像自带 SGLang 早于 GLM-5.3 | 必须注入带 `glm5_next` 的源码树 |
| 62 分片加载完成后 `RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 64, 1, 0]` + `Exception: Capture cuda graph failed` | 用了 `--attention-backend fa3`，该模型必须用 DSA 后端 | 换成 `--attention-backend dsa` + TileLang DSA |
| `Syntax error: ";" unexpected` / `unexpected end of file` | DLC 以 `/bin/sh` 执行 `UserCommand`，多行带引号的命令被拆坏 | 命令改单行、不含引号，逻辑放进 base64 脚本 |

## 尚未验证

- 长上下文（≥262k）单请求：官方注明未修复前 QSA 类长上下文曾崩，本模型需另行实测。
- 并发与吞吐：本表只有单请求数字，未做批量压测。
- PD 分离 / 多机：未接。
- EAS 侧的长上下文与并发：未测。Qwen3.8-Flash-Next 的 EAS 入口同样未接。
