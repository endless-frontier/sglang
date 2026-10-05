# Qwen3.8-Flash-Next 实测结果

记录本配方在实测环境下的数字。日期、作业名与镜像都写清楚，便于复核。

## 环境

- 节点：8 × `L20Z`（H100 级别，80 GiB/卡，sm90），Lingjun 资源
- 镜像：`wangruisi/sglang-glm53-flash:sglang-d6221be-glm53-flash-l20z-1node-20261003-1a4d8b6`
  —— 与 GLM-5.3-flash **同一个镜像**（同源码提交、同依赖集），只是启动参数不同
- 源码：镜像自带 `/opt/sglang/python`，**不注入任何共享 CPFS 源码树**
- 模型目录：`/mnt/data/public_data/public_model/Qwen3.8/Qwen3.8-Flash-Next-1M`
  （131 分片、约 336 GB；`model_type=qwen4_exp`／`Qwen4ExpForConditionalGeneration`，带 MTP）

## 启动与显存（2026-10-03，自有镜像）

| 指标 | 实测值 |
|---|---|
| 权重加载 | 131 分片多线程加载，约 3 分钟（日志显示 ~2.5 s/分片） |
| 每卡显存（加载后） | 约 33.7 GB |
| 宿主内存峰值 | 约 144 GiB（与首次实测的 ~165 GiB 同一量级） |
| 全流程 | 从作业启动到 `/health` 200 约 7–8 分钟（含图捕获） |
| 上下文 | `/v1/models` 报 `max_model_len=1048576` |

## 请求实测（2026-10-03，作业 `qwenimg-20261003-060921`）

| 请求 | 结果 | 延迟 | token |
|---|---|---|---|
| `Reply with exactly: pong` | `content` 以 `pong` 结尾，`finish_reason=stop` | 0.7 s | 57 prompt / 28 completion |
| ~3.9k token 填充 + `reply with exactly: longpong` | `content` 以 `longpong` 结尾，`finish_reason=stop` | 1.9 s | 3860 prompt / 37 completion |

与 2026-10-01 首次实测（团队镜像 + 共享源码树）对比：短请求 0.9 s → 0.7 s，长请求 1.9 s → 1.9 s，
即**自有镜像没有引入任何性能回退**。

该检查点的输出把思考过程写在 `content` 里（形如 `We need ... </think>\n\npong`），不是单独的
`reasoning_content` 字段——调用方如果要纯答案，需自行截断或提示模型只输出结果。

## 长上下文实测（2026-10-03，自有镜像，单机 8 卡）

启动日志给出 **`max_total_num_tokens=1665984`**（KV 池，约 1.67M token）与 `context_len=1048576`。
与 GLM-5.3-flash 同机同卡对比：这份配方（`--mem-fraction-static 0.90` + NEXTN，无 DSA）的 KV 池是 GLM 那套的约两倍，
因此 **~950k token 的单请求在本机可以跑通**。

| 请求长度 | 结果 | 说明 |
|---|---|---|
| ~16k | ✅ | 交付验收要求的那一档 |
| ~300k | ✅ | |
| ~950k | ✅ | 接近 1M 的长请求可服务，单请求耗时 156.5 s |

单请求耗时（同一次加载内实测）：短请求 0.7 s；~16k 3.8 s；~300k 14.6 s；~950k 156.5 s。
这里 ~16k 没有像 GLM 那样被预热拖慢——同一镜像、同一套预热路径，已经跑过一次。

## 与团队脚本的三点区别（都在 `launch/deploy_qwen38_flash_next_1m.sh` 里写明）

1. 源码取自镜像自身，不指向 `/mnt/data/xinyuzhu/sglang` 之类的共享树。
2. **不改写共享模型文件**：该检查点的 `text_config` 已带 YaRN（factor=4.0），只需
   `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` 放行即可；团队脚本会把 `config.json`
   就地重写（留 `config.json.native.bak`），在共享目录上正是他们自己实践笔记里说的风险。
3. 校验由作业日志单独记录（分片数、`text_config` 读数、镜像内版本），脚本本身只管启动。

## 失败尝试（保留记录）

| 现象 | 原因 | 结论 |
|---|---|---|
| 启动即 `ValueError: User-specified context_length (1048576) is greater than the derived context_length (262144)` | 该检查点的 YaRN 在 `text_config` 里，SGLang 推导出的是原生 262144 | 需要 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`（团队脚本也设了这一条） |
| 团队文档给的模型路径不存在（`/mnt/data/public_models/…`） | 文档过期 | 实际权重在 `/mnt/data/public_data/public_model/Qwen3.8/`，且有多份变体 |

## 尚未验证

- 并发与吞吐：本表只有单请求数字。
- 多机与 PD 分离：未接；本目录只覆盖单机 8 卡。
- EAS 侧的长上下文：未测（EAS 上只验证了 GLM 的端点）。
