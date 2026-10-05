# Qwen3.8-Flash-Next：单机 8 卡路线（H100 级 / sm90）

**Qwen3.8-Flash-Next** 在单台 8 卡 H100 级节点上的服务配方，用我们自己的镜像，已实测。

> 本目录只覆盖 **单机 8 卡** 路线（含 PAI 上的 DLC 作业）。同一模型的多机 PD 分离路线见
> 本仓库 dev 分支的 [`qwen3.8-next-flash/`](https://github.com/endless-frontier/sglang/tree/dev/rsw/deploy/endless-frontier/qwen3.8-next-flash)（团队维护）。
> 另注意：**2.4T 旗舰是另一个模型**，见 [`../qwen3.8-2.4t-a95b/`](../qwen3.8-2.4t-a95b/README.md)。

## 目录内容

| 路径 | 作用 |
| --- | --- |
| [`launch/deploy_qwen38_flash_next_1m.sh`](launch/deploy_qwen38_flash_next_1m.sh) | 单机 8 卡启动脚本（1M 上下文） |
| [`aliyun_h100_1node/`](aliyun_h100_1node/) | PAI DLC 作业：`submit_qwen38_flash_next.py` + 配置模板 + `job_entry.py` |
| [`image/`](image/README.md) | 镜像说明（**与 GLM-5.3-flash 同一个镜像**） |
| [`RESULTS.md`](RESULTS.md) | 实测数据（加载、显存、请求延迟、长上下文） |

## 要点（容易踩的）

1. **同一个镜像服务 GLM-5.3-flash 与本模型**：源码提交、依赖集合完全一致，差别只在启动参数。
   镜像定义在 [`../glm5.3-flash/image/`](../glm5.3-flash/image/README.md)。
2. **1M 上下文来自检查点自带 YaRN**：`Qwen3.8-Flash-Next-1M/config.json` 的 `text_config` 里
   `rope_type=yarn, factor=4.0`，所以**不需要改写任何共享文件**；只要求用 `-1M` 那份检查点，
   并把 `max_model_len` 放开。
3. **模型标识是 `qwen4_exp`**（`Qwen4ExpForConditionalGeneration`），带 MTP 头。
4. 权重路径请用自己环境里的实际路径；`pai/configs` 里的模板与 DLC 作业定义都有 `REPLACE_WITH_…`。

## 实测（摘要）

单机 8 卡从自有镜像启动：约 7–8 分钟到 `/health` 200（131 个分片约 3 分钟加载、每卡约 33.7 GB）；
`pong` 0.7 s、~3.9k token 的 `longpong` 1.9 s，均 `finish_reason=stop`。完整数字见 [RESULTS.md](RESULTS.md)。
