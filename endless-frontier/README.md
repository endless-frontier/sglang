# endless-frontier

本目录以 `sgl-project/sglang` 的 `main` 为基线，在 `dev` 分支维护 Qwen3.8
Flash Next（`model_type=qwen4_exp`）在 H200 上的部署配方：原生 262,144 上下文
经 YaRN factor 4 扩展到 1,048,576，支持 PD 分离和阿里云 DLC/EAS 交付。

模型权重、日志、集群 ID 和凭据都不进仓库。

## 方案选择

| 目录 | 资源 | 入口 | 适用场景 |
|---|---|---|---|
| `qwen3.8-next-flash/h200` | 单机 8 GPU | `deploy_qwen38_flash_next_yarn_1m.sh` | 单机验证、微调权重评估、低并发 |
| `qwen3.8-next-flash/h200_2p2d` | 手工 SSH，4 节点 × 8 GPU | `run_qwen38_flash_next_yarn_1m_pd_{worker,router}.sh` + `watch_prefill.sh` | 已有固定内网节点、快速试验与压测 |
| `qwen3.8-next-flash/aliyun_h200_2p2d` | 阿里云 DLC，4 节点 × 8 GPU + EAS 公网转发 | `deployment/qwen38_flash_next_h200/cli.py` | 生产交付、对外提供 OpenAI 兼容 Endpoint |
| `glm5.3-flash/h100` | 单机 8 GPU（H100 级别，sm90） | `deploy_glm53_flash_1m.sh` | GLM-5.3-Flash 单机验证与起服务 |
| `glm5.3-flash/aliyun_h100_1node` | 阿里云 DLC，单节点 8 GPU | `submit_glm53_flash.py` | GLM-5.3-Flash 在 PAI 上起服务 |

三套方案共用同一套模型/YaRN 配置、同一组 parser（`qwen3` / `qwen3_coder`）和
同一套 PD 参数（TP8、Mooncake、Router 限流 200/200/600）。

## 文档导航

- `qwen3.8-next-flash/DEPLOYMENT_PRACTICE.md`：**先读这份**。环境基线、CUDA/
  sglang-kernel/PYTHONPATH 三大坑、YaRN 配置、PD 参数、Router 429、QSA 长上下文
  崩溃、日志与重启手册、交付验证清单。
- `qwen3.8-next-flash/h200/README.md`：单机启动与容量。
- `qwen3.8-next-flash/h200_2p2d/README.md`：四机 PD 的拓扑、端口、参数、
  watchdog、重启运行手册、实测性能、已知问题。
- `qwen3.8-next-flash/aliyun_h200_2p2d/README.md`：DLC 作业 + EAS 代理的
  资源清单、渲染/提交/停止、EAS 控制面命令、10 条实战踩坑。
- `glm5.3-flash/README.md`：**GLM-5.3-Flash 单机配方**——模型要点、DSA/KDA 后端选择、
  三个坑（镜像自带 SGLang 太老 / `fa3` 会在图捕获阶段崩 / DLC 命令必须单行无引号）、验证清单。
  实测数字见 `glm5.3-flash/RESULTS.md`。

## 通用环境

- CUDA 13.0 **devel**（含 `nvcc` 与 toolkit headers）+ PyTorch cu130 +
  SGLang 0.5.18 + `sglang-kernel==0.4.7` + FlashInfer + Triton。
- 四节点/四 Pod 必须使用同一镜像、同一份 SGLang 源码
  （`/mnt/data/xinyu/sglang-qwen38-upstream-1789383617`）、同一模型目录
  （`/mnt/data/public_models/Qwen3.8-Flash-Next`）。
- 跨节点 PD 需要 IB/RDMA + IBGDA/GDRCopy + Mooncake，RoCE GID 默认 3。
- 模型目录首次启动会被写入 YaRN 配置，原生版本备份在 `config.json.native.bak`。

## 仓库卫生

不要提交：AccessKey/Secret、DLC receipt、`.runtime/`、`rendered/` 状态文件、
真实集群/网关/网段 ID、模型权重、编译缓存。模板一律使用
`REPLACE_WITH_...` 占位符；节点地址在文档中以 `<prefill-ip>` 之类的占位符表示。
