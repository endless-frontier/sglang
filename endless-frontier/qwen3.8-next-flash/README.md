# Qwen3.8 Flash Next 部署配方

六套可直接照抄的部署方案。**没接触过这个项目的同学（或 Agent）：先看下面这张表，
选一行，进对应目录读 README。**

| 场景 | 目录 | 机器 | 已验证镜像 |
|---|---|---|---|
| **单机最快起服务** | [`h100/`](h100/README.md) | 8×H100 | `dptech-sh-pai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/dptech-namespace/sglang:sglang-0-5-18-qwen38-next-flash-h100-1m` |
| **PD 分离（手工 SSH，4 机）** | [`h100_2p2d/`](h100_2p2d/README.md) | 4×8×H100 | 同 `h100/` 镜像（flashinfer 0.6.18） |
| **PD 分离（手工 SSH，8 机 4P4D，容量×2 / 并发×2）** | [`h100_4p4d/`](h100_4p4d/README.md) | 8×8×H100 | 同 `h100/` 镜像（flashinfer 0.6.18） |
| 单机 | [`h200/`](h200/README.md) | 8×H200 | `pai-ai-prod-acr-registry.cn-shanghai.cr.aliyuncs.com/acr_namespace/scimaster:sglang-0-5-18-cuda13-qwen38-next-pd` |
| PD 分离（手工 SSH，4 机） | [`h200_2p2d/`](h200_2p2d/README.md) | 4×8×H200 | 同上 |
| PD 分离（阿里云 EAS/DLC） | [`aliyun_h200_2p2d/`](aliyun_h200_2p2d/README.md) | 4×8×H200 | 同上（配置模板里填 image URI） |

共用文档：[`DEPLOYMENT_PRACTICE.md`](DEPLOYMENT_PRACTICE.md) —— 环境基线、四个环境坑、
YaRN 写法、PD 参数、Router 限流、已知缺陷、验证清单。

## 30 秒决策

- **有一台 8 卡 Hopper 机器，想马上验证模型/微调权重？** → `h100/`（H100）或 `h200/`（H200）。
- **要扛线上流量、要 PD 分离？** → `h100_2p2d/`（4×H100）、`h200_2p2d/` 或 `aliyun_h200_2p2d/`。
- **流量大、要更大 KV 总量和更高并发？** → `h100_4p4d/`（8×H100）：
  单请求 TTFT/TPS 与 2P2D 持平，但 KV 总量 ×2（prefill 11.74M / decode 10.86M token）、
  并发 ×2（384），且 Router 默认 `cache_aware` 让长前缀可跨请求复用
  （300k 复现 14.1 s → 1.43 s，800k 复现 99.8 s → 4.88 s）。

## 最小可用命令（H100 单机）

```bash
export QWEN38_MODEL_PATH=<你的 HF 模型目录>
export SCRIPT=/mnt/data/xinyuzhu/sglang/endless-frontier/qwen3.8-next-flash/h100/deploy_qwen38_flash_next_yarn_1m.sh

bash "$SCRIPT" --check-only                       # 1) 校验环境 + 打印启动命令
nohup setsid bash "$SCRIPT" > /tmp/qwen38_single.log 2>&1 &   # 2) 启动
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:40000/health)" = 200 ]; do sleep 10; done
```

## 两条必须知道的红线

1. **flashinfer 版本边界**：镜像里 flashinfer **< 0.6.18** 时，不能让 flashinfer
   当 attention backend，否则启动断言直接失败。Qwen3.8（QSA 压缩注意力）默认就会
   选中 flashinfer，所以要么用带 0.6.18 的镜像（H100 镜像），要么显式
   `--attention-backend fa3`（H200/PD 镜像的默认做法）。
   原理见 `DEPLOYMENT_PRACTICE.md` §2.4。
2. **YaRN 配置只保留一份**：脚本先检测模型目录里的 `config.json` 是否已是 1M
   （YaRN factor 4.0），是就不动文件，不是才备份 `config.json.native.bak` 后改写。
   多个容器共用同一 NAS 模型目录时不要反复重写该文件，否则会出
   `OSError: [Errno 116] Stale file handle`。

## 已知缺陷（上线前必读）

- 单请求 prefill > 约 262k token 时，QSA draft-prefill 曾触发 CUDA illegal access。
  upstream 修复（clamp gather + MTP stream 复用）已随 2026-09-17 的 `main` merge
  进入 `dev`；`h100_2p2d/` 用该树实测 300k / 800k 无崩溃。老的 H200 源码树还没有
  这两个修复，排障时可先 `QWEN38_SPECULATIVE=0` 关掉 NEXTN，并保留健康检查/自动重启。
  详见 `DEPLOYMENT_PRACTICE.md` §6。
