# endless-frontier

本目录收集这次协作产出的**大模型在阿里云 PAI 上的部署配方**：每个模型一个目录，自带镜像定义、
部署路线、实测数据，以及明确的"已验证 / 继承自文档 / 未测"划分。

> 团队自己的 `qwen3.8-next-flash/` 系列（单机、2P2D、4P4D 与 DLC 路线）在 `dev/rsw/deploy`
> 分支上，不在本次提交里；本文件只描述本次提交包含的内容。

## 三个配方

| 目录 | 模型 | 资源 | 入口 | 状态 |
| --- | --- | --- | --- | --- |
| [`glm5.3-flash/`](glm5.3-flash/README.md) | GLM-5.3-flash | 单机 8 卡 H100 级 | `h100/deploy_glm53_flash_1m.sh`、DLC 路由 `aliyun_h100_1node/`、EAS `eas/` | 已实测（长上下文实测约 855k 单机上限） |
| [`qwen3.8-flash-next-1node/`](qwen3.8-flash-next-1node/README.md) | Qwen3.8-Flash-Next | 单机 8 卡 H100 级 | `launch/deploy_qwen38_flash_next_1m.sh`、DLC 路由 `aliyun_h100_1node/` | 已实测（1M 上下文，YaRN 在检查点内） |
| [`qwen3.8-2.4t-a95b/`](qwen3.8-2.4t-a95b/README.md) | Qwen3.8 **2.4T 旗舰** | **6 节点 × 8 卡 = 48 卡**（单实例跨机） | EAS 多机 `aliyun_eas_6node/manage_service.py` | 已实测（240k token 长上下文取证） |

## 三个模型不是一回事，先看清

- `glm5.3-flash` 与 `qwen3.8-flash-next-1node` **共用同一个镜像**：同一份源码提交、同一套依赖，
  只在启动参数上不同；两台单机配方因此互为参照。
- `qwen3.8-2.4t-a95b` 是**旗舰模型**（约 2.45 TB FP8 权重），单机放不下，必须用 EAS 多机分布式
  （一个副本跨 6 台机器）；它和 `qwen3.8-flash-next` 只是名字相近，权重与形态完全不同。

## 通用约定

- 每个目录都写明 **已验证 / 继承 / 未测**：没测的（并发吞吐、超大窗口、EP 等）在文档里点名，不隐含。
- 镜像统一放在团队 ACR 的 `dptech-namespace` 下，按 `<源码提交>-<模型>-<硬件>-<形态>-<日期>` 命名；
  EAS 侧一律用**公网**地址拉取。
- 账号相关信息（VPC / vSwitch / 安全组 / 配额 / 工作空间）在模板里都是 `REPLACE_WITH_…`，
  凭据从环境变量读取，不进仓库。
