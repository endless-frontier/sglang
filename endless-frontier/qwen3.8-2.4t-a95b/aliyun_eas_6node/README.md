# EAS 多机分布式服务：6 台 × 8 卡（TP8 × PP6，48 卡）
**实测通过（2026-10-04）**：一个模型实例跨 6 台 H100 级机器，公网入口可调用，
240k token 的长上下文请求两个暗号都答对。这是 2.4T 模型在本集群唯一可行的服务形态 ——
单节点装不下它。

## 需要准备什么

| 项 | 要求 |
| --- | --- |
| 权重 | `Qwen/Qwen3.8-2.4T-A95B-FP8`，revision `d2dc3565`，**224 个文件 / 2.496 TB**；放在 EAS 能挂载的存储上（CPFS 挂到容器 `/mnt/data` 即可，本配方即如此，权重原地读取、不拷副本） |
| 权重校验 | 发布方为每个分片提供 sha256（213 个分片）；下载后逐个核对再上线 |
| 镜像 | 可直接用 README 里给出的已实测镜像；或自行重建 |
| 配额与网络 | 灵骏配额（可一次排 6 个 8 卡节点）+ 与 EAS 同 VPC/vSwitch/安全组；开 RDMA |

## 目录内容

| 文件 | 作用 |
| --- | --- |
| `service.template.json` | 服务定义模板，所有账号相关值都是 `REPLACE_WITH_…` |
| `manage_service.py` | 创建 / 查看 / 删除服务（`--apply` / `--describe` / `--delete` / `--dry-run`） |
| `deploy_6node.sh` | 容器启动命令，单独放出来方便阅读与本地调试 |

## 使用

```bash
# 1) 先填模板里的占位符：镜像、VPC、vSwitch、安全组、灵骏配额、工作空间
#    （真实值不要提交，命令行的凭据从 ~/.aliyun-ef.env 读）python3 manage_service.py --dry-run        # 打印将要提交的服务定义
python3 manage_service.py --apply          # 创建服务（申请 48 卡，通常要排队）
python3 manage_service.py --describe       # 状态 / 运行实例数 / 公网入口
python3 manage_service.py --delete         # 删除，释放 48 卡
```

验证（拿到 `AccessToken` 后）：

```bash
curl -H "Authorization: <AccessToken>" http://<endpoint>/v1/chat/completions \
  -d '{"model":"qwen3.8-2.4t-a95b","messages":[{"role":"user","content":"Reply with exactly: pong"}],"max_tokens":512}'
```

## 平台做什么、我们做什么

EAS 给分布式单元（Unit）里每个实例注入四个变量，启动命令把它们接到 SGLang 的多机参数上：

| 平台注入 | 引擎需要 |
| --- | --- |
| `unit.size`（单副本机器数） | `--nnodes` |
| `RANK_ID` | `--node-rank` |
| `MASTER_ADDRESS`（0 号实例 IP） | `--dist-init-addr $MASTER_ADDRESS:20000` |
| `COMM_IFNAME`（开 RDMA 时为 `net0`） | `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` |

整个 Unit 只有 0 号实例对外接流量；滚动更新按 Unit 整体重建。

## 踩过的坑（每一条都真花过时间）

1. **镜像要用公网地址**：DLC 用的 `-vpc` 地址在 EAS 侧拉不动（`Unable to pull image`）。
2. **`script` 必须是 shell 引用后的 argv 串**（`bash -c '…'`）；裸字符串会被当成单个 argv 直接崩。
3. **平台日志接口不可靠**：需要时可能返回零字节。本配方的启动命令自己 `tee` 到共享存储，
   排障时去 `/mnt/data/.../eas/` 读自己的日志。
4. **健康检查窗口要给足**：本配方 health 初始延迟 3600 s、startup 600 s、失败阈值 120 × 30 s；
   2.5 TB 权重加载约 18 分钟。
5. **首次部署可能长时间停在 `Waiting`**（入口 503），原样重新部署即正常 —— 记录为偶发。
6. **服务不删就一直在占卡**：48 卡是团队共享池里 48 卡，验证完请 `--delete`。

## 需要一次调度的资源

| 项 | 值 |
| --- | --- |
| 机器 | 6 × 8 卡 H100 级（`L20Z`），单实例 `tp 8 × pp 6` |
| 显存 | 每节点 8 × 80 GB = 640 GB；FP8 权重 2.45 TB 分摊到 6 台 |
| 存储 | 共享 CPFS，权重目录约 2.5 TB |
| 网络 | 灵骏配额 + RDMA（`net0`）；EAS 需与 DLC 同一 VPC/vSwitch/安全组 |
