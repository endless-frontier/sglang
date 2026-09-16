# 阿里云 DLC：2P + 2D（TP8，YaRN 1M）+ EAS 公网转发

用 DLC 作业在 4 个 8 卡 H200 Pod 上跑 2 Prefill + 2 Decode（每实例 TP8、YaRN 1M），
再由一个 EAS CPU 服务把 DLC 内网的 Router 转发到公网 Endpoint。

## 架构

```text
客户端 → EAS 公网 Endpoint → EAS CPU 代理（同 VPC）
        → 读 CPFS 上的 ready.json 得到 DLC Router 私网地址
        → http://<prefill-leader>:8001 → Prefill(×2)/Decode(×2) worker :8000
```

| 组件 | 端口 | 说明 |
|---|---:|---|
| SGLang worker | 8000 | 每 Pod 一个，`0.0.0.0` 监听，Prefill 另开 bootstrap 8998 |
| DLC Router | 8001 | 跑在 Prefill leader Pod 上，`--pd-disaggregation` |
| EAS CPU 代理 | 由网关暴露 | 只做转发与鉴权，不参与推理 |

端口固定用 8000/8001：这是当前集群验证过的组合（自定义端口在 EAS→DLC 的
VPC 链路上会被拦，见“踩坑”）。

## 需要的阿里云资源

| 项 | 示例/说明 |
|---|---|
| PAI Workspace ID | 数字 ID |
| DLC quota/resource ID | 形如 `quota*`，需能一次调度 4 个 8 卡节点 |
| 镜像 | CUDA 13.0 devel + SGLang 0.5.18 + `sglang-kernel==0.4.7`，且 **EAS 侧也能拉取** |
| DataSource | PAI Dataset ID（`d-xxxxxxxx`），不是 CPFS 文件系统 ID |
| CPFS | 文件系统 ID、路径 `/`、挂载点 `/mnt/data`（DLC 与 EAS 共用） |
| VPC / vSwitch / 安全组 | DLC 与 EAS 使用同一套 |
| EAS | CPU quota（可复用 H200 集群的空闲 CPU）、专属网关 ID |
| 凭据 | 通过 `dlc config` / 环境变量注入，**不要写进配置或聊天记录** |

## 目录结构

```text
aliyun_h200_2p2d/
├── configs/
│   ├── qwen38_flash_next_pd_tp8_2p2d.template.json   # DLC 作业模板（占位符）
│   └── eas_profile.template.json                     # EAS profile 模板（占位符）
├── deployment/qwen38_flash_next_h200/
│   ├── cli.py            # render / submit / status / stop
│   ├── schema.py         # 配置校验、路径与 run-id 安全约束
│   ├── capacity.py       # 只读容量预检
│   ├── preflight.py      # 参数与镜像预检
│   ├── request.py        # 渲染 CreateJob / rank 启动命令
│   ├── worker.py         # Pod 内入口：拉起 worker、写 ready.json
│   ├── eas.py            # EAS 服务的创建/更新/验证/删除
│   ├── public_proxy.py   # EAS CPU 代理（流式转发到 DLC Router）
│   └── abort_decode.py   # 中止 Decode 作业的辅助命令
├── render_pd_plan.sh     # 本地离线渲染
└── submit_dlc_later.sh   # 故意禁用的安全占位脚本
```

## 配置与离线渲染

```bash
cp configs/qwen38_flash_next_pd_tp8_2p2d.template.json configs/qwen38_flash_next_pd_tp8_2p2d.json
$EDITOR configs/qwen38_flash_next_pd_tp8_2p2d.json     # 填 workspace/quota/镜像/DataSource/VPC
PYTHONPATH=. python3 -m deployment.qwen38_flash_next_h200.cli render \
  --config configs/qwen38_flash_next_pd_tp8_2p2d.json \
  --run-id qwen38-pd-test --output /tmp/qwen38-create-job.json
```

模板默认值：4 节点 × 8 GPU、TP8、YaRN 1M、`max_total_tokens=6000000`、
`mem_fraction_static=0.85`、`max_running_requests=96`、
`reasoning_parser=qwen3`、`tool_call_parser=qwen3_coder`，
Router 限流 `max_concurrent_requests=200 / queue_size=200 / queue_timeout_secs=600`。

提交前先用一个 1 节点小作业确认 DataSource 里真的有源码和模型：

```bash
# 作业命令示例（只读检查）
ls -l /mnt/data/xinyu/sglang-qwen38-upstream-1789383617/python/sglang/srt/configs/qwen4_exp.py
```

## 提交、观察、停止

```bash
export PYTHONPATH=$PWD
python3 -m deployment.qwen38_flash_next_h200.cli submit \
  --config configs/qwen38_flash_next_pd_tp8_2p2d.json \
  --run-id qwen38-pd-$(date +%Y%m%d-%H%M%S) \
  --receipt .runtime/submit.json --apply \
  --confirm-total-h200 32 --confirm-image-verified

python3 -m deployment.qwen38_flash_next_h200.cli status \
  --source-receipt .runtime/submit.json --output /tmp/status.json
```

停止作业要求同时给出 receipt、作业 ID 和 `--apply`，避免误停其他任务。提交后按顺序确认：
4 个 Pod Running → worker `/health` 200 → `ready.json` 写入 CPFS（含 Router 私网地址）→
再创建 EAS。

## EAS 转发

EAS 子命令和 DLC 子命令在同一个 CLI 里（`cli.py` 会注册 `eas.py` 的 parser）。
profile 默认读 `configs/eas_profile.json`，可用 `QWEN38_EAS_PROFILE` 指定其他路径：

```bash
cp configs/eas_profile.template.json configs/eas_profile.json   # 填 workspace/quota/gateway/VPC/CPFS
export PYTHONPATH=$PWD

# 1) 只读盘点：网关公网状态 + 当前服务
python3 -m deployment.qwen38_flash_next_h200.cli eas-inventory --output /tmp/eas-inventory.json

# 2) DLC ready 之后创建零 GPU 公网代理（会创建云资源，必须 --apply）
python3 -m deployment.qwen38_flash_next_h200.cli proxy-create \
  --source-receipt .runtime/submit.json \
  --service-name qwen38_flash_next_pd --confirm-service-name qwen38_flash_next_pd \
  --receipt .runtime/eas.json --apply

# 3) 查询状态并做一次公网 OpenAI smoke
python3 -m deployment.qwen38_flash_next_h200.cli proxy-status \
  --source-receipt .runtime/submit.json --output /tmp/eas-status.json --probe

# 4) 经公网入口抓一轮 metrics（可选）
python3 -m deployment.qwen38_flash_next_h200.cli proxy-metrics \
  --source-receipt .runtime/submit.json --output /tmp/eas-metrics.json

# 5) 删除代理（需服务名二次确认）
python3 -m deployment.qwen38_flash_next_h200.cli proxy-delete \
  --source-receipt .runtime/submit.json --confirm-service-name qwen38_flash_next_pd \
  --output /tmp/eas-delete.json --apply
```

EAS 代理从 CPFS 的 `ready.json` 读取 DLC Router 地址，请求全程流式转发，支持长上下文。
`/health` 返回 503 表示代理进程活着但连不上 DLC Router。

## 实战踩坑记录

1. **DataSource 必须是 PAI Dataset ID**。把 CPFS 文件系统 ID（`bmcpfs-*`）填进
   `DataSources[].DataSourceId` 会报 `No Such Dataset or No Permission To Operate`。
2. **CPFS 挂载点要对**：`MountPath=/mnt/data`、数据集根目录含 `xinyu/...` 时，
   Pod 内才能看到 `/mnt/data/xinyu/sglang-qwen38-upstream-1789383617`。
3. **PYTHONPATH 必须注入子进程**。只在 shell 里 `export PYTHONPATH=.../python`，
   启动器会把环境覆盖掉，日志表现为 `No module named sglang.srt.configs.qwen4_exp`，
   实际加载了镜像自带的 `/sgl-workspace/sglang`。`worker.py` 已把源码路径写进
   子进程 env（`QWEN38_SGLANG_SOURCE`），改动时不要退回 shell 导出。
4. **参数冲突**：不能同时传 `--cuda-graph-max-bs` 和 `--cuda-graph-max-bs-decode`；
   镜像内 SGLang 不认 `--mamba-scheduler-strategy extra_buffer`。
5. **端口冲突**：`--enable-metrics-for-all-schedulers` 会让同一 Pod 内多个 TP rank
   抢同一端口（`Address already in use`），保持关闭。
6. **镜像拉取**：`...-vpc.` 的 ACR 地址 EAS 的 Lingjun 实例拉不到（ImagePullBackOff），
   改用公网 ACR 地址或 PAI 自定义镜像 ID。
7. **监听地址**：worker 和 Router 都绑 `0.0.0.0`，不要绑 Pod 私网 IP。
8. **EAS 503**：先确认 EAS 与 DLC 在同一 VPC/vSwitch/安全组，且放通 8000/8001/8998。
   当前集群上曾出现代理与 DLC 私网不通、503 无法消除的情况，最终采用“从 VPC 内
   自行转接 Router”的兜底方案；该模式只需要 Router 地址，例如
   `http://<prefill-leader>:8001`。
9. **429**：Router 限流由 `max_concurrent_requests`/`queue_size` 决定；`queue_size=0`
   会在突发时立刻 429（日志 `No tokens available and queuing is disabled, returning 429`）。
   配置里已给 200/200/600，若调小并发请同时给队列。
10. **时区**：Pod 为 `Asia/Shanghai`，但部分组件按 UTC 打日志，跨组件对比时间需换算。

## 安全与仓库卫生

- 不要把 AccessKey/Secret、receipt（`.runtime/`）、`rendered/` 状态文件、真实集群 ID
  提交到 Git；仓库里只保留带 `REPLACE_WITH_...` 占位符的模板。
- 提交作业前用 `--dry-run`/`render` 检查请求体，确认不会误停其他作业。
