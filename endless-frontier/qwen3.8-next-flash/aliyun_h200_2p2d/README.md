# 阿里云 DLC：2P + 2D（TP8）

该配方启动四个 DLC Pod，每 Pod 使用 8 张 H200：两个 Prefill、两个 Decode；每个 SGLang 实例 TP8、PP1。DLC 的 Router 由控制器在 Prefill leader 上启动并监听 `0.0.0.0:8001`，worker API 监听每 Pod 的 `0.0.0.0:8000`。PD 传输后端为 Mooncake，RoCE 使用 IBGDA。

## 前置条件

使用含 CUDA 13.0 devel/toolkit、PyTorch cu130、SGLang 0.5.18 和 `sglang-kernel==0.4.7` 的镜像，例如：

```text
pai-ai-prod-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/acr_namespace/scimaster:sglang-0-5-18-cuda13-qwen38-next-pd
```

镜像必须能在四节点访问 `/usr/local/cuda/bin/nvcc`。DLC 作业需开启 RDMA、NVIDIA IBGDA、GDRCopy，并将 CPFS/DataSource 挂载到 `/mnt/data`。模型和源码的默认位置分别是 `/mnt/data/public_models/Qwen3.8-Flash-Next` 与 `/mnt/data/xinyu/sglang-qwen38-upstream-1789383617`；若路径不同，修改模板中的 `runtime` 字段。

## 配置和渲染

复制模板并填写阿里云专属信息（模板不含密钥）：

```bash
cd endless-frontier/qwen3.8-next-flash/aliyun_h200_2p2d
cp configs/qwen38_flash_next_pd_tp8_2p2d.template.json configs/qwen38_flash_next_pd_tp8_2p2d.json
$EDITOR configs/qwen38_flash_next_pd_tp8_2p2d.json
```

至少填写 `workspace_id`、DLC `resource_id`、`worker_image`、DataSource ID，以及 VPC、vSwitch、安全组 ID。模板已给出 4 节点、每节点 8 GPU、YaRN 1M、`max_total_tokens=6000000`、`reasoning_parser=qwen3` 和 `tool_call_parser=qwen3_coder`。先离线检查请求：

```bash
PYTHONPATH=. python3 -m deployment.qwen38_flash_next_h200.cli render \
  --config configs/qwen38_flash_next_pd_tp8_2p2d.json \
  --run-id qwen38-pd-test --output /tmp/qwen38-create-job.json
```

## 提交、观察和停止

`submit_dlc_later.sh` 是故意禁用的安全占位脚本。确认资源和镜像后，显式执行：

```bash
export PYTHONPATH=$PWD
python3 -m deployment.qwen38_flash_next_h200.cli submit \
  --config configs/qwen38_flash_next_pd_tp8_2p2d.json \
  --run-id qwen38-pd-$(date +%Y%m%d-%H%M%S) \
  --receipt .runtime/submit.json --apply \
  --confirm-total-h200 32 --confirm-image-verified
```

保存 receipt 后可查询状态：

```bash
python3 -m deployment.qwen38_flash_next_h200.cli status \
  --source-receipt .runtime/submit.json --output /tmp/status.json
```

停止作业必须同时提供 receipt、作业 ID 确认和 `--apply`，以避免误停其他任务。服务健康检查使用 Router 的 `http://<prefill-leader>:8001/health`；OpenAI 兼容请求发到 Router 的 `/v1/chat/completions`。

## 常见问题

- `CUDA compiler and CUDA toolkit headers are incompatible`：换 CUDA 13.0 devel 镜像，并确保 `CUDA_HOME`、`CUDACXX` 指向同一套 toolkit；不要混用宿主机 nvcc 与镜像 headers。
- worker 启动后无 ready：检查四 Pod 之间的 IB/RDMA、安全组、端口 8000/8001/8998/29500，以及 `/mnt/data` 是否实际挂载。
- 显存不足：先将 `mem_fraction_static` 从 0.85 调低；减少并发或 CUDA graph batch。保持四个 worker 的 YaRN 和模型配置完全一致。
- 不要在 CPFS 上共享 Triton/Torch 编译缓存；控制器已将缓存放在每 Pod 的 `/tmp`。
