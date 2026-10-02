# GLM-5.3-Flash：阿里云 DLC 单节点 8 卡

在 PAI DLC 上用**一个 8 卡节点**把 GLM-5.3-Flash 起成服务（OpenAI 兼容接口）。
纯文本服务，暂不含 PD 分离与 EAS 公网转发；对外的 EAS 转发参考
`../../qwen3.8-next-flash/aliyun_h200_2p2d/`。

## 需要准备的东西

| 项 | 说明 |
|---|---|
| PAI Workspace ID | 数字 ID |
| DLC quota / resource ID | 形如 `quota*`；单节点 8 卡即可调度 |
| 镜像 | **我们自己的 GLM-5.3 镜像**（占位符 `REPLACE_WITH_OUR_GLM53_IMAGE_URI`）。镜像未就绪前，可用团队 H100 镜像 + 本仓库源码树，但**必须注入源码树**，否则缺 `glm5_next` |
| DataSource | PAI Dataset ID（`d-*`），**不是** `bmcpfs-*` 文件系统 ID；挂到 `/mnt/data` |
| 模型目录 | `/mnt/data/public_data/public_model/GLM5.3/GLM-5.3-Flash` |
| 源码树 | `/mnt/data/<你的账号目录>/sglang`（本仓库源码，含 `glm_next` 实现） |
| 凭据 | 只从环境变量 `ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET` 读取，**不进配置、不进仓库** |

## 目录结构

```text
aliyun_h100_1node/
├── configs/glm53_flash_h100_1node.template.json   # 作业模板（占位符）
├── job_entry.py                                   # 容器内入口：校验 → 启动 → 采样 → 冒烟
└── submit_glm53_flash.py                          # render / submit / status / stop / watch
```

## 用法

```bash
cp configs/glm53_flash_h100_1node.template.json configs/glm53_flash_h100_1node.json
$EDITOR configs/glm53_flash_h100_1node.json      # 填 workspace / quota / 镜像 / DataSource

export ALIBABA_CLOUD_ACCESS_KEY_ID=...
export ALIBABA_CLOUD_ACCESS_KEY_SECRET=...

python3 submit_glm53_flash.py render --config configs/glm53_flash_h100_1node.json   # 只渲染请求体
python3 submit_glm53_flash.py submit --config configs/glm53_flash_h100_1node.json --apply
python3 submit_glm53_flash.py status --job-id <dlc...>
python3 submit_glm53_flash.py stop   --job-id <dlc...> --apply
```

作业起来之后，日志里每 15 秒一行采样（宿主内存、GPU 利用率与显存），
健康后自动跑一次短请求与一次 ~16k 长请求，输出 `RESULT ...` 行。

## 三条必须知道的坑

1. **`UserCommand` 由 `/bin/sh` 执行**：多行、带引号的命令会被拆坏
   （`Syntax error: ";" unexpected` / `unexpected end of file`）。
   本仓库的做法是把 `job_entry.py` 以 base64 落盘执行，命令是单行且完全无引号的：

   ```text
   echo <base64 job_entry.py> | base64 -d > /tmp/job_entry.py; python3 -u /tmp/job_entry.py
   ```

2. **镜像自带 SGLang 早于 GLM-5.3**：直接用镜像内源码会报 `KeyError: 'glm5_next'`。
   必须让进程解析到本仓库源码树（`PYTHONPATH`，必要时覆盖 `/sgl-workspace/sglang/python/sglang`）。

3. **`--attention-backend fa3` 会崩**：62 个分片加载完成后在 decode CUDA-graph capture 阶段报
   `RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 64, 1, 0]`。用 DSA 后端（脚本默认）即可。

## 交付验证

- `curl /health` = 200，`curl /v1/models` 里出现 `glm-5.3-flash` 与 `max_model_len`。
- 短请求与长请求都返回 `finish_reason=stop`（注意 thinking 会占用 token 预算）。
- 采样行里每卡显存与 FP8 权重规模相符（约 38 GB 权重 + KV）。
- 提交的配置与日志里不出现任何 AccessKey、receipt、真实集群/网关 ID。
