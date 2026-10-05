# EAS 在线服务（对外入口）

把 `h100/` 里的单机配方变成一个有公网入口的在线服务。**已实测跑通**：服务起得来、健康检查过、
公网入口能答对内容。

| 请求 | 结果 | 延迟（含网关与网络） |
|---|---|---|
| `/health` | 200（需带服务访问令牌） | — |
| `Reply with exactly: pong` | `content='pong'`，`finish_reason=stop` | **1.2 s** |
| ~3.8k token 填充 + `longpong` | `content='longpong'`，`finish_reason=stop` | **34.8 s** |

对照作业内直连（0.3 s / 33.4 s）：短请求多出约 0.9 s 的网关与网络开销，长请求基本一致。

文件：

| 文件 | 作用 |
|---|---|
| `service.template.json` | EAS v2 的服务定义（body），占位符要替换成真实值 |
| `manage_service.py` | 创建 / 查看 / 日志 / 删除；没有 `--apply` 只做只读或渲染 |
| 本文件 | 说明与踩坑 |

真实值（quota、VPC、CPFS 文件系统 id、工作空间 id）放在仓库外，不要提交进仓库。

## 为什么这样接得通（读现网服务得到的，不是推测）

- **EAS 能直接挂 CPFS**：`storage[0].cpfs` + `mount_path: /mnt/data/`，
  与本配方的模型路径完全一致，300 GB 权重原地读，不需要搬到 OSS。
- **EAS 与 DLC/DSW 共用同一个 Lingjun 资源池**，`metadata.quota_type` 写 `Lingjun`、
  `quota_id` 就是那个额度——所以这不是“额外买一套资源”，而是同一批节点换个调度方式（会互相排队）。
- **健康检查就是 `/health:8000`**，与本配方的就绪判据同一个。

## 用法

```bash
cp service.template.json /path/outside/repo/glm53_eas.json   # 然后替换 REPLACE_WITH_*
python3 manage_service.py create --config /path/outside/repo/glm53_eas.json            # 渲染，不提交
python3 manage_service.py create --config /path/outside/repo/glm53_eas.json --apply    # 创建
python3 manage_service.py status --name ef_glm53_flash_1node
python3 manage_service.py delete --name ef_glm53_flash_1node --apply                   # 验证完删掉
```

创建后返回 `InternetEndpoint`；调用时要带服务自己的令牌（服务列表里的 `AccessToken`），
否则网关返回 `401 Authorization failed.`：

```bash
curl -s -H "Authorization: $TOKEN" https://<endpoint>/api/predict/<name>/health
curl -s -H "Authorization: $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Reply with exactly: pong"}],"max_tokens":256,"temperature":0}' \
  https://<endpoint>/api/predict/<name>/v1/chat/completions
```

## 四个坑（每个都花了一次失败尝试）

1. **`script` 必须是 shell 引用后的 argv 串**，形如 `bash -c '...'`。直接写裸字符串
   `"set -euo pipefail\nexec python3 ..."` 会被当成单个 argv，容器秒崩（`exitCode 1`，反复重启），
   而 EAS 的日志接口返回 0 字节，从外面完全看不出原因。模板里是正确形式。
2. **`cuda-compat` 会遮住 EAS 注入的驱动。** 镜像里为 DLC 准备的 `LD_LIBRARY_PATH` 把
   `/usr/local/cuda/compat`（CUDA 13.0 自带的 580 系列前向兼容库）放在系统目录之前，
   而 EAS 节点注入的是更新的 595 系列驱动。现象很迷惑：
   `torch.cuda.device_count()` 报 8，但 `cuda.is_available()` 是 False，sglang 直接以
   `No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) or platform plugin is available` 退出。
   **服务脚本里必须重设 `LD_LIBRARY_PATH`，不要包含 compat 目录**（模板里已经是这样）。
3. **健康检查窗口要按模型加载时间给。** 该模型启动到就绪约 8 分钟；默认容错只有约两分钟，
   实例会被判为不健康并反复重启。模板给的是 `initial_delay 480s / period 30s / failure_threshold 60`
   （最坏容忍约半小时），并另加了 `startup_check`。
4. **镜像用 ACR 的公网域名拉取**。本工作空间里 `-vpc` 域名报 `Unable to pull image`，
   同一镜像换成公网域名即可。

## 容器日志怎么拿到（上面第 2 个坑就是这样定位的）

EAS 的日志接口在这次实测里返回空。可靠办法：让服务脚本自己把 stdout/stderr `tee` 到 CPFS 上
（`/mnt/data/<account>/eas/*.log`），再用一个一分钟的 DLC 作业 `cat` 出来。
脚本里还包括启动前的 CUDA 自检（打印 `device_count` / `is_available`），
以后遇到“起不来但看不到日志”，照这个模式做即可。

## 花钱的纪律

一个 8 卡 EAS 服务是与队友抢同一批节点，验证完就删；只是 `Waiting` 的服务虽然不占 GPU，
但服务对象会一直留着，同样要清理。
