# H100 4P4D 推荐部署方案（2026-10-15）

四个 Prefill、四个 Decode，每实例 TP8。客户端直接访问 Decode-0 上的 CPU
Router：`http://<D0_IP>:40000/v1`。本配方提供容量调度、原生进展观测和
所属进程组恢复，保留 1M context、P HRRN / D FCFS 与 96 个请求槽。

## 相对原配方的改进

- P 以最多两个上下文交错推进，每请求 quantum 上限 4096、批次预算 8192；
  一批结果完成并 drain 后再推进下一批，保留最老请求的推进机会。
- FULL 池没有余量时停止承诺 chunk，按实际 page 与最终输出余量核算；
  为暂停上下文尚未完成的输入保留完成空间，减少多条长请求相互占满池子的风险。
- 暂停上下文参与 native idle / flush 与负载核算，避免把暂停误当空闲。
- Router 按 P token 工作量、D 已用容量及尚未被原生 ACK 覆盖的 credit 分配；
  只有同一 worker 的较新 admission snapshot 可以退还 credit。
- 健康判定同时检查八个真实 scheduler 和鲜活原生进展。未知观测不报告 HEALTHY、
  不触发故障重启；鲜活连续 300 秒无 chunk / busy 进展才判定暂停 P 僵停。
  Router 不给 SUSPECT / UNKNOWN 新准入，恢复保留 300 秒冷却与最多两个并发槽。
- scheduler 致命异常启用原生进程组退出，避免主线程退出但后台线程残留时继续误报健康。
  进程记录绑定 deployment、主机与启动时刻，避免清理其他进程。

## 准备独立源码与模型目录

需要同镜像、共享目录和八台可互通的内网 IPv4 节点，镜像要求沿用上级 README。
本配方的 P 显存比例为 **0.88**、D 为 **0.93**。HiCache 等模型后端参数沿用
worker 脚本；部署前应确认主机内存足够。

下面所有 `<...>` 和 `/path/to/...` 都要替换为自己的值，完整填写说明见
[上级 README](../README.md#部署前填写自己的环境)。`SOURCE` 是独立源码根目录，
`MODEL` 是含配置、tokenizer 和全部权重分片的 HF 导出，`RUN` 是 Git 工作树外的
空共享目录。八台机器必须以相同绝对路径访问它们。模型别名可以自定，客户端的
`model` 应与 `--model-name` 一致；P/D 各四个地址必须是互不重复的真实内网 IPv4。

原生补丁用逐文件 SHA 校验，适配的 base commit 记录于 `source-manifest.json`。
只在**未运行服务的独立源码副本**上应用；不修改现有部署正在使用的源码。

```bash
export SOURCE='/path/to/sglang-serving'
export DIR="$SOURCE/endless-frontier/qwen3.8-next-flash/h100_4p4d"
export MODEL='/path/to/checkpoints/model/hf'
export RUN='/path/to/shared/deployments/qwen38-pd'

python3 "$DIR/managed/apply_runtime_patch.py" --source "$SOURCE" --check
python3 "$DIR/managed/apply_runtime_patch.py" --source "$SOURCE"

python3 "$DIR/managed/prepare_deployment.py" \
  --source "$SOURCE" --model "$MODEL" --model-name '<SERVED_MODEL_NAME>' \
  --run "$RUN" \
  --prefill-ips '<P0_IP>,<P1_IP>,<P2_IP>,<P3_IP>' \
  --decode-ips '<D0_IP>,<D1_IP>,<D2_IP>,<D3_IP>'
```

`prepare_deployment.py` 不启动服务、不发送请求；拒绝覆盖非空 RUN。
小型 HF 配置复制到 RUN，权重 shard 通过共享链接读取，worker 的 YaRN 配置改写不会
触碰原导出。支持原生 262144 / 524288 context 的 HF 导出。

配置由准备脚本按命令行参数生成，不需要粘贴已有部署的私人 config：

| RUN 内的位置 | 内容及来源 |
|---|---|
| `model/` | HF 小型配置副本及只读权重链接，来源为 `--model`；YaRN 改写仅影响此处副本。 |
| `deployment-settings.json` | 源码目录、模型别名、Router 端口和配方默认参数，来源为 `--source` / `--model-name`。 |
| `state/node-{rank}.json` | 八个节点的 rank、角色、内网地址和端口，来源为 `--prefill-ips` / `--decode-ips`。 |
| `state/fair-prefill-control.json` | 配方默认 `2/4096/8192` 调度参数。 |
| `runtime-source-manifest.json` | 此配方的源码 hash 清单副本，用于部署审计。 |

实际 RUN 中的节点配置、日志和模型配置留在自己的部署环境，不提交到 Git。
本配方不要求云账号 AccessKey 或个人 SSH 私钥。文中的 admission **ACK** 是 worker
运行时动态返回的协议确认，用于退还容量 credit，不是用户授权记录或需要填写的密钥。

## 启动与检查

八台各执行一次对应 rank；P0–P3 为 0–3，D0–D3 为 4–7。显式 rank 与 IP
来自准备配置，不依赖云平台 hostname 或个人 SSH 凭据。D 等待四 P 就绪后开始加载。

```bash
# 在每台节点上分别设置 0..7；使用进程管理器或独立终端持有该进程。
python3 "$RUN/run_node.py" --rank 0
```

节点 supervisor 通过原生 `/health`、`/v1/loads` 和真实 scheduler 进程判断状态；
关闭 health 生成，不发送推理保活。阶段 journal 只含 room、rank、token 数和时刻，
不保存 prompt、completion、工具内容、headers 或密钥。

```bash
curl -fsS 'http://<D0_IP>:40000/health'
curl -fsS 'http://<D0_IP>:40000/v1/models'
curl -fsS 'http://<D0_IP>:40000/workers'
```

`/health` 表示至少一组 P/D 可以接收请求；交付前仍须逐一检查八个 worker 的
HEALTHY、八个 scheduler、同 worker ACK 和 `2/4096/8192` 实际参数。P 的
`num_running_reqs=0` 不能单独证明空闲：还要检查 paused、waiting、传输队列和 credit。
使用方应在独立验收窗口运行有限正确性、SSE、取消和目标负载测试，再开放真实客户端。

停止整套部署可在共享 RUN 下创建 `state/STOP`，各 supervisor 只清理所属进程组。
源码和配置变更应在尾流结清后受控重载，旧 PID 的验收不替代新 PID 的证明。

## 验证与当前边界

CPU 回归覆盖实际 PrefillBudget、10000 个随机 allocator 状态、暂停完成空间、
native idle、未知观测和 300 秒停滞判定，以及隔离 CPU 进程组中的原生 fatal 分支。
运行：

```bash
python3 "$DIR/managed/test_managed_recipe.py"
python3 "$DIR/test_watch_worker.py"
```

同机制的 64 GPU 部署已完成八节点源码/参数/ACK/scheduler 审计；固定
`8 × 327680` 输入的 P 容量临界测试全部完成，并保留连续 chunk span。
另一个固定 `4 × 917504` 长输入窗口为两条完成、两条客户端预算超时，不能称作
所有超长请求成功，也不能用这两组功能测试作性能对照。

**该方案是当前推荐的容量/健康修复配方，不代表零错误或所有负载稳定性已证明。**
高负载仍观测到 D KV 容量中止，D 完整剩余输出预留、端到端取消以及未明的 HTTP400
需要后续处理；当前没有可比重复性能收益或真实 GPU 异常自动恢复的完整证明。
原生空闲和计数守恒也不等于深层 Mamba/COW 所有权证明。上级 `RESULTS.md`
属于原配方的历史测量，不作为本次调度的性能保证。
