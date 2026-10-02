# GLM-5.3-Flash 实测结果

记录本配方在实测环境下的数字。日期、任务 ID 与镜像都写清楚，便于复核。

## 环境

- 节点：8 × `L20Z`（H100 级别，80 GiB/卡，sm90），Lingjun 资源
- 镜像：CUDA 13.0 + SGLang 0.5.18 + sglang-kernel 0.4.7 + flashinfer-python 0.6.18 + torch 2.13.0+cu130
- 源码树：仓库 `dev` 分支（含 `glm5_next`），放在共享 CPFS 上由 `PYTHONPATH` 注入
- 模型目录：`/mnt/data/public_data/public_model/GLM5.3/GLM-5.3-Flash`（62 分片，约 306 GB，FP8 e4m3）

## 启动与显存

| 指标 | 实测值 |
|---|---|
| 权重加载 | `Load weight end. elapsed=272.31 s`（62 分片，多线程加载） |
| 每卡显存（权重加载后） | `mem usage=37.98 GB`，`avail mem=39.56 GB`（TP8、FP8） |
| 每卡显存（图捕获后、空载） | 约 70 GB |
| 宿主内存占用 | 约 30 GiB |
| 从启动到 `/health` 200 | 约 8 分钟（加载 ~4.5 分钟 + DeepGEMM warmup + decode 图捕获） |
| 图捕获期间 GPU 利用率 | 0 ↔ 100% 交替；生成期间各 rank 30%~95% |

## 请求实测

| 请求 | 结果 | 延迟 | token |
|---|---|---|---|
| `Reply with exactly: pong` | `content='pong'`，`finish_reason=stop` | 0.3 s | 17 prompt / 35 completion（含 33 思考） |
| ~3.8k token 填充 + `reply with exactly: longpong` | `content='longpong'`，`finish_reason=stop` | 37.7 s | 3820 prompt / 47 completion（含 44 思考） |

同一批请求里也能看到该检查点的行为特征：答案在 `content`，思考过程在 `reasoning_content`
（上表两例分别为 130 / 206 字符）。因此**调用方必须为 thinking 预留 `max_tokens`**，
否则会以 `finish_reason=length` 结束而拿不到正文——首次冒烟测试就是这样，正文为空。

## 失败尝试（保留记录，便于排障）

| 现象 | 原因 | 结论 |
|---|---|---|
| `KeyError: 'glm5_next'` → `ValueError: ... has model type glm5_next but Transformers does not recognize this architecture` | 镜像自带 SGLang 早于 GLM-5.3 | 必须注入带 `glm5_next` 的源码树 |
| 62 分片加载完成后 `RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 64, 1, 0]` + `Exception: Capture cuda graph failed` | 用了 `--attention-backend fa3`，该模型必须用 DSA 后端 | 换成 `--attention-backend dsa` + TileLang DSA |
| `Syntax error: ";" unexpected` / `unexpected end of file` | DLC 以 `/bin/sh` 执行 `UserCommand`，多行带引号的命令被拆坏 | 命令改单行、不含引号，逻辑放进 base64 脚本 |

## 尚未验证

- 长上下文（≥262k）单请求：官方注明未修复前 QSA 类长上下文曾崩，本模型需另行实测。
- 并发与吞吐：本表只有单请求数字，未做批量压测。
- 对外暴露（EAS）与 PD 分离：未接。
