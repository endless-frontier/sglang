# 镜像

`Dockerfile` 就是实测使用的镜像定义：Ubuntu 24.04（AWS 公共 ECR，构建环境拉不到 Docker Hub）
+ NVIDIA CUDA 13 **devel** 套件 + `uv` 装依赖 + **SGLang 取自 `main` 的 tarball**
（官方模型卡指向尚未发版的 day-0 构建，已发布的 tag 早于本模型的部分代码）。

构建结果：**18 GB**（压缩后 8.86 GB），digest
`sha256:9478bff4fcb257418e7ee7a6e5eba572eddc58e4f9e501e81f10a93289ef1d35`。

## 三个不是装饰的细节

1. **`python3-dev` + 两个 `libcuda.so` 符号链接**：Triton 启动时要现场编译一个小 C 扩展并
   `#include <Python.h>`，而平台不会往镜像里注入驱动库；缺任何一个都会"读完所有权重才崩"。
2. **`uv` 而不是 `pip`**：可达镜像源的 HTTP/2 与 HTTP/1.1 速度差两个数量级，pip 只会说 HTTP/1.1。
3. **构建期断言架构**：最后一层会检查 `Qwen3_5MoeForCausalLM` 与其 `EntryClass` 确实存在，
   并验证 `Python.h` 与 `-l:libcuda.so.1` 可解析 —— 把"pin 错了"变成构建失败，而不是上线后莫名其妙的加载失败。

依赖的完整冻结清单（217 个包，含 torch / flashinfer / triton / tilelang 的精确版本）在
本目录仓库的 `qwen-3.8-2.4t-a95b/image.lock.freeze.txt`（私有仓库 `ef-pai-serving`）。
