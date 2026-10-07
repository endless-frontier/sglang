# 镜像

本模型与 **GLM-5.3-flash 共用同一个镜像**——源码提交与依赖集合完全一致，二者只在启动参数上不同。
镜像定义与构建说明在 [`../../glm5.3-flash/image/`](../../glm5.3-flash/image/README.md)，
本文件只是说明为什么这里没有第二份 Dockerfile。

```text
dptech-sh-pai-acr-registry.cn-shanghai.cr.aliyuncs.com/dptech-namespace/wangruisi/sglang-glm53-flash:sglang-d6221be-glm53-flash-l20z-1node-20261003-1a4d8b6
digest sha256:87ed176b6b58cc97653c9c0b42f5f0d73c8ab58660afc2bc69be16c84070e450
```

这张镜像里，GLM-5.3-flash 与本模型都跑通过（各自的启动参数不同）；两个模型共用意味着一次构建、
一个 pin、一份依赖证据，也意味着任何依赖变更要同时验证两个模型。
