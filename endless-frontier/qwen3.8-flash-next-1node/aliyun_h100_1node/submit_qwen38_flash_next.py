#!/usr/bin/env python3
"""GLM-5.3-Flash 单机 8 卡 DLC 作业的渲染/提交/查询/停止。

凭据只从环境变量读取，绝不写进配置或仓库：
    export ALIBABA_CLOUD_ACCESS_KEY_ID=...
    export ALIBABA_CLOUD_ACCESS_KEY_SECRET=...

用法：
    python3 submit_qwen38_flash_next.py render --config configs/qwen38_flash_next_h100_1node.json
    python3 submit_qwen38_flash_next.py submit --config configs/qwen38_flash_next_h100_1node.json --apply
    python3 submit_qwen38_flash_next.py status --job-id <dlc...>
    python3 submit_qwen38_flash_next.py stop   --job-id <dlc...> --apply

约定（重要）：DLC 用 /bin/sh 执行 UserCommand，多行或带引号的命令会被拆坏。
因此这里把容器内入口脚本以 base64 落盘执行——命令是单行且不含任何引号的。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENTRY = HERE / "job_entry.py"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    missing = [k for k in ("aliyun", "runtime") if k not in config]
    if missing:
        sys.exit(f"配置缺少字段：{missing}")
    aliyun = config["aliyun"]
    for key, value in aliyun.items():
        if isinstance(value, str) and value.startswith("REPLACE_WITH_"):
            sys.exit(f"配置项 aliyun.{key} 仍是占位符：{value}")
    return config


def user_command() -> str:
    payload = base64.b64encode(ENTRY.read_bytes()).decode()
    return (
        f"echo {payload} | base64 -d > /tmp/job_entry.py; python3 -u /tmp/job_entry.py"
    )


def build_request(config: dict, run_id: str) -> dict:
    aliyun, runtime = config["aliyun"], config["runtime"]
    return {
        "Accessibility": "PRIVATE",
        "WorkspaceId": aliyun["workspace_id"],
        "DisplayName": f"{config['name']}-{run_id}"[:63],
        "Description": "recipe=qwen3.8-flash-next;mode=single-node;controller=submit_qwen38_flash_next",
        "JobType": "PyTorchJob",
        "Priority": int(aliyun.get("priority", 5)),
        "ResourceId": aliyun["resource_id"],
        "JobSpecs": [
            {
                "Type": "Worker",
                "Image": aliyun["worker_image"],
                "PodCount": int(aliyun.get("node_count", 1)),
                "ResourceConfig": {
                    "CPU": str(aliyun.get("worker_cpu", 160)),
                    "Memory": aliyun.get("worker_memory", "1500Gi"),
                    "GPU": str(aliyun.get("gpus_per_node", 8)),
                    "SharedMemory": aliyun.get("worker_shared_memory", "256Gi"),
                },
            }
        ],
        "DataSources": [
            {
                "DataSourceId": item["data_source_id"],
                "MountPath": item.get("mount_path", "/mnt/data"),
                **({"Options": item["options"]} if "options" in item else {}),
            }
            for item in aliyun["data_sources"]
        ],
        "Settings": {"EnableRDMA": bool(aliyun.get("enable_rdma", False))},
        "Envs": {
            "PYTHONUNBUFFERED": "1",
            "HOME": "/tmp/qwen38-home",
            "TMPDIR": "/tmp/qwen38-tmp",
            "PYTHONPATH": f"{runtime['sglang_source']}/python",
            "QWEN38_SGLANG_SOURCE": runtime["sglang_source"],
            "QWEN38_MODEL_PATH": runtime["model_path"],
            "QWEN38_SERVED_NAME": runtime.get(
                "served_model_name", "qwen38-flash-next-1m"
            ),
            "QWEN38_PORT": str(runtime.get("port", 40000)),
            "QWEN38_TP_SIZE": str(runtime.get("tp_size", 8)),
            "QWEN38_MEM_FRACTION": str(runtime.get("mem_fraction_static", 0.90)),
            "QWEN38_SPECULATIVE": str(runtime.get("speculative", 1)),
            "QWEN38_SMOKE": str(runtime.get("smoke", 1)),
        },
        "UserCommand": user_command(),
        "JobMaxRunningTimeMinutes": int(
            aliyun.get("job_max_running_time_minutes", 120)
        ),
    }


def client(config: dict):
    from alibabacloud_pai_dlc20201203.client import Client as Dlc
    from alibabacloud_tea_openapi import models as open_api_models

    if not os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID"):
        sys.exit(
            "缺少环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET"
        )
    return Dlc(
        open_api_models.Config(
            access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
            access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
            endpoint=config["aliyun"]["endpoint"],
            region_id=config["aliyun"]["region"],
        )
    )


def tail_logs(dlc, job_id: str, max_lines: int = 120) -> None:
    from alibabacloud_pai_dlc20201203 import models as dlc_models

    body = dlc.get_job(job_id, dlc_models.GetJobRequest()).body
    print(f"status={body.status}")
    lines: list[str] = []
    for pod in body.pods or []:
        pod_id = str(getattr(pod, "pod_id", "") or "")
        if not pod_id:
            continue
        try:
            response = dlc.get_pod_logs(
                job_id, pod_id, dlc_models.GetPodLogsRequest(max_lines=max_lines)
            )
            lines.extend(str(line) for line in (response.body.logs or []))
        except Exception as exc:  # noqa: BLE001
            lines.append(f"[日志读取失败 {pod_id}: {exc}]")
    print("\n".join(lines[-max_lines:]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("render", "submit", "status", "stop", "watch")
    )
    parser.add_argument(
        "--config", default=str(HERE / "configs" / "qwen38_flash_next_h100_1node.json")
    )
    parser.add_argument("--job-id")
    parser.add_argument("--apply", action="store_true", help="真正提交/停止")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    if args.action == "render":
        config = load_config(Path(args.config))
        print(json.dumps(build_request(config, "render"), indent=2, ensure_ascii=False))
        return 0

    if args.action in ("submit", "watch") and not args.job_id:
        config = load_config(Path(args.config))
        request = build_request(config, time.strftime("%Y%m%d-%H%M%S"))
        if not args.apply:
            print(json.dumps(request, indent=2, ensure_ascii=False))
            print("\n（未提交：加 --apply 才会真正创建作业）")
            return 0
        from alibabacloud_pai_dlc20201203 import models as dlc_models

        dlc = client(config)
        job_id = dlc.create_job(
            dlc_models.CreateJobRequest().from_map(request)
        ).body.job_id
        print(f"submitted job: {job_id}")
    else:
        config = (
            load_config(Path(args.config))
            if Path(args.config).exists()
            else {"aliyun": {}}
        )
        dlc = client(config) if config["aliyun"] else None
        job_id = args.job_id
        if dlc is None:
            sys.exit("需要 --config 才能连接")

    if args.action == "stop":
        if not args.apply:
            sys.exit("停止作业需要显式 --apply（并确认 job-id）")
        dlc.stop_job(job_id)
        print(f"stop requested: {job_id}")
        return 0

    if args.action in ("status", "submit", "watch"):
        from alibabacloud_pai_dlc20201203 import models as dlc_models

        while True:
            body = dlc.get_job(job_id, dlc_models.GetJobRequest()).body
            print(f"[{time.strftime('%H:%M:%S')}] status={body.status}")
            if args.action == "status" or body.status in (
                "Succeeded",
                "Failed",
                "Stopped",
            ):
                tail_logs(dlc, job_id)
                break
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
