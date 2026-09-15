from __future__ import annotations

from datetime import datetime
import base64
from io import BytesIO
from pathlib import Path
import shlex
from typing import Any
import zipfile
import zlib

from .schema import (
    GPUS_PER_NODE,
    ROOT,
    LoadedConfig,
    total_gpus,
    total_nodes,
)


def _launch_bundle(loaded: LoadedConfig) -> str:
    files = [
        ROOT / "deployment" / "__init__.py",
        ROOT / "deployment" / "qwen38_flash_next_h200" / "__init__.py",
        ROOT / "deployment" / "qwen38_flash_next_h200" / "schema.py",
        ROOT / "deployment" / "qwen38_flash_next_h200" / "worker.py",
        loaded.path,
    ]
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix())
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return base64.b64encode(zlib.compress(buffer.getvalue(), 9)).decode("ascii")


def build_user_command(loaded: LoadedConfig, run_id: str) -> str:
    runtime = loaded.value["runtime"]
    relative_config = loaded.path.relative_to(ROOT).as_posix()
    bootstrap = (
        "import base64,io,os,zipfile,zlib;"
        "root=os.path.join(os.environ['DEPLOY_LAUNCH_ROOT'],'rank'+os.environ['RANK']);"
        "os.makedirs(root,exist_ok=False);"
        f"data=zlib.decompress(base64.b64decode('{_launch_bundle(loaded)}'));"
        "zipfile.ZipFile(io.BytesIO(data)).extractall(root)"
    )
    extract = shlex.join([runtime["python_executable"], "-c", bootstrap])
    launch_dir = '"$DEPLOY_LAUNCH_ROOT/rank$RANK"'
    config_path = f'"$DEPLOY_LAUNCH_ROOT/rank$RANK/{relative_config}"'
    launch = " ".join(
        [
            f"PYTHONPATH={launch_dir}:$PYTHONPATH",
            shlex.quote(runtime["python_executable"]),
            "-m",
            "deployment.qwen38_flash_next_h200.worker",
            "--config",
            config_path,
            "--run-id",
            shlex.quote(run_id),
        ]
    )
    prepare_local = (
        'mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" '
        '"$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"'
    )
    cuda_home = runtime.get("cuda_home", "/usr/local/cuda")
    sglang_source = runtime.get("sglang_source", "")
    exports = (
        f'export CUDA_HOME={shlex.quote(str(cuda_home))} '
        f'CUDACXX={shlex.quote(str(cuda_home))}/bin/nvcc '
        f'PATH={shlex.quote(str(cuda_home))}/bin:$PATH'
    )
    if sglang_source:
        exports += f' PYTHONPATH={shlex.quote(str(sglang_source))}/python:$PYTHONPATH'
    if sglang_source:
        exports += f' QWEN38_SGLANG_SOURCE={shlex.quote(str(sglang_source))}'
    return f"{exports} && {prepare_local} && {extract} && {launch}"


def build_create_job_request(
    loaded: LoadedConfig,
    *,
    run_id: str,
    model_path: str,
) -> dict[str, Any]:
    config = loaded.value
    aliyun = config["aliyun"]
    runtime = config["runtime"]
    node_count = total_nodes(config)
    gpu_count = total_gpus(config)
    node_local_root = f"/tmp/qwen38_flash_next_h200/{run_id}"
    request: dict[str, Any] = {
        "Accessibility": "PRIVATE",
        "WorkspaceId": aliyun["workspace_id"],
        "DisplayName": f"{config['name']}-{run_id}"[:63],
        "JobType": "PyTorchJob",
        "Priority": int(aliyun["priority"]),
        "ResourceId": aliyun["resource_id"],
        "JobSpecs": [
            {
                "Type": "Worker",
                "Image": aliyun["worker_image"],
                "PodCount": node_count,
                "ResourceConfig": {
                    "CPU": str(aliyun["worker_cpu"]),
                    "Memory": aliyun["worker_memory"],
                    "GPU": str(GPUS_PER_NODE),
                    "SharedMemory": aliyun["worker_shared_memory"],
                },
            }
        ],
        "DataSources": [
            {
                "DataSourceId": item["data_source_id"],
                "MountPath": item["mount_path"],
                **({"Options": item["options"]} if "options" in item else {}),
            }
            for item in aliyun["data_sources"]
        ],
        "Settings": {
            "EnableRDMA": True,
            "AdvancedSettings": {
                "EnableNvidiaIBGDA": "true",
                "EnableNvidiaGDRCopy": "true",
            },
        },
        "Envs": {
            "PYTHONUNBUFFERED": "1",
            "DEPLOY_RUN_ID": run_id,
            "DEPLOY_CONFIG_SHA256": loaded.sha256,
            "DEPLOY_MODEL_PATH": model_path,
            "DEPLOY_LAUNCH_ROOT": (
                runtime["project_root"]
                + "/.runtime/deployment/qwen38_flash_next_h200/"
                + run_id
                + "/launch_bundle"
            ),
            "PROJECT_H200_ACCOUNTING_SCOPE": "explore_xiangruiliu",
            "PROJECT_REQUESTED_H200": str(gpu_count),
            # ZeroMQ IPC and concurrent Triton/Torch compilation require a
            # node-local filesystem. Sharing these caches on CPFS causes
            # EOPNOTSUPP or temp-file rename races across 32 processes.
            "TMPDIR": node_local_root + "/tmp",
            "HF_HOME": node_local_root + "/huggingface",
            "XDG_CACHE_HOME": node_local_root + "/xdg",
            "TRITON_CACHE_DIR": node_local_root + "/triton",
            "TORCHINDUCTOR_CACHE_DIR": node_local_root + "/torchinductor",
        },
        "UserCommand": build_user_command(loaded, run_id),
    }
    if config["mode"] == "pd":
        request["Envs"]["MC_TE_METRIC"] = "1"
    request["Envs"].update(runtime.get("sglang_env") or {})
    job_max_running_time = aliyun.get("job_max_running_time_minutes")
    if job_max_running_time not in (None, "auto"):
        request["JobMaxRunningTimeMinutes"] = int(job_max_running_time)
    user_vpc = aliyun.get("user_vpc")
    if isinstance(user_vpc, dict) and all(
        user_vpc.get(key) for key in ("vpc_id", "switch_id", "security_group_id")
    ):
        request["UserVpc"] = {
            "VpcId": user_vpc["vpc_id"],
            "SwitchId": user_vpc["switch_id"],
            "SecurityGroupId": user_vpc["security_group_id"],
            "ExtendedCIDRs": list(user_vpc.get("extended_cidrs") or []),
        }
    return request


def build_deployment_manifest(
    loaded: LoadedConfig,
    *,
    run_id: str,
    model_path: str,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成供本机审计的可读完整启动清单，不参与云端执行。"""
    from .worker import (
        build_pd_router_args,
        build_sglang_args,
        role_leader_ranks,
    )

    node_count = total_nodes(loaded.value)
    gpu_count = total_gpus(loaded.value)
    nodes = {
        rank: {"ip": f"<rank-{rank}-private-ip>", "visible_gpu_count": GPUS_PER_NODE}
        for rank in range(node_count)
    }
    rank_commands = []
    for rank in range(node_count):
        argv = build_sglang_args(
            loaded,
            rank=rank,
            node_ip=f"<rank-{rank}-private-ip>",
            nodes=nodes,
            chosen_model_path=model_path,
            ib_devices="<discovered-infiniband-devices>",
        )
        rank_commands.append(
            {
                "rank": rank,
                "argv": argv,
                "shell_rendering_for_audit_only": shlex.join(argv),
            }
        )
    router = None
    if loaded.value["mode"] == "pd":
        service_port = loaded.value["runtime"]["service_port"]
        prefill_urls = [
            f"http://<rank-{rank}-private-ip>:{service_port}"
            for rank in role_leader_ranks(loaded.value, "prefill")
        ]
        decode_urls = [
            f"http://<rank-{rank}-private-ip>:{service_port}"
            for rank in role_leader_ranks(loaded.value, "decode")
        ]
        argv = build_pd_router_args(
            loaded,
            prefill_urls=prefill_urls,
            decode_urls=decode_urls,
        )
        router = {
            "argv": argv,
            "shell_rendering_for_audit_only": shlex.join(argv),
        }
    exact_request = request or build_create_job_request(
        loaded, run_id=run_id, model_path=model_path
    )
    return {
        "schema_version": "qwen38-flash-next-h200-deployment-manifest-v1",
        "run_id": run_id,
        "mode": loaded.value["mode"],
        "worker_image": loaded.value["aliyun"]["worker_image"],
        "model_path": model_path,
        "resolved_config": loaded.value,
        "controller_user_command": exact_request["UserCommand"],
        "controller_environment": exact_request["Envs"],
        "rank_launch_commands": rank_commands,
        "pd_router_command": router,
        "note": (
            "IP 与 InfiniBand 设备在 Pod 启动后发现；尖括号字段是审计占位符，"
            "实际 argv 由同一 worker 函数按发现值生成并在 Pod 日志打印。"
        ),
    }


def render_plan(
    loaded: LoadedConfig,
    *,
    run_id: str,
    model_path: str,
    blockers: list[str],
) -> dict[str, Any]:
    gpu_count = total_gpus(loaded.value)
    return {
        "schema_version": "qwen38-flash-next-h200-render-v1",
        "rendered_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": loaded.value["mode"],
        "run_id": run_id,
        "config": {
            "relative_path": loaded.path.relative_to(ROOT).as_posix(),
            "sha256": loaded.sha256,
        },
        "submission_performed": False,
        "submission_ready": not blockers,
        "submission_blockers": blockers,
        "gpu_accounting": {
            "requested": gpu_count,
            "reserved": 0,
            "actually_used": 0,
            "evidence": "本地 render；未调用 Aliyun provider",
        },
        "request": build_create_job_request(
            loaded, run_id=run_id, model_path=model_path
        ),
        "deployment_manifest": build_deployment_manifest(
            loaded, run_id=run_id, model_path=model_path
        ),
    }
