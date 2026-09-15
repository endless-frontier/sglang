from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from uuid import uuid4
import zlib

from .capacity import (
    TERMINAL_JOB_STATES,
    build_capacity_snapshot,
    get_quota,
    list_active_workspace_jobs,
    list_quota_nodes,
    paistudio_client,
    summarize_project_inventory,
)
from .request import build_create_job_request, build_deployment_manifest, render_plan
from .schema import (
    PROJECT_H200_CAP,
    ROOT,
    TASK_APPROVAL_THRESHOLD_H200,
    inside_project,
    load_config,
    model_path,
    safe_run_id,
    submission_blockers,
    total_gpus,
    total_nodes,
)


RUNTIME_ROOT = ROOT / ".runtime" / "deployment" / "qwen38_flash_next_h200"
INVENTORY_PATH = ROOT / ".runtime" / "deployment" / "h200_project_inventory.json"
ADMISSION_LOCK = ROOT / ".runtime" / "deployment" / "h200_project_admission.lock"
TERMINAL_STATES = {"Succeeded", "Failed", "Stopped", "Deleted"}
RUNTIME_MARKER = "QWEN38_FLASH_NEXT_RUNTIME_B64="
READY_MARKER = "QWEN38_FLASH_NEXT_READY_B64="


def atomic_json(path: Path, value: dict[str, Any], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def redact(value: object) -> str:
    text = str(value or "")
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)((?:access.?key|secret|api.?key|token)\s*[:=]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"\b(?:LTAI|STS\.)[A-Za-z0-9._-]{8,}\b", "[REDACTED]", text)
    return text[:64_000]


def provider_error(exc: BaseException) -> dict[str, Any]:
    data = getattr(exc, "data", None)
    return {
        "exception_type": type(exc).__name__,
        "error_full_credential_redacted": redact(exc),
        "response_body_full_credential_redacted": redact(data),
        "http_status": getattr(exc, "status_code", None),
        "request_id": str(
            getattr(exc, "request_id", "")
            or (data.get("RequestId", "") if isinstance(data, dict) else "")
            or ""
        ),
    }


def provider_client():
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_tea_openapi.models import Config
    from .capacity import _credentials
    access_key, secret, token = _credentials()
    return Client(Config(
        access_key_id=access_key,
        access_key_secret=secret,
        region_id="cn-shanghai",
        endpoint="pai-dlc.cn-shanghai.aliyuncs.com",
        protocol="https",
        security_token=token,
    ))


def job_observation(body: Any) -> dict[str, Any]:
    pods = []
    for pod in list(getattr(body, "pods", None) or []):
        pods.append(
            {
                "pod_id": str(getattr(pod, "pod_id", "") or ""),
                "status": str(getattr(pod, "status", "") or ""),
                "type": str(getattr(pod, "type", "") or ""),
            }
        )
    return {
        "job_id": str(getattr(body, "job_id", "") or ""),
        "display_name": str(getattr(body, "display_name", "") or ""),
        "status": str(getattr(body, "status", "") or ""),
        "pods": pods,
    }


def ready_path(run_id: str) -> Path:
    return RUNTIME_ROOT / run_id / "ready.json"


def read_ready(run_id: str) -> dict[str, Any] | None:
    path = ready_path(run_id)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def read_runtime_probes(run_id: str, node_count: int = 4) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = RUNTIME_ROOT / run_id / "runtime"
    for rank in range(node_count):
        path = root / f"rank{rank}.json"
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _latest_json_marker(lines: list[str], prefix: str) -> dict[str, Any] | None:
    for line in reversed(lines):
        if prefix not in line:
            continue
        try:
            encoded = line.split(prefix, 1)[1].strip()
            value = json.loads(zlib.decompress(base64.b64decode(encoded)))
        except (ValueError, json.JSONDecodeError, zlib.error):
            continue
        if isinstance(value, dict):
            return value
    return None


def _log_evidence(
    pod_logs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    by_rank: dict[int, dict[str, Any]] = {}
    recovered_ready = None
    for pod in pod_logs:
        runtime = pod.get("runtime_marker")
        if isinstance(runtime, dict) and isinstance(runtime.get("rank"), int):
            by_rank[int(runtime["rank"])] = runtime
        candidate = pod.get("ready_marker")
        if isinstance(candidate, dict):
            recovered_ready = candidate
    return [by_rank[rank] for rank in sorted(by_rank)], recovered_ready


def _cache_ready_evidence(
    *, run_id: str, job_id: str, ready: dict[str, Any]
) -> dict[str, Any]:
    value = dict(ready)
    value["local_evidence_transport"] = {
        "source": "Aliyun DLC GetPodLogs compressed ready marker",
        "job_id": job_id,
        "cloud_storage_not_locally_mounted": True,
    }
    path = ready_path(run_id)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_id") != run_id:
            raise ValueError(f"本地 ready evidence 与 run_id 不一致：{path}")
        return existing
    atomic_json(path, value)
    return value


def job_gpu_metrics(
    client: Any, job_id: str, job_body: Any | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from alibabacloud_pai_dlc20201203.models import (
        GetJobMetricsRequest,
        GetMetricsRequest,
    )

    now = datetime.now(timezone.utc)
    errors: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    for metric_type in ("GpuMemoryUsage", "GpuCoreUsage"):
        try:
            body = client.get_job_metrics(
                job_id,
                GetJobMetricsRequest(
                    metric_type=metric_type,
                    start_time=(now - timedelta(minutes=15)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    end_time=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    time_step="1m",
                ),
            ).body
            pods = []
            for pod in list(getattr(body, "pod_metrics", None) or []):
                samples = list(getattr(pod, "metrics", None) or [])
                latest = max(
                    samples,
                    key=lambda item: int(getattr(item, "time", 0) or 0),
                    default=None,
                )
                pods.append(
                    {
                        "pod_id": str(getattr(pod, "pod_id", "") or ""),
                        "sample_count": len(samples),
                        "latest_time_unix_ms": (
                            int(getattr(latest, "time", 0) or 0) if latest else None
                        ),
                        "latest_value_percent": (
                            float(getattr(latest, "value", 0.0)) if latest else None
                        ),
                    }
                )
            result[metric_type] = {
                "unit": "percent",
                "source": "Aliyun DLC GetJobMetrics",
                "per_pod": pods,
            }
        except Exception as exc:
            errors.append(
                {"operation": f"get_job_metrics:{metric_type}", **provider_error(exc)}
            )
    # GetJobMetrics is retained for backwards compatibility, but Aliyun routes
    # most DLC resource types to CloudMonitor now.  The proxy GetMetrics API is
    # read-only and avoids treating an empty legacy response as 0% utilization.
    for metric_name in (
        "JOB_GPU_ACCELERATOR_DUTTY_UTIL",
        "CARD_GPU_DRAM_ACTIVE_UTIL",
    ):
        try:
            base_dimension = {
                "jobId": job_id,
                "regionId": str(getattr(client, "_region_id", "") or ""),
                "userId": str(getattr(job_body, "user_id", "") or ""),
                "workspaceId": str(getattr(job_body, "workspace_id", "") or ""),
            }
            dimensions = [base_dimension]
            if metric_name.startswith("CARD_") and job_body is not None:
                dimensions = [
                    {**base_dimension, "pod": str(getattr(pod, "pod_id", "") or "")}
                    for pod in list(getattr(job_body, "pods", None) or [])
                    if getattr(pod, "pod_id", None)
                ]
            body = client.get_metrics(
                GetMetricsRequest(
                    job_id=job_id,
                    dimensions=json.dumps(dimensions, separators=(",", ":")),
                    namespace="acs_pai_dlc",
                    metric_name=metric_name,
                    start_time=(now - timedelta(hours=3)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    end_time=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    period="60",
                    length="1000",
                )
            ).body
            raw = str(getattr(body, "data_points", "") or "[]")
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                decoded = []
            if not isinstance(decoded, list):
                decoded = []
            samples: list[dict[str, Any]] = []
            values: list[float] = []
            for point in decoded:
                if not isinstance(point, dict):
                    continue
                value = point.get("Value", point.get("value"))
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                values.append(numeric)
                samples.append(
                    {
                        key: point[key]
                        for key in (
                            "timestamp",
                            "Timestamp",
                            "jobId",
                            "pod",
                            "podId",
                            "gpu",
                            "gpuId",
                            "Value",
                            "value",
                        )
                        if key in point
                    }
                )
            result[metric_name] = {
                "unit": "percent",
                "source": "Aliyun CloudMonitor via PAI DLC GetMetrics",
                "provider_success": bool(getattr(body, "success", False)),
                "provider_code": str(getattr(body, "code", "") or ""),
                "provider_message_full_credential_redacted": redact(
                    getattr(body, "message", "")
                ),
                "period_seconds_requested": 60,
                "sample_count": len(values),
                "average_percent": sum(values) / len(values) if values else None,
                "maximum_percent": max(values) if values else None,
                "minimum_percent": min(values) if values else None,
                "samples": samples,
            }
        except Exception as exc:
            errors.append(
                {"operation": f"get_metrics:{metric_name}", **provider_error(exc)}
            )
    return result, errors


def job_pod_log_tails(
    client: Any,
    job_id: str,
    body: Any,
    *,
    max_lines: int = 5000,
    tail_lines: int = 200,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from alibabacloud_pai_dlc20201203.models import GetPodLogsRequest

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for pod in list(getattr(body, "pods", None) or []):
        pod_id = str(getattr(pod, "pod_id", "") or "")
        if not pod_id:
            continue
        try:
            response = client.get_pod_logs(
                job_id, pod_id, GetPodLogsRequest(max_lines=max_lines)
            )
            lines = [str(line) for line in (response.body.logs or [])]
            tail = lines[-tail_lines:] if tail_lines else []
            rows.append(
                {
                    "pod_id": pod_id,
                    "line_count_retrieved": len(lines),
                    "runtime_marker": _latest_json_marker(lines, RUNTIME_MARKER),
                    "ready_marker": _latest_json_marker(lines, READY_MARKER),
                    "tail_credential_redacted": [
                        redact(line)[:4000]
                        for line in tail
                        if RUNTIME_MARKER not in line and READY_MARKER not in line
                    ],
                }
            )
        except Exception as exc:
            errors.append(
                {"operation": "get_pod_logs", "pod_id": pod_id, **provider_error(exc)}
            )
    return rows, errors


def render_command(args: argparse.Namespace) -> int:
    loaded = load_config(args.config)
    run_id = safe_run_id(args.run_id)
    chosen_model = model_path(loaded.value, args.model_path)
    blockers = submission_blockers(loaded.value, model_override=args.model_path)
    plan = render_plan(
        loaded,
        run_id=run_id,
        model_path=chosen_model,
        blockers=blockers,
    )
    output = inside_project(args.output)
    atomic_json(output, plan)
    print(json.dumps({"output": str(output), "submission_ready": not blockers, "blockers": blockers}, ensure_ascii=False))
    return 0


def _admission_lock_handle():
    ADMISSION_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = ADMISSION_LOCK.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _empty_inventory() -> dict[str, Any]:
    return {
        "schema_version": "project-h200-inventory-v1",
        "policy": {
            "project_h200_cap": PROJECT_H200_CAP,
            "per_task_approval_threshold_h200": TASK_APPROVAL_THRESHOLD_H200,
        },
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "jobs": [],
    }


def _load_inventory() -> dict[str, Any]:
    if not INVENTORY_PATH.is_file():
        return _empty_inventory()
    value = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("项目 H200 inventory 文件损坏")
    if value.get("schema_version") != "project-h200-inventory-v1":
        raise ValueError("项目 H200 inventory schema_version 不受支持")
    if not isinstance(value.get("jobs"), list):
        raise ValueError("项目 H200 inventory.jobs 必须是 list")
    return value


def _write_inventory(inventory: dict[str, Any]) -> None:
    inventory["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_json(INVENTORY_PATH, inventory, overwrite=True)


def _update_inventory_job(
    inventory: dict[str, Any], job_id: str, *, status: str, observed_at: str
) -> None:
    for item in inventory["jobs"]:
        if isinstance(item, dict) and str(item.get("job_id") or "") == job_id:
            item["status"] = status
            item["last_observed_at"] = observed_at


def _reconcile_inventory(client: Any, inventory: dict[str, Any]) -> list[dict[str, Any]]:
    from alibabacloud_pai_dlc20201203.models import GetJobRequest

    errors: list[dict[str, Any]] = []
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    for item in inventory["jobs"]:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "") in TERMINAL_JOB_STATES:
            continue
        job_id = str(item.get("job_id") or "")
        if not job_id:
            errors.append(
                {
                    "run_id": str(item.get("run_id") or ""),
                    "error": "本地记录停留在 submitting 且没有 job_id，需先在 capacity 的 workspace job 列表中人工核对",
                }
            )
            continue
        try:
            body = client.get_job(job_id, GetJobRequest()).body
        except Exception as exc:
            errors.append({"job_id": job_id, "provider_error": provider_error(exc)})
            continue
        item["status"] = str(getattr(body, "status", "") or "Unknown")
        item["last_observed_at"] = now
    return errors


def _capacity_snapshot(loaded: Any, client: Any, inventory: dict[str, Any]) -> dict[str, Any]:
    config = loaded.value
    resource_id = str(config["aliyun"]["resource_id"])
    workspace_id = str(config["aliyun"]["workspace_id"])
    resource_client = paistudio_client(str(config["aliyun"]["region"]))
    quota = get_quota(resource_client, resource_id)
    nodes = list_quota_nodes(resource_client, resource_id)
    jobs = list_active_workspace_jobs(
        client, workspace_id=workspace_id, quota_id=resource_id
    )
    return build_capacity_snapshot(
        config=config,
        inventory=inventory,
        quota_body=quota,
        node_body=nodes,
        workspace_jobs=jobs,
    )


def _require_admission(
    snapshot: dict[str, Any],
    requested_h200: int,
    *,
    confirmed_provider_gpu_type: str | None = None,
    allow_provider_queue: bool = False,
) -> None:
    project = snapshot["project_policy"]
    if project["active_requested_h200"] is None:
        raise RuntimeError("项目活动任务中存在无法确定 GPU 申请量的 provider job，拒绝提交")
    projected = int(project["active_requested_h200"]) + requested_h200
    if project["unresolved_submission_count"]:
        raise RuntimeError(
            "项目 inventory 中存在没有 job_id 的未决提交；先运行 capacity 并人工核对，拒绝继续提交"
        )
    if projected > PROJECT_H200_CAP:
        raise RuntimeError(
            f"项目活动任务加本次申请将达到 {projected} H200，超过总上限 {PROJECT_H200_CAP}"
        )
    available = snapshot["provider_quota"]["gpu"]["available"]
    free_nodes = snapshot["provider_quota"]["nodes"]["eight_gpu_nodes_free"]
    type_counts = snapshot["provider_quota"]["nodes"]["gpu_type_node_counts"]
    provider_types = {str(item) for item in type_counts if str(item) != "Unknown"}
    required_nodes = requested_h200 // 8
    if not allow_provider_queue:
        if not isinstance(available, (int, float)) or available < requested_h200:
            raise RuntimeError(
                f"Aliyun quota 当前可用 GPU={available!r}，不足本次 {requested_h200} H200"
            )
        if not isinstance(free_nodes, int) or free_nodes < required_nodes:
            raise RuntimeError(
                f"Aliyun quota 当前完整空闲 8 卡节点={free_nodes!r}，不足本次 {required_nodes} 节点"
            )
    if not provider_types:
        raise RuntimeError("Aliyun ListNodes 没有返回可核验的 GPUType，拒绝提交")
    if not all("H200" in item.upper() for item in provider_types):
        confirmed = str(confirmed_provider_gpu_type or "").strip()
        if confirmed not in provider_types or len(provider_types) != 1:
            raise RuntimeError(
                "provider GPUType 与字面 H200 不一致："
                f"{sorted(provider_types)}；核实型号映射后传 --confirm-provider-gpu-type <精确值>"
            )


def capacity_command(args: argparse.Namespace) -> int:
    loaded = load_config(args.config)
    output = inside_project(args.output)
    client = provider_client()
    lock = _admission_lock_handle()
    try:
        inventory = _load_inventory()
        reconciliation_errors = _reconcile_inventory(client, inventory)
        snapshot = _capacity_snapshot(loaded, client, inventory)
        snapshot["inventory_reconciliation_errors"] = reconciliation_errors
        _write_inventory(inventory)
    except Exception as exc:
        snapshot = {
            "schema_version": "project-h200-capacity-v1",
            "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "available": False,
            "provider_error": provider_error(exc),
            "project_policy": summarize_project_inventory(_load_inventory()),
        }
    finally:
        lock.close()
    atomic_json(output, snapshot)
    ok = snapshot.get("available", True) is not False and not snapshot.get(
        "inventory_reconciliation_errors"
    )
    quota = snapshot.get("provider_quota") or {}
    nodes = dict(quota.get("nodes") or {})
    nodes.pop("items", None)
    print(
        json.dumps(
            {
                "available": ok,
                "project": snapshot.get("project_policy"),
                "provider_quota": {
                    "quota_id": quota.get("quota_id"),
                    "quota_name": quota.get("quota_name"),
                    "quota_status": quota.get("quota_status"),
                    "resource_type": quota.get("resource_type"),
                    "gpu": quota.get("gpu"),
                    "nodes": nodes,
                }
                if quota
                else None,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 2


def recent_images_command(args: argparse.Namespace) -> int:
    """只读列出当前 workspace/quota 近期 DLC job 实际使用的容器镜像。"""
    loaded = load_config(args.config)
    config = loaded.value
    output = inside_project(args.output)
    client = provider_client()
    from alibabacloud_pai_dlc20201203.models import GetJobRequest, ListJobsRequest

    response = client.list_jobs(
        ListJobsRequest(
            workspace_id=str(config["aliyun"]["workspace_id"]),
            resource_id=str(config["aliyun"]["resource_id"]),
            start_time=args.start_time,
            page_number=1,
            page_size=args.limit,
            sort_by="GmtCreateTime",
            order="desc",
        )
    )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for summary in list(getattr(response.body, "jobs", None) or [])[: args.limit]:
        job_id = str(getattr(summary, "job_id", "") or "")
        if not job_id:
            continue
        try:
            body = client.get_job(job_id, GetJobRequest()).body
        except Exception as exc:
            errors.append({"job_id": job_id, "provider_error": provider_error(exc)})
            continue
        images = sorted(
            {
                str(getattr(spec, "image", "") or "")
                for spec in list(getattr(body, "job_specs", None) or [])
                if str(getattr(spec, "image", "") or "")
            }
        )
        rows.append(
            {
                "job_id": job_id,
                "display_name": str(getattr(body, "display_name", "") or ""),
                "status": str(getattr(body, "status", "") or ""),
                "created_at": str(getattr(body, "gmt_create_time", "") or ""),
                "priority": getattr(body, "priority", None),
                "images": images,
            }
        )
    image_counts: dict[str, int] = {}
    for row in rows:
        for image in row["images"]:
            image_counts[image] = image_counts.get(image, 0) + 1
    receipt = {
        "schema_version": "qwen38-397b-dlc-recent-images-v1",
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "workspace_id": str(config["aliyun"]["workspace_id"]),
        "resource_id": str(config["aliyun"]["resource_id"]),
        "start_time": args.start_time,
        "limit": args.limit,
        "jobs": rows,
        "image_counts": dict(sorted(image_counts.items())),
        "provider_errors": errors,
    }
    atomic_json(output, receipt)
    print(
        json.dumps(
            {
                "job_count": len(rows),
                "images": receipt["image_counts"],
                "provider_error_count": len(errors),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not errors else 2


def acr_images_command(args: argparse.Namespace) -> int:
    """只读查询配置中精确 ACR repository 的 tag 清单。"""
    from .acr import inventory

    output = inside_project(args.output)
    try:
        receipt = inventory(args.config, max_tags=args.max_tags)
        receipt["available"] = True
    except Exception as exc:
        receipt = {
            "schema_version": "qwen38-397b-acr-image-inventory-v1",
            "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "available": False,
            "provider_error": provider_error(exc),
        }
    atomic_json(output, receipt)
    repositories = receipt.get("repositories") or []
    latest = [
        {
            "instance_name": row.get("instance_name"),
            "namespace": row.get("namespace"),
            "repository": row.get("repository"),
            "latest": (row.get("tags") or [None])[0],
        }
        for row in repositories
    ]
    print(
        json.dumps(
            {
                "available": receipt.get("available"),
                "matching_repository_count": len(repositories),
                "latest_by_repository": latest,
                "provider_error": receipt.get("provider_error"),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if receipt.get("available") else 2


def pai_image_command(args: argparse.Namespace) -> int:
    """按精确 PAI custom image ID 获取部署所需 ImageUri。"""
    loaded = load_config(args.config)
    output = inside_project(args.output)
    region = str(loaded.value["aliyun"]["region"])
    workspace_id = str(loaded.value["aliyun"]["workspace_id"])
    try:
        vendor = ROOT / ".runtime" / "vendor" / "aiworkspace20210204"
        if vendor.is_dir() and str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        ube_root = ROOT / "unified_benchmark_eval"
        if str(ube_root) not in sys.path:
            sys.path.insert(0, str(ube_root))
        from alibabacloud_aiworkspace20210204.client import Client as AIWorkspaceClient
        from alibabacloud_aiworkspace20210204.models import GetImageRequest
        from alibabacloud_tea_openapi.models import Config as OpenAPIConfig
        from scripts.dlc_eval import ALIYUN_CONFIG_PATH
        from ube.aliyun_auth import load_access_key_credentials

        client = AIWorkspaceClient(
            OpenAPIConfig(
                region_id=region,
                endpoint=f"aiworkspace.{region}.aliyuncs.com",
                protocol="https",
                **load_access_key_credentials(ALIYUN_CONFIG_PATH),
            )
        )
        body = client.get_image(
            args.image_id, GetImageRequest(verbose=True)
        ).body.to_map()
        observed_id = str(body.get("ImageId") or args.image_id)
        observed_workspace = str(body.get("WorkspaceId") or "")
        image_uri = str(body.get("ImageUri") or "").strip()
        if observed_id != args.image_id:
            raise RuntimeError(
                f"GetImage 返回 ImageId={observed_id!r}，与请求不一致"
            )
        if observed_workspace and observed_workspace != workspace_id:
            raise RuntimeError(
                f"镜像 workspace={observed_workspace!r}，不是部署 workspace={workspace_id!r}"
            )
        if not image_uri or "/" not in image_uri:
            raise RuntimeError("GetImage 未返回可用 ImageUri")
        receipt = {
            "schema_version": "qwen38-397b-pai-image-v1",
            "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "available": True,
            "source": {
                "provider": "Aliyun AIWorkSpace GetImage",
                "api_version": "2021-02-04",
                "not_inferred_from_dlc_jobs": True,
                "not_inferred_from_acr_tag_order": True,
            },
            "image": {
                "image_id": observed_id,
                "name": str(body.get("Name") or "").strip(),
                "image_uri": image_uri,
                "workspace_id": observed_workspace or workspace_id,
                "gmt_create_time": str(body.get("GmtCreateTime") or "").strip(),
                "gmt_modified_time": str(body.get("GmtModifiedTime") or "").strip(),
                "accessibility": str(body.get("Accessibility") or "").strip(),
                "size_bytes": body.get("Size"),
                "labels": [
                    {
                        "key": str(item.get("Key") or ""),
                        "value": str(item.get("Value") or ""),
                    }
                    for item in (body.get("Labels") or [])
                    if isinstance(item, dict)
                ],
            },
        }
    except Exception as exc:
        receipt = {
            "schema_version": "qwen38-397b-pai-image-v1",
            "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "available": False,
            "requested_image_id": args.image_id,
            "provider_error": provider_error(exc),
        }
    atomic_json(output, receipt)
    print(
        json.dumps(
            {
                "available": receipt.get("available"),
                "image": receipt.get("image"),
                "provider_error": receipt.get("provider_error"),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if receipt.get("available") else 2


def submit_command(args: argparse.Namespace) -> int:
    if not args.apply:
        raise ValueError("真实提交必须显式传 --apply")
    loaded = load_config(args.config)
    requested_h200 = total_gpus(loaded.value)
    if args.confirm_total_h200 != requested_h200:
        raise ValueError(f"必须显式传 --confirm-total-h200 {requested_h200}")
    if not args.confirm_image_verified:
        raise ValueError("必须先验证镜像，再显式传 --confirm-image-verified")
    if requested_h200 > TASK_APPROVAL_THRESHOLD_H200 and not str(
        args.approval_ref or ""
    ).strip():
        raise ValueError(
            f"单任务超过 {TASK_APPROVAL_THRESHOLD_H200} H200，必须传 --approval-ref 记录用户批准依据"
        )
    run_id = safe_run_id(args.run_id)
    if (RUNTIME_ROOT / run_id).exists():
        raise ValueError(f"run_id 已有云端 runtime 目录，必须换新 run_id：{run_id}")
    chosen_model = model_path(loaded.value, args.model_path)
    blockers = submission_blockers(loaded.value, model_override=args.model_path)
    blockers = [item for item in blockers if not item.startswith("候选镜像")]
    if blockers:
        raise ValueError("不能提交：" + "；".join(blockers))
    receipt_path = inside_project(args.receipt)
    if receipt_path.exists():
        raise FileExistsError(f"拒绝覆盖 receipt：{receipt_path}")
    request = build_create_job_request(
        loaded, run_id=run_id, model_path=chosen_model
    )
    client = provider_client()
    lock = _admission_lock_handle()
    try:
        inventory = _load_inventory()
        reconciliation_errors = _reconcile_inventory(client, inventory)
        if reconciliation_errors:
            raise RuntimeError(
                "无法安全核对项目 H200 inventory，拒绝提交："
                + json.dumps(reconciliation_errors, ensure_ascii=False)
            )
        capacity = _capacity_snapshot(loaded, client, inventory)
        _require_admission(
            capacity,
            requested_h200,
            confirmed_provider_gpu_type=args.confirm_provider_gpu_type,
            allow_provider_queue=args.allow_provider_queue,
        )
        submitted_at = datetime.now().astimezone().isoformat(timespec="seconds")
        inventory_entry = {
            "controller": "qwen38_flash_next_h200",
            "run_id": run_id,
            "mode": loaded.value["mode"],
            "display_name": request["DisplayName"],
            "job_id": "",
            "status": "Submitting",
            "requested_h200": requested_h200,
            "approval_ref": str(args.approval_ref or "").strip() or None,
            "submitted_at": submitted_at,
            "last_observed_at": submitted_at,
            "receipt": receipt_path.relative_to(ROOT).as_posix(),
        }
        inventory["jobs"].append(inventory_entry)
        _write_inventory(inventory)
        from alibabacloud_pai_dlc20201203.models import CreateJobRequest

        try:
            response = client.create_job(CreateJobRequest().from_map(request))
            job_id = str(response.body.job_id)
        except Exception as exc:
            inventory_entry["status"] = "SubmitFailed"
            inventory_entry["last_observed_at"] = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
            _write_inventory(inventory)
            failure = {
                "schema_version": "qwen38-flash-next-h200-submit-v1",
                "submitted_at": submitted_at,
                "run_id": run_id,
                "mode": loaded.value["mode"],
                "submitted": False,
                "capacity_before_submit": capacity,
                "gpu_accounting": {
                    "requested": requested_h200,
                    "reserved": 0,
                    "actually_used": 0,
                },
                "provider_error": provider_error(exc),
            }
            atomic_json(receipt_path, failure)
            print(json.dumps({"submitted": False, "receipt": str(receipt_path)}, ensure_ascii=False))
            return 2
        inventory_entry["job_id"] = job_id
        inventory_entry["status"] = "Creating"
        inventory_entry["last_observed_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        inventory_entry["max_running_time_minutes"] = loaded.value["aliyun"][
            "job_max_running_time_minutes"
        ]
        _write_inventory(inventory)
        receipt = {
            "schema_version": "qwen38-flash-next-h200-submit-v1",
            "submitted_at": submitted_at,
            "run_id": run_id,
            "mode": loaded.value["mode"],
            "submitted": True,
            "job_id": job_id,
            "config_sha256": loaded.sha256,
            "capacity_before_submit": capacity,
            "operator_confirmations": {
                "total_h200": args.confirm_total_h200,
                "image_verified_for_qwen38_bf16": args.confirm_image_verified,
                "approval_ref": str(args.approval_ref or "").strip() or None,
                "provider_gpu_type": str(args.confirm_provider_gpu_type or "").strip()
                or None,
                "provider_queue_allowed": bool(args.allow_provider_queue),
            },
            "request": request,
            "deployment_manifest": build_deployment_manifest(
                loaded,
                run_id=run_id,
                model_path=chosen_model,
                request=request,
            ),
            "gpu_accounting": {
                "requested": requested_h200,
                "reserved": 0,
                "actually_used": 0,
                "evidence": (
                    f"CreateJob 已返回 job_id；尚未取得 "
                    f"{total_nodes(loaded.value)} 个 worker/GPU 运行实证"
                ),
            },
            "lifecycle": {
                "max_running_time_minutes": loaded.value["aliyun"]["job_max_running_time_minutes"],
                "stop_command_required": True,
            },
        }
        atomic_json(receipt_path, receipt)
        print(json.dumps({"submitted": True, "job_id": job_id, "receipt": str(receipt_path)}, ensure_ascii=False))
        return 0
    finally:
        lock.close()


def _source_receipt(path: Path) -> dict[str, Any]:
    resolved = inside_project(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "qwen38-flash-next-h200-submit-v1":
        raise ValueError("source receipt 不是本部署包的 submit receipt")
    if value.get("submitted") is not True or not value.get("job_id"):
        raise ValueError("source receipt 没有成功提交的 job_id")
    return value


def _source_requested_h200(source: dict[str, Any]) -> int:
    value = (source.get("gpu_accounting") or {}).get("requested")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    # Backward compatibility for receipts created before dynamic node counts.
    return 32


def status_command(args: argparse.Namespace) -> int:
    source = _source_receipt(args.source_receipt)
    requested_h200 = _source_requested_h200(source)
    node_count = requested_h200 // 8
    output = inside_project(args.output)
    client = provider_client()
    from alibabacloud_pai_dlc20201203.models import GetJobRequest

    ready = read_ready(source["run_id"])
    body: Any = None
    try:
        body = client.get_job(source["job_id"], GetJobRequest()).body
        observation = job_observation(body)
        observation_error = None
    except Exception as exc:
        observation = {
            "job_id": source["job_id"],
            "display_name": "",
            "status": "",
            "pods": [],
        }
        observation_error = provider_error(exc)
    provider_metrics, metric_errors = job_gpu_metrics(
        client, source["job_id"], job_body=body
    )
    runtime_probes = read_runtime_probes(source["run_id"], node_count)
    pod_logs: list[dict[str, Any]] = []
    pod_log_errors: list[dict[str, Any]] = []
    if args.include_logs and body is not None:
        pod_logs, pod_log_errors = job_pod_log_tails(
            client, source["job_id"], body
        )
        recovered_runtime, recovered_ready = _log_evidence(pod_logs)
        if recovered_runtime:
            runtime_probes = recovered_runtime
        if recovered_ready:
            ready = _cache_ready_evidence(
                run_id=source["run_id"],
                job_id=source["job_id"],
                ready=recovered_ready,
            )
    value = {
        "schema_version": "qwen38-flash-next-h200-status-v1",
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": source["run_id"],
        "mode": source["mode"],
        "provider": observation,
        "provider_error": observation_error,
        "provider_gpu_metrics": provider_metrics,
        "provider_gpu_metric_errors": metric_errors,
        "worker_runtime_probes": runtime_probes,
        "pod_log_tails": pod_logs,
        "pod_log_errors": pod_log_errors,
        "ready": ready,
        "gpu_accounting": (
            ready.get("gpu_accounting")
            if ready
            else {
                "requested": requested_h200,
                "reserved": None,
                "actually_used": None,
                "evidence": "尚无 worker ready.json，不从控制面状态推断 GPU 实际使用",
            }
        ),
    }
    atomic_json(output, value)
    if observation_error is None:
        lock = _admission_lock_handle()
        try:
            inventory = _load_inventory()
            _update_inventory_job(
                inventory,
                source["job_id"],
                status=observation["status"] or "Unknown",
                observed_at=value["observed_at"],
            )
            _write_inventory(inventory)
        finally:
            lock.close()
    memory = provider_metrics.get("GpuMemoryUsage", {}).get("per_pod", [])
    print(
        json.dumps(
            {
                "job_id": source["job_id"],
                "status": observation["status"],
                "ready": bool(ready),
                "gpu_memory_latest_percent_by_pod": {
                    row["pod_id"]: row["latest_value_percent"] for row in memory
                },
                "worker_gpu_memory_used_mb_by_rank": {
                    str(row.get("rank")): row.get("gpu_memory_used_mb")
                    for row in runtime_probes
                },
                "metric_error_count": len(metric_errors),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if observation_error is None else 2


def monitor_command(args: argparse.Namespace) -> int:
    sources = [_source_receipt(path) for path in args.source_receipt]
    if len({str(item["job_id"]) for item in sources}) != len(sources):
        raise ValueError("monitor source receipts 含重复 job_id")
    events_path = inside_project(args.events)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    if events_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 monitor events：{events_path}")
    client = provider_client()
    from alibabacloud_pai_dlc20201203.models import GetJobRequest

    deadline = time.monotonic() + args.timeout
    sequence = 0
    with events_path.open("x", encoding="utf-8") as handle:
        while True:
            sequence += 1
            rows = []
            any_failed = False
            all_ready = True
            for source in sources:
                requested_h200 = _source_requested_h200(source)
                node_count = requested_h200 // 8
                observed_at = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
                body: Any = None
                try:
                    body = client.get_job(source["job_id"], GetJobRequest()).body
                    provider = job_observation(body)
                    error = None
                except Exception as exc:
                    provider = {
                        "job_id": source["job_id"],
                        "display_name": "",
                        "status": "",
                        "pods": [],
                    }
                    error = provider_error(exc)
                metrics, metric_errors = job_gpu_metrics(client, source["job_id"])
                ready = read_ready(source["run_id"])
                runtime_probes = read_runtime_probes(source["run_id"], node_count)
                pod_log_errors: list[dict[str, Any]] = []
                if body is not None:
                    pod_logs, pod_log_errors = job_pod_log_tails(
                        client,
                        source["job_id"],
                        body,
                        max_lines=500,
                        tail_lines=0,
                    )
                    recovered_runtime, recovered_ready = _log_evidence(pod_logs)
                    if recovered_runtime:
                        runtime_probes = recovered_runtime
                    if recovered_ready:
                        ready = _cache_ready_evidence(
                            run_id=source["run_id"],
                            job_id=source["job_id"],
                            ready=recovered_ready,
                        )
                proven_ready = bool(
                    ready
                    and (ready.get("gpu_accounting") or {}).get("actually_used")
                    == requested_h200
                )
                all_ready = all_ready and proven_ready
                if provider["status"] in TERMINAL_STATES and not proven_ready:
                    any_failed = True
                rows.append(
                    {
                        "run_id": source["run_id"],
                        "mode": source["mode"],
                        "job_id": source["job_id"],
                        "observed_at": observed_at,
                        "provider": provider,
                        "provider_error": error,
                        "provider_gpu_metrics": metrics,
                        "provider_gpu_metric_errors": metric_errors,
                        "pod_log_errors": pod_log_errors,
                        "worker_runtime_probes": runtime_probes,
                        "ready": ready,
                        "proven_ready": proven_ready,
                    }
                )
            event = {
                "schema_version": "qwen38-flash-next-h200-monitor-event-v1",
                "sequence": sequence,
                "all_ready": all_ready,
                "any_failed": any_failed,
                "jobs": rows,
            }
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(
                json.dumps(
                    {
                        "sequence": sequence,
                        "jobs": [
                            {
                                "mode": row["mode"],
                                "job_id": row["job_id"],
                                "status": row["provider"]["status"],
                                "ready": row["proven_ready"],
                                "runtime_probe_ranks": len(
                                    row["worker_runtime_probes"]
                                ),
                                "gpu_memory_latest_percent_by_pod": {
                                    item["pod_id"]: item["latest_value_percent"]
                                    for item in row["provider_gpu_metrics"]
                                    .get("GpuMemoryUsage", {})
                                    .get("per_pod", [])
                                },
                            }
                            for row in rows
                        ],
                        "events": str(events_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if all_ready:
                return 0
            if any_failed:
                return 2
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 2
            time.sleep(min(args.interval, remaining))


def stop_command(args: argparse.Namespace) -> int:
    source = _source_receipt(args.source_receipt)
    job_id = source["job_id"]
    if not args.apply or args.confirm_job_id != job_id:
        raise ValueError(f"停止任务必须传 --apply --confirm-job-id {job_id}")
    output = inside_project(args.output)
    client = provider_client()
    from alibabacloud_pai_dlc20201203.models import GetJobRequest

    requested_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        client.stop_job(job_id)
        stop_error = None
    except Exception as exc:
        stop_error = provider_error(exc)
    last = {"job_id": job_id, "status": ""}
    observation_error = None
    if stop_error is None:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            try:
                body = client.get_job(job_id, GetJobRequest()).body
            except Exception as exc:
                observation_error = provider_error(exc)
                break
            last = job_observation(body)
            if last["status"] in TERMINAL_STATES:
                break
            time.sleep(5)
    value = {
        "schema_version": "qwen38-flash-next-h200-stop-v1",
        "stop_requested_at": requested_at,
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": source["run_id"],
        "mode": source["mode"],
        "job_id": job_id,
        "stop_error": stop_error,
        "provider_observation_error": observation_error,
        "provider": last,
        "terminal": last["status"] in TERMINAL_STATES,
    }
    atomic_json(output, value)
    if last["status"]:
        lock = _admission_lock_handle()
        try:
            inventory = _load_inventory()
            _update_inventory_job(
                inventory,
                job_id,
                status=last["status"],
                observed_at=value["observed_at"],
            )
            _write_inventory(inventory)
        finally:
            lock.close()
    print(json.dumps({"job_id": job_id, "status": last["status"], "terminal": value["terminal"], "output": str(output)}, ensure_ascii=False))
    return 0 if value["terminal"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.8 Flash Next 多节点 H200 DLC 部署控制器")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="只渲染 CreateJob request，不调用 provider")
    render.add_argument("--config", type=Path, required=True)
    render.add_argument("--run-id", required=True)
    render.add_argument("--model-path")
    render.add_argument("--output", type=Path, required=True)
    render.set_defaults(func=render_command)

    capacity = subparsers.add_parser(
        "capacity", help="只读查询 workspace/quota、活动任务和项目 128 卡余量"
    )
    capacity.add_argument("--config", type=Path, required=True)
    capacity.add_argument("--output", type=Path, required=True)
    capacity.set_defaults(func=capacity_command)

    recent_images = subparsers.add_parser(
        "recent-images", help="只读列出 workspace/quota 近期 DLC job 的实际容器镜像"
    )
    recent_images.add_argument("--config", type=Path, required=True)
    recent_images.add_argument("--start-time", required=True)
    recent_images.add_argument("--limit", type=int, default=100)
    recent_images.add_argument("--output", type=Path, required=True)
    recent_images.set_defaults(func=recent_images_command)

    acr_images = subparsers.add_parser(
        "acr-images", help="只读列出配置中精确 ACR 仓库的镜像 tag"
    )
    acr_images.add_argument("--config", type=Path, required=True)
    acr_images.add_argument("--max-tags", type=int, default=500)
    acr_images.add_argument("--output", type=Path, required=True)
    acr_images.set_defaults(func=acr_images_command)

    pai_image = subparsers.add_parser(
        "pai-image", help="按精确 PAI 工作空间镜像 ID 获取 ImageUri"
    )
    pai_image.add_argument("--config", type=Path, required=True)
    pai_image.add_argument("--image-id", required=True)
    pai_image.add_argument("--output", type=Path, required=True)
    pai_image.set_defaults(func=pai_image_command)

    from .preflight import add_parser as add_preflight_parser

    add_preflight_parser(subparsers)

    from .eas import add_parsers as add_eas_parsers

    add_eas_parsers(subparsers)

    submit = subparsers.add_parser("submit", help="显式确认后真实提交配置所需的 H200 job")
    submit.add_argument("--config", type=Path, required=True)
    submit.add_argument("--run-id", required=True)
    submit.add_argument("--model-path")
    submit.add_argument("--receipt", type=Path, required=True)
    submit.add_argument("--apply", action="store_true")
    submit.add_argument("--confirm-total-h200", type=int)
    submit.add_argument("--confirm-image-verified", action="store_true")
    submit.add_argument(
        "--allow-provider-queue",
        action="store_true",
        help=(
            "项目 128-H200 admission 仍通过时，允许在 Aliyun 瞬时空闲 GPU/节点不足时"
            "先提交到 provider 队列；默认仍要求提交时立即有足够节点和 GPU"
        ),
    )
    submit.add_argument(
        "--confirm-provider-gpu-type",
        help="仅当 capacity 返回的 GPUType 不是字面 H200 时，核实映射后传 provider 精确值",
    )
    submit.add_argument(
        "--approval-ref",
        help="仅单任务超过 32 H200 时必填：记录本次用户批准的非敏感引用",
    )
    submit.set_defaults(func=submit_command)

    status = subparsers.add_parser("status", help="查询一个本项目 submit receipt 绑定的 job")
    status.add_argument("--source-receipt", type=Path, required=True)
    status.add_argument("--output", type=Path, required=True)
    status.add_argument(
        "--include-logs",
        action="store_true",
        help="额外读取每个 Pod 最多 200 行凭据脱敏日志尾部",
    )
    status.set_defaults(func=status_command)

    monitor = subparsers.add_parser(
        "monitor", help="有界轮询本项目任务的状态、显存与 ready 证据"
    )
    monitor.add_argument(
        "--source-receipt", type=Path, action="append", required=True
    )
    monitor.add_argument("--events", type=Path, required=True)
    monitor.add_argument("--interval", type=int, default=30)
    monitor.add_argument("--timeout", type=int, default=7200)
    monitor.set_defaults(func=monitor_command)

    stop = subparsers.add_parser("stop", help="停止并有界等待一个本项目 job")
    stop.add_argument("--source-receipt", type=Path, required=True)
    stop.add_argument("--output", type=Path, required=True)
    stop.add_argument("--apply", action="store_true")
    stop.add_argument("--confirm-job-id")
    stop.add_argument("--timeout", type=int, default=180)
    stop.set_defaults(func=stop_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    timeout_limit = (
        14400
        if args.command == "monitor"
        else (3600 if args.command == "preflight" else 600)
    )
    if getattr(args, "timeout", 1) <= 0 or getattr(args, "timeout", 1) > timeout_limit:
        parser.error(f"timeout 必须位于 1..{timeout_limit} 秒")
    if hasattr(args, "interval") and not 10 <= args.interval <= 300:
        parser.error("interval 必须位于 10..300 秒")
    if hasattr(args, "limit") and not 1 <= args.limit <= 100:
        parser.error("limit 必须位于 1..100")
    if hasattr(args, "max_tags") and not 1 <= args.max_tags <= 500:
        parser.error("max-tags 必须位于 1..500")
    if getattr(args, "stop_wait_timeout", 1) <= 0 or getattr(
        args, "stop_wait_timeout", 1
    ) > 600:
        parser.error("stop-wait-timeout 必须位于 1..600 秒")
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
