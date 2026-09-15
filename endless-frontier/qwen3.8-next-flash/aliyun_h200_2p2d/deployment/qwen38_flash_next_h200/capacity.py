from __future__ import annotations

from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys
import os
import json
from urllib.request import urlopen

def _credentials():
    access_key = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
    secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
    token = os.environ.get("ALIBABA_CLOUD_SECURITY_TOKEN")
    if not access_key or not secret:
        try:
            with urlopen("http://localhost:7002/api/v1/credentials/0", timeout=3) as h:
                value = json.load(h)
            access_key, secret, token = value.get("AccessKeyId"), value.get("AccessKeySecret"), value.get("SecurityToken")
        except Exception:
            pass
    if not access_key or not secret:
        raise RuntimeError("DLC 查询/提交需要有效的 RAM 凭据")
    return access_key, secret, token
from typing import Any
from urllib.parse import quote

from .schema import GPUS_PER_NODE, PROJECT_H100_CAP, ROOT


ACTIVE_JOB_STATES = (
    "Creating",
    "Queuing",
    "Bidding",
    "EnvPreparing",
    "SanityChecking",
    "Running",
    "Restarting",
    "Stopping",
    "SucceededReserving",
)
TERMINAL_JOB_STATES = {"Succeeded", "Failed", "Stopped", "Deleted", "SubmitFailed"}


def _number(value: object) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite():
        return None
    return int(parsed) if parsed == parsed.to_integral_value() else float(parsed)


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _body(value: object) -> dict[str, Any]:
    mapped = _mapping(value)
    body = mapped.get("body")
    return _mapping(body) if isinstance(body, dict) else mapped


def _gpu(amount: object) -> int | float | None:
    return _number(_mapping(amount).get("GPU"))


def paistudio_client(region: str):
    """Create a read-only PAI Resource Service client from env credentials."""
    from alibabacloud_tea_openapi.client import Client
    from alibabacloud_tea_openapi.models import Config
    access_key, secret, token = _credentials()
    return Client(
        Config(
            region_id=region,
            endpoint=f"pai.{region}.aliyuncs.com",
            protocol="https",
            access_key_id=access_key,
            access_key_secret=secret,
            security_token=token,
        )
    )


def _paistudio_get(client: Any, *, action: str, pathname: str, query: dict[str, Any]) -> dict[str, Any]:
    from alibabacloud_tea_openapi.models import OpenApiRequest, Params
    from alibabacloud_tea_openapi.utils import Utils
    from alibabacloud_tea_util.models import RuntimeOptions

    params = Params(
        action=action,
        version="2022-01-12",
        protocol="HTTPS",
        pathname=pathname,
        method="GET",
        auth_type="AK",
        style="ROA",
        req_body_type="json",
        body_type="json",
    )
    request = OpenApiRequest(query=Utils.query(query))
    return _body(client.call_api(params, request, RuntimeOptions()))


def get_quota(client: Any, quota_id: str) -> dict[str, Any]:
    return _paistudio_get(
        client,
        action="GetQuota",
        pathname=f"/api/v1/quotas/{quote(quota_id, safe='')}",
        query={"Verbose": True, "WithNodeMeta": True},
    )


def list_quota_nodes(client: Any, quota_id: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    page_number = 1
    total_count: int | None = None
    while True:
        body = _paistudio_get(
            client,
            action="ListNodes",
            pathname="/api/v1/nodes",
            query={
                "QuotaId": quota_id,
                "AcceleratorType": "GPU",
                "Verbose": True,
                "PageNumber": page_number,
                "PageSize": 100,
            },
        )
        page = body.get("Nodes") or []
        if not isinstance(page, list):
            raise ValueError("ListNodes 返回的 Nodes 不是 list")
        nodes.extend(item for item in page if isinstance(item, dict))
        count = _number(body.get("TotalCount"))
        total_count = int(count) if isinstance(count, int) else total_count
        if not page or len(page) < 100 or (total_count is not None and len(nodes) >= total_count):
            break
        page_number += 1
    return {"Nodes": nodes, "TotalCount": total_count if total_count is not None else len(nodes)}


def list_active_workspace_jobs(
    client: Any, *, workspace_id: str, quota_id: str
) -> list[Any]:
    from alibabacloud_pai_dlc20201203.models import ListJobsRequest

    jobs: list[Any] = []
    seen: set[str] = set()
    for state in ACTIVE_JOB_STATES:
        page_number = 1
        while True:
            response = client.list_jobs(
                ListJobsRequest(
                    workspace_id=workspace_id,
                    resource_id=quota_id,
                    status=state,
                    start_time="2000-01-01T00:00:00Z",
                    page_number=page_number,
                    page_size=100,
                    sort_by="GmtCreateTime",
                    order="desc",
                )
            )
            page = list(getattr(response.body, "jobs", None) or [])
            for job in page:
                job_id = str(getattr(job, "job_id", "") or "")
                key = job_id or f"{state}:{getattr(job, 'display_name', '')}:{page_number}"
                if key not in seen:
                    seen.add(key)
                    jobs.append(job)
            total = getattr(response.body, "total_count", None)
            if not page or len(page) < 100 or (isinstance(total, int) and len(page) * page_number >= total):
                break
            page_number += 1
    return jobs


def requested_job_gpus(job: Any) -> int | float | None:
    direct = _number(getattr(job, "request_gpu", None))
    if direct is not None:
        return direct
    total: int | float = 0
    found = False
    for spec in list(getattr(job, "job_specs", None) or []):
        count = _number(getattr(spec, "pod_count", None))
        resource = getattr(spec, "resource_config", None)
        per_pod = _number(getattr(resource, "gpu", None))
        if count is not None and per_pod is not None:
            total += count * per_pod
            found = True
    return total if found else None


def summarize_jobs(jobs: list[Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    states: Counter[str] = Counter()
    requested_total: int | float = 0
    requested_complete = True
    for job in jobs:
        status = str(getattr(job, "status", "") or "Unknown")
        requested = requested_job_gpus(job)
        states[status] += 1
        if requested is None:
            requested_complete = False
        else:
            requested_total += requested
        entries.append(
            {
                "job_id": str(getattr(job, "job_id", "") or ""),
                "display_name": str(getattr(job, "display_name", "") or ""),
                "status": status,
                "requested_gpu": requested,
                "node_count": _number(getattr(job, "node_count", None)),
                "created_at": str(getattr(job, "gmt_create_time", "") or ""),
            }
        )
    return {
        "active_job_count": len(entries),
        "status_counts": dict(sorted(states.items())),
        "requested_gpu_sum": requested_total if requested_complete else None,
        "requested_gpu_sum_lower_bound": requested_total,
        "requested_gpu_complete": requested_complete,
        "jobs": entries,
    }


def summarize_quota(quota: dict[str, Any], node_page: dict[str, Any]) -> dict[str, Any]:
    details = _mapping(quota.get("QuotaDetails"))
    node_stats = _mapping(details.get("NodeStatistics"))
    allocatable_gpu = _gpu(details.get("AllocatableQuota"))
    guaranteed_gpu = _gpu(details.get("ActualMinQuota"))
    allocated_gpu = _gpu(details.get("AllocatedQuota"))
    submitted_gpu = _gpu(details.get("SelfSubmittedQuota"))
    available_gpu: int | float | None = None
    if allocatable_gpu is not None and allocated_gpu is not None:
        available_gpu = max(allocatable_gpu - allocated_gpu, 0)

    nodes = node_page.get("Nodes") or []
    node_entries: list[dict[str, Any]] = []
    free_eight_gpu_nodes = 0
    total_eight_gpu_nodes = 0
    used_eight_gpu_nodes = 0
    listed_gpu_total: int | float = 0
    listed_gpu_requested: int | float = 0
    listed_gpu_free: int | float = 0
    listed_gpu_complete = True
    gpu_types: Counter[str] = Counter()
    gpu_memories: Counter[str] = Counter()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        total_gpu = _number(node.get("GPU"))
        requested_gpu = _number(node.get("RequestGPU"))
        status = str(node.get("NodeStatus") or "")
        free_gpu = (
            max(total_gpu - requested_gpu, 0)
            if total_gpu is not None and requested_gpu is not None
            else None
        )
        is_eight_gpu = bool(total_gpu is not None and total_gpu >= GPUS_PER_NODE)
        if is_eight_gpu:
            total_eight_gpu_nodes += 1
            if requested_gpu is not None and requested_gpu > 0:
                used_eight_gpu_nodes += 1
            if status == "Ready" and free_gpu is not None and free_gpu >= GPUS_PER_NODE:
                free_eight_gpu_nodes += 1
        if total_gpu is None or requested_gpu is None or free_gpu is None:
            listed_gpu_complete = False
        else:
            listed_gpu_total += total_gpu
            listed_gpu_requested += requested_gpu
            listed_gpu_free += free_gpu
        gpu_type = str(node.get("GPUType") or "Unknown")
        gpu_memory = str(node.get("GPUMemory") or "Unknown")
        gpu_types[gpu_type] += 1
        gpu_memories[gpu_memory] += 1
        node_entries.append(
            {
                "node_name": str(node.get("NodeName") or ""),
                "node_type": str(node.get("NodeType") or ""),
                "status": status,
                "gpu_type": gpu_type,
                "gpu_memory": gpu_memory,
                "gpu_total": total_gpu,
                "gpu_requested": requested_gpu,
                "gpu_free": free_gpu,
                "workload_count": _number(node.get("WorkloadNum")),
            }
        )

    if allocatable_gpu is None and listed_gpu_complete:
        allocatable_gpu = listed_gpu_total
    if allocated_gpu is None and listed_gpu_complete:
        allocated_gpu = listed_gpu_requested
    if listed_gpu_complete:
        available_gpu = listed_gpu_free

    return {
        "quota_id": str(quota.get("QuotaId") or ""),
        "quota_name": str(quota.get("QuotaName") or ""),
        "quota_status": str(quota.get("Status") or ""),
        "resource_type": str(quota.get("ResourceType") or ""),
        "gpu": {
            "guaranteed_total": guaranteed_gpu,
            "allocatable_total": allocatable_gpu,
            "allocated": allocated_gpu,
            "available": available_gpu,
            "submitted_including_queue": submitted_gpu,
            "source": (
                "GetQuota.QuotaDetails；字段缺失时以完整 ListNodes 的 GPU/RequestGPU 求和"
            ),
        },
        "nodes": {
            "guaranteed_total": _number(node_stats.get("ActualMinNodeNum")),
            "allocated": _number(node_stats.get("AllocatedNodeNum")),
            "provider_empty_allocated": _number(node_stats.get("EmptyNodeNum")),
            "listed_gpu_nodes": len(node_entries),
            "eight_gpu_nodes_total": total_eight_gpu_nodes,
            "eight_gpu_nodes_with_any_request": used_eight_gpu_nodes,
            "eight_gpu_nodes_free": free_eight_gpu_nodes,
            "gpu_type_node_counts": dict(sorted(gpu_types.items())),
            "gpu_memory_node_counts": dict(sorted(gpu_memories.items())),
            "source": "GetQuota.NodeStatistics + ListNodes(Verbose=true)",
            "note": "eight_gpu_nodes_free 要求节点 Ready 且 RequestGPU=0；这是容量快照，不是未来调度保证。",
            "items": node_entries,
        },
    }


def summarize_project_inventory(
    inventory: dict[str, Any], workspace_jobs: list[Any] | None = None
) -> dict[str, Any]:
    jobs = inventory.get("jobs") or []
    active = [
        item
        for item in jobs
        if isinstance(item, dict) and str(item.get("status") or "") not in TERMINAL_JOB_STATES
    ]
    tracked_ids = {str(item.get("job_id") or "") for item in active}
    provider_discovered: list[dict[str, Any]] = []
    for job in workspace_jobs or []:
        envs = getattr(job, "envs", None) or {}
        if not isinstance(envs, dict):
            continue
        if envs.get("PROJECT_H100_ACCOUNTING_SCOPE") != "explore_xiangruiliu":
            continue
        job_id = str(getattr(job, "job_id", "") or "")
        if job_id in tracked_ids:
            continue
        requested_gpu = requested_job_gpus(job)
        provider_discovered.append(
            {
                "controller": "provider-discovered",
                "run_id": str(envs.get("DEPLOY_RUN_ID") or ""),
                "job_id": job_id,
                "status": str(getattr(job, "status", "") or "Unknown"),
                "requested_h100": requested_gpu,
                "source": "DLC ListJobs PROJECT_H100_ACCOUNTING_SCOPE marker",
            }
        )
    active.extend(provider_discovered)
    requested_values = [item.get("requested_h100") for item in active]
    requested_complete = all(isinstance(value, (int, float)) for value in requested_values)
    requested = sum(
        int(value) for value in requested_values if isinstance(value, (int, float))
    )
    unresolved = [item for item in active if not item.get("job_id")]
    return {
        "cap_h100": PROJECT_H100_CAP,
        "active_requested_h100": requested if requested_complete else None,
        "active_requested_h100_lower_bound": requested,
        "requested_h100_complete": requested_complete,
        "remaining_by_project_policy_h100": (
            max(PROJECT_H100_CAP - requested, 0) if requested_complete else None
        ),
        "active_job_count": len(active),
        "unresolved_submission_count": len(unresolved),
        "provider_discovered_unregistered_count": len(provider_discovered),
        "active_jobs": active,
    }


def build_capacity_snapshot(
    *,
    config: dict[str, Any],
    inventory: dict[str, Any],
    quota_body: dict[str, Any],
    node_body: dict[str, Any],
    workspace_jobs: list[Any],
) -> dict[str, Any]:
    quota = summarize_quota(quota_body, node_body)
    project = summarize_project_inventory(inventory, workspace_jobs)
    return {
        "schema_version": "project-h100-capacity-v1",
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": {
            "workspace_id": str(config["aliyun"]["workspace_id"]),
            "quota_id": str(config["aliyun"]["resource_id"]),
            "declared_hardware": "H100（用户提供）；provider GPUType 原值另见节点汇总，二者不一致时提交需再次确认。",
        },
        "project_policy": project,
        "provider_quota": quota,
        "workspace_active_jobs": summarize_jobs(workspace_jobs),
        "admission_view": {
            "provider_available_gpu": quota["gpu"]["available"],
            "provider_free_eight_gpu_nodes": quota["nodes"]["eight_gpu_nodes_free"],
            "project_remaining_h100": project["remaining_by_project_policy_h100"],
            "note": "提交还会在短时本地锁内重新获取快照；最终能否调度由 Aliyun 控制面决定。",
        },
    }
