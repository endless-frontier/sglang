"""EAS 公网 CPU 代理的只读盘点、创建、验证和删除控制面。"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
import zlib


from .schema import ROOT, inside_project, safe_run_id
RUNTIME_ROOT = ROOT / ".runtime" / "deployment" / "qwen38_flash_next_h200"
SERVICE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
RUNNING_EAS_STATES = {"Running"}


def _profile() -> dict[str, Any]:
    configured = os.environ.get("QWEN38_EAS_PROFILE", "")
    path = Path(configured) if configured else ROOT / "configs" / "eas_profile.json"
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 EAS profile：{path}；复制 eas_profile.template.json 后填写当前集群 ID"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("EAS profile 必须是 JSON object")
    required = ("region", "workspace_id", "quota_id", "gateway_id", "network", "storage", "container")
    missing = [key for key in required if not value.get(key)]
    if missing:
        raise ValueError("EAS profile 缺少字段：" + ",".join(missing))
    return value


def _client(region: str):
    try:
        from alibabacloud_tea_openapi.models import Config
        from alibabacloud_eas20210701.client import Client
    except ImportError as exc:
        raise RuntimeError(
            "EAS 控制面需要 alibabacloud-eas20210701 与 tea-openapi；仅 render 模式不需要安装"
        ) from exc
    from .capacity import _credentials
    access_key, secret, token = _credentials()
    cfg = Config(access_key_id=access_key, access_key_secret=secret,
                 security_token=token, endpoint=f"eas.{region}.aliyuncs.com")
    return Client(cfg)


def _atomic_json(
    path: Path, value: dict[str, Any], *, overwrite: bool = False, mode: int = 0o644
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def _redact(value: object) -> str:
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


def _provider_error(exc: BaseException) -> dict[str, Any]:
    data = getattr(exc, "data", None)
    return {
        "exception_type": type(exc).__name__,
        "error_full_credential_redacted": _redact(exc),
        "response_body_full_credential_redacted": _redact(data),
        "http_status": getattr(exc, "status_code", None),
        "request_id": str(
            getattr(exc, "request_id", "")
            or (data.get("RequestId", "") if isinstance(data, dict) else "")
            or ""
        ),
    }


def _map(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    converter = getattr(value, "to_map", None)
    mapped = converter() if callable(converter) else {}
    return mapped if isinstance(mapped, dict) else {}


def _safe_tree(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _redact(value) if isinstance(value, str) else value
    if isinstance(value, list):
        return [_safe_tree(item) for item in value]
    raw = value if isinstance(value, dict) else _map(value)
    result = {}
    for key, item in raw.items():
        if re.search(r"(?i)(?:access.?token|secret|authorization)", str(key)):
            result[str(key)] = "[REDACTED]"
        else:
            result[str(key)] = _safe_tree(item)
    return result


def _gateway_rows(body: Any) -> list[dict[str, Any]]:
    rows = []
    for item in list(getattr(body, "gateways", None) or []):
        raw = _map(item)
        rows.append(
            {
                "gateway_id": raw.get("GatewayId"),
                "gateway_name": raw.get("GatewayName"),
                "status": raw.get("Status"),
                "internet_enabled": raw.get("InternetEnabled"),
                "internet_domain": raw.get("InternetDomain"),
                "intranet_domain": raw.get("IntranetDomain"),
                "is_default": raw.get("IsDefault"),
                "replicas": raw.get("Replicas"),
            }
        )
    return rows


def _service_rows(body: Any) -> list[dict[str, Any]]:
    raw = _map(body)
    rows = []
    for item in raw.get("Services", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "service_name": item.get("ServiceName"),
                "service_id": item.get("ServiceId"),
                "status": item.get("Status"),
                "gpu_per_instance": item.get("Gpu"),
                "total_instances": item.get("TotalInstance"),
                "workspace_id": item.get("WorkspaceId"),
                "quota_id": item.get("QuotaId"),
                "gateway": item.get("Gateway"),
                "internet_endpoint": item.get("InternetEndpoint"),
                "create_time": item.get("CreateTime"),
            }
        )
    return rows


def inventory_command(args: argparse.Namespace) -> int:
    from alibabacloud_eas20210701.models import ListGatewayRequest, ListServicesRequest

    profile = _profile()
    client = _client(str(profile["region"]))
    output = inside_project(args.output)
    gateway_error = None
    service_error = None
    gateways: list[dict[str, Any]] = []
    services: list[dict[str, Any]] = []
    try:
        gateways = _gateway_rows(
            client.list_gateway(
                ListGatewayRequest(gateway_id=str(profile["gateway_id"]), page_size=100)
            ).body
        )
    except Exception as exc:
        gateway_error = _provider_error(exc)
    try:
        services = _service_rows(
            client.list_services(
                ListServicesRequest(
                    workspace_id=str(profile["workspace_id"]),
                    quota_id=str(profile["quota_id"]),
                    page_number=1,
                    page_size=100,
                )
            ).body
        )
    except Exception as exc:
        service_error = _provider_error(exc)
    configured = next(
        (row for row in gateways if row["gateway_id"] == profile["gateway_id"]), None
    )
    value = {
        "schema_version": "qwen38-eas-public-inventory-v1",
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "read_only": True,
        "profile": {
            "region": profile["region"],
            "workspace_id": profile["workspace_id"],
            "quota_id": profile["quota_id"],
            "gateway_id": profile["gateway_id"],
        },
        "configured_gateway": configured,
        "configured_gateway_public_ready": bool(
            configured
            and configured.get("status") == "Running"
            and configured.get("internet_enabled") is True
            and configured.get("internet_domain")
        ),
        "gateways": gateways,
        "gateway_error": gateway_error,
        "services": services,
        "service_error": service_error,
    }
    _atomic_json(output, value)
    ok = gateway_error is None and service_error is None
    print(
        json.dumps(
            {
                "query_ok": ok,
                "configured_gateway_public_ready": value[
                    "configured_gateway_public_ready"
                ],
                "active_services": sum(
                    row.get("status") in {"Running", "Creating", "Starting", "Waiting"}
                    for row in services
                ),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 2


def _submit_receipt(path: Path) -> dict[str, Any]:
    resolved = inside_project(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "qwen38-flash-next-h200-submit-v1":
        raise ValueError("source receipt 不是本部署包的 submit receipt")
    if value.get("submitted") is not True or not value.get("job_id"):
        raise ValueError("source receipt 没有成功提交的 job_id")
    return value


def _ready(source: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = RUNTIME_ROOT / str(source["run_id"]) / "ready.json"
    if not path.is_file():
        raise FileNotFoundError(f"DLC worker 尚未写 ready：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "qwen38-flash-next-h200-ready-v1":
        raise ValueError("DLC ready schema 不受支持")
    accounting = value.get("gpu_accounting") or {}
    expected = (source.get("gpu_accounting") or {}).get("requested")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        expected = 32
    if accounting.get("actually_used") != expected:
        raise RuntimeError(
            f"DLC ready 尚无 actually_used={expected} 的完整显存实证"
        )
    return path, value


def _embedded_proxy_command(*, ready_file: str) -> str:
    source = (ROOT / "deployment" / "qwen38_flash_next_h200" / "public_proxy.py").read_bytes()
    payload = base64.b64encode(zlib.compress(source, 9)).decode("ascii")
    bootstrap = (
        "import base64,zlib;exec(compile(zlib.decompress(base64.b64decode("
        + repr(payload)
        + ")), '<qwen38-eas-public-proxy>', 'exec'))"
    )
    return shlex.join(
        [
            "python3",
            "-c",
            bootstrap,
            "--ready-file",
            ready_file,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )


def build_proxy_manifest(
    *, service_name: str, source: dict[str, Any], ready: dict[str, Any]
) -> dict[str, Any]:
    if not SERVICE_NAME_RE.fullmatch(service_name):
        raise ValueError("EAS service name 必须以小写字母开头且只含小写字母、数字、_")
    profile = _profile()
    source_request = source.get("request") or {}
    envs = source_request.get("Envs") or {}
    launch_root = str(envs.get("DEPLOY_LAUNCH_ROOT") or "")
    project_root = launch_root.split("/.runtime/deployment/", 1)[0]
    if not project_root.startswith("/mnt/data/"):
        raise ValueError("submit receipt 的 DEPLOY_LAUNCH_ROOT 中没有可核验的云端 project root")
    ready_file = str(
        PurePosixPath(project_root)
        / ".runtime"
        / "deployment"
        / "qwen38_flash_next_h200"
        / str(source["run_id"])
        / "ready.json"
    )
    storage = []
    for item in profile["storage"]:
        if item["type"] != "cpfs":
            raise ValueError("当前 EAS proxy 只接受已验证 CPFS mount")
        storage.append(
            {
                "cpfs": {
                    "file_system_id": item["file_system_id"],
                    "path": item["path"],
                },
                "enable_cache": bool(item.get("enable_cache", False)),
                "mount_path": item["mount_path"],
            }
        )
    command = _embedded_proxy_command(ready_file=ready_file)
    return {
        "metadata": {
            "name": service_name,
            "workspace_id": str(profile["workspace_id"]),
            "quota_id": str(profile["quota_id"]),
            "instance": 2,
            "cpu": 8,
            "gpu": 0,
            "memory": 32768,
            "disk": "30Gi",
            "rpc": {
                # Milliseconds.  Long-thinking streaming requests may exceed
                # 30 minutes, so keep the public RPC connection for 2 hours.
                "keepalive": 7200000,
                "io_threads": 16,
                "rate_limit": 0,
            },
            "scheduling": {"spread": {"policy": "host"}},
        },
        "cloud": {"networking": dict(profile["network"])},
        "networking": {"gateway": profile["gateway_id"]},
        "storage": storage,
        "containers": [
            {
                "image": profile["container"]["image"],
                "script": command,
                "port": 8000,
            }
        ],
    }


def proxy_create_command(args: argparse.Namespace) -> int:
    if not args.apply:
        raise ValueError("真实创建 EAS proxy 必须显式传 --apply")
    source = _submit_receipt(args.source_receipt)
    _ready_path, ready = _ready(source)
    service_name = str(args.service_name)
    if args.confirm_service_name != service_name:
        raise ValueError(f"必须显式传 --confirm-service-name {service_name}")
    output = inside_project(args.receipt)
    profile = _profile()
    client = _client(str(profile["region"]))

    from alibabacloud_eas20210701.models import (
        CreateServiceRequest,
        ListGatewayRequest,
        ListServicesRequest,
    )

    gateway_rows = _gateway_rows(
        client.list_gateway(
            ListGatewayRequest(gateway_id=str(profile["gateway_id"]), page_size=100)
        ).body
    )
    gateway = next(
        (row for row in gateway_rows if row["gateway_id"] == profile["gateway_id"]), None
    )
    if not (
        gateway
        and gateway.get("status") == "Running"
        and gateway.get("internet_enabled") is True
        and gateway.get("internet_domain")
    ):
        raise RuntimeError("配置的 EAS 专属网关没有同时满足 Running、InternetEnabled 和公网域名")
    existing = _service_rows(
        client.list_services(
            ListServicesRequest(
                workspace_id=str(profile["workspace_id"]),
                service_name=service_name,
                page_number=1,
                page_size=100,
            )
        ).body
    )
    if any(row.get("service_name") == service_name for row in existing):
        raise RuntimeError(f"EAS service 已存在，拒绝覆盖：{service_name}")

    manifest = build_proxy_manifest(
        service_name=service_name, source=source, ready=ready
    )
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        body = client.create_service(
            CreateServiceRequest(body=json.dumps(manifest, ensure_ascii=False))
        ).body
        result = {
            "service_id": getattr(body, "service_id", None),
            "service_name": getattr(body, "service_name", None) or service_name,
            "status": getattr(body, "status", None),
            "internet_endpoint": getattr(body, "internet_endpoint", None),
            "intranet_endpoint": getattr(body, "intranet_endpoint", None),
            "request_id": getattr(body, "request_id", None),
        }
        create_error = None
    except Exception as exc:
        result = {
            "service_id": None,
            "service_name": service_name,
            "status": "",
            "internet_endpoint": None,
            "intranet_endpoint": None,
            "request_id": None,
        }
        create_error = _provider_error(exc)
    receipt = {
        "schema_version": "qwen38-eas-public-proxy-create-v1",
        "created_at": created_at,
        "submitted": create_error is None,
        "dlc": {
            "run_id": source["run_id"],
            "mode": source["mode"],
            "job_id": source["job_id"],
            "source_receipt": inside_project(args.source_receipt).relative_to(ROOT).as_posix(),
            "private_endpoint": ready["endpoint"],
            "served_model_name": ready["served_model_name"],
            "gpu_accounting": ready["gpu_accounting"],
            "metrics_endpoint_counts": {
                role: len(urls)
                for role, urls in (ready.get("metrics_endpoint_groups") or {}).items()
                if isinstance(urls, list)
            },
        },
        "gateway": gateway,
        "manifest": manifest,
        "provider": result,
        "provider_error": create_error,
        "proxy_source_sha256": hashlib.sha256(
            (ROOT / "deployment" / "qwen38_flash_next_h200" / "public_proxy.py").read_bytes()
        ).hexdigest(),
        "gpu_accounting": {
            "requested": 0,
            "reserved": 0,
            "actually_used": 0,
            "evidence": "EAS proxy manifest 的 metadata.gpu=0；模型 GPU 由 DLC receipt 单独核算",
        },
        "lifecycle": {
            "owner": "deployment",
            "stop_condition": "DLC backend 停止时删除；也可由显式 proxy-delete 入口提前删除",
        },
    }
    _atomic_json(output, receipt)
    print(
        json.dumps(
            {
                "submitted": receipt["submitted"],
                "service_name": service_name,
                "status": result["status"],
                "receipt": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if receipt["submitted"] else 2


def proxy_update_command(args: argparse.Namespace) -> int:
    """Replace one bound EAS service config while preserving its identity."""
    source = _proxy_create_receipt(args.source_receipt)
    service_name = str(source["provider"]["service_name"])
    if not args.apply or args.confirm_service_name != service_name:
        raise ValueError(
            f"更新 EAS proxy 必须传 --apply --confirm-service-name {service_name}"
        )
    submit = _submit_receipt(ROOT / str(source["dlc"]["source_receipt"]))
    _ready_path, ready = _ready(submit)
    manifest = build_proxy_manifest(
        service_name=service_name,
        source=submit,
        ready=ready,
    )
    output = inside_project(args.receipt)
    profile = _profile()
    client = _client(str(profile["region"]))
    from alibabacloud_eas20210701.models import UpdateServiceRequest

    try:
        body = client.update_service(
            str(profile["region"]),
            service_name,
            UpdateServiceRequest(
                update_type="replace",
                body=json.dumps(manifest, ensure_ascii=False),
            ),
        ).body
        provider = _map(body)
        update_error = None
    except Exception as exc:
        provider = {}
        update_error = _provider_error(exc)
    value = {
        "schema_version": "qwen38-eas-public-proxy-update-v1",
        "requested_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "service_name": service_name,
        "source_create_receipt": inside_project(args.source_receipt)
        .relative_to(ROOT)
        .as_posix(),
        "update_type": "replace",
        "update_requested": update_error is None,
        "change": {
            "model_request_body_limit_bytes_after": None,
            "model_request_body_limit_mode_after": "transparent_upstream_enforced",
            "model_request_body_limit_added_by_eas_proxy": False,
            "ordinary_request_body_forwarding_after": "streamed",
            "control_request_body_limit_bytes": 64 * 1024,
            "dlc_changed": False,
            "eas_instance_shape_changed": False,
        },
        "manifest": manifest,
        "provider": provider,
        "provider_error": update_error,
        "previous_proxy_source_sha256": source.get("proxy_source_sha256"),
        "updated_proxy_source_sha256": hashlib.sha256(
            (ROOT / "deployment" / "qwen38_flash_next_h200" / "public_proxy.py").read_bytes()
        ).hexdigest(),
    }
    _atomic_json(output, value)
    print(
        json.dumps(
            {
                "service_name": service_name,
                "update_requested": value["update_requested"],
                "receipt": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if update_error is None else 2


def _proxy_create_receipt(path: Path) -> dict[str, Any]:
    resolved = inside_project(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "qwen38-eas-public-proxy-create-v1":
        raise ValueError("source receipt 不是 EAS proxy create receipt")
    if value.get("submitted") is not True:
        raise ValueError("EAS proxy create receipt 没有成功提交")
    return value


def _safe_service(body: Any) -> dict[str, Any]:
    raw = _map(body)
    return {
        "service_id": raw.get("ServiceId"),
        "service_name": raw.get("ServiceName"),
        "status": raw.get("Status"),
        "message": _redact(raw.get("Message")),
        "reason": _redact(raw.get("Reason")),
        "cpu": raw.get("Cpu"),
        "gpu": raw.get("Gpu"),
        "memory": raw.get("Memory"),
        "total_instance": raw.get("TotalInstance"),
        "running_instance": raw.get("RunningInstance"),
        "pending_instance": raw.get("PendingInstance"),
        "quota_id": raw.get("QuotaId"),
        "workspace_id": raw.get("WorkspaceId"),
        "gateway": raw.get("Gateway"),
        "internet_endpoint": raw.get("InternetEndpoint"),
        "intranet_endpoint": raw.get("IntranetEndpoint"),
        "create_time": raw.get("CreateTime"),
        "update_time": raw.get("UpdateTime"),
    }


def _public_request(
    url: str, token: str, *, method: str = "GET", payload: dict[str, Any] | None = None
) -> tuple[int, bytes, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
    last_error: Exception | None = None
    for style, authorization in (("bearer", f"Bearer {token}"), ("raw", token)):
        headers = {"Authorization": authorization}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=1800) as response:
                return response.status, response.read(2 * 1024 * 1024), style
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {401, 403}:
                raise
        except (URLError, TimeoutError) as exc:
            last_error = exc
            raise
    assert last_error is not None
    raise last_error


def proxy_status_command(args: argparse.Namespace) -> int:
    source = _proxy_create_receipt(args.source_receipt)
    output = inside_project(args.output)
    profile = _profile()
    client = _client(str(profile["region"]))
    service_name = str(source["provider"]["service_name"])
    service_error = None
    instance_error = None
    event_error = None
    instances: Any = {}
    events: Any = {}
    endpoints_error = None
    token = ""
    endpoint_rows: list[dict[str, Any]] = []
    try:
        service = _safe_service(
            client.describe_service(str(profile["region"]), service_name).body
        )
    except Exception as exc:
        service = {"service_name": service_name, "status": ""}
        service_error = _provider_error(exc)
    from alibabacloud_eas20210701.models import (
        DescribeServiceEventRequest,
        ListServiceInstancesRequest,
    )

    try:
        instances = _safe_tree(
            client.list_service_instances(
                str(profile["region"]),
                service_name,
                ListServiceInstancesRequest(page_number=1, page_size=20),
            ).body
        )
    except Exception as exc:
        instance_error = _provider_error(exc)
    try:
        events = _safe_tree(
            client.describe_service_event(
                str(profile["region"]),
                service_name,
                DescribeServiceEventRequest(page_num="1", page_size="50"),
            ).body
        )
    except Exception as exc:
        event_error = _provider_error(exc)
    try:
        endpoints_body = client.describe_service_endpoints(
            str(profile["region"]), service_name
        ).body
        token = str(getattr(endpoints_body, "access_token", "") or "")
        for item in list(getattr(endpoints_body, "endpoints", None) or []):
            raw = _map(item)
            endpoint_rows.append(
                {
                    "endpoint_type": raw.get("EndpointType"),
                    "internet_endpoints": raw.get("InternetEndpoints") or [],
                    "intranet_endpoints": raw.get("IntranetEndpoints") or [],
                    "path_type": raw.get("PathType"),
                    "port": raw.get("Port"),
                }
            )
    except Exception as exc:
        endpoints_error = _provider_error(exc)
    public_endpoints = [
        str(url).rstrip("/")
        for row in endpoint_rows
        for url in row["internet_endpoints"]
        if str(url).startswith(("http://", "https://"))
    ]
    if not public_endpoints and service.get("internet_endpoint"):
        public_endpoints = [str(service["internet_endpoint"]).rstrip("/")]

    secret_path = RUNTIME_ROOT / str(source["dlc"]["run_id"]) / "eas_proxy" / "access.json"
    if token:
        _atomic_json(
            secret_path,
            {
                "schema_version": "qwen38-eas-public-access-secret-v1",
                "service_name": service_name,
                "public_base_urls": public_endpoints,
                "access_token": token,
            },
            overwrite=True,
            mode=0o600,
        )

    probe: dict[str, Any] = {
        "attempted": False,
        "health_ok": False,
        "models_ok": False,
        "chat_ok": False,
    }
    if (
        args.probe
        and service.get("status") in RUNNING_EAS_STATES
        and public_endpoints
        and token
    ):
        probe["attempted"] = True
        base = public_endpoints[0]
        try:
            health_status, _health_body, auth_style = _public_request(
                base + "/health", token
            )
            probe["health_status"] = health_status
            probe["health_ok"] = health_status == 200
            models_status, models_body, _ = _public_request(base + "/v1/models", token)
            models = json.loads(models_body)
            model_ids = [
                str(item.get("id"))
                for item in models.get("data", [])
                if isinstance(item, dict) and item.get("id")
            ]
            served = str(source["dlc"]["served_model_name"])
            probe["models_status"] = models_status
            probe["served_model_present"] = served in model_ids
            probe["models_ok"] = models_status == 200 and served in model_ids
            chat_status, chat_body, _ = _public_request(
                base + "/v1/chat/completions",
                token,
                method="POST",
                payload={
                    "model": served,
                    "messages": [{"role": "user", "content": "Reply with pong."}],
                    "max_tokens": 8,
                    "temperature": 0,
                },
            )
            chat = json.loads(chat_body)
            probe["chat_status"] = chat_status
            probe["chat_has_choices"] = bool(chat.get("choices"))
            probe["chat_ok"] = chat_status == 200 and bool(chat.get("choices"))
            probe["auth_style"] = auth_style
        except Exception as exc:
            probe["error"] = _provider_error(exc)
    control_plane_ready = bool(
        service_error is None
        and endpoints_error is None
        and service.get("status") == "Running"
        and public_endpoints
        and token
    )
    public_ready = bool(
        control_plane_ready
        and probe["attempted"]
        and all(probe[key] for key in ("health_ok", "models_ok", "chat_ok"))
    )
    value = {
        "schema_version": "qwen38-eas-public-proxy-status-v1",
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dlc": source["dlc"],
        "service": service,
        "service_error": service_error,
        "instances": instances,
        "instance_error": instance_error,
        "events": events,
        "event_error": event_error,
        "endpoints": endpoint_rows,
        "endpoints_error": endpoints_error,
        "public_base_urls": public_endpoints,
        "access_token_present": bool(token),
        "access_secret_path": (
            secret_path.relative_to(ROOT).as_posix() if token else None
        ),
        "access_secret_mode": "0600" if token else None,
        "probe": probe,
        "control_plane_ready": control_plane_ready,
        "public_ready": public_ready,
        "gpu_accounting": {
            "requested": 0,
            "reserved": 0,
            "actually_used": 0,
            "evidence": f"EAS describe 返回 gpu={service.get('gpu')!r}；模型 GPU 单独见 dlc.gpu_accounting",
        },
    }
    _atomic_json(output, value)
    print(
        json.dumps(
            {
                "service_name": service_name,
                "status": service.get("status"),
                "public_ready": public_ready,
                "public_base_urls": public_endpoints,
                "access_token_present": bool(token),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if service_error is None and endpoints_error is None else 2


def _metric_lines(body: bytes) -> list[str]:
    prefixes = (
        "smg_",
        "sglang_",
        "sglang:",
        "mooncake_",
        "mc_",
        "qwen38_proxy_",
    )
    lines = []
    for raw in body.decode("utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        candidate = stripped.lstrip("# ")
        if candidate.startswith(("HELP ", "TYPE ")):
            candidate = candidate.split(" ", 1)[1]
        if candidate.startswith(prefixes):
            lines.append(stripped[:16_384])
    return lines[:20_000]


def proxy_metrics_command(args: argparse.Namespace) -> int:
    """经 EAS 鉴权公网入口抓取一轮 Router/SGLang Prometheus 指标。"""
    source = _proxy_create_receipt(args.source_receipt)
    run_id = str(source["dlc"]["run_id"])
    secret_path = RUNTIME_ROOT / run_id / "eas_proxy" / "access.json"
    secret = json.loads(secret_path.read_text(encoding="utf-8"))
    token = str(secret.get("access_token") or "")
    bases = list(secret.get("public_base_urls") or [])
    if not token or not bases:
        raise ValueError("缺少 EAS access.json；请先运行 proxy-status")
    mode = str(source["dlc"]["mode"])
    roles: list[tuple[str, str]] = (
        [("router", "/ops/metrics/router")] if mode == "pd" else []
    )
    if mode == "pd":
        counts = source["dlc"].get("metrics_endpoint_counts") or {}
        for role in ("prefill", "decode"):
            count = counts.get(role, 1)
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                count = 1
            for index in range(count):
                label = role if index == 0 else f"{role}_{index}"
                path = (
                    f"/ops/metrics/{role}"
                    if index == 0
                    else f"/ops/metrics/{role}/{index}"
                )
                roles.append((label, path))
    else:
        roles = [("standard", "/ops/metrics/standard")]
    if source.get("proxy_source_sha256"):
        roles.insert(0, ("proxy", "/ops/metrics/proxy"))
    snapshots: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    for role, path in roles:
        try:
            status, body, auth_style = _public_request(
                bases[0].rstrip("/") + path, token
            )
            lines = _metric_lines(body)
            snapshots[role] = {
                "http_status": status,
                "auth_style": auth_style,
                "selected_line_count": len(lines),
                "selected_prometheus_lines": lines,
                "body_sha256": hashlib.sha256(body).hexdigest(),
            }
        except Exception as exc:
            errors.append({"role": role, **_provider_error(exc)})
    output = inside_project(args.output)
    value = {
        "schema_version": "qwen38-eas-public-metrics-snapshot-v1",
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run_id,
        "mode": mode,
        "service_name": source["provider"]["service_name"],
        "public_base_url": bases[0],
        "snapshots": snapshots,
        "errors": errors,
        "access_token_in_receipt": False,
    }
    _atomic_json(output, value)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "roles": list(snapshots),
                "selected_line_counts": {
                    role: row["selected_line_count"]
                    for role, row in snapshots.items()
                },
                "error_count": len(errors),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not errors else 2


def proxy_delete_command(args: argparse.Namespace) -> int:
    source = _proxy_create_receipt(args.source_receipt)
    service_name = str(source["provider"]["service_name"])
    if not args.apply or args.confirm_service_name != service_name:
        raise ValueError(
            f"删除 EAS proxy 必须传 --apply --confirm-service-name {service_name}"
        )
    output = inside_project(args.output)
    profile = _profile()
    client = _client(str(profile["region"]))
    try:
        body = client.delete_service(str(profile["region"]), service_name).body
        delete_error = None
        provider = _map(body)
    except Exception as exc:
        delete_error = _provider_error(exc)
        provider = {}
    value = {
        "schema_version": "qwen38-eas-public-proxy-delete-v1",
        "requested_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "service_name": service_name,
        "delete_requested": delete_error is None,
        "provider": provider,
        "provider_error": delete_error,
    }
    _atomic_json(output, value)
    print(
        json.dumps(
            {
                "service_name": service_name,
                "delete_requested": value["delete_requested"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if delete_error is None else 2


def add_parsers(subparsers: Any) -> None:
    inventory = subparsers.add_parser(
        "eas-inventory", help="只读查询 EAS 专属网关公网状态与当前服务"
    )
    inventory.add_argument("--output", type=Path, required=True)
    inventory.set_defaults(func=inventory_command)

    create = subparsers.add_parser(
        "proxy-create", help="为 ready 的 DLC backend 创建零 GPU EAS 公网代理"
    )
    create.add_argument("--source-receipt", type=Path, required=True)
    create.add_argument("--service-name", required=True)
    create.add_argument("--confirm-service-name", required=True)
    create.add_argument("--receipt", type=Path, required=True)
    create.add_argument("--apply", action="store_true")
    create.set_defaults(func=proxy_create_command)

    update = subparsers.add_parser(
        "proxy-update", help="原地更新一个 receipt 绑定的 EAS proxy"
    )
    update.add_argument("--source-receipt", type=Path, required=True)
    update.add_argument("--confirm-service-name", required=True)
    update.add_argument("--receipt", type=Path, required=True)
    update.add_argument("--apply", action="store_true")
    update.set_defaults(func=proxy_update_command)

    status = subparsers.add_parser(
        "proxy-status", help="查询 EAS proxy、保管 token，并可做公网 OpenAI smoke"
    )
    status.add_argument("--source-receipt", type=Path, required=True)
    status.add_argument("--output", type=Path, required=True)
    status.add_argument("--probe", action="store_true")
    status.set_defaults(func=proxy_status_command)

    metrics = subparsers.add_parser(
        "proxy-metrics", help="经 EAS 公网入口抓取一轮 Router/SGLang metrics"
    )
    metrics.add_argument("--source-receipt", type=Path, required=True)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.set_defaults(func=proxy_metrics_command)

    delete = subparsers.add_parser("proxy-delete", help="显式删除一个本项目 EAS proxy")
    delete.add_argument("--source-receipt", type=Path, required=True)
    delete.add_argument("--confirm-service-name", required=True)
    delete.add_argument("--output", type=Path, required=True)
    delete.add_argument("--apply", action="store_true")
    delete.set_defaults(func=proxy_delete_command)
