#!/usr/bin/env python3
"""零 GPU 验证 DLC 挂载、Qwen3.5 checkpoint 与候选 SGLang 镜像。"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from uuid import uuid4
import zlib


from .schema import ROOT, inside_project, load_config, model_path, safe_run_id


TERMINAL_STATES = {"Succeeded", "Failed", "Stopped", "Deleted"}
MARKER = "QWEN35_397B_PREFLIGHT_B64="


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有 receipt：{path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
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


def _client():
    ube_root = ROOT / "unified_benchmark_eval"
    if str(ube_root) not in sys.path:
        sys.path.insert(0, str(ube_root))
    from scripts.dlc_eval import make_client

    return make_client("h100")


def _probe_source() -> str:
    # This code is compressed into the provider request so the probe can also
    # prove whether the project tree itself is visible on the mounted DataSource.
    return r'''
import importlib.metadata
import importlib.util
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import zlib

model = Path(os.environ["PROBE_MODEL_PATH"])
project = Path(os.environ["PROBE_PROJECT_ROOT"])

def module_exists(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False

def command_help(argv):
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        return {"returncode": result.returncode, "text": text[-120000:]}
    except Exception as exc:
        return {"returncode": None, "text": "", "error": f"{type(exc).__name__}: {exc}"}

config_path = model / "config.json"
config = {}
config_error = None
try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
except Exception as exc:
    config_error = f"{type(exc).__name__}: {exc}"

index_path = model / "model.safetensors.index.json"
index_error = None
indexed_names = []
try:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed_names = sorted(set((index.get("weight_map") or {}).values()))
except Exception as exc:
    index_error = f"{type(exc).__name__}: {exc}"

shards = sorted(model.glob("*.safetensors"))
missing_indexed_shards = [name for name in indexed_names if not (model / name).is_file()]
weights_total_bytes = sum(path.stat().st_size for path in shards)

sglang_version = None
sglang_root = None
sglang_error = None
source_markers = []
try:
    import sglang
    sglang_root = Path(sglang.__file__).resolve().parent
    try:
        sglang_version = importlib.metadata.version("sglang")
    except Exception:
        sglang_version = getattr(sglang, "__version__", None)
    model_root = sglang_root / "srt" / "models"
    for path in model_root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(marker in text for marker in (
            "Qwen3_5Moe", "Qwen3_5", "qwen3_5_moe", "qwen3_5",
        )):
            source_markers.append(str(path.relative_to(sglang_root)))
except Exception as exc:
    sglang_error = f"{type(exc).__name__}: {exc}"

transformers_version = None
auto_config_ok = False
auto_config_error = None
try:
    transformers_version = importlib.metadata.version("transformers")
    from transformers import AutoConfig
    AutoConfig.from_pretrained(str(model), trust_remote_code=True, local_files_only=True)
    auto_config_ok = True
except Exception as exc:
    auto_config_error = f"{type(exc).__name__}: {exc}"

server_help = command_help([sys.executable, "-m", "sglang.launch_server", "--help"])
router_help = command_help([
    sys.executable, "-m", "sglang_router.launch_router", "--help"
])
required_standard_flags = [
    "--tp-size", "--pp-size", "--dist-init-addr", "--nnodes", "--node-rank",
    "--max-mamba-cache-size",
    "--enable-dynamic-chunking", "--enable-mixed-chunk", "--schedule-policy",
]
required_pd_flags = [
    "--disaggregation-mode", "--disaggregation-transfer-backend",
    "--disaggregation-bootstrap-port", "--disaggregation-ib-device",
    "--ep-size", "--dp-size", "--enable-dp-attention", "--moe-a2a-backend",
    "--speculative-algorithm", "--speculative-num-steps",
    "--speculative-eagle-topk", "--speculative-num-draft-tokens",
]
required_router_flags = [
    "--pd-disaggregation", "--prefill", "--decode", "--host", "--port",
    "--queue-size", "--queue-timeout-secs", "--disable-retries",
    "--cb-failure-threshold", "--cb-success-threshold",
    "--cb-timeout-duration-secs", "--cb-window-duration-secs",
    "--health-failure-threshold", "--health-success-threshold",
    "--health-check-timeout-secs", "--health-check-interval-secs",
    "--prometheus-host", "--prometheus-port",
]
standard_flags = {flag: flag in server_help["text"] for flag in required_standard_flags}
pd_flags = {flag: flag in server_help["text"] for flag in required_pd_flags}
router_flags = {flag: flag in router_help["text"] for flag in required_router_flags}

mooncake_modules = {
    name: module_exists(name)
    for name in ("mooncake", "mooncake_transfer_engine")
}
architectures = config.get("architectures") if isinstance(config, dict) else None
text_config = config.get("text_config") if isinstance(config, dict) else None
model_config = text_config if isinstance(text_config, dict) else config
architecture_text = " ".join(str(item) for item in (architectures or []))
model_identified = bool(
    "Qwen3_5" in architecture_text
    or "qwen3_5" in str(config.get("model_type", "")).lower()
    or "qwen3.5" in str(model).lower()
)
raw_bf16_scale = weights_total_bytes >= 600 * 1024**3
checkpoint_ok = bool(
    model.is_dir()
    and config_path.is_file()
    and index_path.is_file()
    and indexed_names
    and not missing_indexed_shards
    and len(shards) >= 2
    and raw_bf16_scale
    and model_identified
)
standard_compatible = bool(
    checkpoint_ok
    and sglang_version
    and source_markers
    and auto_config_ok
    and server_help["returncode"] == 0
    and all(standard_flags.values())
)
pd_compatible = bool(
    standard_compatible
    and all(pd_flags.values())
    and router_help["returncode"] == 0
    and all(router_flags.values())
    and any(mooncake_modules.values())
)

receipt = {
    "schema_version": "qwen35-397b-zero-gpu-worker-preflight-v1",
    "run_id": os.environ["PROBE_RUN_ID"],
    "gpu_requested": 0,
    "weights_loaded": False,
    "inference_started": False,
    "model_path": str(model),
    "model_path_exists": model.is_dir(),
    "config_exists": config_path.is_file(),
    "config_error": config_error,
    "model_metadata": {
        "architectures": architectures,
        "model_type": config.get("model_type") if isinstance(config, dict) else None,
        "torch_dtype": config.get("torch_dtype") if isinstance(config, dict) else None,
        "num_hidden_layers": config.get("num_hidden_layers") if isinstance(config, dict) else None,
        "num_experts": model_config.get("num_experts") if isinstance(model_config, dict) else None,
        "num_experts_per_tok": model_config.get("num_experts_per_tok") if isinstance(model_config, dict) else None,
        "mtp_num_hidden_layers": model_config.get("mtp_num_hidden_layers") if isinstance(model_config, dict) else None,
        "full_attention_interval": model_config.get("full_attention_interval") if isinstance(model_config, dict) else None,
        "linear_num_key_heads": model_config.get("linear_num_key_heads") if isinstance(model_config, dict) else None,
    },
    "checkpoint": {
        "index_exists": index_path.is_file(),
        "index_error": index_error,
        "indexed_shard_count": len(indexed_names),
        "discovered_safetensor_count": len(shards),
        "missing_indexed_shard_count": len(missing_indexed_shards),
        "weights_total_bytes": weights_total_bytes,
        "raw_bf16_scale_at_least_600_gib": raw_bf16_scale,
        "identified_as_qwen35": model_identified,
        "checkpoint_ok": checkpoint_ok,
    },
    "project_root": str(project),
    "project_root_exists": project.is_dir(),
    "formal_launch_bundle": "request-embedded-per-rank",
    "runtime": {
        "python": sys.version.split()[0],
        "sglang_version": sglang_version,
        "sglang_root": str(sglang_root) if sglang_root else None,
        "sglang_error": sglang_error,
        "sglang_qwen35_source_markers": source_markers[:20],
        "transformers_version": transformers_version,
        "auto_config_ok": auto_config_ok,
        "auto_config_error": auto_config_error,
        "server_help_returncode": server_help["returncode"],
        "sglang_router_module_exists": module_exists("sglang_router.launch_router"),
        "router_help_returncode": router_help["returncode"],
        "required_standard_flags": standard_flags,
        "required_pd_flags": pd_flags,
        "required_router_flags": router_flags,
        "mooncake_modules": mooncake_modules,
    },
    "network": {
        "hostname": socket.gethostname(),
        "hostname_ips": subprocess.run(
            ["hostname", "-I"], check=False, capture_output=True, text=True
        ).stdout.split(),
        "infiniband_devices": sorted(
            path.name for path in Path("/sys/class/infiniband").glob("*") if path.is_dir()
        ),
    },
    "standard_compatible": standard_compatible,
    "pd_compatible": pd_compatible,
}
receipt["passed"] = standard_compatible and pd_compatible
encoded = base64.b64encode(
    zlib.compress(json.dumps(receipt, ensure_ascii=False, sort_keys=True).encode("utf-8"), 9)
).decode("ascii")
print("QWEN35_397B_PREFLIGHT_B64=" + encoded, flush=True)
sys.exit(0 if receipt["passed"] else 3)
'''


def _encoded_command() -> str:
    payload = base64.b64encode(zlib.compress(_probe_source().encode("utf-8"), 9)).decode(
        "ascii"
    )
    return (
        "python3 -c \"import base64,zlib;"
        f"exec(compile(zlib.decompress(base64.b64decode('{payload}')),"
        "'<qwen35-397b-preflight>','exec'))\""
    )


def _request(config: dict[str, Any], *, run_id: str, chosen_model: str) -> dict[str, Any]:
    aliyun = config["aliyun"]
    runtime = config["runtime"]
    request: dict[str, Any] = {
        "Accessibility": "PRIVATE",
        "WorkspaceId": aliyun["workspace_id"],
        "DisplayName": f"qwen35-397b-zero-gpu-preflight-{run_id}"[:63],
        "Description": (
            "project=explore_xiangruiliu;controller=qwen35_397b_preflight;"
            "requested_h100=0"
        ),
        "JobType": "PyTorchJob",
        "Priority": int(aliyun["priority"]),
        "ResourceId": aliyun["resource_id"],
        "JobSpecs": [
            {
                "Type": "Worker",
                "Image": aliyun["worker_image"],
                "PodCount": 1,
                "ResourceConfig": {
                    "CPU": "8",
                    "Memory": "32Gi",
                    "GPU": "0",
                    "SharedMemory": "8Gi",
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
        "Settings": {"EnableRDMA": False},
        "Envs": {
            "PYTHONUNBUFFERED": "1",
            "PROBE_RUN_ID": run_id,
            "PROBE_MODEL_PATH": chosen_model,
            "PROBE_PROJECT_ROOT": runtime["project_root"],
            "PROJECT_H100_ACCOUNTING_SCOPE": "explore_xiangruiliu",
            "PROJECT_REQUESTED_H100": "0",
            "HOME": runtime["project_root"] + "/.runtime/home",
            "TMPDIR": runtime["project_root"] + "/.runtime/tmp",
            "HF_HOME": runtime["project_root"] + "/.runtime/cache/huggingface",
            "XDG_CACHE_HOME": runtime["project_root"] + "/.runtime/cache",
        },
        "UserCommand": _encoded_command(),
        "JobMaxRunningTimeMinutes": 30,
    }
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


def _parse_worker(lines: list[str]) -> dict[str, Any] | None:
    for line in reversed(lines):
        if MARKER not in str(line):
            continue
        try:
            encoded = str(line).split(MARKER, 1)[1].strip()
            value = json.loads(zlib.decompress(base64.b64decode(encoded)))
        except (ValueError, json.JSONDecodeError, zlib.error):
            continue
        if isinstance(value, dict):
            return value
    return None


def _fetch_logs(client: Any, job_id: str, body: Any) -> tuple[list[str], list[dict[str, Any]]]:
    from alibabacloud_pai_dlc20201203.models import GetJobRequest, GetPodLogsRequest

    errors: list[dict[str, Any]] = []
    pods = list(getattr(body, "pods", None) or []) if body is not None else []
    for _ in range(24):
        if pods:
            break
        time.sleep(5)
        try:
            body = client.get_job(job_id, GetJobRequest()).body
            pods = list(getattr(body, "pods", None) or [])
        except Exception as exc:
            errors.append({"operation": "refresh_pods", **_provider_error(exc)})
            break
    lines: list[str] = []
    for attempt in range(24 if pods else 1):
        lines = []
        for pod in pods:
            pod_id = str(getattr(pod, "pod_id", "") or "")
            if not pod_id:
                continue
            try:
                response = client.get_pod_logs(
                    job_id, pod_id, GetPodLogsRequest(max_lines=5000)
                )
                lines.extend(str(line) for line in (response.body.logs or []))
            except Exception as exc:
                errors.append({"operation": "get_pod_logs", **_provider_error(exc)})
        if _parse_worker(lines) is not None or attempt == 23:
            break
        time.sleep(5)
    return lines, errors


def run_preflight(args: argparse.Namespace) -> int:
    if not args.apply:
        raise ValueError("零 GPU 云端预检必须显式传 --apply")
    loaded = load_config(args.config)
    run_id = safe_run_id(args.run_id)
    chosen_model = model_path(loaded.value, args.model_path)
    if not chosen_model.startswith("/"):
        raise ValueError("--model-path 必须是 DLC 容器内绝对路径")
    receipt_path = inside_project(args.receipt)
    if receipt_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 receipt：{receipt_path}")
    request = _request(loaded.value, run_id=run_id, chosen_model=chosen_model)
    request_audit = dict(request)
    request_audit["UserCommand"] = {
        "sha256": _sha256(request["UserCommand"].encode("utf-8")),
        "persisted": False,
    }

    from alibabacloud_pai_dlc20201203.models import CreateJobRequest, GetJobRequest

    client = _client()
    job_id = ""
    body: Any = None
    timed_out = False
    stop_requested = False
    provider_attempts: list[dict[str, Any]] = []
    try:
        response = client.create_job(CreateJobRequest().from_map(request))
        job_id = str(response.body.job_id)
        provider_attempts.append(
            {"operation": "create_job", "status": "succeeded", "job_id": job_id}
        )
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            body = client.get_job(job_id, GetJobRequest()).body
            status = str(getattr(body, "status", "") or "")
            print(json.dumps({"job_id": job_id, "status": status}), flush=True)
            if status in TERMINAL_STATES:
                break
            time.sleep(15)
        else:
            timed_out = True
            client.stop_job(job_id)
            stop_requested = True
            stop_deadline = time.monotonic() + args.stop_wait_timeout
            while time.monotonic() < stop_deadline:
                body = client.get_job(job_id, GetJobRequest()).body
                status = str(getattr(body, "status", "") or "")
                print(
                    json.dumps({"job_id": job_id, "status": status, "cleanup": True}),
                    flush=True,
                )
                if status in TERMINAL_STATES:
                    break
                time.sleep(5)
    except Exception as exc:
        provider_attempts.append(
            {"operation": "provider_control_plane", "status": "failed", **_provider_error(exc)}
        )
        if job_id and str(getattr(body, "status", "") or "") not in TERMINAL_STATES:
            try:
                client.stop_job(job_id)
                stop_requested = True
            except Exception as stop_exc:
                provider_attempts.append(
                    {"operation": "stop_after_error", "status": "failed", **_provider_error(stop_exc)}
                )

    lines, log_errors = _fetch_logs(client, job_id, body) if job_id else ([], [])
    provider_attempts.extend(log_errors)
    worker = _parse_worker(lines)
    terminal_status = str(getattr(body, "status", "") or "") if body else ""
    receipt = {
        "schema_version": "qwen35-397b-zero-gpu-preflight-v1",
        "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run_id,
        "config": {
            "relative_path": loaded.path.relative_to(ROOT).as_posix(),
            "sha256": loaded.sha256,
        },
        "request": request_audit,
        "request_sha256": _sha256(_canonical_json(request_audit).encode("utf-8")),
        "job_id": job_id,
        "terminal_status": terminal_status,
        "timed_out": timed_out,
        "stop_requested": stop_requested,
        "cleanup_terminal": terminal_status in TERMINAL_STATES,
        "gpu_accounting": {
            "requested": 0,
            "reserved": 0,
            "actually_used": 0,
            "evidence": "CreateJob request 明确 GPU=0；worker 不加载权重、不启动推理",
        },
        "worker_receipt": worker,
        "provider_attempts": provider_attempts,
        "error_lines_credential_redacted": [
            _redact(line)
            for line in lines
            if re.search(r"Traceback|Error|Exception|No such file", str(line))
        ][-30:],
    }
    receipt["passed"] = bool(
        terminal_status == "Succeeded"
        and not timed_out
        and isinstance(worker, dict)
        and worker.get("passed") is True
        and worker.get("standard_compatible") is True
        and worker.get("pd_compatible") is True
        and receipt["cleanup_terminal"]
    )
    receipt["content_sha256"] = _sha256(_canonical_json(receipt).encode("utf-8"))
    _atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "job_id": job_id,
                "terminal_status": terminal_status,
                "passed": receipt["passed"],
                "gpu_requested": 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if receipt["passed"] else 2


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "preflight", help="提交一个限时零 GPU Job 验证 checkpoint 与候选镜像"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--stop-wait-timeout", type=int, default=180)
    parser.add_argument("--apply", action="store_true")
    parser.set_defaults(func=run_preflight)

    diagnose = subparsers.add_parser(
        "preflight-diagnose", help="只读获取本部署零 GPU preflight job 的状态与脱敏日志"
    )
    diagnose.add_argument("--job-id", required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.set_defaults(func=diagnose_preflight)

    stop = subparsers.add_parser(
        "preflight-stop", help="停止一个由本入口创建的错误零 GPU preflight"
    )
    stop.add_argument("--job-id", required=True)
    stop.add_argument("--confirm-job-id", required=True)
    stop.add_argument("--output", type=Path, required=True)
    stop.add_argument("--timeout", type=int, default=180)
    stop.add_argument("--apply", action="store_true")
    stop.set_defaults(func=stop_preflight)


def stop_preflight(args: argparse.Namespace) -> int:
    from alibabacloud_pai_dlc20201203.models import GetJobRequest

    if not args.apply:
        raise ValueError("停止 preflight 必须显式传 --apply")
    if args.job_id != args.confirm_job_id:
        raise ValueError("--confirm-job-id 必须与 --job-id 完全一致")
    output = inside_project(args.output)
    client = _client()
    body = client.get_job(args.job_id, GetJobRequest()).body
    display_name = str(getattr(body, "display_name", "") or "")
    description = str(getattr(body, "description", "") or "")
    if not display_name.startswith("qwen35-397b-zero-gpu-preflight-") or (
        "controller=qwen35_397b_preflight" not in description
    ):
        raise ValueError("job 不是本部署入口创建的零 GPU preflight，拒绝停止")
    status = str(getattr(body, "status", "") or "")
    stop_requested = False
    if status not in TERMINAL_STATES:
        client.stop_job(args.job_id)
        stop_requested = True
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            body = client.get_job(args.job_id, GetJobRequest()).body
            status = str(getattr(body, "status", "") or "")
            if status in TERMINAL_STATES:
                break
            time.sleep(5)
    receipt = {
        "schema_version": "qwen35-397b-zero-gpu-preflight-stop-v1",
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "job_id": args.job_id,
        "display_name": display_name,
        "status": status,
        "stop_requested": stop_requested,
        "terminal": status in TERMINAL_STATES,
        "gpu_accounting": {"requested": 0, "reserved": 0, "actually_used": 0},
    }
    _atomic_json(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["terminal"] else 2


def diagnose_preflight(args: argparse.Namespace) -> int:
    from alibabacloud_pai_dlc20201203.models import GetJobRequest, GetPodLogsRequest

    output = inside_project(args.output)
    client = _client()
    body = client.get_job(args.job_id, GetJobRequest()).body
    display_name = str(getattr(body, "display_name", "") or "")
    description = str(getattr(body, "description", "") or "")
    if not display_name.startswith("qwen35-397b-zero-gpu-preflight-") or (
        "controller=qwen35_397b_preflight" not in description
    ):
        raise ValueError("job 不是本部署入口创建的零 GPU preflight，拒绝读取")
    pod_rows = []
    log_lines: list[str] = []
    log_errors: list[dict[str, Any]] = []
    for pod in list(getattr(body, "pods", None) or []):
        pod_id = str(getattr(pod, "pod_id", "") or "")
        pod_rows.append(
            {
                "pod_id": pod_id,
                "status": str(getattr(pod, "status", "") or ""),
                "sub_status": str(getattr(pod, "sub_status", "") or ""),
                "type": str(getattr(pod, "type", "") or ""),
                "ip": str(getattr(pod, "ip", "") or ""),
            }
        )
        if not pod_id:
            continue
        try:
            response = client.get_pod_logs(
                args.job_id, pod_id, GetPodLogsRequest(max_lines=5000)
            )
            log_lines.extend(str(line) for line in (response.body.logs or []))
        except Exception as exc:
            log_errors.append(_provider_error(exc))
    worker = _parse_worker(log_lines)
    value = {
        "schema_version": "qwen35-397b-zero-gpu-preflight-diagnosis-v1",
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "job_id": args.job_id,
        "display_name": display_name,
        "status": str(getattr(body, "status", "") or ""),
        "reason_code": str(getattr(body, "reason_code", "") or ""),
        "reason_message_credential_redacted": _redact(
            getattr(body, "reason_message", "")
        ),
        "pods": pod_rows,
        "worker_receipt": worker,
        "log_line_count": len(log_lines),
        "log_tail_credential_redacted": [_redact(line) for line in log_lines[-300:]],
        "log_errors": log_errors,
        "gpu_accounting": {
            "requested": 0,
            "reserved": 0,
            "actually_used": 0,
            "evidence": "目标 job 的已核验 request 为 GPU=0 preflight",
        },
    }
    _atomic_json(output, value)
    print(
        json.dumps(
            {
                "job_id": args.job_id,
                "status": value["status"],
                "reason_code": value["reason_code"],
                "log_line_count": value["log_line_count"],
                "worker_receipt_present": worker is not None,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0
