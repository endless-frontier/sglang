#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import ipaddress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.request import urlopen
from uuid import uuid4
import zlib


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.qwen38_flash_next_h200.schema import (  # noqa: E402
    DEFAULT_PD_ROUTER,
    GPUS_PER_NODE,
    LoadedConfig,
    instance_nodes,
    load_config,
    model_path,
    safe_run_id,
    total_gpus,
    total_nodes,
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def emit_json_marker(prefix: str, value: dict[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = base64.b64encode(zlib.compress(payload, 9)).decode("ascii")
    print(prefix + encoded, flush=True)


def local_ip(cidr: str = "") -> str:
    completed = subprocess.run(
        ["hostname", "-I"], check=False, capture_output=True, text=True
    )
    if completed.returncode == 0 and completed.stdout.split():
        candidates = completed.stdout.split()
        if cidr:
            network = ipaddress.ip_network(cidr, strict=False)
            for candidate in candidates:
                if ipaddress.ip_address(candidate) in network:
                    return candidate
            raise RuntimeError(f"没有本机 IP 落在 worker_ip_cidr={cidr}")
        return candidates[0]
    return socket.gethostbyname(socket.gethostname())


def visible_gpu_memory_mb() -> list[int] | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        return [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    except ValueError:
        return None


def role_layout(config_or_mode: dict[str, Any] | str, rank: int) -> dict[str, Any]:
    """Map a DLC rank to an independent SGLang cell.

    The string form preserves the original four-node API used by older tests
    and receipts. New deployments always pass the resolved config.
    """

    legacy_api = isinstance(config_or_mode, str)
    if legacy_api:
        config: dict[str, Any] = {
            "mode": config_or_mode,
            "aliyun": {"nodes": 4, "gpus_per_node": GPUS_PER_NODE},
            "topology": {
                "prefill_nodes": 2,
                "decode_nodes": 2,
                "tensor_parallel_size": 16,
                "pipeline_parallel_size": 1 if config_or_mode == "pd" else 2,
            },
        }
    else:
        config = config_or_mode
    node_count = total_nodes(config)
    if not 0 <= rank < node_count:
        raise ValueError(f"RANK={rank} 不在 [0,{node_count})")
    if config["mode"] == "pd":
        topology = config["topology"]
        prefill_nodes = int(topology["prefill_nodes"])
        cell_nodes = instance_nodes(config)
        role = "prefill" if rank < prefill_nodes else "decode"
        role_start = 0 if role == "prefill" else prefill_nodes
        cell_index = (rank - role_start) // cell_nodes
        leader_rank = role_start + cell_index * cell_nodes
        result = {
            "role": role,
            "cell_index": cell_index,
            "leader_rank": leader_rank,
            "node_rank": rank - leader_rank,
            "nnodes": cell_nodes,
        }
        if legacy_api:
            result.pop("cell_index")
        return result
    result = {
        "role": "standard",
        "cell_index": 0,
        "leader_rank": 0,
        "node_rank": rank,
        "nnodes": node_count,
    }
    if legacy_api:
        result.pop("cell_index")
    return result


def role_leader_ranks(config: dict[str, Any], role: str) -> list[int]:
    return [
        rank
        for rank in range(total_nodes(config))
        if (layout := role_layout(config, rank))["role"] == role
        and layout["node_rank"] == 0
    ]


def discover_ib_devices() -> str:
    root = Path("/sys/class/infiniband")
    if not root.is_dir():
        return ""
    return ",".join(sorted(path.name for path in root.iterdir() if path.is_dir()))


def build_sglang_args(
    loaded: LoadedConfig,
    *,
    rank: int,
    node_ip: str,
    nodes: dict[int, dict[str, Any]],
    chosen_model_path: str,
    ib_devices: str,
) -> list[str]:
    config = loaded.value
    base_runtime = config["runtime"]
    topology = config["topology"]
    layout = role_layout(config, rank)
    runtime = dict(base_runtime)
    if config["mode"] == "pd":
        runtime.update((base_runtime.get("role_overrides") or {}).get(layout["role"], {}))
    leader_ip = str(nodes[layout["leader_rank"]]["ip"])
    args = [
        runtime["python_executable"],
        "-m",
        runtime["sglang_module"],
        "--model-path",
        chosen_model_path,
        "--served-model-name",
        runtime["served_model_name"],
        # Listen on every container interface; node_ip remains the advertised
        # address in ready.json and for distributed initialization.
        "--host",
        "0.0.0.0",
        "--port",
        str(runtime["service_port"]),
        "--trust-remote-code",
        "--dtype",
        runtime["dtype"],
        "--mem-fraction-static",
        str(runtime["mem_fraction_static"]),
        "--schedule-conservativeness",
        str(runtime["schedule_conservativeness"]),
        "--page-size",
        str(runtime["page_size"]),
        "--reasoning-parser",
        runtime["reasoning_parser"],
        "--tool-call-parser",
        runtime["tool_call_parser"],
        "--watchdog-timeout",
        str(runtime["watchdog_timeout_seconds"]),
        "--model-loader-extra-config",
        json.dumps(
            {"enable_multithread_load": "true", "num_threads": 64},
            separators=(",", ":"),
        ),
        "--tp-size",
        str(topology["tensor_parallel_size"]),
        "--pp-size",
        str(topology["pipeline_parallel_size"]),
        "--dist-init-addr",
        f"{leader_ip}:{runtime['dist_port']}",
        "--nnodes",
        str(layout["nnodes"]),
        "--node-rank",
        str(layout["node_rank"]),
    ]
    mamba_ssm_dtype = runtime.get("mamba_ssm_dtype")
    if mamba_ssm_dtype:
        args.extend(["--mamba-ssm-dtype", str(mamba_ssm_dtype)])
    mamba_full_memory_ratio = runtime.get("mamba_full_memory_ratio")
    if mamba_full_memory_ratio not in (None, "auto"):
        args.extend(
            ["--mamba-full-memory-ratio", str(mamba_full_memory_ratio)]
        )
    optional_integer_options = (
        ("context_length", "--context-length"),
        ("max_total_tokens", "--max-total-tokens"),
        ("max_running_requests", "--max-running-requests"),
        ("max_queued_requests", "--max-queued-requests"),
        ("chunked_prefill_size", "--chunked-prefill-size"),
        ("cuda_graph_max_bs", "--cuda-graph-max-bs"),
        ("max_mamba_cache_size", "--max-mamba-cache-size"),
    )
    for key, option in optional_integer_options:
        value = runtime.get(key)
        if value not in (None, "auto"):
            args.extend([option, str(value)])
    load_balance_method = runtime.get("load_balance_method")
    if load_balance_method:
        args.extend(["--load-balance-method", str(load_balance_method)])
    if runtime["enable_metrics"]:
        args.append("--enable-metrics")
    if runtime["enable_metrics_for_all_schedulers"]:
        args.append("--enable-metrics-for-all-schedulers")
    if runtime["enable_mfu_metrics"]:
        args.append("--enable-mfu-metrics")
    if runtime["enable_request_time_stats_logging"]:
        args.append("--enable-request-time-stats-logging")
    if runtime.get("disable_prefill_cuda_graph") is True:
        args.append("--disable-prefill-cuda-graph")
    if config["mode"] == "pd":
        if not ib_devices:
            raise RuntimeError("PD 模式没有发现 /sys/class/infiniband 设备")
        args.extend(
            [
                "--disaggregation-mode",
                layout["role"],
                "--disaggregation-transfer-backend",
                runtime["disaggregation_transfer_backend"],
                "--disaggregation-bootstrap-port",
                str(runtime["bootstrap_port"]),
                "--disaggregation-ib-device",
                ib_devices,
            ]
        )
    args.extend(runtime["extra_sglang_args"])
    return args


def build_pd_router_args(
    loaded: LoadedConfig,
    *,
    prefill_urls: list[str] | None = None,
    decode_urls: list[str] | None = None,
    prefill_url: str | None = None,
    decode_url: str | None = None,
) -> list[str]:
    runtime = loaded.value["runtime"]
    router = {**DEFAULT_PD_ROUTER, **(loaded.value.get("router") or {})}
    resolved_prefill_urls = list(prefill_urls or ([] if prefill_url is None else [prefill_url]))
    resolved_decode_urls = list(decode_urls or ([] if decode_url is None else [decode_url]))
    if not resolved_prefill_urls or not resolved_decode_urls:
        raise ValueError("PD Router 至少需要一个 Prefill URL 和一个 Decode URL")
    args = [
        runtime["python_executable"],
        "-m",
        "sglang_router.launch_router",
        "--host",
        "0.0.0.0",
        "--port",
        str(runtime["router_port"]),
        "--pd-disaggregation",
        "--prefill-policy",
        str(router["prefill_policy"]),
        "--decode-policy",
        str(router["decode_policy"]),
        "--request-timeout-secs",
        str(router["request_timeout_secs"]),
        "--worker-startup-timeout-secs",
        str(router["worker_startup_timeout_secs"]),
        "--queue-size",
        str(router["queue_size"]),
        "--queue-timeout-secs",
        str(router["queue_timeout_secs"]),
        "--cb-failure-threshold",
        str(router["cb_failure_threshold"]),
        "--cb-success-threshold",
        str(router["cb_success_threshold"]),
        "--cb-timeout-duration-secs",
        str(router["cb_timeout_duration_secs"]),
        "--cb-window-duration-secs",
        str(router["cb_window_duration_secs"]),
        "--health-failure-threshold",
        str(router["health_failure_threshold"]),
        "--health-success-threshold",
        str(router["health_success_threshold"]),
        "--health-check-timeout-secs",
        str(router["health_check_timeout_secs"]),
        "--health-check-interval-secs",
        str(router["health_check_interval_secs"]),
        "--prometheus-host",
        "0.0.0.0",
        "--prometheus-port",
        str(runtime["router_prometheus_port"]),
    ]
    for url in resolved_prefill_urls:
        args.extend(["--prefill", url, str(runtime["bootstrap_port"])])
    for url in resolved_decode_urls:
        args.extend(["--decode", url])
    max_concurrent = router.get("max_concurrent_requests")
    if max_concurrent not in (None, "auto"):
        args.extend(["--max-concurrent-requests", str(max_concurrent)])
    if router["disable_retries"]:
        args.append("--disable-retries")
    return args


def wait_for_nodes(
    registry: Path,
    *,
    timeout: int,
    config_sha256: str,
    mode: str,
    node_count: int = 4,
) -> dict[int, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found: dict[int, dict[str, Any]] = {}
        for rank in range(node_count):
            path = registry / "nodes" / f"rank{rank}.json"
            if not path.is_file():
                break
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("config_sha256") != config_sha256 or value.get("mode") != mode:
                raise RuntimeError(f"节点注册信息与本次配置不一致：{path}")
            found[rank] = value
        if len(found) == node_count:
            return found
        time.sleep(2)
    raise TimeoutError(f"等待 {node_count} 个 DLC 节点注册超时：{registry}")


def health_ready(url: str) -> bool:
    try:
        with urlopen(url.rstrip("/") + "/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def tcp_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False


class ChildProcesses:
    def __init__(self) -> None:
        self.children: list[subprocess.Popen[Any]] = []

    def start(self, argv: list[str]) -> subprocess.Popen[Any]:
        print("[launch] " + " ".join(json.dumps(item) for item in argv), flush=True)
        child_env = os.environ.copy()
        source = str(child_env.get("QWEN38_SGLANG_SOURCE", "") or "").strip()
        if source:
            source_python = str(Path(source) / "python")
            child_env["PYTHONPATH"] = source_python + os.pathsep + child_env.get("PYTHONPATH", "")
        process = subprocess.Popen(argv, env=child_env, start_new_session=True)
        self.children.append(process)
        return process

    def stop_all(self) -> None:
        for process in reversed(self.children):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and any(
            child.poll() is None for child in self.children
        ):
            time.sleep(0.5)
        for process in reversed(self.children):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def write_runtime_probe(registry: Path, rank: int, child: subprocess.Popen[Any]) -> None:
    memory = visible_gpu_memory_mb()
    value = {
        "rank": rank,
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "child_alive": child.poll() is None,
        "visible_gpu_count": len(memory) if memory is not None else None,
        "gpu_memory_used_mb": memory,
        "gpus_with_more_than_1gib_used": (
            sum(item >= 1024 for item in memory) if memory is not None else None
        ),
    }
    atomic_json(registry / "runtime" / f"rank{rank}.json", value)
    emit_json_marker("QWEN38_FLASH_NEXT_RUNTIME_B64=", value)


def gpu_accounting(
    registry: Path,
    timeout: int,
    *,
    node_count: int = 4,
    gpu_count: int = 32,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        latest = []
        for rank in range(node_count):
            path = registry / "runtime" / f"rank{rank}.json"
            if not path.is_file():
                break
            latest.append(json.loads(path.read_text(encoding="utf-8")))
        if len(latest) == node_count and all(
            item.get("visible_gpu_count") == GPUS_PER_NODE
            and item.get("gpus_with_more_than_1gib_used") == GPUS_PER_NODE
            for item in latest
        ):
            return {
                "requested": gpu_count,
                "reserved": gpu_count,
                "actually_used": gpu_count,
                "evidence": (
                    f"{node_count} 个 worker 均可见 {GPUS_PER_NODE} 卡，且 "
                    "SGLang ready 后每卡显存使用超过 1 GiB"
                ),
                "per_rank": latest,
            }
        time.sleep(5)
    return {
        "requested": gpu_count,
        "reserved": gpu_count,
        "actually_used": None,
        "evidence": (
            f"{node_count} 个 DLC worker 已注册，但 nvidia-smi "
            "显存证据未在有界窗口内齐全"
        ),
        "per_rank": latest,
    }


def invalidate_ready(
    registry: Path, *, run_id: str, config_sha256: str
) -> bool:
    """只删除属于当前 run/config 的 ready，避免 EAS 继续读取已死亡后端。"""
    path = registry / "ready.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if (
        value.get("schema_version") != "qwen38-flash-next-h200-ready-v1"
        or value.get("run_id") != run_id
        or value.get("config_sha256") != config_sha256
    ):
        return False
    path.unlink(missing_ok=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DLC 内部的 Qwen3.8 Flash Next 多机 worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run_id = safe_run_id(args.run_id)
    loaded = load_config(args.config)
    config = loaded.value
    runtime = config["runtime"]
    chosen_model = model_path(config, os.environ.get("DEPLOY_MODEL_PATH"))
    if not chosen_model or not chosen_model.startswith("/"):
        raise ValueError("DEPLOY_MODEL_PATH 必须是非空的容器内绝对路径")
    model_dir = Path(chosen_model)
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"模型目录或 config.json 不存在：{model_dir}")

    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    node_count = total_nodes(config)
    gpu_count = total_gpus(config)
    if world_size != node_count:
        raise ValueError(f"WORLD_SIZE={world_size}，期望 {node_count}")
    layout = role_layout(config, rank)
    node_ip = local_ip(runtime["worker_ip_cidr"])
    registry = (
        Path(runtime["project_root"])
        / ".runtime"
        / "deployment"
        / "qwen38_flash_next_h200"
        / run_id
    )
    memory = visible_gpu_memory_mb()
    atomic_json(
        registry / "nodes" / f"rank{rank}.json",
        {
            "rank": rank,
            "world_size": world_size,
            "mode": config["mode"],
            "role": layout["role"],
            "ip": node_ip,
            "hostname": socket.gethostname(),
            "config_sha256": loaded.sha256,
            "visible_gpu_count": len(memory) if memory is not None else None,
            "registered_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )
    nodes = wait_for_nodes(
        registry,
        timeout=runtime["node_discovery_timeout_seconds"],
        config_sha256=loaded.sha256,
        mode=config["mode"],
        node_count=node_count,
    )
    if any(item.get("visible_gpu_count") != GPUS_PER_NODE for item in nodes.values()):
        raise RuntimeError("至少一个 DLC worker 没有观测到 8 张 H200")

    children = ChildProcesses()
    shutting_down = False

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        print(f"[signal] 收到 {signum}，停止子进程", flush=True)
        children.stop_all()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    ib_devices = discover_ib_devices()
    sglang = children.start(
        build_sglang_args(
            loaded,
            rank=rank,
            node_ip=node_ip,
            nodes=nodes,
            chosen_model_path=chosen_model,
            ib_devices=ib_devices,
        )
    )
    leader_for_readiness = rank == 0
    router: subprocess.Popen[Any] | None = None
    try:
        if leader_for_readiness:
            if config["mode"] == "pd":
                prefill_urls = [
                    f"http://{nodes[leader]['ip']}:{runtime['service_port']}"
                    for leader in role_leader_ranks(config, "prefill")
                ]
                decode_urls = [
                    f"http://{nodes[leader]['ip']}:{runtime['service_port']}"
                    for leader in role_leader_ranks(config, "decode")
                ]
                prefill_url = prefill_urls[0]
                decode_url = decode_urls[0]
                health_urls = prefill_urls + decode_urls
            else:
                prefill_urls = []
                decode_urls = []
                prefill_url = ""
                decode_url = ""
                health_urls = [f"http://{nodes[0]['ip']}:{runtime['service_port']}"]
            deadline = time.monotonic() + runtime["startup_timeout_seconds"]
            while time.monotonic() < deadline:
                write_runtime_probe(registry, rank, sglang)
                if sglang.poll() is not None:
                    raise RuntimeError(
                        f"本节点 SGLang 在 ready 前退出：rc={sglang.returncode}"
                    )
                if all(health_ready(url) for url in health_urls):
                    break
                time.sleep(10)
            else:
                raise TimeoutError(
                    f"SGLang 启动超过 {runtime['startup_timeout_seconds']} 秒"
                )

            if config["mode"] == "pd":
                router = children.start(
                    build_pd_router_args(
                        loaded,
                        prefill_urls=prefill_urls,
                        decode_urls=decode_urls,
                    )
                )
                endpoint = f"http://{nodes[0]['ip']}:{runtime['router_port']}"
                router_deadline = time.monotonic() + 120
                while time.monotonic() < router_deadline:
                    if router.poll() is not None:
                        raise RuntimeError(
                            f"PD sglang_router 启动失败：rc={router.returncode}"
                        )
                    # 两个后端已分别通过 /health，这里确认 model gateway 开始监听。
                    if tcp_ready(str(nodes[0]["ip"]), int(runtime["router_port"])):
                        break
                    time.sleep(3)
                else:
                    raise TimeoutError("PD sglang_router 在 120 秒内没有监听 TCP")
            else:
                endpoint = health_urls[0]

            accounting = gpu_accounting(
                registry,
                runtime["gpu_usage_probe_timeout_seconds"],
                node_count=node_count,
                gpu_count=gpu_count,
            )
            ready_value = {
                "schema_version": "qwen38-flash-next-h200-ready-v1",
                "ready_at": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "run_id": run_id,
                "mode": config["mode"],
                "endpoint": endpoint,
                "prefill_endpoint": prefill_url or None,
                "decode_endpoint": decode_url or None,
                "prefill_endpoints": prefill_urls,
                "decode_endpoints": decode_urls,
                "metrics_endpoints": {
                    "router": (
                        f"http://{nodes[0]['ip']}:{runtime['router_prometheus_port']}/metrics"
                        if config["mode"] == "pd"
                        else None
                    ),
                    "prefill": (
                        f"{prefill_url}/metrics" if config["mode"] == "pd" else None
                    ),
                    "decode": (
                        f"{decode_url}/metrics" if config["mode"] == "pd" else None
                    ),
                    "standard": (
                        f"{endpoint}/metrics" if config["mode"] == "standard" else None
                    ),
                },
                "metrics_endpoint_groups": {
                    "prefill": [f"{url}/metrics" for url in prefill_urls],
                    "decode": [f"{url}/metrics" for url in decode_urls],
                },
                "served_model_name": runtime["served_model_name"],
                "config_sha256": loaded.sha256,
                "gpu_accounting": accounting,
            }
            atomic_json(registry / "ready.json", ready_value)
            emit_json_marker("QWEN38_FLASH_NEXT_READY_B64=", ready_value)
            print(f"[ready] endpoint={endpoint}", flush=True)

        while sglang.poll() is None:
            write_runtime_probe(registry, rank, sglang)
            if router is not None and router.poll() is not None:
                raise RuntimeError(f"PD sglang_router 意外退出：rc={router.returncode}")
            time.sleep(15)
        return int(sglang.returncode or 0)
    finally:
        children.stop_all()
        if rank == 0:
            invalidated = invalidate_ready(
                registry,
                run_id=run_id,
                config_sha256=loaded.sha256,
            )
            print(f"[ready] invalidated={invalidated}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
