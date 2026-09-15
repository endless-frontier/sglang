from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "qwen38-flash-next-h200-deployment-v1"
TOTAL_NODES = 4
GPUS_PER_NODE = 8
TOTAL_GPUS = TOTAL_NODES * GPUS_PER_NODE
PROJECT_H200_CAP = 128
TASK_APPROVAL_THRESHOLD_H200 = 32
# Compatibility aliases for the copied read-only capacity helpers.
PROJECT_H100_CAP = PROJECT_H200_CAP
TASK_APPROVAL_THRESHOLD_H100 = TASK_APPROVAL_THRESHOLD_H200
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}")
LOAD_BALANCE_METHODS = {
    "round_robin",
    "follow_bootstrap_room",
    "total_requests",
    "total_tokens",
}
DEFAULT_PD_ROUTER = {
    "prefill_policy": "cache_aware",
    "decode_policy": "round_robin",
    "max_concurrent_requests": None,
    "queue_size": 100,
    "queue_timeout_secs": 60,
    "request_timeout_secs": 1800,
    "worker_startup_timeout_secs": 1800,
    "disable_retries": False,
    "cb_failure_threshold": 10,
    "cb_success_threshold": 3,
    "cb_timeout_duration_secs": 60,
    "cb_window_duration_secs": 120,
    "health_failure_threshold": 3,
    "health_success_threshold": 2,
    "health_check_timeout_secs": 5,
    "health_check_interval_secs": 60,
}


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    value: dict[str, Any]
    sha256: str


def total_nodes(config: dict[str, Any]) -> int:
    """Return the configured DLC worker count.

    TOTAL_NODES/TOTAL_GPUS remain the legacy four-node defaults for callers that
    import those names, while all deployment paths use these config-derived
    helpers.
    """

    return int(config["aliyun"]["nodes"])


def total_gpus(config: dict[str, Any]) -> int:
    return total_nodes(config) * int(config["aliyun"]["gpus_per_node"])


def instance_nodes(config: dict[str, Any]) -> int:
    topology = config["topology"]
    instance_gpus = int(topology["tensor_parallel_size"]) * int(
        topology["pipeline_parallel_size"]
    )
    return instance_gpus // int(config["aliyun"]["gpus_per_node"])


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inside_project(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"路径必须位于项目内：{resolved}") from exc
    return resolved


def safe_run_id(value: str) -> str:
    if not RUN_ID_RE.fullmatch(value):
        raise ValueError(
            "run_id 必须以字母或数字开头，只能包含字母、数字、点、下划线、连字符，"
            "且最长 63 字符"
        )
    return value


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 JSON object")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} 必须是正数")
    return float(value)


def _optional_positive_int(value: object, name: str) -> int | None:
    """接受正整数或自动值；None/"auto" 都表示不向下游强制传参。"""
    if value is None or value == "auto":
        return None
    return _positive_int(value, name)


def _nonnegative_int_or_auto(value: object, name: str) -> int | None:
    """DLC 时长允许 0（不限时）以及 None/"auto"（省略 provider 字段）。"""
    if value is None or value == "auto":
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数、null 或 auto")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 不能为空")
    return value.strip()


def _cli_option_int(args: list[str], option: str) -> int | None:
    for index, token in enumerate(args):
        if token == option:
            if index + 1 >= len(args):
                raise ValueError(f"{option} 缺少值")
            try:
                value = int(args[index + 1])
            except ValueError as exc:
                raise ValueError(f"{option} 必须是整数") from exc
            if value <= 0:
                raise ValueError(f"{option} 必须是正整数")
            return value
        if token.startswith(option + "="):
            try:
                value = int(token.split("=", 1)[1])
            except ValueError as exc:
                raise ValueError(f"{option} 必须是整数") from exc
            if value <= 0:
                raise ValueError(f"{option} 必须是正整数")
            return value
    return None


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version 必须为 {SCHEMA_VERSION}")
    mode = config.get("mode")
    if mode not in {"pd", "standard"}:
        raise ValueError("mode 必须是 pd 或 standard")
    _nonempty(config.get("name"), "name")

    aliyun = _object(config.get("aliyun"), "aliyun")
    for key in (
        "region",
        "endpoint",
        "workspace_id",
        "resource_id",
        "worker_image",
    ):
        _nonempty(aliyun.get(key), f"aliyun.{key}")
    priority = _positive_int(aliyun.get("priority"), "aliyun.priority")
    if priority > 9:
        raise ValueError("aliyun.priority 必须位于 1..9")
    nodes = _positive_int(aliyun.get("nodes"), "aliyun.nodes")
    if nodes * GPUS_PER_NODE > PROJECT_H200_CAP:
        raise ValueError(
            f"aliyun.nodes 最多为 {PROJECT_H200_CAP // GPUS_PER_NODE}，"
            f"对应项目上限 {PROJECT_H200_CAP} H200"
        )
    if (
        _positive_int(aliyun.get("gpus_per_node"), "aliyun.gpus_per_node")
        != GPUS_PER_NODE
    ):
        raise ValueError(f"aliyun.gpus_per_node 固定为 {GPUS_PER_NODE}")
    if aliyun.get("enable_rdma") is not True:
        raise ValueError("aliyun.enable_rdma 必须为 true")
    _nonnegative_int_or_auto(
        aliyun.get("job_max_running_time_minutes"),
        "aliyun.job_max_running_time_minutes",
    )
    if not isinstance(aliyun.get("image_verified_for_qwen38_bf16"), bool):
        raise ValueError("aliyun.image_verified_for_qwen38_bf16 必须是 boolean")
    data_sources = aliyun.get("data_sources")
    if not isinstance(data_sources, list) or not data_sources:
        raise ValueError("aliyun.data_sources 必须是非空 list")
    for index, source in enumerate(data_sources):
        item = _object(source, f"aliyun.data_sources[{index}]")
        _nonempty(item.get("data_source_id"), f"data_sources[{index}].data_source_id")
        mount = _nonempty(item.get("mount_path"), f"data_sources[{index}].mount_path")
        if not mount.startswith("/"):
            raise ValueError("DataSource mount_path 必须是绝对路径")

    runtime = _object(config.get("runtime"), "runtime")
    project_root = _nonempty(runtime.get("project_root"), "runtime.project_root")
    if not project_root.startswith("/"):
        raise ValueError("runtime.project_root 必须是容器内绝对路径")
    model_path = runtime.get("model_path")
    if not isinstance(model_path, str):
        raise ValueError("runtime.model_path 必须是字符串；模板中允许留空")
    for key in (
        "served_model_name",
        "python_executable",
        "sglang_module",
        "dtype",
        "reasoning_parser",
        "tool_call_parser",
        "mamba_scheduler_strategy",
    ):
        _nonempty(runtime.get(key), f"runtime.{key}")
    mamba_ssm_dtype = runtime.get("mamba_ssm_dtype")
    if mamba_ssm_dtype is not None and mamba_ssm_dtype not in {
        "float32",
        "bfloat16",
        "float16",
    }:
        raise ValueError(
            "runtime.mamba_ssm_dtype 必须是 float32、bfloat16、float16 或 null"
        )
    if not isinstance(runtime.get("worker_ip_cidr"), str):
        raise ValueError("runtime.worker_ip_cidr 必须是字符串；不限定网段时填空串")
    if runtime["worker_ip_cidr"]:
        try:
            ipaddress.ip_network(runtime["worker_ip_cidr"], strict=False)
        except ValueError as exc:
            raise ValueError("runtime.worker_ip_cidr 不是合法 CIDR") from exc
    for key in (
        "service_port",
        "router_port",
        "bootstrap_port",
        "dist_port",
        "page_size",
        "watchdog_timeout_seconds",
        "node_discovery_timeout_seconds",
        "startup_timeout_seconds",
        "gpu_usage_probe_timeout_seconds",
        "router_prometheus_port",
    ):
        _positive_int(runtime.get(key), f"runtime.{key}")
    for key in (
        "enable_metrics",
        "enable_metrics_for_all_schedulers",
        "enable_mfu_metrics",
        "enable_request_time_stats_logging",
    ):
        if not isinstance(runtime.get(key), bool):
            raise ValueError(f"runtime.{key} 必须是 bool")
    if runtime["enable_metrics_for_all_schedulers"] and not runtime["enable_metrics"]:
        raise ValueError(
            "runtime.enable_metrics_for_all_schedulers 需要同时启用 enable_metrics"
        )
    if runtime["enable_mfu_metrics"] and not runtime["enable_metrics"]:
        raise ValueError("runtime.enable_mfu_metrics 需要同时启用 enable_metrics")
    for key in (
        "context_length",
        "max_total_tokens",
        "max_running_requests",
        "max_queued_requests",
        "chunked_prefill_size",
        "cuda_graph_max_bs",
        "max_mamba_cache_size",
    ):
        _optional_positive_int(runtime.get(key), f"runtime.{key}")
    fraction = runtime.get("mem_fraction_static")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
        raise ValueError("runtime.mem_fraction_static 必须是数字")
    if not 0 < float(fraction) < 1:
        raise ValueError("runtime.mem_fraction_static 必须位于 (0, 1)")
    _positive_number(
        runtime.get("schedule_conservativeness"),
        "runtime.schedule_conservativeness",
    )
    mamba_full_memory_ratio = runtime.get("mamba_full_memory_ratio")
    if mamba_full_memory_ratio not in (None, "auto"):
        ratio = _positive_number(
            mamba_full_memory_ratio,
            "runtime.mamba_full_memory_ratio",
        )
        if ratio > 1:
            raise ValueError("runtime.mamba_full_memory_ratio 必须位于 (0, 1]")
    extra_args = runtime.get("extra_sglang_args")
    if not isinstance(extra_args, list) or not all(
        isinstance(item, str) and item for item in extra_args
    ):
        raise ValueError("runtime.extra_sglang_args 必须是字符串 list")
    protected_options = (
        "--model-path",
        "--tp",
        "--tp-size",
        "--pp",
        "--pp-size",
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
        "--disaggregation-mode",
        "--disaggregation-transfer-backend",
        "--load-balance-method",
        "--host",
        "--port",
    )
    if any(
        token == option or token.startswith(option + "=")
        for token in extra_args
        for option in protected_options
    ):
        raise ValueError("runtime.extra_sglang_args 不得覆盖模型路径、并行、网络或 PD 拓扑")

    sglang_env = runtime.get("sglang_env", {})
    sglang_env = _object(sglang_env, "runtime.sglang_env")
    invalid_env_names = [
        key
        for key in sglang_env
        if not isinstance(key, str) or not re.fullmatch(r"SGLANG_[A-Z0-9_]+", key)
    ]
    if invalid_env_names:
        raise ValueError(
            "runtime.sglang_env 只允许 SGLANG_* 环境变量："
            + ",".join(sorted(map(str, invalid_env_names)))
        )
    if not all(isinstance(value, str) and value for value in sglang_env.values()):
        raise ValueError("runtime.sglang_env 的值必须是非空字符串")

    role_overrides = runtime.get("role_overrides", {})
    if role_overrides is None:
        role_overrides = {}
    role_overrides = _object(role_overrides, "runtime.role_overrides")
    if mode != "pd" and role_overrides:
        raise ValueError("runtime.role_overrides 只允许用于 PD 模式")
    unknown_roles = set(role_overrides) - {"prefill", "decode"}
    if unknown_roles:
        raise ValueError(
            "runtime.role_overrides 含未知角色：" + ",".join(sorted(unknown_roles))
        )
    allowed_override_keys = {
        "mem_fraction_static",
        "schedule_conservativeness",
        "max_running_requests",
        "chunked_prefill_size",
        "cuda_graph_max_bs",
        "max_mamba_cache_size",
        "load_balance_method",
        "disable_prefill_cuda_graph",
    }
    for role, raw_overrides in role_overrides.items():
        overrides = _object(raw_overrides, f"runtime.role_overrides.{role}")
        unknown_keys = set(overrides) - allowed_override_keys
        if unknown_keys:
            raise ValueError(
                f"runtime.role_overrides.{role} 含不允许字段："
                + ",".join(sorted(unknown_keys))
            )
        if "mem_fraction_static" in overrides:
            value = overrides["mem_fraction_static"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < float(value) < 1
            ):
                raise ValueError(
                    f"runtime.role_overrides.{role}.mem_fraction_static 必须位于 (0, 1)"
                )
        if "schedule_conservativeness" in overrides:
            _positive_number(
                overrides["schedule_conservativeness"],
                f"runtime.role_overrides.{role}.schedule_conservativeness",
            )
        if "load_balance_method" in overrides:
            method = _nonempty(
                overrides["load_balance_method"],
                f"runtime.role_overrides.{role}.load_balance_method",
            )
            if method not in LOAD_BALANCE_METHODS:
                raise ValueError(
                    f"runtime.role_overrides.{role}.load_balance_method "
                    f"必须是 {sorted(LOAD_BALANCE_METHODS)} 之一"
                )
        if "disable_prefill_cuda_graph" in overrides:
            if not isinstance(overrides["disable_prefill_cuda_graph"], bool):
                raise ValueError(
                    f"runtime.role_overrides.{role}.disable_prefill_cuda_graph "
                    "必须是 bool"
                )
        for key in allowed_override_keys - {
            "mem_fraction_static",
            "schedule_conservativeness",
            "load_balance_method",
            "disable_prefill_cuda_graph",
        }:
            if key in overrides:
                _optional_positive_int(
                    overrides[key], f"runtime.role_overrides.{role}.{key}"
                )

    router = config.get("router")
    if mode == "pd":
        # 旧模板没有独立 router block；保留其历史默认行为，同时允许新模板
        # 显式记录全部 Model Gateway 参数。
        router = {**DEFAULT_PD_ROUTER, **_object(router or {}, "router")}
        if router.get("prefill_policy") not in {"cache_aware", "round_robin"}:
            raise ValueError(
                "router.prefill_policy 必须是 cache_aware 或 round_robin"
            )
        if router.get("decode_policy") not in {"cache_aware", "round_robin"}:
            raise ValueError(
                "router.decode_policy 必须是 cache_aware 或 round_robin"
            )
        _optional_positive_int(
            router.get("max_concurrent_requests"),
            "router.max_concurrent_requests",
        )
        _nonnegative_int(router.get("queue_size"), "router.queue_size")
        for key in (
            "queue_timeout_secs",
            "request_timeout_secs",
            "worker_startup_timeout_secs",
            "cb_failure_threshold",
            "cb_success_threshold",
            "cb_timeout_duration_secs",
            "cb_window_duration_secs",
            "health_failure_threshold",
            "health_success_threshold",
            "health_check_timeout_secs",
            "health_check_interval_secs",
        ):
            _positive_int(router.get(key), f"router.{key}")
        if not isinstance(router.get("disable_retries"), bool):
            raise ValueError("router.disable_retries 必须是 bool")
    elif router not in (None, {}):
        raise ValueError("router 配置只允许用于 PD 模式")

    topology = _object(config.get("topology"), "topology")
    tp = _positive_int(topology.get("tensor_parallel_size"), "topology.tensor_parallel_size")
    pp = _positive_int(
        topology.get("pipeline_parallel_size"), "topology.pipeline_parallel_size"
    )
    instance_gpus = tp * pp
    if instance_gpus % GPUS_PER_NODE:
        raise ValueError("TP×PP 必须能整除每节点 8 GPU")
    nodes_per_instance = instance_gpus // GPUS_PER_NODE
    if mode == "pd":
        prefill_nodes = _positive_int(
            topology.get("prefill_nodes"), "topology.prefill_nodes"
        )
        decode_nodes = _positive_int(
            topology.get("decode_nodes"), "topology.decode_nodes"
        )
        if prefill_nodes + decode_nodes != nodes:
            raise ValueError("PD 的 prefill_nodes + decode_nodes 必须等于 aliyun.nodes")
        if instance_gpus != 8 or tp != 8 or pp != 1:
            raise ValueError(
                "当前 Qwen3.8 Flash Next PD cell 固定使用单节点 TP8/PP1"
            )
        if prefill_nodes % nodes_per_instance or decode_nodes % nodes_per_instance:
            raise ValueError(
                "PD 的 Prefill/Decode 节点数都必须是单个 16-GPU cell 节点数的整数倍"
            )
        if pp > 1 and "--disable-overlap-schedule" not in extra_args:
            raise ValueError("PP>1 时必须显式关闭 overlap schedule")
        dp = _cli_option_int(extra_args, "--dp-size")
        ep = _cli_option_int(extra_args, "--ep-size")
        if "--enable-dp-attention" in extra_args:
            if dp is None or tp % dp:
                raise ValueError("DP attention 要求 --dp-size 存在且能整除 TP")
        if ep is not None and (ep > tp or tp % ep):
            raise ValueError("--ep-size 必须不大于 TP 且能整除 TP")
        if "--moe-a2a-backend" in extra_args:
            backend_index = extra_args.index("--moe-a2a-backend")
            backend = (
                extra_args[backend_index + 1]
                if backend_index + 1 < len(extra_args)
                else ""
            )
            if backend == "deepep" and (dp is None or dp < 2):
                raise ValueError("DeepEP 当前部署契约要求 DP attention 且 --dp-size >= 2")
    else:
        if instance_gpus != nodes * GPUS_PER_NODE:
            raise ValueError("普通部署的 TP×PP 必须等于全部 DLC worker 可见 GPU 总数")


def load_config(path: Path) -> LoadedConfig:
    resolved = inside_project(path)
    raw = resolved.read_bytes()
    value = json.loads(raw)
    config = _object(value, "config")
    validate_config(config)
    return LoadedConfig(path=resolved, value=config, sha256=sha256_bytes(raw))


def model_path(config: dict[str, Any], override: str | None = None) -> str:
    chosen = override if override is not None else config["runtime"]["model_path"]
    if not isinstance(chosen, str):
        raise ValueError("model path 必须是字符串")
    stripped = chosen.strip()
    if any(ord(character) < 32 for character in stripped):
        raise ValueError("model path 不得包含控制字符")
    return stripped


def submission_blockers(
    config: dict[str, Any], *, model_override: str | None = None
) -> list[str]:
    blockers: list[str] = []
    chosen_model = model_path(config, model_override)
    if not chosen_model:
        blockers.append("model_path 为空")
    elif not chosen_model.startswith("/"):
        blockers.append("model_path 不是容器内绝对路径")
    if config["aliyun"]["image_verified_for_qwen38_bf16"] is not True:
        blockers.append("候选镜像尚未验证 Qwen3.8 Flash Next BF16 / SGLang PD")
    return blockers
