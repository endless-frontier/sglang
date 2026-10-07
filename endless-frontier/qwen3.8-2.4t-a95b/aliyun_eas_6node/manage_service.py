"""The 2.4T serving service — create / describe / delete, for the EAS distributed shape.

Prepared so that the deployment is one command when the weights land. Every flag comes from the
official SGLang cookbook recipe for this model (see
../2026-10-04-official-sglang-recipe.md), and every environment variable comes from the EAS
multi-machine contract (see ../../../context/serving-design-rules.md):

    EAS injects        ->  SGLang wants
    unit.size          ->  --nnodes
    RANK_ID            ->  --node-rank
    MASTER_ADDRESS     ->  --dist-init-addr <ip>:20000
    COMM_IFNAME        ->  NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME

  --dry-run   print the service body
  --apply     create the service
  --describe  current state
  --delete    delete it
"""

import argparse
import json
from pathlib import Path

ENV = Path.home() / ".aliyun-ef.env"
REGION = "cn-shanghai"
WORKSPACE = "REPLACE_WITH_WORKSPACE_ID"
QUOTA = "REPLACE_WITH_LINGJUN_QUOTA_ID"
CPFS_FS = "REPLACE_WITH_CPFS_FILESYSTEM_ID"
NETWORK = {
    "security_group_id": "REPLACE_WITH_SECURITY_GROUP_ID",
    "vpc_id": "REPLACE_WITH_VPC_ID",
    "vswitch_id": "REPLACE_WITH_VSWITCH_ID",
}
MODEL_DIR = "/mnt/data/<你的目录>/models/Qwen3.8-2.4T-A95B-FP8"
NAME = "ef_qwen38_24t_a95b"
# machines per replica: the H200 recipe uses 4 nodes of 141 GB; ours are 80 GB, so the same weight
# set and comparable headroom need 6 (3.84 TB) — see the recipe evidence for the arithmetic
MACHINES = 6
GPUS_PER_MACHINE = 8

# the cookbook's verified H200 cell, with the pipeline dimension carrying the machine count instead
SGLANG_FLAGS = (
    "--tp-size 8 --pp-size {machines} --dist-timeout 1800 "
    "--linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer "
    "--mamba-full-memory-ratio 0.95 --mamba-ssm-dtype bfloat16 --max-prefill-tokens 8192 "
    "--page-size 64 --reasoning-parser qwen3 --tool-call-parser qwen3_coder "
    "--max-running-requests 48 --host 0.0.0.0 --port 8000 --trust-remote-code"
).format(machines=MACHINES)

SCRIPT = (
    "bash -c 'set -uo pipefail; "
    "export HOME=/tmp/eas-home TMPDIR=/tmp/eas-tmp TRITON_CACHE_DIR=/tmp/triton-cache "
    "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor XDG_CACHE_HOME=/tmp/xdg-cache; "
    'mkdir -p "$HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"; '
    "export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64; "
    "mkdir -p /mnt/data/<你的目录>/eas; "
    "LOG=/mnt/data/<你的目录>/eas/qwen38-24t-$(hostname).log; "
    '{ echo "=== start $(date -Is) host=$(hostname)"; '
    'echo "--- injected contract ---"; '
    'for v in RANK_ID COMM_IFNAME RANK_IP MASTER_ADDRESS; do eval "echo $v=\\${$v:-MISSING}"; done; '
    'echo "--- fail fast rather than load 2.45 TB blind ---"; '
    'if [ -z "${RANK_ID:-}" ] || [ -z "${MASTER_ADDRESS:-}" ]; then '
    '  echo "CONTRACT-MISSING: EAS did not inject RANK_ID/MASTER_ADDRESS into this custom image"; '
    '  echo "--- environment (filtered) ---"; env | sort | grep -iE "rank|master|comm|world|node|nccl" || true; '
    "  exit 42; fi; "
    "export NCCL_SOCKET_IFNAME=${COMM_IFNAME:-net0} GLOO_SOCKET_IFNAME=${COMM_IFNAME:-net0}; "
    "python3 -c \"import torch; print('devices', torch.cuda.device_count(), 'available', torch.cuda.is_available())\" 2>&1 | tail -2; "
    "export PYTHONPATH=/opt/sglang/python PYTHONUNBUFFERED=1; "
    "exec python3 -m sglang.launch_server "
    "--model-path " + MODEL_DIR + " "
    "--served-model-name qwen3.8-2.4t-a95b "
    "--nnodes " + str(MACHINES) + " --node-rank ${RANK_ID:-0} "
    "--dist-init-addr ${MASTER_ADDRESS}:20000 "
    + SGLANG_FLAGS
    + '; } 2>&1 | tee -a "$LOG"\''
)


def body() -> dict:
    return {
        "cloud": {"networking": dict(NETWORK)},
        "containers": [
            {
                "image": "REPLACE_WITH_OUR_IMAGE_URI_PUBLIC",
                "port": 8000,
                "script": SCRIPT,
                "health_check": {
                    "failure_threshold": 120,
                    "http_get": {"path": "/health", "port": 8000},
                    "initial_delay_seconds": 3600,  # a 2.45 TB load is not a warm-up
                    "period_seconds": 30,
                    "success_threshold": 1,
                    "timeout_seconds": 10,
                },
                "startup_check": {
                    "failure_threshold": 120,
                    "http_get": {"path": "/health", "port": 8000},
                    "initial_delay_seconds": 600,
                    "period_seconds": 30,
                    "success_threshold": 1,
                    "timeout_seconds": 10,
                },
            }
        ],
        "labels": {
            "project": "qwen3.8-2.4t-a95b",
            "owner": "REPLACE_WITH_OWNER",
            "purpose": "serving",
        },
        "metadata": {
            "cpu": 128,
            "disk": "30Gi",
            "gpu": GPUS_PER_MACHINE,
            "instance": 1,
            "memory": 1500000,
            "name": NAME,
            "quota_id": QUOTA,
            "quota_type": "Lingjun",
            "resource_burstable": False,
            "workspace_id": WORKSPACE,
        },
        # the documented distributed switch: machines per model instance
        "unit": {"size": MACHINES},
        "options": {"priority": 5},
        "storage": [
            {
                "cpfs": {"file_system_id": CPFS_FS, "path": "/mnt/cpfs/"},
                "mount_path": "/mnt/data/",
            }
        ],
    }


def load_creds() -> dict:
    creds = {}
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip()
    return creds


def client(creds):
    from alibabacloud_eas20210701.client import Client as Eas
    from alibabacloud_tea_openapi import models as om

    return Eas(
        om.Config(
            access_key_id=creds["ALIBABA_CLOUD_ACCESS_KEY_ID"],
            access_key_secret=creds["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
            endpoint=f"pai-eas.{REGION}.aliyuncs.com",
            region_id=REGION,
            connect_timeout=20000,
            read_timeout=60000,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--update", action="store_true")
    g.add_argument("--describe", action="store_true")
    g.add_argument("--delete", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print(json.dumps(body(), ensure_ascii=False, indent=1))
        return 0

    from alibabacloud_eas20210701 import models as eas

    c = client(load_creds())
    if args.apply:
        r = c.create_service(
            eas.CreateServiceRequest(develop=False, workspace_id=WORKSPACE, body=body())
        ).body.to_map()
        print("create:", r.get("ServiceName"), r.get("Status"))
    if args.describe or args.apply or args.update:
        b = c.describe_service(REGION, NAME).body.to_map()
        keys = (
            "ServiceName",
            "Status",
            "TotalInstance",
            "RunningInstance",
            "Gpu",
            "Cpu",
            "Message",
            "InternetEndpoint",
            "QuotaId",
            "UpdateTime",
        )
        print(
            json.dumps(
                {k: b.get(k) for k in keys if k in b}, ensure_ascii=False, indent=1
            )[:900]
        )
    if args.update:
        from alibabacloud_eas20210701 import models as eas

        r = c.update_service(
            REGION, NAME, eas.UpdateServiceRequest(body=body())
        ).body.to_map()
        print("update:", r.get("ServiceName") or r)

    if args.delete:
        print("delete:", c.delete_service(REGION, NAME).body.to_map())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
