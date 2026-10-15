"""Prepare one shared deployment directory; never launch workers or send requests."""

import argparse
import ipaddress
import json
import shutil
from pathlib import Path

from apply_runtime_patch import verify

HERE = Path(__file__).resolve().parent


def addresses(value):
    values = value.split(",")
    if len(values) != 4 or len(set(values)) != 4:
        raise argparse.ArgumentTypeError(
            "Exactly four distinct IPv4 addresses are required"
        )
    try:
        for value in values:
            ipaddress.IPv4Address(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use IPv4 addresses") from error
    return values


def prepare(args):
    source, model, run = args.source.resolve(), args.model.resolve(), args.run.resolve()
    verify(source, "after")
    if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
        raise ValueError(
            "Expected an HF export with config.json and safetensors weights"
        )
    ips = args.prefill_ips + args.decode_ips
    if len(set(ips)) != 8:
        raise ValueError("All eight workers must have distinct addresses")
    if run.exists() and any(run.iterdir()):
        raise ValueError(
            "Use a new empty deployment directory; never replace a running deployment"
        )
    run.mkdir(parents=True, exist_ok=True)
    state = run / "state"
    state.mkdir()
    # Copy small configuration files so the worker's YaRN adaptation never edits
    # the original HF export. Large weight shards remain shared, read-only links.
    private_model = run / "model"
    private_model.mkdir()
    for path in model.iterdir():
        target = private_model / path.name
        if path.suffix == ".safetensors":
            target.symlink_to(path.resolve())
        elif path.is_dir():
            shutil.copytree(path, target, symlinks=True)
        else:
            shutil.copy2(path, target)
    for path in (HERE / "runtime").glob("*.py"):
        shutil.copy2(path, run / path.name)
    for name in ("watch_worker.py", "run_qwen38_flash_next_yarn_1m_pd_worker.sh"):
        shutil.copy2(
            HERE.parent / name,
            run / ("watch_worker_original.py" if name == "watch_worker.py" else name),
        )
    settings = dict(
        model_name=args.model_name,
        source=str(source),
        router_port=40000,
        worker_max_running_requests=96,
        prefill_mem_fraction_static=0.88,
        decode_mem_fraction_static=0.93,
        prefill_batch_contexts=2,
        prefill_quantum_ceiling=4096,
        prefill_batch_token_budget=8192,
        require_worker_load_admission_ack=True,
    )
    (run / "deployment-settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (state / "fair-prefill-control.json").write_text(
        json.dumps(dict(contexts=2, quantum=4096, budget=8192, mode="fair")) + "\n"
    )
    for rank, ip in enumerate(ips):
        node = dict(
            rank=rank,
            role="prefill" if rank < 4 else "decode",
            ip=ip,
            vpc_ip=ip,
            port=41000 if rank < 4 else 42000,
            bootstrap_port=8998,
        )
        (state / f"node-{rank}.json").write_text(json.dumps(node) + "\n")
    shutil.copy2(HERE / "source-manifest.json", run / "runtime-source-manifest.json")
    return dict(
        run=str(run), nodes=8, router=f"http://{ips[4]}:40000/v1", model=args.model_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prefill-ips", type=addresses, required=True)
    parser.add_argument("--decode-ips", type=addresses, required=True)
    print(json.dumps(prepare(parser.parse_args())))
