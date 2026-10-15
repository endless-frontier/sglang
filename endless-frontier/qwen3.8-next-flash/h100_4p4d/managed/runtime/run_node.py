"""Run the owned supervisor on one of eight explicitly configured nodes."""

import argparse
import fcntl
import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path

from pd_state import read_json, save

RUN = Path(__file__).resolve().parent


def main(rank):
    state = RUN / "state"
    settings = read_json(RUN / "deployment-settings.json")
    node = read_json(state / f"node-{rank}.json")
    source = Path(settings["source"])
    manifest = read_json(RUN / "runtime-source-manifest.json")
    for relative, hashes in manifest["files"].items():
        if (
            hashlib.sha256((source / relative).read_bytes()).hexdigest()
            != hashes["after"]
        ):
            raise RuntimeError("Runtime source mismatch: " + relative)
    env = dict(
        os.environ,
        QWEN38_NODE_RANK=str(rank),
        QWEN38_STATE=str(state),
        QWEN38_MODEL_PATH=str(RUN / "model"),
        QWEN38_SGLANG_SOURCE=str(source),
        QWEN38_SERVED_MODEL_NAME=settings["model_name"],
        PYTHONUNBUFFERED="1",
        SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="false",
        SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION="1",
        QWEN38_MEM_FRACTION_STATIC=str(
            settings[
                "prefill_mem_fraction_static"
                if rank < 4
                else "decode_mem_fraction_static"
            ]
        ),
        QWEN38_MAX_RUNNING_REQUESTS="96",
        QWEN38_CHUNKED_PREFILL_SIZE="8192",
        QWEN38_SCHEDULE_POLICY="hrrn" if rank < 4 else "fcfs",
        QWEN38_FAIR_PREFILL="1" if rank < 4 else "0",
    )
    env["PYTHONPATH"] = (
        str(RUN) + ":" + str(source / "python") + ":" + env.get("PYTHONPATH", "")
    )
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)
    command = [
        "bash",
        str(RUN / "run_qwen38_flash_next_yarn_1m_pd_worker.sh"),
        node["role"],
        node["ip"],
        str(node["port"]),
        str(node["bootstrap_port"]),
    ]
    subprocess.run(command + ["--check-only"], env=env, check=True, timeout=120)
    save(
        state / f"watch-command-{rank}.json",
        dict(worker=command, router=["python3", str(RUN / "capacity_router.py")]),
    )
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Prefill loads first. Waiting checks shared state only, never inference.
    if rank >= 4:
        deadline = time.monotonic() + 2100
        while not stopping:
            try:
                ready = all(
                    read_json(state / f"watchdog-{i}.json")["worker"]["state"]
                    == "HEALTHY"
                    and time.time()
                    - read_json(state / f"watchdog-{i}.json")["updated_at"]
                    < 45
                    for i in range(4)
                )
            except (OSError, ValueError, KeyError):
                ready = False
            if ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Prefill startup deadline exceeded")
            time.sleep(5)
    child = None
    with (state / f"node-{rank}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (state / f"watchdog-{rank}.log").open("a", buffering=1) as log:
            try:
                while not stopping and not (state / "STOP").exists():
                    if child is None or child.poll() is not None:
                        child = subprocess.Popen(
                            ["python3", str(RUN / "node_supervisor.py")],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    time.sleep(5)
            finally:
                if child is not None and child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=range(8), required=True)
    main(parser.parse_args().rank)
