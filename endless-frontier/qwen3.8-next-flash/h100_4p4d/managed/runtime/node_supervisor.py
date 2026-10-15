"""Node supervisor using the shared 4P4D recovery policy.

No cloud API or SSH credential is needed. Local worker process groups are owned
by this supervisor; restart leases are shared by the eight pods of this job.
"""

import fcntl
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from pd_state import read_json, save

RUN = Path(__file__).resolve().parent
STATE = RUN / "state"
spec = importlib.util.spec_from_file_location(
    "original_watch", RUN / "watch_worker_original.py"
)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def schedulers(pid):
    result = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            d = dict(
                x.split(":", 1)
                for x in (p / "status").read_text().splitlines()
                if ":" in x
            )
            if int(d.get("PPid", -1)) == pid and d.get("Name", "").strip().startswith(
                "sglang::schedul"
            ):
                result.append(
                    {"pid": int(p.name), "state": d["State"].strip().split()[0]}
                )
        except (OSError, ValueError):
            pass
    return result


def pid_alive(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().split()[2] != "Z"
    except (OSError, ValueError):
        return False


class Leases:
    def __init__(self):
        self.token = uuid.uuid4().hex

    def change(self, acquire=False, release=False):
        with (STATE / "restart-leases.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = STATE / "restart-leases.json"
            now = time.time()
            leases = read_json(path) if path.exists() else {}
            leases = {k: v for k, v in leases.items() if v["expires"] > now}
            if release:
                leases.pop(self.token, None)
            if acquire and self.token not in leases and len(leases) < 2:
                leases[self.token] = {
                    "rank": RANK,
                    "expires": now + w.STARTUP_TIMEOUT + 120,
                }
            save(path, leases)
            return self.token in leases, len(leases)


def group_members(pgid):
    rows = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            st = (p / "stat").read_text().rsplit(")", 1)[1].split()
            if int(st[2]) == pgid and st[0] not in ("Z", "X"):
                rows.append(int(p.name))
        except (OSError, ValueError, IndexError):
            pass
    return rows


def stop_process(pid):
    if not pid or not group_members(pid):
        return
    # Only groups started by this supervisor, whose group ID is their leader PID.
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    until = time.monotonic() + float(os.environ.get("QWEN38_STOP_GRACE", "25"))
    while group_members(pid) and time.monotonic() < until:
        time.sleep(0.1)
    if group_members(pid):
        os.killpg(pid, signal.SIGKILL)
    until = time.monotonic() + 10
    while group_members(pid) and time.monotonic() < until:
        time.sleep(0.1)
    if group_members(pid):
        raise RuntimeError("Owned process group did not stop")


def last_batch(path):
    from datetime import datetime, timedelta, timezone

    if not path.exists():
        return -1
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 256 * 1024))
        text = f.read().decode(errors="replace")
    hits = re.findall(
        r"\[(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)[^\n]*(?:Prefill|Decode) batch", text
    )
    return (
        datetime.strptime(hits[-1], "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=timezone(timedelta(hours=8)))
        .timestamp()
        if hits
        else -1
    )


def main():
    global RANK
    RANK = int(os.environ["QWEN38_NODE_RANK"])
    node = read_json(STATE / f"node-{RANK}.json")
    config = read_json(STATE / f"watch-command-{RANK}.json")
    w.INTERVAL = float(config.get("test_interval", 10))
    w.HTTP_TIMEOUT = 5
    w.STARTUP_TIMEOUT = float(config.get("test_startup_timeout", 1800))
    w.FAIL_THRESHOLD = 3
    w.MIN_SOFT_DURATION = float(config.get("test_soft_duration", 180))
    w.COOLDOWN = float(config.get("test_cooldown", 300))
    lease = Leases()
    children = {}
    handles = []
    stopping = False
    own = w.Node(f"{node['role']}-{RANK}", node["role"], node["ip"], node["port"])
    nodes = [own]
    if RANK == 4:
        nodes.append(
            w.Node(
                "router",
                "router",
                node["vpc_ip"],
                read_json(RUN / "deployment-settings.json")["router_port"],
                kind="local",
            )
        )

    def start(n):
        name = "router" if n.role == "router" else "worker"
        log = (
            STATE
            / (f"router.log" if name == "router" else f"{node['role']}-{RANK}.log")
        ).open("a", buffering=1)
        handles.append(log)
        command = config[name]
        env = dict(os.environ)
        env["SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION"] = "1"
        if name == "router":
            env["CUDA_VISIBLE_DEVICES"] = ""
        p = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children[name] = p
        save(
            STATE / f"owned-process-{RANK}-{name}.json",
            {
                "pid": p.pid,
                "rank": RANK,
                "run": str(RUN),
                "hostname": socket.gethostname(),
                "start_ticks": (Path("/proc") / str(p.pid) / "stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()[19],
            },
        )
        return p

    def process(n):
        return children.get("router" if n.role == "router" else "worker")

    def alive(n):
        p = process(n)
        if not p or p.poll() is not None:
            return False
        if n.role == "router":
            return True
        rows = schedulers(p.pid)
        return len(rows) == 8 and all(r["state"] not in ("Z", "X") for r in rows)

    from pd_progress import ProgressGuard

    progress_guard = ProgressGuard(300)
    progress_state = {
        "verdict": "unknown",
        "detected_samples": 0,
        "last_detected_at": None,
    }
    health_cache = {}

    def probe_health(url, timeout=5):
        n = next(x for x in nodes if x.url == url)
        if not alive(n):
            return (
                False,
                "http-503-scheduler-gone"
                if n.role != "router"
                else "http-503-process-gone",
            )
        try:
            with HTTP.open(url + "/health", timeout=timeout) as reply:
                ok = reply.status == 200
                status = reply.status
            if ok and n.role == "prefill":
                try:
                    with HTTP.open(url + "/v1/loads", timeout=timeout) as reply:
                        sample = json.load(reply)["loads"][0]
                except (OSError, ValueError, KeyError, IndexError):
                    sample = None
                stalled = progress_guard.observe(sample, time.monotonic())
                progress_state["verdict"] = (
                    "unknown"
                    if stalled is None
                    else ("stalled" if stalled else "observed")
                )
                if stalled is None:
                    return None, "unknown-native-progress"
                if stalled:
                    progress_state["detected_samples"] += 1
                    progress_state["last_detected_at"] = time.time()
                    return False, "http-503-paused-prefill-no-progress"
            return ok, f"http-{status}"
        except Exception as e:
            return False, type(e).__name__

    def health(url, timeout=5):
        return (
            health_cache.pop(url) if url in health_cache else probe_health(url, timeout)
        )

    def restart(n):
        if n.role != "router" and not lease.change(acquire=True)[0]:
            n.state = "SUSPECT"
            n.last_restart = 0
            return False, "global restart slots busy; retry without killing worker"
        p = process(n)
        if n.role != "router":
            (STATE / f"ready-{RANK}.json").unlink(missing_ok=True)
        if p:
            stop_process(p.pid)
            p.wait(timeout=10)
        p = start(n)
        return (
            True,
            f"process launched pid={p.pid}; awaiting health AND 8 live schedulers",
        )

    w.http_health = health
    w.remote_process_alive = lambda ip: alive(own)
    w.remote_last_work_epoch = lambda ip, role: last_batch(STATE / f"{role}-{RANK}.log")
    w.restart_remote = restart
    w.restart_router = restart
    w._local_proc_alive = lambda n: alive(n)
    w.restarting_count = lambda _: lease.change()[1]
    w.log = lambda msg, also_print=True: print(
        json.dumps({"watchdog_event": msg, "rank": RANK, "ts": time.time()}), flush=True
    )
    for name in ("worker", "router"):
        old = STATE / f"owned-process-{RANK}-{name}.json"
        if old.exists():
            info = read_json(old)
            if info.get("run") != str(RUN):
                raise RuntimeError("Ownership record mismatch")
            if group_members(info["pid"]):
                proc = Path("/proc") / str(info["pid"])
                if info.get("hostname") != socket.gethostname():
                    raise RuntimeError("Ownership record belongs to another host")
                if not proc.exists() or proc.joinpath("stat").read_text().rsplit(
                    ")", 1
                )[1].split()[19] != info.get("start_ticks"):
                    raise RuntimeError(
                        "Unknown or reused process group; refusing cleanup"
                    )
            stop_process(info["pid"])
    for n in nodes:
        start(n)
        n.state = "RESTARTING"
        n.restarting_since = time.time()

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping and not (STATE / "STOP").exists():
            for n in nodes:
                if (
                    n is own
                    and n.state == "RESTARTING"
                    and time.time() - n.restarting_since >= w.STARTUP_TIMEOUT
                ):
                    lease.change(release=True)
                evidence = probe_health(n.url)
                if evidence[0] is None:
                    if n.state != "RESTARTING":
                        n.state = "UNKNOWN"
                    n.fail_count = 0
                    n.first_fail_ts = None
                else:
                    health_cache[n.url] = evidence
                    w.check_node(n, nodes, False)
            if own.state == "HEALTHY" and alive(own):
                lease.change(release=True)
                if not (STATE / f"ready-{RANK}.json").exists():
                    save(
                        STATE / f"ready-{RANK}.json",
                        dict(
                            node,
                            watchdog_pid=os.getpid(),
                            worker_pid=process(own).pid,
                            ready_at=time.time(),
                        ),
                    )
            elif own.state != "HEALTHY" or not alive(own):
                (STATE / f"ready-{RANK}.json").unlink(missing_ok=True)
            save(
                STATE / f"watchdog-{RANK}.json",
                {
                    "pid": os.getpid(),
                    "updated_at": time.time(),
                    "rank": RANK,
                    "adapter": "owned-node-supervisor-v1",
                    "auto_restart": True,
                    "worker_pid": process(own).pid,
                    "scheduler_processes": schedulers(process(own).pid),
                    "worker": own.as_dict(),
                    "router": nodes[1].as_dict() if len(nodes) > 1 else None,
                    "passive_prefill_progress": dict(progress_state),
                    "interval": w.INTERVAL,
                    "cooldown": w.COOLDOWN,
                    "soft_duration": w.MIN_SOFT_DURATION,
                    "startup_timeout": w.STARTUP_TIMEOUT,
                    "max_concurrent_worker_restarts": 2,
                },
            )
            time.sleep(w.INTERVAL)
    finally:
        (STATE / f"ready-{RANK}.json").unlink(missing_ok=True)
        lease.change(release=True)
        for p in children.values():
            stop_process(p.pid)
        for h in handles:
            h.close()


if __name__ == "__main__":
    main()
