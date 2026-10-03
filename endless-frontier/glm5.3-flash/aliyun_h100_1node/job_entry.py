"""容器内入口：启动 GLM-5.3-Flash 配方脚本，并按固定间隔采样资源与打印进度。

由 DLC 的 UserCommand 以「单行、无引号」的方式落盘后执行：
    echo <base64 job_entry.py> | base64 -d > /tmp/job_entry.py; python3 -u /tmp/job_entry.py

环境变量（由提交端注入）：
    GLM53_SGLANG_SOURCE   源码树根目录（含 python/sglang/srt/models/glm5_next.py）
    GLM53_MODEL_PATH      模型目录
    GLM53_PORT            服务端口，默认 8000
    GLM53_SMOKE           1=健康后跑一次短请求与一次长请求
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SRC = os.environ.get("GLM53_SGLANG_SOURCE", "/mnt/data/REPLACE_WITH_ACCOUNT_DIR/sglang")
MODEL = os.environ.get("GLM53_MODEL_PATH", "/mnt/data/public_data/public_model/GLM5.3/GLM-5.3-Flash")
PORT = os.environ.get("GLM53_PORT", "8000")
SMOKE = os.environ.get("GLM53_SMOKE", "1") == "1"
SCRIPT = f"{SRC}/endless-frontier/glm5.3-flash/h100/deploy_glm53_flash_1m.sh"
LOG = Path("/tmp/glm53_server.log")
RES = Path("/tmp/glm53_resources.log")


def log(message) -> None:
    print(str(message), flush=True)


def host_memory() -> str:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        info[key] = int(value.split()[0])
    used = (info["MemTotal"] - info["MemAvailable"]) // 1048576
    total = info["MemTotal"] // 1048576
    return f"hostmem={used}/{total}GiB"


def gpu_state() -> str:
    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
    except Exception as exc:  # noqa: BLE001
        return f"gpu-error {exc}"
    if not rows:
        return "gpu-empty"
    return f"gpu0={rows[0].replace(' ', '')} n={len(rows)}"


def sampler(stop: threading.Event) -> None:
    with RES.open("w") as handle:
        while not stop.is_set():
            handle.write(f"{time.strftime('%H:%M:%S')} {host_memory()} {gpu_state()}\n")
            handle.flush()
            stop.wait(15)


def health() -> int:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as response:
            return response.status
    except Exception:  # noqa: BLE001
        return 0


def request(label: str, content: str, max_tokens: int = 200) -> None:
    body = json.dumps(
        {
            "model": os.environ.get("GLM53_SERVED_NAME", "glm-5.3-flash"),
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    started = time.time()
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=1800).read().decode())
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        log(f"RESULT {label} elapsed={time.time() - started:.1f}s finish={choice.get('finish_reason')}")
        log(f"  content={(message.get('content') or '')[:300]!r} usage={data.get('usage')}")
    except Exception as exc:  # noqa: BLE001
        log(f"RESULT {label} ERROR {type(exc).__name__} {str(exc)[:250]}")


def main() -> int:
    log(f"JOB-ENTRY-START {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log(f"model={MODEL}")
    log(f"source={SRC}")
    log(host_memory())
    log(gpu_state())

    check = subprocess.run(["bash", SCRIPT, "--check-only"], capture_output=True, text=True)
    log(f"=== check-only rc={check.returncode} ===")
    log((check.stdout or "")[-2000:])
    if check.returncode != 0:
        log((check.stderr or "")[-800:])
        log("JOB-ENTRY-END check-failed")
        return 1

    stop = threading.Event()
    threading.Thread(target=sampler, args=(stop,), daemon=True).start()

    env = dict(os.environ)
    env["GLM53_SGLANG_SOURCE"] = SRC
    env["GLM53_MODEL_PATH"] = MODEL
    env["GLM53_PORT"] = PORT
    handle = LOG.open("w")
    proc = subprocess.Popen(
        ["bash", SCRIPT], stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )
    log(f"launched recipe pid={proc.pid}")

    ready = False
    for tick in range(1, 161):
        time.sleep(15)
        code = health()
        alive = proc.poll() is None
        log(f"[tick {tick}] {time.strftime('%H:%M:%S')} health={code} alive={alive} "
            f"{host_memory()} {gpu_state()}")
        if tick % 4 == 0:
            tail = LOG.read_text(errors="replace").splitlines()[-3:]
            log("  srv: " + " | ".join(tail)[:600])
        if code == 200:
            ready = True
            log("HEALTH-OK")
            break
        if not alive:
            log(f"PROCESS-GONE rc={proc.returncode}")
            break

    log("=== server log tail ===")
    log("\n".join(LOG.read_text(errors="replace").splitlines()[-60:]))

    if ready and SMOKE:
        request("short", "Reply with exactly: pong")
        filler = "The quick brown fox jumps over the lazy dog. " * 1600
        log(f"smoke long prompt: {len(filler)} chars (~16k tokens expected)")
            request("long16k", filler + " Now reply with exactly: longpong")
    elif not ready:
        log("NOT-READY")

    stop.set()
    log("JOB-ENTRY-END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
