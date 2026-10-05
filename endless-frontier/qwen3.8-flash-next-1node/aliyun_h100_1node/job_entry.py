"""容器内入口：启动 Qwen3.8-Flash-Next 单机 8 卡服务，并采样资源与跑冒烟请求。

由 DLC 的 UserCommand 以「单行、无引号」的方式落盘后执行：
    echo <base64 job_entry.py> | base64 -d > /tmp/job_entry.py; python3 -u /tmp/job_entry.py

这里刻意不依赖任何共享 CPFS 源码树：SGLang 取自镜像内的 /opt/sglang/python
（提交作业用的镜像必须是我们自己构建的那一个），启动参数直接写在本文件里，
与 launch/deploy_qwen38_flash_next_1m.sh 保持一致，便于对照复核。

环境变量（可选覆盖）：
    QWEN38_SGLANG_SOURCE  默认 /opt/sglang/python
    QWEN38_MODEL_PATH     默认 /mnt/data/public_data/public_model/Qwen3.8/Qwen3.8-Flash-Next-1M
    QWEN38_PORT           默认 40000
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SRC = os.environ.get("QWEN38_SGLANG_SOURCE", "/opt/sglang/python")
MODEL = os.environ.get(
    "QWEN38_MODEL_PATH", "/mnt/data/public_data/public_model/Qwen3.8/Qwen3.8-Flash-Next-1M"
)
SERVED = os.environ.get("QWEN38_SERVED_NAME", "qwen38-flash-next-1m")
PORT = os.environ.get("QWEN38_PORT", "40000")
TP_SIZE = os.environ.get("QWEN38_TP_SIZE", "8")
MEM_FRACTION = os.environ.get("QWEN38_MEM_FRACTION", "0.90")
SPECULATIVE = os.environ.get("QWEN38_SPECULATIVE", "1") == "1"
SMOKE = os.environ.get("QWEN38_SMOKE", "1") == "1"
URL = f"http://127.0.0.1:{PORT}"
LOG = Path("/tmp/qwen38_server.log")
RES = Path("/tmp/qwen38_resources.log")

ARGS = [
    "--model-path", MODEL,
    "--served-model-name", SERVED,
    "--host", "0.0.0.0",
    "--port", PORT,
    "--tp-size", TP_SIZE,
    "--trust-remote-code",
    "--context-length", "1048576",
    "--mem-fraction-static", MEM_FRACTION,
    "--cuda-graph-max-bs-decode", "32",
    "--max-running-requests", "96",
]
if SPECULATIVE:
    ARGS += [
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
    ]


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
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
    except Exception as exc:  # noqa: BLE001
        return f"gpu-query-error {exc}"
    util = [row.split(",")[1].strip().replace(" %", "") for row in rows]
    used = [row.split(",")[2].strip().replace(" MiB", "") for row in rows]
    return f"gpu_util={util} gpu_mem_MiB={used}"


def sampler(stop: threading.Event) -> None:
    with RES.open("w") as handle:
        while not stop.is_set():
            handle.write(f"{time.strftime('%H:%M:%S')} {host_memory()} {gpu_state()}\n")
            handle.flush()
            stop.wait(15)


def health() -> int:
    try:
        with urllib.request.urlopen(f"{URL}/health", timeout=5) as response:
            return response.status
    except Exception:  # noqa: BLE001
        return 0


def tail(path: Path, count: int) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-count:])
    except Exception:  # noqa: BLE001
        return "(no log)"


def request(label: str, content: str, max_tokens: int = 256) -> None:
    body = json.dumps({
        "model": SERVED,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=1800).read().decode())
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        log(f"RESULT {label} elapsed={time.time() - started:.1f}s "
            f"finish={choice.get('finish_reason')}")
        log(f"  content={(message.get('content') or '')[:300]!r}")
        log(f"  usage={data.get('usage')}")
    except Exception as exc:  # noqa: BLE001
        log(f"RESULT {label} ERROR {type(exc).__name__} {str(exc)[:300]}")


def main() -> int:
    log(f"ENTRY-START {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log(f"source={SRC} model={MODEL} port={PORT}")
    if not Path(SRC, "sglang").is_dir():
        log(f"ERROR: 镜像内没有 SGLang 源码：{SRC}")
        return 2
    if not Path(MODEL).is_dir():
        log(f"ERROR: 模型目录不存在：{MODEL}")
        return 2

    stop = threading.Event()
    threading.Thread(target=sampler, args=(stop,), daemon=True).start()

    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + ":" + env.get("PYTHONPATH", "")
    # 该检查点的 YaRN 在 text_config 里，SGLang 推导出原生 262144；不放行就无法请求 1M。
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-m", "sglang.launch_server", *ARGS],
        stdout=LOG.open("w"), stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )
    log(f"launched pid={proc.pid}")

    ready = False
    for tick in range(1, 71):
        time.sleep(15)
        code = health()
        alive = proc.poll() is None
        log(f"[tick {tick}] {time.strftime('%H:%M:%S')} health={code} alive={alive} "
            f"{host_memory()} {gpu_state()}")
        if tick % 4 == 0:
            log("  srv: " + " | ".join(tail(LOG, 4).splitlines()[-4:])[:700])
        if code == 200:
            ready = True
            log("HEALTH-OK")
            break
        if not alive:
            log(f"PROCESS-GONE rc={proc.returncode}")
            break

    log("--- server log tail ---")
    log(tail(LOG, 70))

    if ready:
        try:
            with urllib.request.urlopen(f"{URL}/v1/models", timeout=30) as response:
                log("MODELS " + response.read().decode()[:400])
        except Exception as exc:  # noqa: BLE001
            log(f"MODELS-ERROR {exc}")
        if SMOKE:
            request("short", "Reply with exactly: pong", 256)
            # 长请求按交付验收要求做到 ~16k token：45 字符 × 1600 ≈ 72k 字符 ≈ 15–16k token。
            # （早期版本只填了 380 次，约 3.9k token，却把这一步叫做 long16k——名不副实，已纠正。）
            filler = "The quick brown fox jumps over the lazy dog. " * 1600
            log(f"smoke long prompt: {len(filler)} chars (~16k tokens expected)")
            request("long16k", filler + " Now reply with exactly: longpong", 256)
            log("--- sampler tail ---")
            log(tail(RES, 8))

    try:
        proc.terminate()
        time.sleep(10)
        if proc.poll() is None:
            proc.kill()
    except Exception:  # noqa: BLE001
        pass
    stop.set()
    log(f"ENTRY-END ready={ready}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
