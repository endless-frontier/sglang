#!/usr/bin/env python3
"""Qwen3.8-Flash-Next 1M —— H100 4P4D 看门狗（跑在 Router 那台机器上）

监控 4 台 prefill + 4 台 decode（外加本机 Router），异常时重启对应节点的服务。

判定两类故障（缺一不可）：

  1. process-gone     进程没了 —— SSH 上去 `pgrep sglang.launch_server` 查不到；
                      通常是 OOM / 段错误 / 被人手动 kill。
  2. process-wedged   进程僵死 —— 进程还在，但 `/health` 连续 N 次非 200 或超时。
                      这是 QSA 长上下文故障的形态：进程活着、`/health` 卡住、
                      `/metrics` 还有响应，光靠 pgrep 完全查不出来。
  3. node-unreachable SSH 都连不上（整机挂了）—— 只告警，不做重启（也没法重启）。

为什么用 paramiko 而不是 ssh 命令：这些节点镜像里**没有 ssh 客户端**
（`/usr/bin/ssh` 不存在，apt 也装不了 openssh-client），所以走 Python 的
paramiko + 已有的 id_rsa 私钥，直连内网 `10.0.x.x:22`（已验证 0.1s 可连）。

用法：
    python3 watch_worker.py --once --dry-run     # 单次巡检，只打印不重启
    python3 watch_worker.py --once               # 单次巡检，该重启就重启
    python3 watch_worker.py                      # 常驻（默认 30s 一轮）
    nohup setsid python3 watch_worker.py > /tmp/qwen38_watchdog.out 2>&1 &

常用环境变量（都有默认值，见下面 ENV 一节）：
    QWEN38_WATCH_PREFILL_IPS / QWEN38_WATCH_DECODE_IPS   节点列表（必填）
    QWEN38_WATCH_INTERVAL            巡检间隔秒，默认 30
    QWEN38_WATCH_FAIL_THRESHOLD      /health 连续失败几次判定僵死，默认 3
    QWEN38_WATCH_COOLDOWN            同一节点两次重启最小间隔秒，默认 300（5 分钟）
    QWEN38_WATCH_STARTUP_TIMEOUT     重启后等它起来的窗口秒，默认 900（15 分钟）
    QWEN38_WATCH_MAX_CONCURRENT      同时处于重启中的节点数上限，默认 2（错开 NAS 读盘）
    QWEN38_WATCH_DRY_RUN=1           只判定不重启
    QWEN38_WATCH_MONITOR_ROUTER=0    不监控本机 Router
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def env(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


PREFILL_IPS = env("QWEN38_WATCH_PREFILL_IPS", "<PREFILL1_IP>,<PREFILL2_IP>,<PREFILL3_IP>,<PREFILL4_IP>")
DECODE_IPS = env("QWEN38_WATCH_DECODE_IPS", "<DECODE1_IP>,<DECODE2_IP>,<DECODE3_IP>,<DECODE4_IP>")
PREFILL_PORT = int(env("QWEN38_WATCH_PREFILL_PORT", "41000"))
DECODE_PORT = int(env("QWEN38_WATCH_DECODE_PORT", "42000"))
BOOTSTRAP_PORT = env("QWEN38_WATCH_BOOTSTRAP_PORT", "8998")
ROUTER_PORT = int(env("QWEN38_WATCH_ROUTER_PORT", "40000"))

SSH_KEY = env("QWEN38_WATCH_SSH_KEY", "/mnt/data/xinyuzhu/id_rsa")
SSH_USER = env("QWEN38_WATCH_SSH_USER", "root")
SSH_PORT = int(env("QWEN38_WATCH_SSH_PORT", "22"))

WORKER_DIR = env("QWEN38_WATCH_WORKER_DIR",
                 "/mnt/data/xinyuzhu/sglang/endless-frontier/qwen3.8-next-flash/h100_4p4d")
WORKER_SCRIPT = env("QWEN38_WATCH_WORKER_SCRIPT", "run_qwen38_flash_next_yarn_1m_pd_worker.sh")
ROUTER_SCRIPT = env("QWEN38_WATCH_ROUTER_SCRIPT", "run_qwen38_flash_next_yarn_1m_pd_router.sh")

INTERVAL = float(env("QWEN38_WATCH_INTERVAL", "30"))
FAIL_THRESHOLD = int(env("QWEN38_WATCH_FAIL_THRESHOLD", "3"))
COOLDOWN = float(env("QWEN38_WATCH_COOLDOWN", "300"))
STARTUP_TIMEOUT = float(env("QWEN38_WATCH_STARTUP_TIMEOUT", "900"))
HTTP_TIMEOUT = float(env("QWEN38_WATCH_HTTP_TIMEOUT", "30"))
MAX_CONCURRENT_RESTARTS = int(env("QWEN38_WATCH_MAX_CONCURRENT", "2"))
# 「超时」比「明确返回非 200」弱得多：一台正在猛干活的 prefill，HTTP 线程可能被
# 大 chunk 堵住十几秒，10s 超时会把健康节点误判成僵死（2026-09-17 实测踩过：
# prefill-2 真挂掉后流量压到 prefill-3，prefill-3 立刻被误判重启）。
# 所以：非 200 = 硬失败，按阈值判；超时 = 软失败，还要额外持续 MIN_SOFT_DURATION。
MIN_SOFT_DURATION = float(env("QWEN38_WATCH_MIN_SOFT_DURATION", "180"))
# 进程还活着时，看它「首次失败之后有没有再干过活」：干过 = 忙，没干过 = 僵死。
# 用「首次失败时刻」而不是固定窗口，避免把失败前几分钟的正常干活误判成还活着。
MONITOR_ROUTER = env("QWEN38_WATCH_MONITOR_ROUTER", "1") not in ("0", "false", "no")

LOG_PATH = env("QWEN38_WATCH_LOG", "/tmp/qwen38_watchdog.log")
STATUS_PATH = env("QWEN38_WATCH_STATUS", "/tmp/qwen38_watchdog_status.json")
PID_PATH = env("QWEN38_WATCH_PID", "/tmp/qwen38_watchdog.pid")


def log(msg, also_print=True):
    line = "[%s] %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    if also_print:
        print(line, flush=True)


# ---------------------------------------------------------------------------
# 节点定义
# ---------------------------------------------------------------------------


class Node:
    __slots__ = ("name", "role", "ip", "port", "kind",
                 "fail_count", "last_restart", "restarting_since", "restarts", "state",
                 "first_fail_ts")

    def __init__(self, name, role, ip, port, kind="remote"):
        self.name = name
        self.role = role
        self.ip = ip
        self.port = port
        self.kind = kind          # remote | local（router 在本机）
        self.fail_count = 0
        self.last_restart = 0.0
        self.restarting_since = None
        self.restarts = 0
        self.state = "HEALTHY"
        self.first_fail_ts = None

    @property
    def url(self):
        host = "127.0.0.1" if self.kind == "local" else self.ip
        return "http://%s:%d" % (host, self.port)

    def as_dict(self):
        return {
            "name": self.name, "role": self.role, "ip": self.ip, "port": self.port,
            "state": self.state, "fail_count": self.fail_count, "restarts": self.restarts,
            "last_restart": (datetime.fromtimestamp(self.last_restart).strftime("%Y-%m-%d %H:%M:%S")
                             if self.last_restart else None),
        }


def build_nodes():
    prefills = [ip.strip() for ip in PREFILL_IPS.split(",") if ip.strip()]
    decodes = [ip.strip() for ip in DECODE_IPS.split(",") if ip.strip()]
    missing = [ip for ip in prefills + decodes if ip.startswith("<")]
    if missing:
        raise SystemExit(
            "节点列表还是占位符 %s\n"
            "请设置 QWEN38_WATCH_PREFILL_IPS / QWEN38_WATCH_DECODE_IPS，例如：\n"
            "  export QWEN38_WATCH_PREFILL_IPS=10.0.1.120,10.0.1.125,10.0.0.45,10.0.0.46\n"
            "  export QWEN38_WATCH_DECODE_IPS=10.0.1.126,10.0.1.127,10.0.0.47,10.0.0.48"
            % missing[0])
    nodes = []
    for idx, ip in enumerate(prefills, 1):
        nodes.append(Node("prefill-%d" % idx, "prefill", ip, PREFILL_PORT))
    for idx, ip in enumerate(decodes, 1):
        nodes.append(Node("decode-%d" % idx, "decode", ip, DECODE_PORT))
    if MONITOR_ROUTER:
        nodes.append(Node("router", "router", "127.0.0.1", ROUTER_PORT, kind="local"))
    return nodes


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


def http_health(url, timeout=HTTP_TIMEOUT):
    """返回 (是否健康, 说明)。"""
    req = urllib.request.Request(url + "/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (resp.status == 200), "http-%d" % resp.status
    except urllib.error.HTTPError as exc:
        return False, "http-%d" % exc.code
    except Exception as exc:  # noqa: BLE001  超时/连接被拒/连接重置都算不健康
        return False, type(exc).__name__


# ---------------------------------------------------------------------------
# SSH（paramiko）
# ---------------------------------------------------------------------------

_KEY = None


def get_key():
    global _KEY
    if _KEY is None:
        import paramiko

        try:
            _KEY = paramiko.RSAKey.from_private_key_file(SSH_KEY)
        except Exception:  # noqa: BLE001
            _KEY = paramiko.Ed25519Key.from_private_key_file(SSH_KEY)
    return _KEY


def ssh_run(ip, script, timeout=120):
    """在远端执行一段 bash（脚本从 stdin 喂进去，避免引号地狱）。返回 (rc, out, err)。"""
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(ip, port=SSH_PORT, username=SSH_USER, pkey=get_key(),
                       timeout=15, banner_timeout=30, auth_timeout=30,
                       allow_agent=False, look_for_keys=False)
        stdin, stdout, stderr = client.exec_command("bash -s", timeout=timeout)
        stdin.write(script)
        stdin.channel.shutdown_write()
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err
    finally:
        client.close()


# 远端：数一下 sglang.launch_server 还在不在
PS_COUNT = r"""
ps -eo pid,pgid,args | grep -E '[s]glang\.launch_server' | wc -l
"""


def remote_process_alive(ip):
    """返回 True/False/None（None = SSH 不通，判不了）。"""
    try:
        rc, out, _ = ssh_run(ip, PS_COUNT, timeout=45)
    except Exception:  # noqa: BLE001
        return None
    if rc != 0:
        return None
    try:
        return int(out.strip().splitlines()[-1]) > 0
    except (ValueError, IndexError):
        return None


# 远端：最后一条「真干活」的批次日志的时间戳（epoch 秒）。
# 只看 Prefill/Decode batch，不看日志尾部那些 "Health check failed"
# —— 僵死的服务会一直刷那种行，拿它当活跃度会被骗。
LAST_WORK = r"""
f=/tmp/qwen38___ROLE__.log
[ -f "$f" ] || { echo -1; exit 0; }
last=$(grep -E '(Prefill|Decode) batch' "$f" | tail -1 | grep -oE '^\[20[0-9-]{8} [0-9:]{8}' | tr -d '[')
[ -n "$last" ] && date -d "$last" +%s || echo -1
"""


def remote_last_work_epoch(ip, role):
    """最后一批活的时间（epoch 秒）；-1 = 日志里没有；None = SSH/读取失败。"""
    script = LAST_WORK.replace("__ROLE__", role)
    try:
        rc, out, _ = ssh_run(ip, script, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    if rc != 0:
        return None
    try:
        return int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None




def restart_remote(node):
    """杀掉远端 worker 的整个进程组再重新拉起。返回 (ok, 说明)。"""
    script = r"""
set -u
python3 - <<'PY'
import os, signal, subprocess, time

def targets():
    out = subprocess.run(['ps', '-eo', 'pid,pgid,args'], capture_output=True, text=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, pgid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        args = parts[2]
        if 'sglang.launch_server' in args or 'run_qwen38_flash_next_yarn_1m_pd_worker.sh' in args:
            hits.append((pid, pgid))
    return hits

# 1) 先按进程组 TERM（setsid 起的 worker，PGID == 组长 PID，子进程 sglang::sched 一起带走）
killed = []
for pid, pgid in targets():
    try:
        os.killpg(pgid, signal.SIGTERM)
        killed.append(pid)
    except ProcessLookupError:
        pass
print('TERM sent to pgids of pids:', killed)
time.sleep(8)

# 2) 还赖着的（包括 sglang::* 子进程）直接 KILL
leftover = []
out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout
for line in out.splitlines():
    parts = line.split(None, 1)
    if len(parts) < 2:
        continue
    try:
        pid = int(parts[0])
    except ValueError:
        continue
    if pid == os.getpid():
        continue
    if 'sglang' in parts[1] or 'run_qwen38_flash_next_yarn_1m_pd_worker.sh' in parts[1]:
        leftover.append(pid)
for pid in leftover:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
print('KILL sent to leftover pids:', leftover)
time.sleep(3)
print('remaining:', subprocess.run(['bash','-c',
      "ps -eo pid,args | grep -cE '[s]glang' "], capture_output=True, text=True).stdout.strip())
PY

# 3) 拉起
cd __WORKER_DIR__ || exit 3
nohup setsid bash __WORKER_SCRIPT__ __ROLE__ __IP__ __PORT__ __BOOTSTRAP__ \
    > /tmp/qwen38___ROLE__.log 2>&1 < /dev/null &
sleep 12
echo "--- 启动后进程数 ---"
ps -eo pid,pgid,etime,args | grep -E '[s]glang\.launch_server' | head -1 | cut -c1-160
"""
    script = (script
              .replace("__WORKER_DIR__", shlex.quote(WORKER_DIR))
              .replace("__WORKER_SCRIPT__", shlex.quote(WORKER_SCRIPT))
              .replace("__ROLE__", node.role)
              .replace("__IP__", node.ip)
              .replace("__PORT__", str(node.port))
              .replace("__BOOTSTRAP__", BOOTSTRAP_PORT))
    rc, out, err = ssh_run(node.ip, script, timeout=300)
    tail = " | ".join(out.strip().splitlines()[-3:])
    return rc == 0 and "sglang.launch_server" in out, "%s %s" % (tail, err.strip()[-200:])


def restart_router(node):
    """Router 跑在本机，直接本地重启（注意别把自己杀了）。"""
    me = os.getpid()
    out = subprocess.run(["ps", "-eo", "pid,pgid,args"], capture_output=True, text=True).stdout
    killed = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, pgid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if pid == me or pid == os.getppid():
            continue
        args = parts[2]
        if "sglang_router.launch_router" in args or "sglang::router" in args:
            try:
                os.killpg(pgid, signal.SIGTERM)
                killed.append(pid)
            except ProcessLookupError:
                pass
    time.sleep(5)
    env = dict(os.environ)
    env.setdefault("QWEN38_PD_PREFILL_IPS", PREFILL_IPS)
    env.setdefault("QWEN38_PD_DECODE_IPS", DECODE_IPS)
    logf = open("/tmp/qwen38_router.log", "ab")
    subprocess.Popen(["setsid", "bash", os.path.join(WORKER_DIR, ROUTER_SCRIPT)],
                     cwd=WORKER_DIR, env=env, stdout=logf, stderr=logf,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(10)
    ok, detail = http_health(node.url, timeout=HTTP_TIMEOUT)
    return ok, "killed=%s health=%s" % (killed, detail)


# ---------------------------------------------------------------------------
# 巡检主流程
# ---------------------------------------------------------------------------


def restarting_count(nodes):
    return sum(1 for n in nodes if n.state == "RESTARTING")


def check_node(node, nodes, dry_run):
    healthy, detail = http_health(node.url)
    now = time.time()

    if healthy:
        if node.state == "RESTARTING":
            took = now - (node.restarting_since or now)
            log("%-10s 已恢复（重启后 %.0fs）" % (node.name, took))
        elif node.state == "SUSPECT":
            log("%-10s 恢复正常（此前连续失败 %d 次）" % (node.name, node.fail_count))
        node.state = "HEALTHY"
        node.fail_count = 0
        node.first_fail_ts = None
        node.restarting_since = None
        return

    # 重启中：给足启动窗口（加载 336GB 权重要 5~6 分钟），别在加载过程中又判死
    if node.state == "RESTARTING":
        waited = now - (node.restarting_since or now)
        if waited < STARTUP_TIMEOUT:
            if int(waited) // 60 != int(waited - INTERVAL) // 60:   # 每分钟打一条
                log("%-10s 重启中（已等 %.0fs / %ds，%s）" % (node.name, waited, STARTUP_TIMEOUT, detail))
            return
        log("%-10s 启动窗口 %ds 内没起来，判定重启失败，转入重新判定" % (node.name, int(STARTUP_TIMEOUT)))
        node.state = "SUSPECT"
        node.fail_count = FAIL_THRESHOLD
        node.first_fail_ts = now - (MIN_SOFT_DURATION + 1)

    if node.fail_count == 0:
        node.first_fail_ts = now
    node.fail_count += 1
    node.state = "SUSPECT"

    # 明确拿到 HTTP 响应但非 200（如 503）是硬失败；超时/连接错误是软失败
    hard = detail.startswith("http-")
    kind = "硬失败" if hard else "软失败(超时类)"
    if node.fail_count < FAIL_THRESHOLD:
        log("%-10s 不健康（%s，%s），连续 %d/%d 次" % (node.name, detail, kind, node.fail_count, FAIL_THRESHOLD))
        return

    elapsed = now - (node.first_fail_ts or now)
    if not hard and elapsed < MIN_SOFT_DURATION:
        log("%-10s 不健康（%s，%s）已 %.0fs < %ds → 继续观察（可能是忙，不是僵死）"
            % (node.name, detail, kind, elapsed, int(MIN_SOFT_DURATION)))
        return

    # 够阈值了，先分类故障
    if node.kind == "local":
        reason = "process-wedged" if _local_proc_alive(node) else "process-gone"
    else:
        alive = remote_process_alive(node.ip)
        if alive is None:
            log("%-10s 连续 %d 次不健康（%s），但 SSH 连不上 → 判为 node-unreachable，只告警不重启"
                % (node.name, node.fail_count, detail))
            return
        if not alive:
            reason = "process-gone"
        else:
            # 进程还在：看它「第一次失败之后」有没有再干过活。
            # 干过 → 只是忙/慢（大 chunk 把 HTTP 线程堵住），不是僵死；
            # 一次都没干过 → 真僵死（detokenizer 心跳停掉 / scheduler 卡死）。
            last_work = remote_last_work_epoch(node.ip, node.role)
            if last_work is None:
                log("%-10s 进程在但读不到日志，无法确认是否僵死 → 本轮不重启（下一轮再看）" % node.name)
                return
            # last_work 是整秒（日志只有秒级精度），first_fail_ts 是浮点，
            # 留 1s 容差，免得同一秒内判定被截断误差带偏。
            if last_work + 1 >= (node.first_fail_ts or now):
                log("%-10s /health 不正常（%s）但首次失败后仍在干活（最后一批 %ds 前）→ 判定为「忙」而非僵死，不重启"
                    % (node.name, detail, max(0, int(now - last_work))))
                node.state = "HEALTHY"
                node.fail_count = 0
                node.first_fail_ts = None
                return
            reason = "process-wedged"

    delta = now - node.last_restart
    if node.last_restart and delta < COOLDOWN:
        log("%-10s %s，但距上次重启仅 %.0fs（冷却 %ds）→ 本轮跳过"
            % (node.name, reason, delta, int(COOLDOWN)))
        return

    if restarting_count(nodes) >= MAX_CONCURRENT_RESTARTS:
        log("%-10s %s，但已有 %d 个节点在重启中（上限 %d）→ 本轮跳过"
            % (node.name, reason, restarting_count(nodes), MAX_CONCURRENT_RESTARTS))
        return

    if dry_run:
        log("[dry-run] %-10s %s → 会重启（%s:%d）" % (node.name, reason, node.ip, node.port))
        node.fail_count = 0
        node.first_fail_ts = None
        return

    log("%-10s %s → 开始重启（%s:%d）" % (node.name, reason, node.ip, node.port))
    node.last_restart = now
    node.restarting_since = now
    node.state = "RESTARTING"
    node.restarts += 1
    node.fail_count = 0
    node.first_fail_ts = None
    try:
        if node.kind == "local":
            ok, info = restart_router(node)
        else:
            ok, info = restart_remote(node)
    except Exception as exc:  # noqa: BLE001
        ok, info = False, "%s: %s" % (type(exc).__name__, exc)
    log("%-10s 重启%s：%s" % (node.name, "成功" if ok else "失败", info))


def _local_proc_alive(node):
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    if node.role == "router":
        return ("sglang::router" in out) or ("sglang_router.launch_router" in out)
    return "sglang.launch_server" in out


def write_status(nodes, started):
    payload = {
        "pid": os.getpid(),
        "started": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "interval": INTERVAL,
        "cooldown": COOLDOWN,
        "fail_threshold": FAIL_THRESHOLD,
        "nodes": [n.as_dict() for n in nodes],
    }
    tmp = STATUS_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, STATUS_PATH)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description="Qwen3.8-Flash-Next 4P4D watchdog")
    ap.add_argument("--once", action="store_true", help="只巡检一轮就退出")
    ap.add_argument("--dry-run", action="store_true", help="只判定不重启")
    ap.add_argument("--status", action="store_true", help="打印当前状态文件后退出")
    args = ap.parse_args()

    if args.status:
        try:
            with open(STATUS_PATH, encoding="utf-8") as fh:
                print(fh.read())
        except OSError as exc:
            raise SystemExit("读不到状态文件 %s：%s" % (STATUS_PATH, exc))
        return 0

    dry_run = args.dry_run or env("QWEN38_WATCH_DRY_RUN", "0") not in ("0", "false", "no")
    nodes = build_nodes()

    # 单实例保护
    if not args.once:
        if os.path.exists(PID_PATH):
            try:
                with open(PID_PATH, encoding="utf-8") as fh:
                    old = int(fh.read().strip())
                os.kill(old, 0)
                raise SystemExit("已经有一个看门狗在跑（PID %d，%s）；先 kill 它再启动" % (old, PID_PATH))
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        with open(PID_PATH, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))

    log("看门狗启动：%d 个节点（%s）；间隔 %ds / 阈值 %d / 冷却 %ds / 启动窗口 %ds / 并发上限 %d%s"
        % (len(nodes), ", ".join("%s=%s" % (n.name, n.ip) for n in nodes),
           int(INTERVAL), FAIL_THRESHOLD, int(COOLDOWN), int(STARTUP_TIMEOUT),
           MAX_CONCURRENT_RESTARTS, "；DRY-RUN" if dry_run else ""))

    started = time.time()
    try:
        while True:
            for node in nodes:
                try:
                    check_node(node, nodes, dry_run)
                except Exception as exc:  # noqa: BLE001  单个节点出问题不影响其它节点
                    log("%-10s 巡检异常：%s: %s" % (node.name, type(exc).__name__, exc))
            write_status(nodes, started)
            if args.once:
                break
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        log("收到中断，看门狗退出")
    finally:
        if not args.once:
            try:
                os.unlink(PID_PATH)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
