#!/usr/bin/env python3
"""看门狗判定逻辑的离线单测（不碰真机，靠打桩驱动状态机）。

两个用例直接来自 2026-09-17 的真实事故：

  A. prefill-2 真故障：scheduler 在 19:37:33 心跳停掉、20s 后 /health 返 503、
     父进程 sglang.launch_server 还活着、日志里再也没有新的 Prefill batch
     → 期望：判定 process-wedged 并重启。

  B. prefill-3 假故障：prefill-2 停了以后它的 78% 流量压过来，HTTP 线程被大 chunk
     堵死，/health 连续超时，但日志里 Prefill batch 一直在涨
     → 期望：判定为「忙」而非僵死，**不重启**。

跑法：python3 test_watch_worker.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys


def load_module():
    path = pathlib.Path(__file__).with_name("watch_worker.py")
    spec = importlib.util.spec_from_file_location("watch_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import time as _time


def _ago(seconds):
    """模拟「最后一批活发生在 seconds 秒之前」。"""
    return int(_time.time()) - seconds


def drive(w, node, health, proc, work, cycles=6):
    """按给定信号连续跑若干轮，返回 (是否重启过, 日志行)。"""
    w.http_health = lambda url, timeout=None: health
    w.remote_process_alive = lambda ip: proc
    w.remote_last_work_epoch = lambda ip, role: (int(_time.time()) if work is None else work)
    restarted = []
    w.restart_remote = lambda n: (restarted.append(n.name), (True, "stub"))[1]
    lines = []
    w.log = lambda msg, also_print=False: lines.append(msg)
    nodes = [node]
    for _ in range(cycles):
        w.check_node(node, nodes, dry_run=False)
    return bool(restarted), lines


def main():
    w = load_module()
    w.FAIL_THRESHOLD = 3
    w.MIN_SOFT_DURATION = 0        # 单测里不等真实时间
    w.COOLDOWN = 0
    w.MAX_CONCURRENT_RESTARTS = 2
    w.STARTUP_TIMEOUT = 900
    failures = []

    # --- A: 真故障（503 + 进程在 + 没在干活）→ 应该重启 ---
    node = w.Node("prefill-2", "prefill", "10.0.1.125", 41000)
    ok, lines = drive(w, node, (False, "http-503"), True, _ago(600))
    print("A 真故障（503，进程在，最近 0 批活）")
    print("   最后一行:", lines[-1])
    print("   -> 重启:", ok)
    if not ok:
        failures.append("A 应该重启但没有")

    # --- B: 假故障（超时 + 进程在 + 一直在干活）→ 不应该重启 ---
    node = w.Node("prefill-3", "prefill", "10.0.0.45", 41000)
    ok, lines = drive(w, node, (False, "TimeoutError"), True, None)   # 用 now：失败后还在干活
    print("B 假故障（超时，进程在，最近 42 批活）")
    print("   最后一行:", lines[-1])
    print("   -> 重启:", ok)
    if ok:
        failures.append("B 不应该重启但重启了")

    # --- C: 进程没了 → 应该重启 ---
    node = w.Node("decode-1", "decode", "10.0.1.126", 42000)
    ok, lines = drive(w, node, (False, "URLError"), False, None)
    print("C 进程没了（URLError + pgrep 不到）")
    print("   最后一行:", lines[-1])
    print("   -> 重启:", ok)
    if not ok:
        failures.append("C 应该重启但没有")

    # --- D: SSH 不通（整机挂了）→ 只告警不重启 ---
    node = w.Node("prefill-4", "prefill", "10.0.0.46", 41000)
    ok, lines = drive(w, node, (False, "TimeoutError"), None, None)
    print("D 整机不可达（SSH 不通）")
    print("   最后一行:", lines[-1])
    print("   -> 重启:", ok)
    if ok:
        failures.append("D 不应该重启但重启了")

    # --- E: 单次失败不该触发重启（阈值保护）---
    node = w.Node("prefill-1", "prefill", "10.0.1.120", 41000)
    ok, _ = drive(w, node, (False, "http-503"), True, _ago(600), cycles=2)
    print("E 只失败 2 次（阈值 3）")
    print("   -> 重启:", ok)
    if ok:
        failures.append("E 未到阈值就重启了")

    if failures:
        print("\n失败:", "; ".join(failures))
        return 1
    print("\n全部通过：真故障会重启，忙/不可达/未达阈值不会重启")
    return 0


if __name__ == "__main__":
    sys.exit(main())
