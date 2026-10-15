import __future__

import ast
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as N

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "runtime")]
import hashlib
import tempfile

import apply_runtime_patch as patcher
import prepare_deployment as preparation

SOURCE = next(p for p in HERE.parents if (p / "python/sglang/srt").is_dir())
TEMP = tempfile.TemporaryDirectory(prefix="pd-recipe-cpu-")
ROOT = Path(TEMP.name) / "source"
ROOT.mkdir()
manifest = json.loads((HERE / "source-manifest.json").read_text())
for rel, hashes in manifest["files"].items():
    if hashes["before"] is not None:
        q = ROOT / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        # Tests must also work after applying the recipe to the working tree.
        raw = subprocess.check_output(["git", "show", "HEAD:" + rel], cwd=SOURCE)
        assert hashlib.sha256(raw).hexdigest() == hashes["before"]
        q.write_bytes(raw)
assert patcher.install(ROOT) == "installed"
PIN = ROOT / "python/sglang/srt"
H = HERE
ns = {}
tree = ast.parse((PIN / "mem_cache/prefill_budget.py").read_text())
exec(
    compile(
        ast.Module(
            body=[
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "PrefillBudget"
            ],
            type_ignores=[],
        ),
        "actual_fixed_budget",
        "exec",
        flags=__future__.annotations.compiler_flag,
    ),
    ns,
)
Budget = ns["PrefillBudget"]
ns = {}
exec((PIN / "managers/scheduler_components/prefill_capacity.py").read_text(), ns)
debt = ns["paused_completion_debt"]
sys.path.insert(0, str(H))
from pd_progress import ProgressGuard


def budget(free):
    return Budget(
        N(page_size=64, available_size=lambda: free),
        N(supports_mamba=lambda: True, full_evictable_size=lambda: 0),
    )


def req(key, n, allocated, end=None):
    return N(
        rid=key,
        origin_input_ids=range(n),
        kv=N(kv_allocated_len=allocated),
        extend_range=N(end=allocated if end is None else end),
        sampling_params=N(max_new_tokens=1),
        output_ids=[],
        finished=lambda: False,
        is_retracted=False,
    )


class Tests(unittest.TestCase):
    def test_actual_4160_vs_64_boundary(self):
        b = budget(64)
        self.assertIsNone(
            b.fit_chunk(
                extend_input_len=10000,
                max_new_tokens=0,
                chunk_limit=b.available_chunk_tokens(4096),
            )
        )
        b.reserve(64, 0)
        self.assertEqual(b.remaining_total, -64)
        self.assertIsNone(b.available_chunk_tokens(4096))

    def test_actual_10000_random_allocator_reservations(self):
        rng = random.Random(20261015)
        for _ in range(10000):
            free = rng.randrange(0, 100000) // 64 * 64
            b = budget(free)
            total = 0
            for _ in range(4):
                remaining = rng.randrange(1, 30000)
                out = rng.randrange(1, 32000)
                limit = b.available_chunk_tokens(4096)
                fit = (
                    b.fit_chunk(
                        extend_input_len=remaining,
                        max_new_tokens=out,
                        chunk_limit=limit,
                    )
                    if limit
                    else None
                )
                if fit:
                    total += (
                        ((fit + 63) // 64) * 64 + 64 + (out if fit >= remaining else 0)
                    )
                    self.assertLessEqual(total, free)
                    b.reserve(fit, out if fit >= remaining else 0)

    def test_full_completion_capacity_prevents_pause_deadlock(self):
        a = req("a", 1000000, 4096)
        b = req("b", 1000000, 4096)
        used = 8192
        capacity = 2617472
        self.assertGreater(used + debt([a, b], [], 64) + 1000000, capacity)
        self.assertLess(used + debt([a], [], 64) + 1000000, capacity)

    def test_debt_deduplicates_and_uses_selected_span(self):
        r = req("a", 10000, 4096, 8192)
        self.assertEqual(debt([r, r], [r], 64), 1921)
        r.is_retracted = True
        self.assertEqual(debt([r], [], 64), 0)

    def test_fair_order_and_budget_unchanged(self):
        new = PIN / "managers/scheduler_components/fair_prefill.py"
        fn = lambda p: next(
            n
            for n in ast.parse(p.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "order_requests"
        )
        text = new.read_text()
        self.assertEqual(
            hashlib.sha256(ast.get_source_segment(text, fn(new)).encode()).hexdigest(),
            "9c1766d9b201bf76b40c706e4f3863e2abbbb8dffcaf4dcf452e7d38f66a4512",
        )

    def test_paused_not_native_idle_or_health_idle(self):
        t = ast.parse((PIN / "managers/scheduler.py").read_text())
        c = next(
            n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
        )
        f = next(
            n
            for n in c.body
            if isinstance(n, ast.FunctionDef) and n.name == "is_fully_idle"
        )
        scope = {}
        exec(
            compile(
                ast.Module(body=[f], type_ignores=[]),
                "actual_idle",
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            scope,
        )
        for health in (False, True):
            self.assertFalse(scope["is_fully_idle"](N(fair_paused=[1]), health))

    def sample(self, wall=1000, busy=100, paused=2):
        return {
            "timestamp": wall,
            "total_prefill_busy_us": busy,
            "pd_admission": {
                "fair_prefill": {"paused_contexts": paused, "last_completed_ts": 900}
            },
        }

    def test_fresh_300_second_stall(self):
        g = ProgressGuard(300)
        for i in range(0, 300, 10):
            self.assertFalse(g.observe(self.sample(1000 + i), i, 1000 + i))
        self.assertTrue(g.observe(self.sample(1300), 300, 1300))

    def test_busy_and_idle_are_not_stalls(self):
        g = ProgressGuard(300)
        for i in range(0, 1000, 10):
            self.assertFalse(
                g.observe(self.sample(1000 + i, busy=100 + i), i, 1000 + i)
            )
        self.assertFalse(g.observe(self.sample(2000, paused=0), 1000, 2000))

    def test_unknown_breaks_continuity(self):
        for bad in (
            None,
            {},
            self.sample(1000),
            dict(self.sample(), timestamp=float("nan")),
        ):
            g = ProgressGuard(300)
            g.observe(self.sample(), 0, 1000)
            self.assertIsNone(g.observe(bad, 300, 1300))
            self.assertFalse(g.observe(self.sample(1310), 310, 1310))

    def test_invalid_numeric_is_not_zero_or_stall(self):
        for bad in ("2", True, -1, float("inf"), float("nan")):
            g = ProgressGuard()
            s = self.sample()
            s["pd_admission"]["fair_prefill"]["paused_contexts"] = bad
            self.assertIsNone(g.observe(s, 0, 1000))

    def test_actual_native_exception_kills_group_despite_background_thread(self):
        pin = PIN / "managers/scheduler.py"
        fn = next(
            n
            for n in ast.parse(pin.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "run_scheduler_process"
        )
        handler = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.ExceptHandler)
            and any(
                isinstance(x, ast.If) and "SGLANG_KILLPG" in ast.dump(x) for x in n.body
            )
        )
        module = ast.Module(body=handler.body, type_ignores=[])
        program = """import os,signal,threading
from types import SimpleNamespace as N
logger=N(error=lambda x:None)
get_exception_traceback=lambda:'bounded synthetic exception'
parent_process=N(send_signal=lambda x:None)
envs=N(SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION=N(get=lambda:True))
threading.Thread(target=lambda:threading.Event().wait(),daemon=False).start()
""" + ast.unparse(module)
        p = subprocess.Popen(
            [sys.executable, "-c", program],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        p.communicate(timeout=10)
        self.assertEqual(p.returncode, -9)

    def test_actual_watchdog_unknown_skips_fault_recovery(self):
        t = ast.parse((H / "runtime/node_supervisor.py").read_text())
        main = next(
            n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        loop = next(
            n
            for n in ast.walk(main)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id == "n"
            and any(
                isinstance(x, ast.Assign)
                and any(
                    isinstance(y, ast.Name) and y.id == "evidence" for y in x.targets
                )
                for x in n.body
            )
        )
        statements = loop.body[1:]
        calls = []
        n = N(url="private", state="HEALTHY", fail_count=2, first_fail_ts=1)
        scope = {
            "n": n,
            "probe_health": lambda u: (None, "unknown-native-progress"),
            "health_cache": {},
            "w": N(check_node=lambda *a: calls.append(a)),
            "nodes": [n],
        }
        exec(
            compile(
                ast.Module(body=statements, type_ignores=[]),
                "actual_watchdog_unknown",
                "exec",
            ),
            scope,
        )
        self.assertEqual(n.state, "UNKNOWN")
        self.assertEqual(n.fail_count, 0)
        self.assertEqual(calls, [])


class DeploymentTests(unittest.TestCase):
    def test_install_is_idempotent(self):
        self.assertEqual(patcher.install(ROOT), "already_installed")

    def test_drift_rejected_without_overwriting(self):
        relative = next(k for k, v in manifest["files"].items() if v["before"])
        p = ROOT / relative
        original = p.read_bytes()
        try:
            p.write_bytes(original + b"\n# unexpected change\n")
            with self.assertRaises(RuntimeError):
                patcher.install(ROOT)
            self.assertTrue(p.read_bytes().endswith(b"# unexpected change\n"))
        finally:
            p.write_bytes(original)

    def args(self, home):
        model = home / "hf"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "weights.safetensors").write_bytes(b"fixture")
        return N(
            source=ROOT,
            model=model,
            run=home / "run",
            model_name="test-model",
            prefill_ips=["192.0.2." + str(i) for i in range(1, 5)],
            decode_ips=["192.0.2." + str(i) for i in range(5, 9)],
        )

    def test_prepare_preserves_export_and_copies_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.args(Path(td))
            result = preparation.prepare(args)
            self.assertEqual(result["nodes"], 8)
            self.assertEqual(result["router"], "http://192.0.2.5:40000/v1")
            self.assertTrue((args.run / "model/weights.safetensors").is_symlink())
            (args.run / "model/config.json").write_text('{"private":true}')
            self.assertEqual((args.model / "config.json").read_text(), "{}")
            self.assertTrue((args.run / "node_supervisor.py").is_file())
            self.assertTrue((args.run / "watch_worker_original.py").is_file())
            config = json.loads((args.run / "deployment-settings.json").read_text())
            self.assertTrue(config["require_worker_load_admission_ack"])
            self.assertEqual(
                [
                    config[k]
                    for k in (
                        "prefill_batch_contexts",
                        "prefill_quantum_ceiling",
                        "prefill_batch_token_budget",
                    )
                ],
                [2, 4096, 8192],
            )
            before = (args.run / "deployment-settings.json").read_bytes()
            with self.assertRaises(ValueError):
                preparation.prepare(args)
            self.assertEqual(
                (args.run / "deployment-settings.json").read_bytes(), before
            )

    def test_duplicate_workers_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.args(Path(td))
            args.decode_ips[0] = args.prefill_ips[0]
            with self.assertRaises(ValueError):
                preparation.prepare(args)
            self.assertFalse(args.run.exists())


if __name__ == "__main__":
    unittest.main()
