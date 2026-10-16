"""Offline checks for explicit deployment paths; never connect to a worker."""

import importlib.util
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load_watchdog():
    spec = importlib.util.spec_from_file_location(
        "portable_watchdog", HERE / "watch_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PortableConfigTests(unittest.TestCase):
    def test_worker_without_model_stops_before_imports(self):
        with patch.dict(os.environ, {}, clear=True):
            result = subprocess.run(
                [
                    "bash",
                    str(HERE / "run_qwen38_flash_next_yarn_1m_pd_worker.sh"),
                    "prefill",
                    "192.0.2.1",
                    "41000",
                    "8998",
                    "--check-only",
                ],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Set QWEN38_MODEL_PATH", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_benchmark_without_tokenizer_stops_before_imports(self):
        with patch.dict(os.environ, {}, clear=True):
            result = subprocess.run(
                [sys.executable, str(HERE / "bench_pd.py"), "--input-tokens", "1"],
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("set --tokenizer or QWEN38_MODEL_PATH", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_missing_ssh_key_fails_before_paramiko(self):
        with patch.dict(os.environ, {}, clear=True):
            watchdog = load_watchdog()
        with self.assertRaisesRegex(RuntimeError, "Set QWEN38_WATCH_SSH_KEY"):
            watchdog.get_key()

    def test_missing_model_does_not_contact_or_stop_remote_worker(self):
        with patch.dict(os.environ, {}, clear=True):
            watchdog = load_watchdog()
            with patch.object(watchdog, "ssh_run") as ssh:
                ok, detail = watchdog.restart_remote(
                    watchdog.Node("prefill-1", "prefill", "192.0.2.1", 41000)
                )
                ssh.assert_not_called()
        self.assertFalse(ok)
        self.assertIn("QWEN38_MODEL_PATH", detail)

    def test_remote_restart_forwards_only_quoted_path_settings(self):
        settings = {
            "QWEN38_MODEL_PATH": "/shared/model with spaces/'$(touch NOT_EXECUTED)'/hf",
            "QWEN38_SGLANG_SOURCE": "/shared/source tree",
            "QWEN38_MOE_CONFIG_DIR": "/shared/moe configs",
            "UNRELATED_PRIVATE_SETTING": "do-not-forward",
        }
        with patch.dict(os.environ, settings, clear=True):
            watchdog = load_watchdog()
            with patch.object(
                watchdog, "ssh_run", return_value=(0, "sglang.launch_server", "")
            ) as ssh:
                ok, _ = watchdog.restart_remote(
                    watchdog.Node("prefill-1", "prefill", "192.0.2.1", 41000)
                )
        self.assertTrue(ok)
        script = ssh.call_args.args[1]
        self.assertEqual(
            subprocess.run(["bash", "-n"], input=script, text=True).returncode, 0
        )
        # Parse the generated launch command; do not execute its remote process cleanup.
        launch = next(
            line for line in script.splitlines() if line.startswith("nohup setsid env ")
        )
        words = shlex.split(launch.removesuffix("\\"))
        bash_at = words.index("bash", 3)
        self.assertEqual(
            dict(item.split("=", 1) for item in words[3:bash_at]),
            {
                key: value
                for key, value in settings.items()
                if key != "UNRELATED_PRIVATE_SETTING"
            },
        )
        self.assertNotIn("do-not-forward", script)


if __name__ == "__main__":
    unittest.main()
