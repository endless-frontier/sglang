#!/usr/bin/env python3
"""EAS CPU service: authenticate at the EAS gateway and proxy to a DLC backend."""

from __future__ import annotations

import argparse
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import signal
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit


HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

METRICS_PATHS = {
    "/ops/metrics/router": "router",
    "/ops/metrics/prefill": "prefill",
    "/ops/metrics/decode": "decode",
    "/ops/metrics/standard": "standard",
}
ABORT_DECODE_PATH = "/ops/abort/decode"
ABORT_DECODE_ALL_PATH = "/ops/abort/decode-all"
ABORT_RID_RE = re.compile(r"[0-9a-f]{32}")
CONTROL_REQUEST_BODY_BYTES = 64 * 1024


class LimitedReader:
    """Expose exactly one HTTP request body without buffering it in memory."""

    def __init__(self, source: Any, length: int) -> None:
        if length < 0:
            raise ValueError("length must be non-negative")
        self.source = source
        self.remaining = length
        self.bytes_read = 0

    def read(self, amount: int = -1) -> bytes:
        if self.remaining == 0:
            return b""
        if amount is None or amount < 0:
            amount = self.remaining
        wanted = min(amount, self.remaining)
        chunk = self.source.read(wanted)
        if not chunk:
            raise EOFError("client request body ended before Content-Length")
        if len(chunk) > wanted:
            raise RuntimeError("request body reader returned more data than requested")
        self.remaining -= len(chunk)
        self.bytes_read += len(chunk)
        return chunk


def _read_buffered_body(source: Any, length: int) -> bytes:
    reader = LimitedReader(source, length)
    chunks = []
    while reader.remaining:
        chunks.append(reader.read(min(reader.remaining, 64 * 1024)))
    return b"".join(chunks)


def _backend(ready_file: Path) -> tuple[str, int]:
    value = json.loads(ready_file.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "qwen38-flash-next-h200-ready-v1":
        raise ValueError("ready file schema 不受支持")
    endpoint = str(value.get("endpoint") or "")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise ValueError("DLC backend endpoint 必须是显式 http://host:port")
    return parsed.hostname, parsed.port


def _metrics_backend(ready_file: Path, request_path: str) -> tuple[str, int, str]:
    key = METRICS_PATHS.get(request_path)
    index = 0
    if key is None:
        match = re.fullmatch(r"/ops/metrics/(prefill|decode)/([0-9]+)", request_path)
        if not match:
            raise ValueError("metrics path 不受支持")
        key = match.group(1)
        index = int(match.group(2))
    value = json.loads(ready_file.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "qwen38-flash-next-h200-ready-v1":
        raise ValueError("ready file schema 不受支持")
    groups = value.get("metrics_endpoint_groups") or {}
    endpoints = groups.get(key) if isinstance(groups, dict) else None
    if isinstance(endpoints, list) and index < len(endpoints):
        endpoint = str(endpoints[index] or "")
    elif index == 0:
        endpoint = str((value.get("metrics_endpoints") or {}).get(key) or "")
    else:
        endpoint = ""
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise ValueError(f"{key} metrics endpoint 尚不可用")
    return parsed.hostname, parsed.port, parsed.path or "/metrics"


def _decode_control_backend(ready_file: Path) -> tuple[str, int, str]:
    value = json.loads(ready_file.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "qwen38-flash-next-h200-ready-v1":
        raise ValueError("ready file schema 不受支持")
    endpoint = str((value.get("metrics_endpoints") or {}).get("decode") or "")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise ValueError("decode endpoint 尚不可用")
    return parsed.hostname, parsed.port, "/abort_request"


def _ready_run_id(ready_file: Path) -> str:
    value = json.loads(ready_file.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "qwen38-flash-next-h200-ready-v1":
        raise ValueError("ready file schema 不受支持")
    run_id = str(value.get("run_id") or "")
    if not run_id:
        raise ValueError("ready file 缺少 run_id")
    return run_id


def _router_active_requests(ready_file: Path, timeout: float = 5) -> list[int]:
    host, port, path = _metrics_backend(ready_file, "/ops/metrics/router")
    connection = HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read(2 * 1024 * 1024)
        if response.status != 200:
            raise ValueError("router metrics 不可用")
    finally:
        connection.close()
    values: list[int] = []
    for raw_line in body.decode("utf-8", errors="replace").splitlines():
        if not raw_line.startswith("smg_worker_requests_active{"):
            continue
        try:
            value = float(raw_line.rsplit(" ", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("router active metric 格式无效") from exc
        if value < 0 or not value.is_integer():
            raise ValueError("router active metric 不是非负整数")
        values.append(int(value))
    if not values:
        raise ValueError("router active metric 缺失")
    return values


def _backend_health(ready_file: Path, timeout: float = 3) -> bool:
    try:
        host, port = _backend(ready_file)
        connection = HTTPConnection(host, port, timeout=timeout)
        connection.request("GET", "/health")
        response = connection.getresponse()
        response.read()
        connection.close()
        return response.status == 200
    except Exception:
        return False


def _is_event_stream(response: Any) -> bool:
    content_type = str(response.getheader("Content-Type", "") or "").lower()
    return "text/event-stream" in content_type


def _response_chunks(response: Any, *, event_stream: bool) -> Iterator[bytes]:
    # HTTPResponse.read(amt) deliberately accumulates across HTTP chunks until
    # amt bytes or EOF.  For SSE this turns token chunks into large bursts.
    # read1() performs at most one underlying read and returns buffered data
    # immediately, which preserves upstream streaming cadence.
    reader = response.read1 if event_stream else response.read
    while True:
        chunk = reader(64 * 1024)
        if not chunk:
            return
        yield chunk


class ProxyStats:
    """Small local metric set that never records request bodies or credentials."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_requests = 0
        self.max_active_requests = 0
        self.requests_total = 0
        self.upstream_errors_total = 0
        self.request_duration_seconds_sum = 0.0
        self.request_body_bytes_total = 0
        self.request_body_bytes_max = 0
        self.request_body_rejections_total = 0

    def begin(self) -> float:
        with self._lock:
            self.requests_total += 1
            self.active_requests += 1
            self.max_active_requests = max(
                self.max_active_requests, self.active_requests
            )
        return time.monotonic()

    def finish(self, started_at: float, *, upstream_error: bool) -> None:
        elapsed = max(0.0, time.monotonic() - started_at)
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            if upstream_error:
                self.upstream_errors_total += 1
            self.request_duration_seconds_sum += elapsed

    def observe_body(self, length: int, *, rejected: bool) -> None:
        with self._lock:
            self.request_body_bytes_total += max(0, length)
            self.request_body_bytes_max = max(self.request_body_bytes_max, length)
            if rejected:
                self.request_body_rejections_total += 1

    def prometheus(self) -> bytes:
        with self._lock:
            rows = (
                ("qwen38_proxy_active_requests", self.active_requests),
                ("qwen38_proxy_max_active_requests", self.max_active_requests),
                ("qwen38_proxy_requests_total", self.requests_total),
                ("qwen38_proxy_upstream_errors_total", self.upstream_errors_total),
                (
                    "qwen38_proxy_request_duration_seconds_sum",
                    self.request_duration_seconds_sum,
                ),
                (
                    "qwen38_proxy_request_body_bytes_total",
                    self.request_body_bytes_total,
                ),
                (
                    "qwen38_proxy_request_body_bytes_max",
                    self.request_body_bytes_max,
                ),
                (
                    "qwen38_proxy_request_body_rejections_total",
                    self.request_body_rejections_total,
                ),
            )
        return (
            "\n".join(f"{name} {value}" for name, value in rows) + "\n"
        ).encode("ascii")


class HighConcurrencyHTTPServer(ThreadingHTTPServer):
    # TCPServer defaults to a very small listen backlog on Python <= 3.14.
    # The EAS gateway can deliver large bursts, so make the accept queue explicit.
    request_queue_size = 1024
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Qwen38EasProxy/5"

    @property
    def ready_file(self) -> Path:
        return self.server.ready_file  # type: ignore[attr-defined]

    @property
    def stats(self) -> ProxyStats:
        return self.server.stats  # type: ignore[attr-defined]

    def setup(self) -> None:
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, format: str, *args: Any) -> None:
        # Never log request headers or bodies because they may include EAS tokens/prompts.
        print(
            json.dumps(
                {
                    "client": self.client_address[0],
                    "method": self.command,
                    "path": self.path.split("?", 1)[0],
                    "message": format % args,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def _json(self, status: int, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def _health(self) -> None:
        healthy = _backend_health(self.ready_file)
        self._json(200 if healthy else 503, {"ready": healthy})

    def _proxy_metrics(self) -> None:
        payload = self.stats.prometheus()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def _proxy(
        self,
        *,
        metrics_path: str | None = None,
        abort_decode: bool = False,
        abort_decode_all: bool = False,
    ) -> None:
        started_at = self.stats.begin()
        upstream_error = False
        try:
            if abort_decode or abort_decode_all:
                host, port, upstream_path = _decode_control_backend(self.ready_file)
            elif metrics_path is None:
                host, port = _backend(self.ready_file)
                upstream_path = self.path
            else:
                host, port, upstream_path = _metrics_backend(
                    self.ready_file, metrics_path
                )
        except Exception as exc:
            upstream_error = True
            self._json(503, {"error": "backend_not_ready", "detail": type(exc).__name__})
            self.stats.finish(started_at, upstream_error=upstream_error)
            return
        length_text = self.headers.get("Content-Length", "0")
        try:
            length = int(length_text)
        except ValueError:
            self._json(400, {"error": "invalid_content_length"})
            self.stats.finish(started_at, upstream_error=False)
            return
        control_request = abort_decode or abort_decode_all
        if length < 0:
            self._json(400, {"error": "invalid_content_length"})
            self.stats.finish(started_at, upstream_error=False)
            return
        # Model-facing requests are deliberately not capped here.  This proxy
        # streams exactly Content-Length bytes and lets the provider gateway and
        # the version-matched SGLang Router enforce their own effective limits.
        # The small, project-owned abort API remains buffered because its JSON
        # must be fully authenticated and validated before forwarding.
        if control_request and length > CONTROL_REQUEST_BODY_BYTES:
            self.stats.observe_body(length, rejected=True)
            self._json(
                413,
                {
                    "error": "request_too_large",
                    "max_request_body_bytes": CONTROL_REQUEST_BODY_BYTES,
                },
            )
            self.stats.finish(started_at, upstream_error=False)
            return
        self.stats.observe_body(length, rejected=False)
        try:
            body = (
                _read_buffered_body(self.rfile, length)
                if control_request and length
                else None
            )
        except EOFError:
            self._json(400, {"error": "incomplete_request_body"})
            self.stats.finish(started_at, upstream_error=False)
            return
        if abort_decode:
            try:
                value = json.loads((body or b"").decode("utf-8"))
                if (
                    not isinstance(value, dict)
                    or set(value) != {"rid"}
                    or not ABORT_RID_RE.fullmatch(str(value.get("rid") or ""))
                ):
                    raise ValueError("rid")
                body = json.dumps(
                    {"rid": str(value["rid"])},
                    separators=(",", ":"),
                ).encode("ascii")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                self._json(400, {"error": "invalid_abort_request"})
                self.stats.finish(started_at, upstream_error=False)
                return
        elif abort_decode_all:
            try:
                value = json.loads((body or b"").decode("utf-8"))
                if (
                    not isinstance(value, dict)
                    or set(value) != {"run_id", "expected_active_requests"}
                    or not isinstance(value.get("expected_active_requests"), int)
                    or isinstance(value.get("expected_active_requests"), bool)
                    or int(value["expected_active_requests"]) < 1
                    or str(value.get("run_id") or "") != _ready_run_id(self.ready_file)
                ):
                    raise ValueError("confirmation")
                expected = int(value["expected_active_requests"])
                observed = _router_active_requests(self.ready_file)
                if any(item != expected for item in observed):
                    self._json(
                        409,
                        {
                            "error": "active_request_count_mismatch",
                            "observed_worker_active": observed,
                        },
                    )
                    self.stats.finish(started_at, upstream_error=False)
                    return
                body = json.dumps(
                    {
                        "abort_all": True,
                        "abort_message": (
                            "owned validation cleanup for "
                            + str(value["run_id"])
                        ),
                    },
                    separators=(",", ":"),
                ).encode("ascii")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                self._json(400, {"error": "invalid_abort_all_confirmation"})
                self.stats.finish(started_at, upstream_error=False)
                return
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP
            and key.lower() not in {"host", "authorization", "content-length"}
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))
            upstream_body: Any = body
        elif length:
            # http.client accepts a readable object. LimitedReader returns EOF
            # exactly at Content-Length, so a large JSON/image upload is sent
            # incrementally and can never consume hundreds of MiB of proxy RAM.
            headers["Content-Length"] = str(length)
            upstream_body = LimitedReader(self.rfile, length)
        else:
            upstream_body = None
        # Long-thinking FSR requests can legitimately stream for well over
        # 30 minutes.  Keep this aligned with the Router request timeout and
        # the EAS RPC keepalive so the proxy is not the first layer to cut an
        # otherwise healthy chunked response.
        connection = HTTPConnection(host, port, timeout=7200)
        headers_sent = False
        try:
            connection.request(
                self.command,
                upstream_path,
                body=upstream_body,
                headers=headers,
            )
            upstream = connection.getresponse()
            event_stream = _is_event_stream(upstream)
            self.send_response(upstream.status, upstream.reason)
            for key, value in upstream.getheaders():
                if key.lower() in HOP_BY_HOP or key.lower() in {"content-length"}:
                    continue
                self.send_header(key, value)
            if event_stream:
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            headers_sent = True
            for chunk in _response_chunks(upstream, event_stream=event_stream):
                if event_stream:
                    self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()
            if event_stream:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except Exception as exc:
            upstream_error = True
            if not headers_sent and not self.wfile.closed:
                try:
                    self._json(502, {"error": "backend_proxy_error", "detail": type(exc).__name__})
                except Exception:
                    pass
        finally:
            connection.close()
            self.close_connection = True
            self.stats.finish(started_at, upstream_error=upstream_error)

    def do_GET(self) -> None:  # noqa: N802
        request_path = self.path.split("?", 1)[0]
        if request_path in {"/", "/health"}:
            self._health()
        elif request_path == "/ops/metrics/proxy":
            self._proxy_metrics()
        elif request_path in METRICS_PATHS or re.fullmatch(
            r"/ops/metrics/(prefill|decode)/[0-9]+", request_path
        ):
            self._proxy(metrics_path=request_path)
        else:
            self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        request_path = self.path.split("?", 1)[0]
        if request_path == ABORT_DECODE_PATH:
            self._proxy(abort_decode=True)
        elif request_path == ABORT_DECODE_ALL_PATH:
            self._proxy(abort_decode_all=True)
        else:
            self._proxy()

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port 必须位于 1..65535")
    server = HighConcurrencyHTTPServer((args.host, args.port), ProxyHandler)
    server.ready_file = args.ready_file  # type: ignore[attr-defined]
    server.stats = ProxyStats()  # type: ignore[attr-defined]

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        json.dumps(
            {
                "event": "proxy_listening",
                "host": args.host,
                "port": args.port,
                "ready_file": str(args.ready_file),
                "server_version": ProxyHandler.server_version,
                "request_queue_size": server.request_queue_size,
                "model_request_body_limit": "none_in_eas_proxy",
                "control_request_body_bytes": CONTROL_REQUEST_BODY_BYTES,
                "request_body_forwarding": "streamed",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever(poll_interval=0.5)
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
