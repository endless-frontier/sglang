from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schema import inside_project


RID_RE = re.compile(r"[0-9a-f]{32}")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_access(path: Path) -> tuple[str, str]:
    if path.stat().st_mode & 0o077:
        raise PermissionError("access file must be mode 0600")
    value = json.loads(path.read_text(encoding="utf-8"))
    urls = value.get("public_base_urls") or []
    token = str(value.get("access_token") or "")
    if not urls or not token:
        raise ValueError("access file lacks public_base_urls/access_token")
    return str(urls[0]).rstrip("/"), token


def _abort(base_url: str, token: str, rid: str) -> dict:
    payload = json.dumps({"rid": rid}, separators=(",", ":")).encode("ascii")
    last: dict | None = None
    for auth_style, authorization in (
        ("bearer", f"Bearer {token}"),
        ("raw", token),
    ):
        request = Request(
            base_url + "/ops/abort/decode",
            data=payload,
            method="POST",
            headers={
                "Authorization": authorization,
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read(64 * 1024)
                return {
                    "rid": rid,
                    "auth_style": auth_style,
                    "http_status": int(response.status),
                    "response_bytes": len(body),
                    "response_sha256": hashlib.sha256(body).hexdigest(),
                    "success": int(response.status) == 200,
                    "error_type": None,
                }
        except HTTPError as exc:
            body = exc.read(64 * 1024)
            last = {
                "rid": rid,
                "auth_style": auth_style,
                "http_status": int(exc.code),
                "response_bytes": len(body),
                "response_sha256": hashlib.sha256(body).hexdigest(),
                "success": False,
                "error_type": type(exc).__name__,
            }
            if exc.code not in {401, 403}:
                return last
        except (URLError, TimeoutError) as exc:
            return {
                "rid": rid,
                "auth_style": auth_style,
                "http_status": None,
                "response_bytes": 0,
                "response_sha256": None,
                "success": False,
                "error_type": type(exc).__name__,
            }
    return last or {
        "rid": rid,
        "auth_style": None,
        "http_status": None,
        "response_bytes": 0,
        "response_sha256": None,
        "success": False,
        "error_type": "authentication_failed",
    }


def _abort_all(
    base_url: str,
    token: str,
    *,
    run_id: str,
    expected_active_requests: int,
) -> dict:
    payload = json.dumps(
        {
            "run_id": run_id,
            "expected_active_requests": expected_active_requests,
        },
        separators=(",", ":"),
    ).encode("ascii")
    last: dict | None = None
    for auth_style, authorization in (
        ("bearer", f"Bearer {token}"),
        ("raw", token),
    ):
        request = Request(
            base_url + "/ops/abort/decode-all",
            data=payload,
            method="POST",
            headers={
                "Authorization": authorization,
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read(64 * 1024)
                return {
                    "operation": "abort_all",
                    "run_id": run_id,
                    "expected_active_requests": expected_active_requests,
                    "auth_style": auth_style,
                    "http_status": int(response.status),
                    "response_bytes": len(body),
                    "response_sha256": hashlib.sha256(body).hexdigest(),
                    "success": int(response.status) == 200,
                    "error_type": None,
                }
        except HTTPError as exc:
            body = exc.read(64 * 1024)
            last = {
                "operation": "abort_all",
                "run_id": run_id,
                "expected_active_requests": expected_active_requests,
                "auth_style": auth_style,
                "http_status": int(exc.code),
                "response_bytes": len(body),
                "response_sha256": hashlib.sha256(body).hexdigest(),
                "success": False,
                "error_type": type(exc).__name__,
            }
            if exc.code not in {401, 403}:
                return last
        except (URLError, TimeoutError) as exc:
            return {
                "operation": "abort_all",
                "run_id": run_id,
                "expected_active_requests": expected_active_requests,
                "auth_style": auth_style,
                "http_status": None,
                "response_bytes": 0,
                "response_sha256": None,
                "success": False,
                "error_type": type(exc).__name__,
            }
    return last or {
        "operation": "abort_all",
        "run_id": run_id,
        "expected_active_requests": expected_active_requests,
        "auth_style": None,
        "http_status": None,
        "response_bytes": 0,
        "response_sha256": None,
        "success": False,
        "error_type": "authentication_failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Abort exact owned Decode request IDs through the EAS ops proxy."
    )
    parser.add_argument("--access-file", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rid", action="append")
    group.add_argument("--abort-all-expected-active", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    access_file = inside_project(args.access_file)
    output = inside_project(args.output)
    base_url, token = _load_access(access_file)
    if args.abort_all_expected_active is not None:
        if args.abort_all_expected_active < 1:
            parser.error("--abort-all-expected-active must be at least 1")
        if not args.run_id:
            parser.error("--run-id is required with --abort-all-expected-active")
        rows = [
            _abort_all(
                base_url,
                token,
                run_id=str(args.run_id),
                expected_active_requests=args.abort_all_expected_active,
            )
        ]
    else:
        rids = list(dict.fromkeys(str(rid).lower() for rid in (args.rid or [])))
        if any(not RID_RE.fullmatch(rid) for rid in rids):
            parser.error("every --rid must be exactly 32 lowercase hexadecimal characters")
        rows = [_abort(base_url, token, rid) for rid in rids]
    payload = {
        "schema_version": "qwen35-decode-abort-receipt-v2",
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "access_token_in_receipt": False,
        "requested": len(rows),
        "succeeded": sum(bool(row["success"]) for row in rows),
        "rows": rows,
    }
    _atomic_json(output, payload)
    print(
        json.dumps(
            {
                "requested": payload["requested"],
                "succeeded": payload["succeeded"],
                "access_token_present": bool(token),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["succeeded"] == payload["requested"] else 2


if __name__ == "__main__":
    sys.exit(main())
