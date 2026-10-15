"""Shared deployment metadata I/O. ESTALE during rename/read is a retryable observation."""

import errno
import json
import os
import time
import uuid
from pathlib import Path

TRANSIENT = {errno.ESTALE, errno.EINTR, errno.EIO, errno.ENOENT}


def read_json(path, retries=8):
    path = Path(path)
    for attempt in range(retries):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            if isinstance(e, OSError) and e.errno not in TRANSIENT:
                raise
            if attempt + 1 == retries:
                raise
            time.sleep(min(0.5, 0.05 * (attempt + 1)))


def save(path, obj):
    path = Path(path)
    data = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    for attempt in range(8):
        tmp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(data)
            tmp.replace(path)
            return
        except OSError as e:
            if e.errno not in TRANSIENT or attempt == 7:
                raise
            time.sleep(min(0.5, 0.05 * (attempt + 1)))
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
