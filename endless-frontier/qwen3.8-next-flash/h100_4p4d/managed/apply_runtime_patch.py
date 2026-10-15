"""Install the recipe's runtime changes into an offline, private source checkout."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify(source, expected):
    source = Path(source).resolve()
    for relative, hashes in json.loads((HERE / "source-manifest.json").read_text())[
        "files"
    ].items():
        path = source / relative
        actual = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        )
        if actual != hashes[expected]:
            raise RuntimeError(f"Unexpected {expected} source: {relative}")


def install(source, check=False):
    source = Path(source).resolve()
    try:
        verify(source, "after")
        return "already_installed"
    except RuntimeError:
        verify(source, "before")
    patch = str(HERE / "runtime.patch")
    subprocess.run(
        ["git", "apply", "--unidiff-zero", "--check", patch], cwd=source, check=True
    )
    if check:
        return "ready_to_apply"
    subprocess.run(["git", "apply", "--unidiff-zero", patch], cwd=source, check=True)
    verify(source, "after")
    return "installed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="Validate without changing source"
    )
    args = parser.parse_args()
    print(json.dumps({"source_status": install(args.source, args.check)}))
