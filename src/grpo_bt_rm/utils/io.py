from __future__ import annotations

import os
from typing import Optional


def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_run_dir(run_dir: str) -> str:
    """
    If run_dir contains exactly one v0-* subdir, return that.
    Otherwise return run_dir unchanged.
    """
    if not os.path.isdir(run_dir):
        return run_dir
    v0s = sorted(
        d for d in (os.path.join(run_dir, x) for x in os.listdir(run_dir))
        if os.path.isdir(d) and os.path.basename(d).startswith("v0-")
    )
    if len(v0s) == 1:
        return v0s[0]
    return run_dir
