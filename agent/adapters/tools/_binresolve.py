"""Shared CLI-binary resolution for the generative-media tools (image/video).

A uv tool install bundles the generator packages (mflux, mlx-video) inside the
kas tool venv, whose bin dir is NOT on the user's PATH — only kas/kas-server get
linked out. So a bare binary name is searched on PATH first and then next to our
own interpreter; an explicit path is taken as-is.
"""

import os
import pathlib
import shutil
import sys


def _runnable(p: pathlib.Path) -> bool:
    # match shutil.which's contract: exists AND executable — a non-X file
    # would raise PermissionError deep in a worker thread instead of giving
    # the caller its friendly "not found → install hint" path
    return p.is_file() and os.access(p, os.X_OK)


def resolve_bin(bin_: str) -> str | None:
    """Locate a generator CLI, or None if absent/not executable."""
    p = pathlib.Path(bin_)
    if len(p.parts) > 1:  # explicit path (absolute or relative)
        return bin_ if _runnable(p) else None
    found = shutil.which(bin_)
    if found:
        return found
    sibling = pathlib.Path(sys.executable).parent / bin_
    return str(sibling) if _runnable(sibling) else None
