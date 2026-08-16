"""Shared CLI-binary resolution for the generative-media tools (image/video).

A uv tool install bundles the generator packages (mflux, mlx-video) inside the
kas tool venv, whose bin dir is NOT on the user's PATH — only kas/kas-server get
linked out. So a bare binary name is searched on PATH first and then next to our
own interpreter; an explicit path is taken as-is.
"""

import pathlib
import shutil
import sys


def resolve_bin(bin_: str) -> str | None:
    """Locate a generator CLI, or None if absent."""
    p = pathlib.Path(bin_)
    if len(p.parts) > 1:  # explicit path (absolute or relative)
        return bin_ if p.exists() else None
    found = shutil.which(bin_)
    if found:
        return found
    sibling = pathlib.Path(sys.executable).parent / bin_
    return str(sibling) if sibling.exists() else None
