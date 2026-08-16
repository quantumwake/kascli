"""Opt-in local video generation (--art): a thin wrapper over the mlx-video CLI
(LTX-2 on the Apple GPU, MLX-native). The LLM directs — it writes the prompt;
this tool renders an MP4 to disk and returns the path. Like the image tool, the
bytes never pass through the token stream.

mlx-video-with-audio is an optional dependency (the 'video' extra; installed
from git — the PyPI name `mlx-video` is an unrelated placeholder package). The
command is assembled from env-tunable config (KAS_VIDEO_BIN / KAS_VIDEO_MODEL /
KAS_VIDEO_FRAMES / KAS_VIDEO_FPS / KAS_VIDEO_SIZE / KAS_VIDEO_STEPS) and the
exact command is echoed on failure. First use downloads the model weights
(~45 GB for the default LTX-2 export) — slow once, cached after.
"""

import pathlib
import subprocess

from ... import config
from ._binresolve import resolve_bin
from .image import _slug


def find_bin() -> str | None:
    """Locate the video-generator CLI (PATH, then beside our interpreter)."""
    return resolve_bin(config.VIDEO_BIN)


def _missing_hint() -> str:
    return (
        f"video backend {config.VIDEO_BIN!r} not found — install the 'video' extra "
        "(`uv pip install "
        "'mlx-video-with-audio @ git+https://github.com/james-see/mlx-video-with-audio.git'`), "
        "or set KAS_VIDEO_BIN to your generator"
    )


def build_command(
    prompt: str,
    out_path,
    *,
    seed=None,
    frames=None,
    image=None,
    bin_path=None,
) -> list[str]:
    """Assemble the mlx-video CLI invocation from the (env-tunable) config."""
    w, _, h = config.VIDEO_SIZE.lower().partition("x")
    cmd = [bin_path or config.VIDEO_BIN, "--prompt", prompt, "--output-path", str(out_path)]
    if config.VIDEO_MODEL:
        cmd += ["--model-repo", config.VIDEO_MODEL]
    if config.VIDEO_TEXT_ENCODER:
        cmd += ["--text-encoder-repo", config.VIDEO_TEXT_ENCODER]
    cmd += ["--num-frames", str(frames or config.VIDEO_FRAMES)]
    cmd += ["--fps", str(config.VIDEO_FPS)]
    if w and h:
        cmd += ["--width", w, "--height", h]
    if config.VIDEO_STEPS:
        cmd += ["--num-inference-steps", config.VIDEO_STEPS]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if image:
        cmd += ["--image", str(image)]
    return cmd


def resolve_out(workdir, prompt: str, path: str | None) -> pathlib.Path:
    """Resolve the output MP4 path (default assets/generated/<slug>.mp4 under workdir)."""
    out = (
        pathlib.Path(path) if path else pathlib.Path(config.ART_OUTPUT_DIR) / f"{_slug(prompt)}.mp4"
    )
    return out if out.is_absolute() else pathlib.Path(workdir) / out


def render(
    prompt: str,
    out: pathlib.Path,
    *,
    seed: int | None = None,
    frames: int | None = None,
    image: str | None = None,
) -> tuple[str, bool]:
    """BLOCKING render of one video to `out` via the mlx-video CLI. (Run
    off-thread for async; see ToolRunner.tool_generate_video.)"""
    if not prompt or not prompt.strip():
        return "generate_video requires a non-empty 'prompt'", True
    bin_path = find_bin()
    if bin_path is None:
        return _missing_hint(), True
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(prompt, out, seed=seed, frames=frames, image=image, bin_path=bin_path)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=config.VIDEO_TIMEOUT)
    except FileNotFoundError:
        return _missing_hint(), True
    except subprocess.TimeoutExpired:
        return (
            f"video generation timed out ({config.VIDEO_TIMEOUT}s). command: {' '.join(cmd)}",
            True,
        )
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        return (
            f"video generation failed (exit {proc.returncode}).\ncommand: {' '.join(cmd)}\n{tail}",
            True,
        )
    note = f" (seed {seed})" if seed is not None else ""
    return f"wrote video to {out}{note}", False
