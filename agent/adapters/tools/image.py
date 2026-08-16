"""Opt-in local image generation (--art): a thin wrapper over the mflux CLI
(MLX-native FLUX / FLUX.2 on the Apple GPU). The LLM is the art director — it
writes the prompt; this tool renders a PNG to disk and returns the path. The
image bytes never pass through the token stream.

mflux is an optional dependency (`uv add mflux` / the 'art' extra). Its CLI is
per-model and versioned, so the command is assembled from env-tunable config
(KAS_ART_BIN / KAS_ART_MODEL / KAS_ART_STEPS / KAS_ART_QUANTIZE / KAS_ART_LORAS /
KAS_ART_STYLE) and the exact command is echoed on failure — so a wrong flag is
trivial to correct, and the agent can even probe `mflux-generate --help` via bash.

Consistency (matters for game asset sets): diffusion is deterministic given a
fixed (model, prompt, seed, steps), so reuse a fixed `seed` per asset. For a
whole set to share look/angle/scale, set KAS_ART_STYLE (a preamble prepended to
every prompt) and/or KAS_ART_LORAS (a locked style LoRA).
"""

import pathlib
import re
import subprocess

from ... import config
from ._binresolve import resolve_bin


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "image"


def find_bin() -> str | None:
    """Locate the image-generator CLI (PATH, then beside our interpreter)."""
    return resolve_bin(config.ART_BIN)


def build_command(prompt: str, out_path, *, seed=None, steps=None, bin_path=None) -> list[str]:
    """Assemble the mflux CLI invocation from the (env-tunable) config."""
    full = f"{config.ART_STYLE}, {prompt}" if config.ART_STYLE else prompt
    cmd = [bin_path or config.ART_BIN]
    if config.ART_MODEL:
        cmd += ["--model", config.ART_MODEL]
    cmd += ["--prompt", full, "--output", str(out_path), "--steps", str(steps or config.ART_STEPS)]
    if config.ART_QUANTIZE:
        cmd += ["-q", str(config.ART_QUANTIZE)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if config.ART_LORAS:
        cmd += ["--lora-paths", *config.ART_LORAS]
        cmd += ["--lora-scales", *(["1.0"] * len(config.ART_LORAS))]
    return cmd


def _missing_hint() -> str:
    return (
        f"image backend {config.ART_BIN!r} not found — install the 'art' extra "
        "(`uv add mflux`) and pull a model, or set KAS_ART_BIN to your generator"
    )


def resolve_out(workdir, prompt: str, path: str | None) -> pathlib.Path:
    """Resolve the output PNG path (default assets/generated/<slug>.png under workdir)."""
    out = (
        pathlib.Path(path) if path else pathlib.Path(config.ART_OUTPUT_DIR) / f"{_slug(prompt)}.png"
    )
    return out if out.is_absolute() else pathlib.Path(workdir) / out


def render(
    prompt: str, out: pathlib.Path, *, seed: int | None = None, steps: int | None = None
) -> tuple[str, bool]:
    """BLOCKING render of one image to `out` via the mflux CLI. (Run off-thread
    for async; see ToolRunner.tool_generate_image.)"""
    if not prompt or not prompt.strip():
        return "generate_image requires a non-empty 'prompt'", True
    bin_path = find_bin()
    if bin_path is None:
        return _missing_hint(), True
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(prompt, out, seed=seed, steps=steps, bin_path=bin_path)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except FileNotFoundError:
        return _missing_hint(), True
    except subprocess.TimeoutExpired:
        return f"image generation timed out (900s). command: {' '.join(cmd)}", True
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        return (
            f"image generation failed (exit {proc.returncode}).\ncommand: {' '.join(cmd)}\n{tail}",
            True,
        )
    note = f" (seed {seed})" if seed is not None else ""
    return f"wrote image to {out}{note}", False


def generate_image(
    prompt: str, workdir, path: str | None = None, seed: int | None = None, steps: int | None = None
) -> tuple[str, bool]:
    """Blocking convenience wrapper (resolve path + render)."""
    return render(prompt, resolve_out(workdir, prompt, path), seed=seed, steps=steps)
