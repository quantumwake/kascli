"""Agent configuration: env-overridable knobs, server probes, and small shared
helpers. Values here are mutated by the CLI composition root (agent/cli.py)
from parsed args at startup, so other modules read them as `config.MODEL` etc.
rather than importing the values by name (which would freeze them at import).
"""

import os

import httpx

BASE_URL = os.environ.get("KAS_BASE_URL", "http://127.0.0.1:8765")
MODEL = os.environ.get("KAS_MODEL")  # default: ask the server what it loaded
MAX_TOKENS = int(os.environ.get("KAS_MAX_TOKENS", "16384"))
# Compaction is a decode-speed relief valve, not a context-window necessity —
# KV continuation makes prefill cheap and quantization eases long-context
# decode, so trigger it rarely and high. Too low + a large project = a
# compact->read->compact thrash that never makes progress.
COMPACT_AT = int(os.environ.get("KAS_COMPACT_AT", "120000"))
# Hard floor on turns between compactions — guarantees no tight loop even if
# the work keeps refilling context.
COMPACT_COOLDOWN = int(os.environ.get("KAS_COMPACT_COOLDOWN", "5"))
# Decode-rate trigger: compaction exists to relieve the slowdown a GROWING context
# causes, so trigger on a RELATIVE drop from the model's own baseline decode speed
# — not an absolute tok/s (a model whose baseline is ~8 tok/s shouldn't look like
# it needs compacting just for running at 8). Trip when smoothed decode falls below
# this fraction of the learned peak/low-context baseline. 0 disables.
COMPACT_TPS_FRAC = float(os.environ.get("KAS_COMPACT_TPS_FRAC", "0.55"))
# Deprecated absolute threshold (kept for back-compat; the trigger is relative now).
COMPACT_TPS = float(os.environ.get("KAS_COMPACT_TPS", "8.0"))

MAX_TOOL_OUTPUT = 8_000

# --- Opt-in image generation (--art): local mflux/FLUX backend -------------
# mflux's CLI is per-model and versioned, so every knob is env-tunable and the
# tool echoes the exact command it ran on failure (easy to correct / self-fix).
ART_MODEL = os.environ.get("KAS_ART_MODEL", "flux2-klein-4b")  # small, fast, fits beside the LLM


def _art_bin_for(model: str) -> str:
    """mflux ships one CLI entry point per model family — the generic
    mflux-generate REFUSES FLUX.2/Qwen/z-image models with an error telling you
    which binary to use. Mirror that routing so the default just works."""
    m = model.lower()
    if "flux2" in m or "flux.2" in m:
        return "mflux-generate-flux2-edit" if "edit" in m else "mflux-generate-flux2"
    if "qwen" in m:
        return "mflux-generate-qwen-edit" if "edit" in m else "mflux-generate-qwen"
    if "z-image" in m or "z_image" in m:
        return "mflux-generate-z-image-turbo" if "turbo" in m else "mflux-generate-z-image"
    return "mflux-generate"  # FLUX.1 dev/schnell/krea and friends


ART_BIN = os.environ.get("KAS_ART_BIN", _art_bin_for(ART_MODEL))
ART_STEPS = int(os.environ.get("KAS_ART_STEPS", "4"))  # distilled FLUX needs few steps
ART_QUANTIZE = os.environ.get("KAS_ART_QUANTIZE", "8")  # "" to disable
ART_OUTPUT_DIR = os.environ.get("KAS_ART_OUTPUT_DIR", "assets/generated")
# Consistency levers (see docs): a style preamble prepended to EVERY prompt so a
# whole sprite set shares look/angle/scale, plus LoRA files (path[,path]) for a
# locked style. Pair with a fixed `seed` per asset.
ART_STYLE = os.environ.get("KAS_ART_STYLE", "")
ART_LORAS = [p for p in os.environ.get("KAS_ART_LORAS", "").split(os.pathsep) if p]

# --- Opt-in video generation (--art): local mlx-video/LTX-2 backend --------
# Backend is the mlx-video-with-audio fork (james-see): its generate_av entry
# point is SELF-CONTAINED with the unified ~45 GB MLX export below — the
# Blaizzy original expects a split layout + separate text-encoder repo, and the
# official Lightricks/LTX-2 repo is 314 GB of every variant (never default to
# either). Frames must be 4n+1.
VIDEO_BIN = os.environ.get("KAS_VIDEO_BIN", "mlx_video.generate_av")
VIDEO_MODEL = os.environ.get("KAS_VIDEO_MODEL", "notapalindrome/ltx2-mlx-av")
VIDEO_TEXT_ENCODER = os.environ.get("KAS_VIDEO_TEXT_ENCODER", "")  # "" = derived
VIDEO_FRAMES = int(os.environ.get("KAS_VIDEO_FRAMES", "33"))  # ~1.4 s @ 24 fps
VIDEO_FPS = int(os.environ.get("KAS_VIDEO_FPS", "24"))
VIDEO_SIZE = os.environ.get("KAS_VIDEO_SIZE", "512x512")  # WxH
VIDEO_STEPS = os.environ.get("KAS_VIDEO_STEPS", "")  # "" = pipeline default
VIDEO_TIMEOUT = int(os.environ.get("KAS_VIDEO_TIMEOUT", "3600"))  # seconds


def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + f"\n... [truncated {len(text) - MAX_TOOL_OUTPUT} chars]"


def served_model(base_url: str) -> str | None:
    """Ask the server which model it actually has loaded."""
    return served_info(base_url)[0]


def served_info(base_url: str) -> tuple[str | None, int | None]:
    """Return (model_id, context_length) from the server, or (None, None)."""
    try:
        d = httpx.get(base_url.rstrip("/") + "/v1/models", timeout=5).json()["data"][0]
        return d.get("id"), d.get("context_length")
    except Exception:
        return None, None
