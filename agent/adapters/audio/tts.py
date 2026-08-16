"""Text→speech for spoken assistant replies — with a voice-FX layer.

Pipeline:  synth (Kokoro / native) → optional ffmpeg FX → play.

  - engine: KAS_TTS = mlx | native | auto (default auto = Kokoro when mlx-audio
    is installed, else the native OS voice — macOS `say`, Linux `espeak-ng`).
  - character: KAS_TTS_FX = warrior (default) | none, or KAS_TTS_FILTER for a raw
    ffmpeg -af chain. The default "warrior" preset pitches the voice down and
    adds hall reverb + a chorus shimmer + compression — deep, powerful, alien.
    Needs ffmpeg (already required for voice capture); without it, the dry voice
    plays.
  - voice: KAS_KOKORO_VOICE (Kokoro id, default am_onyx = deep male) ·
    KAS_TTS_VOICE (a macOS `say` voice, default Daniel) · KAS_TTS_PITCH /
    KAS_TTS_RATE tune the native pitch/rate before FX.

Speech runs in a detached process group so it never blocks the UI; a new
utterance interrupts the previous one. Everything degrades to (message, True).
"""

import json
import os
import pathlib
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile

_proc: subprocess.Popen | None = None

_KOKORO_DEFAULT = "am_onyx"  # deep male; bm_george (UK) / af_heart (warm) etc.
_NATIVE_DEFAULT = "Daniel"  # en_GB — the deepest real voice usually present

# --- persistent voice settings (/voice) -------------------------------------
# Precedence per setting: env var (session override) > ~/.kascode/voice.json
# (the /voice command's durable store) > built-in default. Read at synth time,
# not import time, so /voice changes apply to the very next utterance.

_VOICE_STORE = pathlib.Path.home() / ".kascode" / "voice.json"

# Kokoro voice ids encode dialect+gender in the prefix: a=American, b=British,
# e=Spanish, f=French, h=Hindi, i=Italian, j=Japanese, p=Portuguese, z=Chinese;
# then f/m for gender (af_heart, bm_george, ...).
DIALECTS = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Portuguese",
    "z": "Chinese",
}


def _store() -> dict:
    try:
        return json.loads(_VOICE_STORE.read_text())
    except Exception:
        return {}


def save_setting(key: str, value) -> None:
    """Persist one /voice setting (None deletes it)."""
    d = _store()
    if value is None:
        d.pop(key, None)
    else:
        d[key] = value
    _VOICE_STORE.parent.mkdir(exist_ok=True)
    _VOICE_STORE.write_text(json.dumps(d, indent=2) + "\n")


def _setting(env: str, key: str, default: str) -> str:
    v = os.environ.get(env)
    if v is not None:
        return v
    return str(_store().get(key, default))


def voice_settings() -> dict:
    """The effective settings, for /voice status display."""
    return {
        "voice": _setting("KAS_KOKORO_VOICE", "voice", _KOKORO_DEFAULT),
        "say_voice": _setting("KAS_TTS_VOICE", "say_voice", _NATIVE_DEFAULT),
        "speed": _setting("KAS_TTS_SPEED", "speed", "1.0"),
        "pitch": _setting("KAS_TTS_PITCH_SEMITONES", "pitch", "0"),
        "fx": _setting("KAS_TTS_FX", "fx", "warrior"),
        "engine": "kokoro" if _mlx_available() else "native",
    }


def kokoro_voices() -> list[str]:
    """Voice ids shipped with the cached Kokoro model (empty if not cached)."""
    model = os.environ.get("KAS_TTS_MODEL", "mlx-community/Kokoro-82M-bf16")
    hub = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    d = hub / ("models--" + model.replace("/", "--")) / "snapshots"
    names = {f.stem for f in d.glob("*/voices/*") if f.suffix in (".safetensors", ".pt", ".bin")}
    return sorted(names)


# The voice-character FX chains (ffmpeg -af). Sample-rate-independent: normalise
# to 44.1k, drop the pitch ~18% (asetrate), restore tempo, then space + shimmer +
# power. "warrior" = deep/alien/powerful.
_FX_PRESETS = {
    "warrior": (
        "aresample=44100,asetrate=36162,aresample=44100,atempo=1.18,"
        "aecho=0.8:0.85:55:0.35,chorus=0.6:0.9:50:0.4:0.25:2,"
        "acompressor=threshold=-18dB:ratio=4,alimiter"
    ),
    "none": "",
}


def _pitch_stage(semitones: float) -> str:
    """An ffmpeg stage shifting pitch by N semitones, tempo-compensated — the
    engine-agnostic pitch control (Kokoro has no native pitch knob)."""
    if not semitones:
        return ""
    factor = 2.0 ** (semitones / 12.0)  # ±12 semi → 0.5–2.0, atempo's range
    rate, tempo = round(44100 * factor), f"{1 / factor:.4f}"
    return f"aresample=44100,asetrate={rate},aresample=44100,atempo={tempo}"


def _fx_filter() -> str:
    """The ffmpeg -af chain: user pitch shift (semitones) + the character
    preset. KAS_TTS_FILTER (raw chain) replaces the preset but keeps pitch."""
    raw = os.environ.get("KAS_TTS_FILTER")
    if raw is None:
        preset = _setting("KAS_TTS_FX", "fx", "warrior").lower()
        raw = _FX_PRESETS.get(preset, _FX_PRESETS["warrior"])
    try:
        semis = float(_setting("KAS_TTS_PITCH_SEMITONES", "pitch", "0"))
    except ValueError:
        semis = 0.0
    pitch = _pitch_stage(semis)
    return ",".join(s for s in (pitch, raw) if s)


def _mlx_available() -> bool:
    import importlib.util

    # Kokoro hard-requires misaki (its G2P text processor) at RUNTIME even
    # though mlx-audio ships without it — with mlx_audio present but misaki
    # missing, synthesis errors out after model load. Treat that as "not
    # available" so the pipeline builds the native voice instead of silence.
    return all(importlib.util.find_spec(m) is not None for m in ("mlx_audio", "misaki"))


def _native_synth(text: str, out: str) -> list[str] | None:
    """Argv that synthesises `text` to the audio file `out` with a native engine,
    or None if none is available."""
    try:
        speed = float(_setting("KAS_TTS_SPEED", "speed", "1.0"))
    except ValueError:
        speed = 1.0
    if platform.system() == "Darwin" and shutil.which("say"):
        voice = _setting("KAS_TTS_VOICE", "say_voice", _NATIVE_DEFAULT)
        pitch = os.environ.get("KAS_TTS_PITCH", "18")  # say [[pbas]] base pitch
        # cadence: KAS_TTS_RATE wins; else the shared speed scales the base wpm
        rate = os.environ.get("KAS_TTS_RATE") or str(round(155 * speed))
        body = f"[[pbas {pitch}]] [[rate {rate}]] {text}"
        # `say -o x.wav` FAILS ("fmt?") without an explicit linear-PCM data
        # format — WAVE only accepts PCM, and say's default codec isn't.
        return ["say", "-v", voice, "--data-format=LEF32@22050", "-o", out, body]
    for bin_ in ("espeak-ng", "espeak"):
        if shutil.which(bin_):
            rate = os.environ.get("KAS_TTS_RATE") or str(round(150 * speed))
            return [bin_, "-p", "20", "-s", rate, "-w", out, text]
    return None


def _kokoro_synth(text: str, out: str) -> str | None:
    """A shell snippet that synthesises `text` to `out` via mlx-audio (Kokoro),
    or None if not installed. (Kokoro writes <prefix>*.wav; we take the newest
    and move it to `out` so the rest of the pipeline is engine-agnostic.)"""
    if not _mlx_available():
        return None
    model = os.environ.get("KAS_TTS_MODEL", "mlx-community/Kokoro-82M-bf16")
    voice = _setting("KAS_KOKORO_VOICE", "voice", _KOKORO_DEFAULT)
    speed = _setting("KAS_TTS_SPEED", "speed", "1.0")
    prefix = out + ".koko"
    # OUR interpreter, not bare `python` — the detached sh resolves `python`
    # from PATH, which may be a different env without mlx_audio (or absent).
    py = sys.executable
    gen = (
        f"{shlex.quote(py)} -m mlx_audio.tts.generate --model {shlex.quote(model)} "
        f"--voice {shlex.quote(voice)} --text {shlex.quote(text)} "
        f"--speed {shlex.quote(speed)} "
        # Kokoro's lang code IS the voice prefix (a=en-US, b=en-GB, ...): pass
        # it so a British/etc voice gets the matching G2P dialect, not en-US.
        f"--lang_code {shlex.quote(voice[:1])} "
        f"--file_prefix {shlex.quote(prefix)}"
    )
    # move the produced wav to the canonical out path
    return f'{gen} && mv "$(ls -t {shlex.quote(prefix)}*.wav | head -1)" {shlex.quote(out)}'


def _play_argv(path: str) -> list[str] | None:
    if platform.system() == "Darwin" and shutil.which("afplay"):
        return ["afplay", path]
    for bin_, args in (
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("aplay", []),
        ("paplay", []),
    ):
        if shutil.which(bin_):
            return [bin_, *args, path]
    return None


def _pipeline(text: str) -> list[str] | None:
    """Build the full detached `sh -c` command: synth → (ffmpeg FX) → play, or
    None if no engine/player is available."""
    engine = os.environ.get("KAS_TTS", "auto").lower()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="kas-tts-"))
    raw, played = str(tmp / "raw.wav"), str(tmp / "out.wav")

    kokoro: str | None = None
    if engine != "native":  # auto / mlx: prefer Kokoro when present
        kokoro = _kokoro_synth(text, raw)
    argv = _native_synth(text, raw)
    native = shlex.join(argv) if argv else None
    if kokoro and native:
        # Runtime fallback: a Kokoro that errors mid-flight (missing model,
        # broken dep) must degrade to the native voice, not to silence.
        synth = f"{{ {kokoro} ; }} || {native}"
    else:
        synth = kokoro or native
    if synth is None:
        return None

    fx = _fx_filter()
    use_fx = bool(fx) and shutil.which("ffmpeg") is not None
    target = played if use_fx else raw
    play = _play_argv(target)
    if play is None:
        return None

    steps = [synth]
    if use_fx:
        steps.append(
            f"ffmpeg -y -loglevel quiet -i {shlex.quote(raw)} -af {shlex.quote(fx)} "
            f"{shlex.quote(played)}"
        )
    steps.append(shlex.join(play))
    steps.append(f"rm -rf {shlex.quote(str(tmp))}")  # best-effort cleanup
    return ["sh", "-c", " && ".join(steps[:-1]) + "; " + steps[-1]]


def available() -> bool:
    return _mlx_available() or _native_synth("x", "/tmp/x") is not None


def stop() -> None:
    """Interrupt any in-flight speech (the whole process group: synth/ffmpeg/play)."""
    global _proc
    if _proc is not None and _proc.poll() is None:
        try:
            os.killpg(os.getpgid(_proc.pid), 15)
        except Exception:
            try:
                _proc.terminate()
            except Exception:
                pass
    _proc = None


def wait() -> None:
    """Block until the current utterance finishes (or is stopped). Off-thread only."""
    proc = _proc
    if proc is not None:
        try:
            proc.wait()
        except Exception:
            pass


def speak(text: str) -> tuple[str, bool]:
    """Speak `text` in the background (non-blocking). Returns ("", False) on
    success or (message, True) when no engine is available."""
    stop()
    text = (text or "").strip()
    if not text:
        return "", False
    cmd = _pipeline(text)
    if cmd is None:
        return (
            "no TTS engine — macOS has `say`; on Linux install espeak-ng; or add "
            "mlx-audio for neural Kokoro voices (kas doctor --install)",
            True,
        )
    global _proc
    try:
        _proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
        )
    except OSError as exc:
        return f"could not start TTS: {exc}", True
    return "", False
