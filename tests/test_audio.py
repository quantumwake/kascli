"""Voice→text adapter: ffmpeg record-command construction and transcription
gating. No mic or model — the interactive capture/transcribe paths are mocked.

Run:  uv run python tests/test_audio.py
"""

import sys

sys.path.insert(0, ".")

from agent.adapters.audio import record as rec
from agent.adapters.audio import stt

# record_command: 16 kHz mono, time-limited, platform input format.
cmd = rec.record_command("/tmp/x.wav", 5)
if cmd is not None:  # None only on an unknown OS
    assert cmd[0] == "ffmpeg" and "/tmp/x.wav" in cmd
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
    assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "5"
    assert any(f in cmd for f in ("avfoundation", "pulse"))
print("record_command: OK")

# VAD endpoint coefficient: 2.5s default (long-sentence pauses don't cut off),
# env-overridable; signature takes spoke_secs for the future dynamic version.
assert rec.silence_hold_secs() == 2.5
import os as _os  # noqa: E402

_os.environ["KAS_STT_SILENCE"] = "1.0"
try:
    assert rec.silence_hold_secs(spoke_secs=8.0) == 1.0
finally:
    del _os.environ["KAS_STT_SILENCE"]
print("silence_hold_secs: OK")

# record() degrades when ffmpeg is missing (simulate via PATH lookup miss).
orig_which = rec.shutil.which
rec.shutil.which = lambda _name: None
try:
    path, err = rec.record("/tmp/x.wav", 1)
    assert path is None and "ffmpeg" in err, (path, err)
finally:
    rec.shutil.which = orig_which
print("record ffmpeg-missing: OK")


# transcribe(): missing mlx-whisper -> graceful error, not a raise.
if not stt.whisper_available():
    text, is_err = stt.transcribe("/nonexistent.wav")
    assert is_err and "mlx-whisper" in text, (text, is_err)
    print("transcribe missing-dep: OK")
else:
    # Installed: a missing file is reported, still no raise.
    text, is_err = stt.transcribe("/definitely/not/here.wav")
    assert is_err and "no audio file" in text, (text, is_err)
    print("transcribe missing-file: OK")

# transcribe() now goes through the persistent Transcriber (a serve-mode worker
# subprocess). Stub the subprocess with a fake that speaks the serve protocol:
# emits {"ready"} on spawn, then {"transcribing"}+{"done"} after a wav is written.
import json  # noqa: E402
import tempfile  # noqa: E402

import agent.adapters.audio.transcriber as trans  # noqa: E402

stt.importlib.util.find_spec = lambda name: object() if name == "mlx_whisper" else None
wav = tempfile.mktemp(suffix=".wav")
open(wav, "wb").close()


class _FakeServe:
    """Acts as the Popen handle AND its stdin/stdout: ready first, then a
    transcript per wav written to stdin."""

    def __init__(self, done_text="  hello world ", error=None):
        self._done_text, self._error = done_text, error
        self._q = ['{"event": "ready"}\n']
        self.returncode = None

    # stdin
    def write(self, s):
        if self._error is not None:
            self._q.append(json.dumps({"event": "error", "msg": "Trace\n" + self._error}) + "\n")
        else:
            self._q.append('{"event": "transcribing", "audio_secs": 1.0}\n')
            self._q.append(json.dumps({"event": "done", "text": self._done_text}) + "\n")
        return len(s)

    def flush(self):
        pass

    # process handle
    def poll(self):
        return None  # alive

    def kill(self):
        self.returncode = -9

    # streams (stdin and stdout are both this object)
    stdin = property(lambda self: self)
    stdout = property(lambda self: self)

    def __iter__(self):
        return self

    def __next__(self):
        if self._q:
            return self._q.pop(0)
        raise StopIteration


stt._TRANSCRIBER = None
_orig_trans_popen = trans.subprocess.Popen
trans.subprocess.Popen = lambda *a, **k: _FakeServe()
try:
    seen = []
    text, is_err = stt.transcribe(wav, on_progress=lambda e: seen.append(e["event"]))
    assert not is_err and text == "hello world", (text, is_err)
    assert seen == ["transcribing"], seen  # ready consumed at spawn; done at end
    # a worker error surfaces with the last traceback line
    stt._TRANSCRIBER = None
    trans.subprocess.Popen = lambda *a, **k: _FakeServe(error="ValueError: boom")
    text, is_err = stt.transcribe(wav)
    assert is_err and "boom" in text, (text, is_err)
finally:
    stt._TRANSCRIBER = None
    trans.subprocess.Popen = _orig_trans_popen  # restore (it's the global module)
print("transcribe via warm worker (stubbed): OK")


# --- text→voice (TTS) -------------------------------------------------------
# Hermetic: every tts assert below must see THIS test's settings — not the
# developer's real ~/.kascode/voice.json (e.g. someone who ran `/voice fx none`)
# nor exported KAS_TTS_* session overrides.
import pathlib as _pl  # noqa: E402
import tempfile as _tf  # noqa: E402

from agent.adapters.audio import tts  # noqa: E402

tts._VOICE_STORE = _pl.Path(_tf.mkdtemp(prefix="kas-test-voice-")) / "voice.json"
for _k in (
    "KAS_TTS",
    "KAS_TTS_FX",
    "KAS_TTS_FILTER",
    "KAS_TTS_VOICE",
    "KAS_TTS_RATE",
    "KAS_TTS_SPEED",
    "KAS_TTS_PITCH_SEMITONES",
    "KAS_KOKORO_VOICE",
):
    _os.environ.pop(_k, None)

# native synth writes the spoken text to an output file (macOS say / linux espeak).
cmd = tts._native_synth("hello world", "/tmp/o.wav")
assert cmd is None or (cmd[0] in ("say", "espeak-ng", "espeak") and "/tmp/o.wav" in cmd)
print("tts native_synth: OK")

# the FX layer: default is the "warrior" pitch-down chain; KAS_TTS_FX=none disables.
assert "asetrate" in tts._fx_filter() and "aecho" in tts._fx_filter()
import os as _os2  # noqa: E402

_os2.environ["KAS_TTS_FX"] = "none"
try:
    assert tts._fx_filter() == ""
finally:
    del _os2.environ["KAS_TTS_FX"]
print("tts fx_filter: OK")

# speak() with no engine -> graceful error; with a stubbed pipeline -> launches it.
orig_synth = tts._native_synth
orig_mlx = tts._mlx_available
tts._native_synth = lambda *_a, **_k: None
tts._mlx_available = lambda: False
try:
    msg, is_err = tts.speak("hi")
    assert is_err and "TTS" in msg, (msg, is_err)
finally:
    tts._native_synth = orig_synth
    tts._mlx_available = orig_mlx

launched = {}
orig_popen = tts.subprocess.Popen
orig_pipeline = tts._pipeline
tts._pipeline = lambda t: ["true", t]


def _fake_popen(cmd, **kw):
    launched["cmd"] = cmd
    return type("P", (), {"poll": lambda self: None, "terminate": lambda self: None})()


tts.subprocess.Popen = _fake_popen
try:
    msg, is_err = tts.speak("speak this")
    assert not is_err and launched["cmd"] == ["true", "speak this"], (msg, launched)
    # empty text is a no-op success (no pipeline built)
    assert tts.speak("   ") == ("", False)
finally:
    tts._pipeline = orig_pipeline
    tts.subprocess.Popen = orig_popen
print("tts speak: OK")

# _reply_text pulls the last assistant text out of a transcript (for /say).
from agent.tui.loops import WorkerLoops  # noqa: E402

msgs = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": [{"type": "text", "text": "hello there"}]},
    {"role": "user", "content": "bye"},
    {"role": "assistant", "content": "goodbye now"},
]
assert WorkerLoops._reply_text(msgs) == "goodbye now"
assert WorkerLoops._reply_text(msgs[:2]) == "hello there"
assert WorkerLoops._reply_text([{"role": "user", "content": "x"}]) == ""
# content blocks may be OBJECTS (SDK/pydantic), not dicts — thinking skipped, no crash
import types as _t  # noqa: E402

blocks = [
    _t.SimpleNamespace(type="thinking", thinking="hmm"),
    _t.SimpleNamespace(type="text", text="the answer"),
]
assert WorkerLoops._reply_text([{"role": "assistant", "content": blocks}]) == "the answer"
print("tts reply_text (dict + object blocks): OK")


# --- wav decode (the fds_to_keep fix: hand whisper samples, not a path) ------
# numpy rides the voice extra (via mlx-whisper); on a core-only env (e.g. CI) it
# can be absent, so the in-process decode helper isn't exercisable — gate on it.
import importlib.util as _ilu  # noqa: E402
import pathlib  # noqa: E402

from agent.adapters.audio.stt import model_present  # noqa: E402

if _ilu.find_spec("numpy") is not None:
    import wave  # noqa: E402

    import numpy as np  # noqa: E402

    from agent.adapters.audio.stt import _load_wav_16k_mono  # noqa: E402

    wav = tempfile.mktemp(suffix=".wav")
    samp = (
        0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000, endpoint=False)) * 32767
    ).astype(np.int16)
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(samp.tobytes())
    arr = _load_wav_16k_mono(pathlib.Path(wav))
    assert arr.dtype == np.float32 and len(arr) == 8000 and -1.0 <= arr.min() and arr.max() <= 1.0
    # stereo @ 8 kHz downmixes to mono and resamples to 16 kHz
    wav2 = tempfile.mktemp(suffix=".wav")
    st = np.repeat(samp[:4000, None], 2, axis=1)
    with wave.open(wav2, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(st.tobytes())
    arr2 = _load_wav_16k_mono(pathlib.Path(wav2))
    assert arr2.ndim == 1 and abs(len(arr2) - 8000) <= 2
    print("wav decode: OK")
else:
    print("wav decode: skipped (numpy not installed — voice extra)")

assert isinstance(model_present(), bool)
print("model_present: OK")


# --- voice indicator override (owns the live bar during /listen + /say) ------
from agent.tui.app import AgentApp  # noqa: E402


class IndApp:
    def __init__(self):
        self.fx_mode = "idle"
        self.fx_override = None


app = IndApp()
AgentApp.voice_indicator(app, "listening", conn="🎙 listening", work="5s")
assert app.fx_override["mode"] == "listening" and app.fx_mode == "listening"
assert app.fx_override["conn"] == "🎙 listening"
AgentApp.voice_indicator(app, None)  # release
assert app.fx_override is None
print("voice_indicator: OK")

# --- /voice settings: persistence, precedence, and command assembly ----------
import os
import tempfile

from agent.adapters.audio import tts

with tempfile.TemporaryDirectory() as td:
    _saved_store = tts._VOICE_STORE
    tts._VOICE_STORE = pathlib.Path(td) / "voice.json"
    # (KAS_* env already scrubbed at the top of the TTS section)

    # defaults
    s = tts.voice_settings()
    assert s["voice"] == "am_onyx" and s["speed"] == "1.0" and s["pitch"] == "0", s

    # persisted settings apply to the synth commands
    tts.save_setting("voice", "bm_george")
    tts.save_setting("speed", "1.2")
    tts.save_setting("pitch", "-3")
    if tts._mlx_available():
        cmd = tts._kokoro_synth("hello", "/tmp/x.wav")
        assert "bm_george" in cmd and "--speed 1.2" in cmd, cmd
        assert "--lang_code b" in cmd, cmd  # dialect follows the voice prefix
    fx = tts._fx_filter()
    assert "asetrate" in fx, fx  # pitch stage present
    argv = tts._native_synth("hello", "/tmp/x.wav")
    if argv and argv[0] == "say":
        assert "rate 186" in argv[-1], argv  # 155 * 1.2 cadence scaling
        assert any(a.startswith("--data-format=") for a in argv), argv  # WAV needs PCM

    # env override beats the store; deleting a key restores the default
    os.environ["KAS_TTS_SPEED"] = "0.8"
    assert tts.voice_settings()["speed"] == "0.8"
    del os.environ["KAS_TTS_SPEED"]
    tts.save_setting("pitch", None)
    assert tts.voice_settings()["pitch"] == "0"

    # env pitch is clamped to ±12 (beyond that, atempo leaves ffmpeg's range and
    # the play step silently never runs) and non-finite values are rejected
    tts.save_setting("fx", "none")
    os.environ["KAS_TTS_PITCH_SEMITONES"] = "15"
    try:
        chain = tts._fx_filter()
        assert "atempo=0.5000" in chain, chain  # clamped to +12, not 15
        os.environ["KAS_TTS_PITCH_SEMITONES"] = "nan"
        assert tts._fx_filter() == "", tts._fx_filter()  # nan -> pitch off
    finally:
        del os.environ["KAS_TTS_PITCH_SEMITONES"]
    tts.save_setting("fx", None)
    tts._VOICE_STORE = _saved_store
print("voice settings: OK")

# /voice is registered and matches
from agent.tui.commands import REGISTRY

assert any(c.name == "/voice" for c in REGISTRY)
print("/voice registered: OK")

print("all audio tests passed")
