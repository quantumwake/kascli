"""/voice — control the spoken voice: dialect/voice id, pitch, cadence, FX.

Settings persist to ~/.kascode/voice.json (env vars still override per
session) and apply from the very next utterance — no restart needed.
"""

import re

from rich.text import Text

from ...adapters.audio import tts
from .base import Command


class VoiceCommand(Command):
    name = "/voice"
    summary = "voice controls: dialect/voice, pitch, speed, fx"
    usage = "[<voice-id> | list | pitch <±semitones> | speed <0.5-2.0> | fx <preset> | test]"
    subcommands = (
        ("list", "list installed Kokoro voices by dialect"),
        ("pitch", "shift pitch in semitones (e.g. -3; 0 = off)"),
        ("speed", "cadence multiplier (e.g. 0.9 slower, 1.2 faster)"),
        ("fx", "voice character: warrior | none"),
        ("test", "speak a sample with the current settings"),
    )

    def run(self, app, arg: str) -> None:
        arg = arg.strip()
        word, _, rest = arg.partition(" ")
        word, rest = word.lower(), rest.strip()

        if not arg:
            self._status(app)
            return
        if word == "list":
            self._list(app)
            return
        if word == "pitch":
            self._set_number(app, "pitch", rest, -12.0, 12.0, "semitones")
            return
        if word == "speed":
            self._set_number(app, "speed", rest, 0.5, 2.0, "x")
            return
        if word == "fx":
            preset = rest.lower()
            if preset not in tts._FX_PRESETS:
                opts = " | ".join(tts._FX_PRESETS)
                app.body_write(Text(f"unknown fx {preset!r} — options: {opts}", style="yellow"))
                return
            tts.save_setting("fx", preset)
            app.body_write(Text(f"voice fx → {preset}", style="cyan"))
            return
        if word == "test":
            s = tts.voice_settings()
            msg, err = tts.speak(
                f"This is {s['voice']}, speed {s['speed']}, pitch {s['pitch']} semitones."
            )
            app.body_write(Text(msg if err else "[speaking a sample…]", style="cyan"))
            return
        # anything else: treat as a voice id
        self._set_voice(app, arg)

    # -- helpers -------------------------------------------------------------

    def _status(self, app) -> None:
        s = tts.voice_settings()
        dialect = tts.DIALECTS.get(s["voice"][:1], "?")
        app.body_write(Text(f"voice   : {s['voice']}  ({dialect})", style="cyan"))
        app.body_write(Text(f"speed   : {s['speed']}x   pitch: {s['pitch']} st", style="cyan"))
        app.body_write(Text(f"fx      : {s['fx']}   engine: {s['engine']}", style="cyan"))
        app.body_write(
            Text("/voice list · /voice <id> · pitch <±n> · speed <x> · fx <p> · test", style="dim")
        )

    def _list(self, app) -> None:
        voices = tts.kokoro_voices()
        if not voices:
            app.body_write(
                Text("no Kokoro voices cached (first /say downloads them)", style="yellow")
            )
            return
        current = tts.voice_settings()["voice"]
        by_dialect: dict[str, list[str]] = {}
        for v in voices:
            label = tts.DIALECTS.get(v[:1], "other")
            gender = {"f": "female", "m": "male"}.get(v[1:2], "")
            by_dialect.setdefault(f"{label} ({gender})", []).append(v)
        for group in sorted(by_dialect):
            ids = "  ".join(f"[{v}]" if v == current else v for v in sorted(by_dialect[group]))
            app.body_write(Text(f"{group:24s} {ids}", style="cyan"))

    def _set_voice(self, app, vid: str) -> None:
        voices = tts.kokoro_voices()
        if voices:
            if vid not in voices:
                app.body_write(Text(f"unknown voice {vid!r} — see /voice list", style="yellow"))
                return
        elif not re.fullmatch(r"[a-z]{2}_[a-z]+", vid):
            # No cached voice list to validate against (fresh install, relocated
            # cache): still require the Kokoro id SHAPE, or a typo'd subcommand
            # ("/voice hlep") would be persisted as the voice and silently kill
            # the neural engine on every later utterance.
            app.body_write(
                Text(
                    f"{vid!r} doesn't look like a voice id (e.g. am_onyx) — /voice list",
                    style="yellow",
                )
            )
            return
        tts.save_setting("voice", vid)
        dialect = tts.DIALECTS.get(vid[:1], "?")
        app.body_write(Text(f"voice → {vid} ({dialect}) — /voice test to hear it", style="cyan"))

    def _set_number(self, app, key: str, raw: str, lo: float, hi: float, unit: str) -> None:
        try:
            val = float(raw)
        except ValueError:
            app.body_write(Text(f"usage: /voice {key} <number {lo}..{hi}>", style="yellow"))
            return
        val = max(lo, min(hi, val))
        tts.save_setting(key, "0" if val == 0 else f"{val:g}")
        app.body_write(Text(f"voice {key} → {val:g} {unit}", style="cyan"))
