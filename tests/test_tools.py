"""Characterization tests for the agent's ToolRunner: file tools, dispatch,
and error surfacing. Locks behaviour before the hexagonal refactor extracts
these into agent/adapters/tools/*. No model or server needed.

Run:  uv run python tests/test_tools.py
"""

import pathlib
import sys
import tempfile

sys.path.insert(0, ".")

from agent.main import ToolRunner


class FakeIO:
    def __init__(self, answer="y"):
        self.answer = answer
        self.notices = []

    def confirm(self, command):
        return self.answer

    def notice(self, text):
        self.notices.append(text)


def runner(tmp, **kw):
    return ToolRunner(pathlib.Path(tmp), yolo=True, io=FakeIO(), **kw)


with tempfile.TemporaryDirectory() as tmp:
    r = runner(tmp)

    # write_file -> read_file round-trip.
    out, err = r.tool_write_file("a.txt", "hello\nworld\n")
    assert not err and "wrote" in out, (out, err)
    out, err = r.tool_read_file("a.txt")
    assert not err and out == "hello\nworld\n", repr(out)

    # append mode.
    out, err = r.tool_write_file("a.txt", "again\n", append=True)
    assert not err and "appended" in out, out
    assert r.tool_read_file("a.txt")[0] == "hello\nworld\nagain\n"

    # ranged read is line-number prefixed.
    out, err = r.tool_read_file("a.txt", start_line=2, end_line=2)
    assert not err and "world" in out and "[lines 2-2 of 3]" in out, repr(out)

    # edit_file: unique match required.
    out, err = r.tool_edit_file("a.txt", "world", "WORLD")
    assert not err and "edited" in out, (out, err)
    assert "WORLD" in r.tool_read_file("a.txt")[0]

    out, err = r.tool_edit_file("a.txt", "missing", "x")
    assert err and "not found" in out, (out, err)

    r.tool_write_file("dup.txt", "x\nx\n")
    out, err = r.tool_edit_file("dup.txt", "x", "y")
    assert err and "appears 2 times" in out, (out, err)

    # list_dir.
    out, err = r.tool_list_dir(".")
    assert not err and "a.txt" in out and "dup.txt" in out, out
    print("file tools: OK")

    # apply_patch: surgical unified-diff edit (git apply, works outside a repo).
    r.tool_write_file("p.txt", "alpha\nbeta\ngamma\n")
    patch = "--- a/p.txt\n+++ b/p.txt\n@@ -1,3 +1,3 @@\n alpha\n-beta\n+BETA\n gamma\n"
    out, err = r.tool_apply_patch(patch)
    assert not err and "applied" in out, (out, err)
    assert r.tool_read_file("p.txt")[0] == "alpha\nBETA\ngamma\n", r.tool_read_file("p.txt")[0]

    # context that doesn't match -> error carrying git's message (so the model can retry)
    out, err = r.tool_apply_patch("--- a/p.txt\n+++ b/p.txt\n@@ -1 +1 @@\n-nope\n+x\n")
    assert err and "git apply failed" in out, (out, err)

    # empty patch -> guarded
    out, err = r.tool_apply_patch("")
    assert err and "non-empty" in out, (out, err)
    print("apply_patch: OK")


# ---------------------------------------------------------------------------
# run() dispatch + error surfacing
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    r = runner(tmp)

    # unknown tool name.
    out, err = r.run("does_not_exist", {})
    assert err and "unknown tool" in out, (out, err)

    # non-mutating dispatch goes through and does not flip `mutated`.
    out, err = r.run("list_dir", {"path": "."})
    assert not err and not r.mutated, (out, err, r.mutated)

    # exceptions inside a handler are surfaced as a tool error, not raised.
    out, err = r.run("read_file", {"path": "nope.txt"})
    assert err and ("FileNotFoundError" in out or "No such file" in out), (out, err)
    print("dispatch + errors: OK")


# ---------------------------------------------------------------------------
# --sandbox: file tools jailed to the workdir
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    # Default (sandbox off): absolute paths outside workdir still work.
    r = runner(tmp)
    out, err = r.tool_write_file("/tmp/kas_sandbox_off_probe.txt", "x")
    assert not err, (out, err)  # legacy behaviour preserved
    pathlib.Path("/tmp/kas_sandbox_off_probe.txt").unlink(missing_ok=True)

    # Sandbox on: in-workdir paths allowed, escapes refused via run() error.
    rs = runner(tmp, sandbox=True)
    out, err = rs.tool_write_file("inside.txt", "ok")
    assert not err, (out, err)

    out, err = rs.run("write_file", {"path": "/etc/kas_should_not_write", "content": "x"})
    assert err and "sandbox" in out.lower(), (out, err)

    out, err = rs.run("read_file", {"path": "../../../../etc/hosts"})
    assert err and "sandbox" in out.lower(), (out, err)

    out, err = rs.run("list_dir", {"path": "/"})
    assert err and "sandbox" in out.lower(), (out, err)
    print("sandbox jail: OK")


# ---------------------------------------------------------------------------
# generate_image: graceful when the backend is absent + command construction
# ---------------------------------------------------------------------------

from agent import config as _cfg
from agent.adapters.tools.image import build_command, generate_image

with tempfile.TemporaryDirectory() as tmp:
    # Backend missing -> a helpful error, never a crash.
    _saved_bin = _cfg.ART_BIN
    _cfg.ART_BIN = "definitely-not-a-real-binary-xyz"
    out, err = generate_image("a red truck", pathlib.Path(tmp))
    assert err and "not found" in out and "art" in out, (out, err)

    # Empty prompt is rejected.
    out, err = generate_image("   ", pathlib.Path(tmp))
    assert err and "non-empty" in out, (out, err)
    _cfg.ART_BIN = _saved_bin

    # Command assembly: prompt + output + model + seed + the style preamble.
    _saved_style, _cfg.ART_STYLE = _cfg.ART_STYLE, "isometric pixel sprite"
    cmd = build_command("a red truck", pathlib.Path(tmp) / "truck.png", seed=7)
    _cfg.ART_STYLE = _saved_style
    assert "--prompt" in cmd and "isometric pixel sprite, a red truck" in cmd, cmd
    assert "--output" in cmd and "--seed" in cmd and "7" in cmd, cmd
    print("generate_image: OK")

# ---------------------------------------------------------------------------
# /kv command (toggle + status), no server needed
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    r = runner(tmp)
    assert r.persist_kv is True
    out = r.kv_status("off")
    assert r.persist_kv is False and "OFF" in out, (out, r.persist_kv)
    out = r.kv_status("on")
    assert r.persist_kv is True and "ON" in out, (out, r.persist_kv)
    out = r.kv_status("")  # query only; count is 0 in a fresh tmp workdir
    assert "delta file(s)" in out, out
    print("kv_status: OK")


# ---------------------------------------------------------------------------
# generate_image is ASYNC: returns a task id immediately; image_status tracks it
# ---------------------------------------------------------------------------

import time as _time

with tempfile.TemporaryDirectory() as tmp:
    r = runner(tmp)
    assert r.tool_image_status()[0] == "no image tasks this session"

    # fire-and-continue: returns at once with a task id (doesn't block on render)
    _saved = _cfg.ART_BIN
    _cfg.ART_BIN = "definitely-not-a-real-binary-xyz"  # render will fail fast in the bg
    out, err = r.tool_generate_image("a red truck")
    assert not err and "task #1 started" in out, (out, err)

    # the background task settles to error (backend missing); image_status reflects it
    line = ""
    for _ in range(60):
        line, _e = r.tool_image_status(1)
        if "running" not in line:
            break
        _time.sleep(0.05)
    _cfg.ART_BIN = _saved
    assert "#1" in line and "error" in line, line
    assert "#1" in r.tool_image_status()[0]  # list-all view
    assert r.tool_image_status(99)[1] and "no image task" in r.tool_image_status(99)[0]
    print("async image tasks: OK")


# ---------------------------------------------------------------------------
# generate_video: graceful when the backend is absent + command construction
# ---------------------------------------------------------------------------

from agent.adapters.tools.video import build_command as video_command
from agent.adapters.tools.video import render as video_render

with tempfile.TemporaryDirectory() as tmp:
    # Backend missing -> a helpful error, never a crash.
    _saved_vbin = _cfg.VIDEO_BIN
    _cfg.VIDEO_BIN = "definitely-not-a-real-binary-xyz"
    out, err = video_render("a rocket launch", pathlib.Path(tmp) / "rocket.mp4")
    assert err and "not found" in out and "video" in out, (out, err)

    # Empty prompt is rejected.
    out, err = video_render("   ", pathlib.Path(tmp) / "x.mp4")
    assert err and "non-empty" in out, (out, err)
    _cfg.VIDEO_BIN = _saved_vbin

    # Command assembly: prompt + output + model repo + frames + size + seed + image.
    cmd = video_command(
        "a rocket launch", pathlib.Path(tmp) / "rocket.mp4", seed=7, frames=33, image="still.png"
    )
    assert "--prompt" in cmd and "a rocket launch" in cmd, cmd
    assert "--output-path" in cmd and "--model-repo" in cmd, cmd
    assert "--num-frames" in cmd and "33" in cmd, cmd
    assert "--seed" in cmd and "7" in cmd and "--image" in cmd, cmd
    print("generate_video: OK")


# ---------------------------------------------------------------------------
# generate_video is ASYNC: returns a task id immediately; video_status tracks it
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    r = runner(tmp)
    assert r.tool_video_status()[0] == "no video tasks this session"

    _saved_vbin = _cfg.VIDEO_BIN
    _cfg.VIDEO_BIN = "definitely-not-a-real-binary-xyz"  # render will fail fast in the bg
    out, err = r.tool_generate_video("a rocket launch")
    assert not err and "task #1 started" in out, (out, err)

    line = ""
    for _ in range(60):
        line, _e = r.tool_video_status(1)
        if "running" not in line:
            break
        _time.sleep(0.05)
    _cfg.VIDEO_BIN = _saved_vbin
    assert "#1" in line and "error" in line, line
    assert "#1" in r.tool_video_status()[0]  # list-all view
    assert r.tool_video_status(99)[1] and "no video task" in r.tool_video_status(99)[0]
    print("async video tasks: OK")

print("all tool tests passed")
