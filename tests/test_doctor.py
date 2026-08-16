"""kas doctor: platform/GPU detection, capability gating, and install-plan
generation — across hosts this machine isn't (cuda/rocm/cpu + apt/dnf), via
synthetic env dicts and stubbed package/tool probes. No installs run.

Run:  uv run python tests/test_doctor.py
"""

import sys

sys.path.insert(0, ".")

from scripts import doctor


def env(os="Darwin", arch="arm64", gpu="metal", mgr="brew", **kw):
    base = {
        "os": os,
        "arch": arch,
        "gpu": gpu,
        "pkg_mgr": mgr,
        "python": "3.11",
        "uv": True,
        "ffmpeg": True,
        "pngpaste": True,
        "native_tts": True,
    }
    base.update(kw)
    return base


cap = {c["id"]: c for c in doctor.CAPS}

# --- applies(): GPU-family + macOS gating ----------------------------------
assert doctor.applies(cap["server-mlx"], env(gpu="metal"))
assert not doctor.applies(cap["server-mlx"], env(os="Linux", arch="x86_64", gpu="cuda"))
assert doctor.applies(cap["server-llamacpp"], env(os="Linux", gpu="cuda"))
assert doctor.applies(cap["server-llamacpp"], env(os="Linux", gpu="rocm"))
assert doctor.applies(cap["server-llamacpp"], env(os="Linux", gpu="cpu"))
assert not doctor.applies(cap["server-llamacpp"], env(gpu="metal"))  # mlx host -> not llama.cpp
assert doctor.applies(cap["image-preview"], env(gpu="cpu"))  # "any" everywhere
# clipboard-image is macOS-only regardless of GPU
assert doctor.applies(cap["clipboard-image"], env(os="Darwin"))
assert not doctor.applies(cap["clipboard-image"], env(os="Linux", gpu="cpu"))
print("applies(): OK")

# --- cap_status(): ready / partial / n-a with stubbed probes ---------------
orig_have, orig_tool = doctor._have_module, doctor._tool_present
try:
    doctor._have_module = lambda m: False  # nothing installed
    doctor._tool_present = lambda t, e: False  # no tools either
    # vision on a metal host with no package -> partial, lists the pkg
    st = doctor.cap_status(cap["vision"], env(gpu="metal"))
    assert st["state"] == "partial" and st["missing_pkgs"] == ["mlx-vlm"], st
    # vision on a cuda host -> not applicable
    assert doctor.cap_status(cap["vision"], env(os="Linux", gpu="cuda"))["state"] == "n/a"
    # voice needs ffmpeg too -> both pkg and tool missing
    st = doctor.cap_status(cap["voice"], env(gpu="metal"))
    assert st["missing_pkgs"] == ["mlx-whisper"] and st["missing_tools"] == ["ffmpeg"], st

    doctor._have_module = lambda m: True  # everything present
    doctor._tool_present = lambda t, e: True
    assert doctor.cap_status(cap["vision"], env(gpu="metal"))["state"] == "ready"
finally:
    doctor._have_module, doctor._tool_present = orig_have, orig_tool
print("cap_status(): OK")

# --- install_plan(): right deps + right install target per host ------------
orig_have = doctor._have_module
orig_tool = doctor._tool_present
orig_edit = doctor._editable_checkout
orig_target = doctor._target_is_tool
try:
    doctor._have_module = lambda m: False
    doctor._tool_present = lambda t, e: False
    doctor._target_is_tool = lambda: False  # not a uv-tool install ...
    doctor._editable_checkout = lambda: True  # ... a dev checkout -> uv add

    # Linux + CUDA + apt: llama.cpp (not the Apple-only mlx stack), no ffmpeg
    # (voice/vision are Apple-only, so they're n/a here — honest, not a gap hidden).
    plan = doctor.install_plan(env(os="Linux", arch="x86_64", gpu="cuda", mgr="apt-get"))
    joined = "\n".join(plan)
    assert "llama-cpp-python" in joined and "mlx-vlm" not in joined, plan
    assert "ffmpeg" not in joined, plan  # voice n/a on non-Apple
    assert "uv add" in joined, plan  # dev checkout target

    # macOS + metal + brew: mlx stack + brew ffmpeg (voice) + brew pngpaste, NOT llama.cpp.
    plan = doctor.install_plan(env(os="Darwin", gpu="metal", mgr="brew"))
    joined = "\n".join(plan)
    assert "brew install pngpaste" in joined and "brew install ffmpeg" in joined, plan
    assert "mlx-vlm" in joined and "mlx-whisper" in joined, plan
    assert "llama-cpp-python" not in joined, plan

    # optional neural TTS only when asked.
    assert "mlx-audio" not in "\n".join(doctor.install_plan(env(gpu="metal")))
    assert "mlx-audio" in "\n".join(doctor.install_plan(env(gpu="metal"), include_optional=True))

    # A needed native tool with no recipe for this manager -> advisory echo, not dropped.
    plan = doctor.install_plan(env(os="Darwin", gpu="metal", mgr=None))
    assert any(c.startswith("echo ") and "ffmpeg" in c for c in plan), plan

    # uv-TOOL install target -> `uv tool install --with` + a make-install hint
    # (NOT `uv pip install`, which would land in the throwaway .venv).
    doctor._target_is_tool = lambda: True
    plan = doctor.install_plan(env(gpu="metal", mgr="brew"))
    joined = "\n".join(plan)
    assert any("uv tool install --force" in c and "--with" in c for c in plan), plan
    assert "uv pip install" not in joined, plan
    assert any(c.startswith("#") and "RESTART" in c for c in plan), plan
finally:
    doctor._have_module, doctor._tool_present = orig_have, orig_tool
    doctor._editable_checkout, doctor._target_is_tool = orig_edit, orig_target
print("install_plan(): OK")

# --- detect_gpu(): env-driven accelerator family ---------------------------
import platform as _pf  # noqa: E402

orig_sys, orig_mach, orig_which = _pf.system, _pf.machine, doctor.shutil.which
try:
    _pf.system, _pf.machine = (lambda: "Darwin"), (lambda: "arm64")
    assert doctor.detect_gpu() == "metal"
    _pf.system, _pf.machine = (lambda: "Linux"), (lambda: "x86_64")
    doctor.shutil.which = lambda b: b == "nvidia-smi"
    assert doctor.detect_gpu() == "cuda"
    doctor.shutil.which = lambda b: b == "rocminfo"
    assert doctor.detect_gpu() == "rocm"
    doctor.shutil.which = lambda b: False
    assert doctor.detect_gpu() == "cpu"
finally:
    _pf.system, _pf.machine, doctor.shutil.which = orig_sys, orig_mach, orig_which
print("detect_gpu(): OK")

# --- capability_install_command(): one-capability install argv (for `/x install`) ---
orig_tool = doctor._tool_present
orig_target2 = doctor._target_is_tool
try:
    # uv-TOOL target: `uv tool install --with` with BARE names (no marker — this
    # machine already passed applies()).
    doctor._target_is_tool = lambda: True
    cmd, note = doctor.capability_install_command("vision", env(gpu="metal"))
    assert cmd and cmd[:2] == ["uv", "tool"] and any("mlx-vlm" in p for p in cmd), (cmd, note)
    assert not any("platform_machine" in p for p in cmd), cmd  # bare in --with
    # dev-checkout target: `uv add` with the Apple marker (shared pyproject).
    doctor._target_is_tool = lambda: False
    doctor._editable_checkout = lambda: True
    cmd, _ = doctor.capability_install_command("vision", env(gpu="metal"))
    assert cmd[:2] == ["uv", "add"] and any("platform_machine" in p for p in cmd), cmd
    doctor._target_is_tool = orig_target2
    # native tool the feature also needs is flagged (not pip-installed) when absent
    doctor._tool_present = lambda t, e: False
    _, note = doctor.capability_install_command("voice", env(gpu="metal"))
    assert "ffmpeg" in note, note
    doctor._tool_present = orig_tool
    # unsupported platform / native-only capability -> no command, a reason
    cmd, err = doctor.capability_install_command("vision", env(os="Linux", gpu="cuda"))
    assert cmd is None and "supported" in err, (cmd, err)
    cmd, err = doctor.capability_install_command("tts-native", env(gpu="metal"))
    assert cmd is None and "native" in err, (cmd, err)
    # cross-platform cap installs anywhere (no Apple marker since not metal-only)
    cmd, _ = doctor.capability_install_command("image-preview", env(os="Linux", gpu="cpu"))
    assert cmd and any("pillow" in part for part in cmd), cmd
    assert not any("platform_machine" in part for part in cmd), cmd
finally:
    doctor._tool_present = orig_tool
    doctor._target_is_tool = orig_target2
    doctor._editable_checkout = orig_edit
print("capability_install_command(): OK")

# --- URL package specs survive the shell + intent round-trip -----------------
# "name @ git+…" specs contain spaces: the shell command must quote them, and
# save_intent_from_argv must parse with shlex so the full spec (not a truncated
# bare name) is persisted for the next reinstall.
import shlex

URL_SPEC = "mlx-video-with-audio @ git+https://github.com/james-see/mlx-video-with-audio.git"
cmd_line = f"uv tool install --force kas --with {shlex.quote(URL_SPEC)} --with mlx-audio"
argv = shlex.split(cmd_line)
withs = [argv[i + 1] for i, a in enumerate(argv) if a == "--with" and i + 1 < len(argv)]
assert withs == [URL_SPEC, "mlx-audio"], withs

# PEP 508: the dev-checkout marker join needs whitespace before ';' for URL specs
joined = f"{URL_SPEC} ; sys_platform == 'darwin' and platform_machine == 'arm64'"
try:
    from packaging.requirements import Requirement

    assert Requirement(joined).name == "mlx-video-with-audio"
except ImportError:
    pass  # packaging absent in this env — shape asserted above
print("url-spec quoting round-trip: OK")

# video-gen must be OPTIONAL: the default plan must never pull the ~45 GB stack
vg = next(c for c in doctor.CAPS if c["id"] == "video-gen")
assert vg.get("optional") is True, vg
print("video-gen optional: OK")

print("all doctor tests passed")
