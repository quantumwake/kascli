"""kas doctor — detect this machine, report what each capability needs, and
(guided) install the missing pieces for the detected platform.

The hard part of a local-first multimodal tool is that every feature has a
different runtime, and the right one depends on OS + CPU arch + GPU vendor +
which system tools/peripherals are present. This centralizes all of that into
one capability registry (the same gating the backends/adapters do at import
time, made explicit and installable) plus an environment probe.

Pure functions (probe_env, cap_status, install_plan) carry no I/O so they're
unit-testable without a machine of every kind; report()/guided_install() do the
printing and (with consent) run the commands.

  python scripts/doctor.py            # status report + the commands to fix gaps
  python scripts/doctor.py --install  # walk the plan, confirming each step
  python scripts/doctor.py --json     # machine-readable status
"""

import importlib.util
import json
import os
import pathlib
import platform
import shlex
import shutil
import subprocess
import sys

# --- environment probe ------------------------------------------------------


def detect_gpu() -> str:
    """Best-effort accelerator family: metal / cuda / rocm / cpu."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal"
    if shutil.which("nvidia-smi"):
        return "cuda"
    if shutil.which("rocminfo") or shutil.which("rocm-smi"):
        return "rocm"
    return "cpu"


def detect_pkg_mgr() -> str | None:
    """The system package manager for installing native tools (ffmpeg, …)."""
    for mgr in ("brew", "apt-get", "dnf", "pacman", "zypper"):
        if shutil.which(mgr):
            return mgr
    return None


def probe_env() -> dict:
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "gpu": detect_gpu(),
        "pkg_mgr": detect_pkg_mgr(),
        "python": platform.python_version(),
        "uv": bool(shutil.which("uv")),
        # peripherals / native tools that features lean on
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "pngpaste": bool(shutil.which("pngpaste")),
        "native_tts": _native_tts_tool() is not None,
    }


def _native_tts_tool() -> str | None:
    if platform.system() == "Darwin" and shutil.which("say"):
        return "say"
    for b in ("espeak-ng", "espeak", "spd-say"):
        if shutil.which(b):
            return b
    return None


# How to install a native tool with each package manager (None = unavailable there).
SYS_INSTALL: dict[str, dict[str, str]] = {
    "ffmpeg": {
        "brew": "brew install ffmpeg",
        "apt-get": "sudo apt-get install -y ffmpeg",
        "dnf": "sudo dnf install -y ffmpeg",
        "pacman": "sudo pacman -S --noconfirm ffmpeg",
        "zypper": "sudo zypper install -y ffmpeg",
    },
    "pngpaste": {"brew": "brew install pngpaste"},  # macOS clipboard images only
    "espeak-ng": {
        "apt-get": "sudo apt-get install -y espeak-ng",
        "dnf": "sudo dnf install -y espeak-ng",
        "pacman": "sudo pacman -S --noconfirm espeak-ng",
        "zypper": "sudo zypper install -y espeak-ng",
        "brew": "brew install espeak-ng",
    },
}


# --- capability registry ----------------------------------------------------
# Each capability: what feature it enables, the python module that proves it's
# installed (+ the pip packages to get it), required native tools, and which GPU
# families it applies to ("any" = all). `optional` capabilities improve a feature
# that already has a working fallback (e.g. neural TTS over the native voice).

CAPS: list[dict] = [
    {
        "id": "server-mlx",
        "label": "Inference server (MLX / Apple GPU)",
        "enables": "running models locally on Apple Silicon",
        "module": "mlx_lm",
        "pkgs": ["mlx-lm"],
        "tools": [],
        "gpus": ["metal"],
    },
    {
        "id": "server-llamacpp",
        "label": "Inference server (llama.cpp / GGUF)",
        "enables": "running GGUF models on CPU/CUDA/ROCm (non-Apple path)",
        "module": "llama_cpp",
        "pkgs": ["llama-cpp-python"],
        "tools": [],
        "gpus": ["cuda", "rocm", "cpu"],
    },
    {
        "id": "vision",
        "label": "Image → text (vision / VLM)",
        "enables": "/image, drag-drop images, VLM models",
        "module": "mlx_vlm",
        "pkgs": ["mlx-vlm"],
        "tools": [],
        "gpus": ["metal"],
    },
    {
        "id": "voice",
        "label": "Voice → text (Whisper)",
        "enables": "/listen — mic transcription",
        "module": "mlx_whisper",
        "pkgs": ["mlx-whisper"],
        "tools": ["ffmpeg"],
        "gpus": ["metal"],
    },
    {
        "id": "tts-native",
        "label": "Text → voice (native)",
        "enables": "/say — spoken replies (no model download)",
        "module": None,
        "pkgs": [],
        "tools": ["__native_tts__"],
        "gpus": ["any"],
    },
    {
        "id": "tts-neural",
        "label": "Text → voice (neural, optional)",
        "enables": "higher-quality /say via mlx-audio/Kokoro",
        "module": "mlx_audio",
        # misaki = Kokoro's G2P; mlx-audio omits it but Kokoro needs it at
        # runtime — and misaki's English pipeline needs spaCy's small model,
        # which spaCy can't self-install into a running process.
        "pkgs": [
            "mlx-audio",
            "misaki[en]",
            "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
        ],
        "tools": [],
        "gpus": ["metal"],
        "optional": True,
    },
    {
        "id": "image-gen",
        "label": "Text → image (FLUX / mflux)",
        "enables": "generate_image tool (--art)",
        "module": "mflux",
        "pkgs": ["mflux"],
        "tools": [],
        "gpus": ["metal"],
    },
    {
        "id": "video-gen",
        "label": "Text → video (LTX-2 / mlx-video)",
        "enables": "generate_video tool (--art)",
        "module": "mlx_video",
        # PyPI `mlx-video` is an unrelated placeholder — install from git.
        "pkgs": [
            "mlx-video-with-audio @ git+https://github.com/james-see/mlx-video-with-audio.git"
        ],
        "tools": [],
        "gpus": ["metal"],
        # optional: the default plan must NOT pull a git package whose first
        # use downloads ~45 GB of weights — video is a deliberate opt-in
        "optional": True,
    },
    {
        "id": "image-preview",
        "label": "Inline image preview",
        "enables": "/show — half-block render in the TUI",
        "module": "PIL",
        "pkgs": ["pillow"],
        "tools": [],
        "gpus": ["any"],
    },
    {
        "id": "clipboard-image",
        "label": "Clipboard image paste",
        "enables": "/image with no path (raw copied pixels, macOS)",
        "module": None,
        "pkgs": [],
        "tools": ["pngpaste"],
        "gpus": ["any"],
        "macos_only": True,
    },
    {
        "id": "memory",
        "label": "Semantic recall (vector store)",
        "enables": "/memory — sqlite-vec + portable CPU embedder",
        "module": "sqlite_vec",
        "pkgs": ["sqlite-vec", "model2vec"],
        "tools": [],
        "gpus": ["any"],
    },
    {
        "id": "web",
        "label": "Web search / fetch",
        "enables": "web_search, web_fetch (--net)",
        "module": "ddgs",
        "pkgs": ["ddgs", "trafilatura"],
        "tools": [],
        "gpus": ["any"],
    },
]


def _have_module(module: str | None) -> bool:
    if module is None:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _tool_present(tool: str, env: dict) -> bool:
    if tool == "__native_tts__":
        return env["native_tts"]
    return bool(shutil.which(tool))


def applies(cap: dict, env: dict) -> bool:
    """Is this capability relevant on this host (GPU family + macOS gating)?"""
    if cap.get("macos_only") and env["os"] != "Darwin":
        return False
    gpus = cap["gpus"]
    return "any" in gpus or env["gpu"] in gpus


def cap_status(cap: dict, env: dict) -> dict:
    """Status of one capability: ready / partial / n-a, plus what's missing."""
    if not applies(cap, env):
        return {"id": cap["id"], "state": "n/a", "missing_pkgs": [], "missing_tools": []}
    have_py = _have_module(cap["module"])
    missing_tools = [t for t in cap["tools"] if not _tool_present(t, env)]
    missing_pkgs = [] if have_py else list(cap["pkgs"])
    state = "ready" if (have_py and not missing_tools) else "partial"
    return {
        "id": cap["id"],
        "state": state,
        "missing_pkgs": missing_pkgs,
        "missing_tools": missing_tools,
    }


def capability_install_command(
    cap_id: str, env: dict | None = None
) -> tuple[list[str] | None, str]:
    """Build the pip/uv argv to install ONE capability's Python packages on this
    host (so a feature can offer `/<cmd> install`, like `/memory install`).

    Returns (argv, note) on success — note flags any native tool the feature also
    needs (those aren't pip-installable; we hint, not run sudo). Returns
    (None, reason) when the capability is unknown, native-only, or unsupported
    here. Targets sys.executable so it works for a dev checkout AND the global
    `kas` tool's env.
    """
    env = env or probe_env()
    cap = next((c for c in CAPS if c["id"] == cap_id), None)
    if cap is None:
        return None, f"unknown capability {cap_id!r}"
    if not applies(cap, env):
        return None, f"{cap['label']} isn't supported on {env['os']}/{env['arch']} ({env['gpu']})"
    if not cap["pkgs"]:
        return None, f"{cap['label']} uses a native tool, not a pip package"
    missing_tools = [
        ("native-tts" if t == "__native_tts__" else t)
        for t in cap["tools"]
        if not _tool_present(t, env)
    ]
    note = f" — also needs {', '.join(missing_tools)} (install separately)" if missing_tools else ""

    # Pick an install that PERSISTS for how kas is actually run, else a plain
    # `uv pip install` gets wiped (uv re-syncs `uv run`, and reinstalling the uv
    # tool drops anything pip-installed). The uv-tool path is COMPREHENSIVE (keeps
    # every other feature) and records intent; dev gets a markered `uv add`.
    if _target_is_tool():
        pkgs = comprehensive_with(env, add=cap["pkgs"])  # additive: keep other features
        withs = [x for p in pkgs for x in ("--with", p)]
        return ["uv", "tool", "install", "--force", *_kas_source(), *withs], note + " (persists)"
    if _editable_checkout():  # dev checkout via `uv run` -> add to pyproject so syncs keep it
        marker = (
            "; sys_platform == 'darwin' and platform_machine == 'arm64'"
            if (cap["gpus"] == ["metal"])
            else ""
        )
        # PEP 508 requires whitespace between a URL spec and its ';marker' —
        # bare concatenation breaks "name @ https://…" pkgs.
        return ["uv", "add", *[f"{p} {marker}".strip() for p in cap["pkgs"]]], note + (
            " (added to pyproject)"
        )
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable, *cap["pkgs"]], note
    return [sys.executable, "-m", "pip", "install", *cap["pkgs"]], note


def _in_uv_tool() -> bool:
    """Is the running interpreter a uv-managed tool env (~/.../uv/tools/<name>)?"""
    return "uv/tools" in sys.prefix.replace(os.sep, "/")


def _kas_tool_installed() -> bool:
    """Is kas installed as a uv tool? Check the uv tools DIRECTORY directly —
    `which('kas')` is unreliable here because `make doctor` runs via `uv run`,
    whose subprocess PATH may not include ~/.local/bin. We detect it so we advise
    installing into the tool, not the throwaway .venv."""
    xdg = os.environ.get("XDG_DATA_HOME")
    roots = [pathlib.Path(xdg) / "uv" / "tools"] if xdg else []
    roots.append(pathlib.Path.home() / ".local" / "share" / "uv" / "tools")
    return any((r / "kas").exists() for r in roots)


def _target_is_tool() -> bool:
    """Should installs go into the kas uv tool? True if we ARE the tool, or a
    kas tool is installed alongside this (dev) process."""
    return _in_uv_tool() or _kas_tool_installed()


def _kas_repo_root():
    """The source checkout for an editable install (so `uv tool install
    --editable <root>` keeps it editable), or None."""
    try:
        import agent

        root = pathlib.Path(agent.__file__).resolve().parent.parent
        return root if (root / "pyproject.toml").exists() else None
    except Exception:
        return None


# The git source for a from-scratch install (curl install.sh) — so `uv tool
# install` works WITHOUT a local clone. A local editable checkout takes priority.
KAS_GIT = "git+https://github.com/quantumwake/kas"


def _kas_source() -> list[str]:
    """Source spec for `uv tool install`: the local editable checkout if present,
    else the git URL (NEVER the bare name `kas` — that's a different PyPI tool)."""
    repo = _kas_repo_root()
    return ["--editable", str(repo)] if repo else [KAS_GIT]


# --- desired-feature persistence: the ONE source of truth -------------------
# All installers (kas doctor --install, /x install, make install/install.sh)
# read+write this so a feature set, once chosen, survives every reinstall and no
# path drops another's packages.
DESIRED_FILE = pathlib.Path.home() / ".kascode" / "features.json"


def load_desired() -> list[str]:
    try:
        data = json.loads(DESIRED_FILE.read_text())
        return [str(x) for x in data.get("with", [])]
    except (OSError, json.JSONDecodeError, AttributeError):
        return []


def save_desired(pkgs) -> None:
    try:
        DESIRED_FILE.parent.mkdir(parents=True, exist_ok=True)
        DESIRED_FILE.write_text(json.dumps({"with": sorted(set(pkgs))}, indent=2) + "\n")
    except OSError:
        pass


def comprehensive_with(env: dict, add=(), include_optional: bool = False) -> list[str]:
    """The COMPLETE --with set for a uv-tool install — additive, never subtractive
    (a `--force` reinstall would otherwise drop any package not listed). Union of:
    the persisted intent, `add`, every applicable feature already installed (keep
    it), and every applicable non-optional feature (+ optional if asked)."""
    want = set(load_desired()) | set(add)
    for cap in CAPS:
        if not (applies(cap, env) and cap["pkgs"]):
            continue
        if cap_status(cap, env)["state"] == "ready":  # installed -> keep it
            want |= set(cap["pkgs"])
        elif not cap.get("optional") or include_optional:  # wanted feature -> add it
            want |= set(cap["pkgs"])
    want.discard("mlx-lm")  # core dep; never bundle redundantly
    return sorted(want)


def default_bundle(env: dict) -> list[str]:
    """Packages a FRESH install bundles: the persisted set if any, else a light
    default (voice + memory + preview + web; the heavy vision/tts/art are opt-in
    via `kas doctor --install`)."""
    saved = load_desired()
    if saved:
        return saved
    light = {"voice", "memory", "image-preview", "web"}
    out: list[str] = []
    for cap in CAPS:
        if cap["id"] in light and applies(cap, env) and cap["pkgs"]:
            out += cap["pkgs"]
    return sorted(dict.fromkeys(out))


def _editable_checkout() -> bool:
    """Are we running inside the kas source tree (dev) vs an installed tool?"""
    try:
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        return (root / "pyproject.toml").exists()
    except OSError:
        return False


def install_plan(env: dict, include_optional: bool = False) -> list[str]:
    """Shell commands to make every applicable capability ready on this host.

    Python deps go through uv (editable `uv pip install` in a checkout, else a
    global `uv tool install --with`); native tools use the detected package
    manager. A native tool with no install recipe for this manager is skipped
    with an explanatory echo so the plan never silently drops a gap.
    """
    cmds: list[str] = []
    pkgs: list[str] = []
    tools: set[str] = set()
    for cap in CAPS:
        if cap.get("optional") and not include_optional:
            continue
        st = cap_status(cap, env)
        if st["state"] != "partial":
            continue
        pkgs += st["missing_pkgs"]
        tools.update(st["missing_tools"])

    # native tools first (a python feature may depend on one, e.g. voice->ffmpeg)
    mgr = env["pkg_mgr"]
    for tool in sorted(tools):
        if tool == "__native_tts__":
            cmds.append(
                "echo 'install a TTS voice: macOS has `say`; Linux: espeak-ng "
                "(see your package manager)'"
            )
            continue
        recipe = SYS_INSTALL.get(tool, {})
        if mgr and mgr in recipe:
            cmds.append(recipe[mgr])
        else:
            avail = ", ".join(sorted(recipe)) or "n/a"
            cmds.append(
                f"echo 'install {tool} manually (no recipe for {mgr or 'your OS'}; has: {avail})'"
            )

    if not pkgs:  # no MISSING python packages -> only native-tool advice (if any)
        return cmds
    if _target_is_tool():
        # COMPREHENSIVE (additive): `uv tool install --force` replaces the --with
        # set, so we list the COMPLETE desired set (keep installed + add wanted),
        # never just the missing — otherwise a reinstall drops other features.
        uniq = comprehensive_with(env, include_optional=include_optional)
        src = " ".join(_kas_source())
        # quote every spec: these strings run through sh, and URL specs
        # ("name @ git+…") contain spaces that would split into stray tokens
        withs = " ".join(f"--with {shlex.quote(p)}" for p in uniq)
        cmds.append(f"uv tool install --force {src} {withs}")
        cmds.append("# then RESTART kas so it picks up the new packages")
    elif _editable_checkout():  # dev checkout via uv run -> pyproject
        cmds.append("uv add " + " ".join(shlex.quote(p) for p in dict.fromkeys(pkgs)))
    else:
        cmds.append("uv pip install " + " ".join(shlex.quote(p) for p in dict.fromkeys(pkgs)))
    return cmds


# --- reporting + guided install (I/O) ---------------------------------------

_MARK = {"ready": "\033[32m✓\033[0m", "partial": "\033[33m⚠\033[0m", "n/a": "\033[90m–\033[0m"}


def report(env: dict) -> None:
    print("kas doctor\n")
    print(
        f"  host: {env['os']}/{env['arch']}  ·  gpu: {env['gpu']}  ·  "
        f"python {env['python']}  ·  pkg-mgr: {env['pkg_mgr'] or 'none'}"
    )
    tts_mark = "✓" if env["native_tts"] else "✗"
    print(
        f"  tools: uv {'✓' if env['uv'] else '✗'}  ffmpeg {'✓' if env['ffmpeg'] else '✗'}  "
        f"pngpaste {'✓' if env['pngpaste'] else '✗'}  native-tts {tts_mark}\n"
    )
    for cap in CAPS:
        st = cap_status(cap, env)
        mark = _MARK[st["state"]]
        line = f"  {mark} {cap['label']:34s} {cap['enables']}"
        print(line)
        if st["state"] == "partial":
            need = []
            if st["missing_pkgs"]:
                need.append("pkgs: " + ", ".join(st["missing_pkgs"]))
            if st["missing_tools"]:
                tools = [
                    ("native-tts" if t == "__native_tts__" else t) for t in st["missing_tools"]
                ]
                need.append("tools: " + ", ".join(tools))
            print(f"      \033[33mneeds {' · '.join(need)}\033[0m")
    plan = install_plan(env)
    if plan:
        print("\n  to enable the missing pieces:\n")
        for c in plan:
            print(f"    {c}")
        print("\n  run `python scripts/doctor.py --install` to do it guided.")
    else:
        print("\n  everything applicable to this host is installed. ✓")


def guided_install(env: dict, include_optional: bool = False, assume_yes: bool = False) -> None:
    plan = install_plan(env, include_optional)
    if not plan:
        print("nothing to install — this host is fully set up.")
        return
    print("planned steps:\n")
    for c in plan:
        print(f"  {c}")
    print()
    for cmd in plan:
        if cmd.startswith("#"):  # advisory comment (e.g. the make-install hint)
            print(f"  {cmd}")
            continue
        if cmd.startswith("echo "):  # advisory note, not a real install
            subprocess.run(cmd, shell=True)
            continue
        if not assume_yes:
            ans = input(f"run: {cmd}\n  [y/N] ").strip().lower()
            if ans != "y":
                print("  skipped")
                continue
        rc = subprocess.run(cmd, shell=True).returncode
        print("  ok" if rc == 0 else f"  failed (exit {rc}) — continuing")
        if rc == 0 and cmd.startswith("uv tool install"):
            # shlex, not str.split: quoted URL specs ("name @ git+…") must
            # survive as single --with values, not be truncated to bare names
            save_intent_from_argv(shlex.split(cmd))  # remember it for next reinstall


def save_intent_from_argv(argv) -> None:
    """Persist the --with packages from a `uv tool install …` command, so the
    chosen feature set survives the next `make install`/reinstall."""
    withs = [argv[i + 1] for i, a in enumerate(argv) if a == "--with" and i + 1 < len(argv)]
    if withs:
        save_desired(withs)


_FEATURE_ALIASES = {
    "tts": "tts-neural",
    "art": "image-gen",
    "image": "image-gen",
    "preview": "image-preview",
}


def remove_feature(env: dict, name: str) -> int:
    """Drop a feature: remove its packages from the desired set and reinstall the
    tool without them. The reverse of `/x install`."""
    cap_id = _FEATURE_ALIASES.get(name, name)
    cap = next((c for c in CAPS if c["id"] == cap_id and c["pkgs"]), None)
    if cap is None:
        names = ", ".join(sorted(c["id"] for c in CAPS if c["pkgs"]))
        print(f"unknown feature {name!r} — removable: {names}")
        return 1
    drop = set(cap["pkgs"])
    keep = [p for p in comprehensive_with(env) if p not in drop]
    save_desired(keep)
    print(f"removed {name} ({', '.join(sorted(drop))}) from the feature set")
    if not _target_is_tool():
        print("  (run an install to apply)")
        return 0
    cmd = ["uv", "tool", "install", "--force", *_kas_source()] + [
        x for p in keep for x in ("--with", p)
    ]
    print(f"  reinstalling without it: {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    print("  done — RESTART kas" if rc == 0 else f"  failed (exit {rc})")
    return rc


def main(argv: list[str]) -> int:
    env = probe_env()
    if "--bundle" in argv:  # space-separated packages for install.sh to --with (no repo needed)
        print(" ".join(default_bundle(env)))
        return 0
    if "--json" in argv:
        out = {"env": env, "caps": [cap_status(c, env) for c in CAPS], "plan": install_plan(env)}
        print(json.dumps(out, indent=2))
        return 0
    if "--remove" in argv:
        i = argv.index("--remove")
        if i + 1 >= len(argv):
            print("usage: kas doctor --remove <feature>  (vision|voice|tts|art|memory|web|preview)")
            return 1
        return remove_feature(env, argv[i + 1])
    if "--install" in argv:
        guided_install(env, include_optional="--optional" in argv, assume_yes="--yes" in argv)
        return 0
    report(env)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
