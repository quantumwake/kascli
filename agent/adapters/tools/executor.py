"""ToolRunner — the ToolExecutor adapter: dispatches a tool call to its handler
and coordinates per-turn workspace checkpointing.

The collaborators it wires (BashSession, PathResolver, Memory, web fns,
GitWorkspace) live in sibling modules, and the tool_* handlers themselves are
grouped into per-area mixins (_bash_tools / _file_tools / _image_tools). This
class is the composition point: it owns the shared state in __init__, runs the
dispatch, and keeps the few tiny handlers (recall/web/kv) that don't warrant a
module of their own.
"""

import pathlib
from collections import deque

from ..workspace.git import GitWorkspace
from ._bash_tools import BashToolsMixin
from ._file_tools import FileToolsMixin
from ._image_tools import ImageToolsMixin
from ._video_tools import VideoToolsMixin
from .bash import BashSession
from .files import PathResolver
from .memory import Memory
from .web import web_fetch, web_search


# The tool_* handlers come from the per-group mixins; run() finds each by name
# across the MRO via getattr(self, "tool_<name>"), so adding a tool group is a
# new mixin, not a change here.
class ToolRunner(BashToolsMixin, FileToolsMixin, ImageToolsMixin, VideoToolsMixin):
    MUTATING_TOOLS = (
        "write_file",
        "edit_file",
        "apply_patch",
        "bash",
        "bash_send_input",
        "generate_image",
        "generate_video",
    )

    def __init__(
        self,
        workdir: pathlib.Path,
        yolo: bool,
        io=None,
        checkpoint: bool = False,
        net: bool = False,
        rag: bool = False,
        context_limit: int | None = None,
        sandbox: bool = False,
        compact_at: int | None = None,
        art: bool = False,
    ) -> None:
        self.workdir = workdir
        self.yolo = yolo
        self.art = art  # generate_image tool available only when True
        if io is None:
            from ...config import BASE_URL
            from ..ui.console import ConsoleIO

            io = ConsoleIO(BASE_URL)
        self.io = io
        self.net = net  # web_search / web_fetch available only when True
        self.rag = rag  # recall tool available only when True
        self.context_limit = context_limit  # model's native context window (overflow safety)
        self.tps_window: deque = deque(maxlen=4)  # recent decode tok/s, for the trigger
        self.tps_baseline = 0.0  # learned fast/low-context decode rate (relative trigger)
        self.session: BashSession | None = None
        self.mutated = False  # any tool may have changed files this turn
        self._paths = PathResolver(workdir, sandbox=sandbox)
        self.memory = Memory(workdir)  # pluggable recall backends (bm25 today)
        self.git = GitWorkspace(workdir, io, force_checkpoint=checkpoint)
        # context size right after the last compaction; auto-compaction
        # triggers on GROWTH beyond this, not on an absolute threshold —
        # otherwise post-compaction re-reads immediately re-trigger it
        self.compact_floor = 0
        self.compact_cooldown = 0  # turns remaining before compaction may fire again
        # Compaction policy (mutable at runtime via the /ctx command):
        from ...config import COMPACT_AT

        self.compact_at = (
            COMPACT_AT if compact_at is None else compact_at
        )  # soft size cap (0 = off)
        self.tps_valve = True  # decode-speed relief valve on/off
        self.hard_limit_frac = 0.85  # fraction of native window that forces compaction
        self.last_input_tokens = 0  # most recent prompt size, for /ctx display
        self.persist_kv = True  # send the session dir so the server persists KV (/kv)
        # Async media generation: tasks render off-thread so the loop never
        # waits. Image and video share one bounded pool (GPU jobs), but keep
        # separate task tables so image_status / video_status stay per-kind.
        self._art_tasks: dict[int, dict] = {}
        self._art_seq = 0
        self._art_pool = None
        self._video_tasks: dict[int, dict] = {}
        self._video_seq = 0
        self._video_pool = None  # separate 1-worker pool: videos must not starve images

    # -- workspace checkpointing ----------------------------------------------

    def checkpoint(self, label: str) -> str | None:
        """Commit this turn's changes; returns the short sha or None."""
        mutated, self.mutated = self.mutated, False
        return self.git.checkpoint(mutated, label)

    @property
    def sandbox(self) -> bool:
        """Whether the FILE tools are jailed to the workdir. Proxies the
        PathResolver so /sandbox can toggle it live. NB: this confines the file
        tools only — bash is NOT sandboxed (see the /sandbox notice)."""
        return self._paths.sandbox

    @sandbox.setter
    def sandbox(self, on: bool) -> None:
        self._paths.sandbox = on

    # -- dispatch -------------------------------------------------------------

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Returns (output, is_error)."""
        try:
            handler = getattr(self, f"tool_{name}", None)
            if handler is None:
                return f"unknown tool: {name}", True
            if name in self.MUTATING_TOOLS:
                # Ensure the workspace repo (and its pre-agent baseline commit)
                # exists BEFORE the first change lands.
                try:
                    self.git.ready()
                except Exception:
                    pass
            output, is_error = handler(**args)
            if not is_error and name in self.MUTATING_TOOLS:
                self.mutated = True  # may have changed the workspace
            return output, is_error
        except Exception as exc:  # surface errors to the model, don't crash
            return f"{type(exc).__name__}: {exc}", True

    # -- opt-in tools ---------------------------------------------------------

    def tool_recall(self, query: str, k: int = 8) -> tuple[str, bool]:
        return self.memory.recall(query, k)

    def tool_web_search(self, query: str, max_results: int = 5) -> tuple[str, bool]:
        return web_search(query, max_results)

    def tool_web_fetch(self, url: str) -> tuple[str, bool]:
        return web_fetch(url)

    def kv_status(self, arg: str = "") -> str:
        """Handle /kv [on|off]: toggle whether this session persists its KV cache
        to disk (for warm --resume) and report the on-disk delta count."""
        import pathlib

        a = (arg or "").strip().lower()
        if a in ("on", "enable"):
            self.persist_kv = True
        elif a in ("off", "disable"):
            self.persist_kv = False
        d = pathlib.Path(self.workdir) / ".agent" / "sessions"
        # best-effort count across this workdir's kvcache dirs (main thread)
        n = (
            sum(len(list(p.glob("*.safetensors"))) for p in d.glob("*/kvcache/main"))
            if d.exists()
            else 0
        )
        return (
            f"KV-resume {'ON' if self.persist_kv else 'OFF'} "
            f"(server must also have KAS_KV_PERSIST!=0) · {n} delta file(s) on disk"
        )
