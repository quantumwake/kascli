"""TUI slash commands: a registry of one-class-per-command handlers plus the
input router that dispatches to them.

CommandHandler is mixed into AgentApp. on_input_submitted routes every submitted
line: confirmations, /commands (via REGISTRY), exit, and — when busy — steering
vs (when idle) a new turn. Adding a command = a new module in this package + one
line in REGISTRY; the dispatcher and AgentApp don't change.
"""

from rich.text import Text
from textual.widgets import Input

from .ai_wellbeing import AiWellbeingCommand
from .art import ArtCommand
from .compact import CompactCommand
from .converse import ConverseCommand
from .ctx import CtxCommand
from .fx import FxCommand
from .help import HelpCommand
from .image import ImageCommand
from .kv import KvCommand
from .listen import ListenCommand
from .memory import MemoryCommand
from .model import ModelCommand
from .models import ModelsCommand
from .pause import PauseCommand
from .sandbox import SandboxCommand
from .say import SayCommand
from .self_skill import SelfSkillCommand
from .show import ShowCommand
from .spec import SpecCommand
from .stats import StatsCommand
from .status import StatusCommand
from .stop import StopCommand
from .subagent import SubagentCommand
from .theme import ThemeCommand
from .viz import VizCommand
from .voice import VoiceCommand
from .yolo import YoloCommand

# Ordered: prefix-matching commands (/model, /memory, /subagent) keep their relative
# order so none shadows another — this mirrors the historical dispatch order.
REGISTRY = [
    StopCommand(),
    PauseCommand(),
    ModelsCommand(),  # BEFORE ModelCommand — "/models" must not be shadowed by "/model"
    ModelCommand(),
    CompactCommand(),
    SelfSkillCommand(),
    AiWellbeingCommand(),
    SpecCommand(),
    YoloCommand(),
    SubagentCommand(),
    FxCommand(),
    ThemeCommand(),
    VizCommand(),
    MemoryCommand(),
    StatsCommand(),
    CtxCommand(),
    KvCommand(),
    ArtCommand(),
    ShowCommand(),
    ListenCommand(),
    ImageCommand(),
    SayCommand(),
    VoiceCommand(),
    ConverseCommand(),
    SandboxCommand(),
    StatusCommand(),
]
_HELP = HelpCommand()

__all__ = ["CommandHandler", "REGISTRY", "command_completions"]


def command_completions() -> list[str]:
    """Every Tab-complete candidate: each command's name + `/name <subcommand>`
    (and any aliases), plus `exit`. Feeds both the inline suggester and Tab."""
    out: list[str] = []
    for cmd in REGISTRY:
        out.extend(cmd.completions())
    out.append("exit")
    return out


class CommandHandler:
    """Mixin on AgentApp: the input router + /command dispatch."""

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if getattr(self, "_fx_browsing", False):  # Enter keeps the browsed effect
            self.fx_browse_end(keep=True)
            return
        # confirmations and slash-commands act on the typed line only; staged
        # pastes (if any) stay staged for the next real message.
        if self.confirming:
            self.io.confirm_q.put(text)
            return
        if not text and not self._pastes:
            return
        if text in ("exit", "quit"):
            self.exit()
            return
        if text.startswith("/") and not self._pastes:
            self._dispatch_command(text)
            return
        # attach staged multiline paste(s): typed instruction first, blob after
        if self._pastes:
            blob = "\n\n".join(self._pastes)
            self._pastes = []
            text = f"{text}\n\n{blob}" if text else blob
        self._submit_message(text)

    def _submit_message(self, text: str) -> None:
        """Send a finished user message: steer it in if the agent is busy, else
        start a new turn. Shared by the input router and the Composer."""
        if self.busy:
            self.io.steer_q.put(text)
            self.body_write(
                Text("[queued steering — applies at the next tool boundary]", style="magenta")
            )
            return
        text = self._capture_dropped_images(text)
        preview = text.splitlines()[0][:80] + (" …" if "\n" in text or len(text) > 80 else "")
        if getattr(self, "_mdui_rule", False):  # gated; default OFF
            self.turn_rule("you", "#3fb950")
            self.body_write(Text(preview))
            self._agent_header_pending = True
        else:
            self.body_write(Text(f"\nyou> {preview}", style="bold"))
        self.msg_q.put(text)

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic")

    def _capture_dropped_images(self, text: str) -> str:
        """Pull dragged/typed image paths out of `text`, stage them for the next
        user turn (attached by reference — no base64), and return the remaining
        text. Dropping a file onto the terminal pastes its (shell-escaped) path,
        so shlex resolves quotes/backslash-escapes the way the terminal wrote
        them; non-path prose is left untouched."""
        import os
        import shlex

        try:
            tokens = shlex.split(text)
        except ValueError:
            return text  # unbalanced quotes -> treat as plain prose, no capture
        imgs, rest = [], []
        for tok in tokens:
            p = os.path.expanduser(tok)
            if tok.lower().endswith(self.IMAGE_EXTS) and os.path.isfile(p):
                imgs.append(p)
            else:
                rest.append(tok)
        if not imgs:
            return text
        self._pending_images.extend(imgs)
        names = ", ".join(os.path.basename(i) for i in imgs)
        self.body_write(Text(f"📎 attached {len(imgs)} image(s): {names}", style="cyan"))
        return " ".join(rest)

    def _dispatch_command(self, text: str) -> None:
        """Route a /slash command to the first registry entry that matches; the
        help line is the fallback for an unknown command."""
        for cmd in REGISTRY:
            arg = cmd.match(text)
            if arg is not None:
                cmd.run(self, arg)
                return
        _HELP.run(self, "")
