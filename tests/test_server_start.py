"""Tests for the "offer to start the server" flow added to the agent CLI:
the local-vs-remote / TTY gating, accept/decline outcomes, and the model picker
(numbered local list, typed HF id, or default). The real spawn + wait are
stubbed so no process is launched. No model or server needed.

Run:  uv run python tests/test_server_start.py
"""

import builtins
import sys

sys.path.insert(0, ".")

import agent.cli as cli
import scripts.select_model as sm


class FakeStdin:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def offer(url, tty, answer, *, model=None, can_start=True, served=("served-model", 4096)):
    """Drive _offer_to_start_server with a controlled TTY, input, and stubbed
    (never real) spawn/wait/probe/picker."""
    saved = (
        sys.stdin,
        builtins.input,
        cli._spawn_server,
        cli._wait_for_server,
        cli.served_info,
        cli._pick_model,
    )
    sys.stdin = FakeStdin(tty)
    builtins.input = lambda *a: answer
    cli._spawn_server = lambda port, model=None: "fake-proc"
    cli._wait_for_server = lambda base, proc, logf=None, stall_polls=120: can_start
    cli.served_info = lambda base: served
    cli._pick_model = lambda: None  # picker covered separately
    try:
        return cli._offer_to_start_server(url, model)
    finally:
        sys.stdin, builtins.input = saved[0], saved[1]
        cli._spawn_server, cli._wait_for_server = saved[2], saved[3]
        cli.served_info, cli._pick_model = saved[4], saved[5]


# --- _is_local_url ---------------------------------------------------------
assert cli._is_local_url("http://127.0.0.1:8765")
assert cli._is_local_url("http://localhost:9000")
assert not cli._is_local_url("http://example.com:8765")
assert not cli._is_local_url("https://api.remote.net")
print("is_local_url: OK")

# --- gating: never offer for a remote URL or a non-TTY ---------------------
assert offer("http://example.com:8765", True, "y") == (None, None)  # remote
assert offer("http://127.0.0.1:8765", False, "y") == (None, None)  # not a TTY
print("gating (remote / non-tty): OK")

# --- decline ---------------------------------------------------------------
assert offer("http://127.0.0.1:8765", True, "n") == (None, None)
assert offer("http://127.0.0.1:8765", True, "no") == (None, None)
print("decline: OK")

# --- accept (blank or yes) -> starts and returns the server's model --------
assert offer("http://127.0.0.1:8765", True, "") == ("served-model", 4096)  # Enter = default yes
assert offer("http://127.0.0.1:8765", True, "y") == ("served-model", 4096)
print("accept -> start: OK")

# --- accept but the server fails to come up -> (None, None) -----------------
assert offer("http://127.0.0.1:8765", True, "y", can_start=False) == (None, None)
print("accept but start fails: OK")


# --- _pick_model: number / typed HF id / default / out-of-range ------------
def pick(answer, models):
    saved_input, saved_info = builtins.input, sm.model_info
    builtins.input = lambda *a: answer
    sm.model_info = lambda: models
    try:
        return cli._pick_model()
    finally:
        builtins.input, sm.model_info = saved_input, saved_info


MODELS = [
    {"id": "org/m1", "size_h": "4GB", "complete": True},
    {"id": "org/m2", "size_h": "8GB", "complete": True},
]
assert pick("2", MODELS) == "org/m2"  # numbered selection
assert pick("mlx-community/Custom-7B", MODELS) == "mlx-community/Custom-7B"  # typed HF id
assert pick("", MODELS) is None  # Enter = server default
assert pick("99", MODELS) == "99"  # out of range -> treated as a typed id
assert pick("only/this", []) == "only/this"  # no local models -> still accepts a typed id
print("pick_model: OK")

print("all server-start tests passed")
