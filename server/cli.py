"""kas-server — the local MLX inference server (Anthropic Messages API).

Installed as its own console script so `kas` (the agent/orchestrator) and the
server are separate binaries. `kas serve` manages this as a background daemon;
run `kas-server` directly to run it in the foreground.
"""

import argparse
import os
import socket


def _port_in_use(host: str, port: int) -> bool:
    """Is something already LISTENING on host:port? Uses a connect probe (not a
    test bind) so it's reliable regardless of SO_REUSEADDR — connect succeeds
    only if a server is actually accepting there."""
    probe = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((probe, port)) == 0


def main() -> None:
    import uvicorn

    from scripts.version import kas_version

    ap = argparse.ArgumentParser(prog="kas-server", description="kas inference server")
    ap.add_argument("--version", action="version", version=f"kas {kas_version()}")
    ap.add_argument("--port", type=int, default=int(os.environ.get("KAS_PORT", "8765")))
    ap.add_argument("--host", default=os.environ.get("KAS_HOST", "127.0.0.1"))
    ap.add_argument("--model", default=None, help="model repo to load")
    ap.add_argument(
        "--quant", default=None, help="GGUF quant to load (e.g. Q4_K_M); else auto-picked"
    )
    a = ap.parse_args()

    # Preflight: refuse to start onto an occupied port. Without this the bind
    # fails AFTER the model loads, uvicorn exits, and any pre-existing (orphan)
    # server keeps answering — so callers see "ready" while our process is dead.
    # Fail loud and early instead.
    if _port_in_use(a.host, a.port):
        raise SystemExit(
            f"kas-server: port {a.port} is already in use — a server is already "
            f"running there.\n"
            f"  stop it:        kas serve --stop   "
            f"(or:  lsof -ti:{a.port} | xargs kill)\n"
            f"  use another:    --port <N>   (or KAS_PORT)"
        )

    if a.model:
        os.environ["KAS_MODEL"] = a.model
    if a.quant:
        os.environ["KAS_GGUF_QUANT"] = a.quant
    os.environ["KAS_PORT"] = str(a.port)
    # In-process model downloads (mlx_lm/hf snapshot_download) must not use the
    # xet backend: it stalls silently and hides progress (same policy as the
    # Makefile's download target). setdefault so an explicit opt-in
    # (KAS_XET=1 / your own HF_HUB_DISABLE_XET) still wins.
    if os.environ.get("KAS_XET") != "1":
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    uvicorn.run("server.app:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
