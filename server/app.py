"""Anthropic Messages API-compatible server backed by a local MLX model.

Composition root: wires the MLX engine (driven adapter) to the HTTP routes
(driving adapter) and the generate->events pipeline (core). The request
translation, continuation memo, and SSE framing live in server/{prompting,core,
adapters}; this module holds only the FastAPI app, lifecycle, shared state
(the engine + per-thread continuation memos), and the route handlers.

Run:  uv run uvicorn server.app:app --port 8765

Point any official Anthropic SDK at it:

    client = anthropic.Anthropic(base_url="http://127.0.0.1:8765", api_key="local")
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters.http.complete import complete
from .adapters.http.sse import stream_safe
from .config import KV_PERSIST, MODEL_ID
from .core import kvpersist
from .core.continuation import echo_matches, norm_blocks, req_key, try_continuation
from .core.pipeline import run
from .core.ports import EngineLike
from .registry import ModelRegistry
from .schema import MessagesRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("kas")

# Shared server state. Models are held by REGISTRY (multiple resident at once,
# routed by request model id, evicted LRU under a count cap + GPU memory budget).
# `engine`/`_memos` remain as back-compat globals: when REGISTRY is None (unit
# tests that inject a fake engine directly), the routes fall back to them.
REGISTRY: ModelRegistry | None = None
engine: EngineLike | None = None
_memos: dict[str, dict[str, Any]] = {}


def _resolve(model_id: str | None) -> EngineLike | None:
    """The engine for a request's model. The registry routes/loads by id — but a
    request's `model` is often a placeholder ("x") or simply doesn't match what's
    loaded (e.g. the server was started from a local .gguf PATH, or the client
    sends a default name). Only a DIFFERENT, repo-id-shaped model triggers an
    actual load; anything else — and any load that fails — falls back to the
    default resident model instead of 500-ing the request.

    Absent a registry (injected-engine unit tests) fall back to the global engine.
    """
    if REGISTRY is None:
        return engine
    default = REGISTRY.default_model
    # "/" gates a load attempt: repo ids (ns/name) and paths have one; a bare
    # placeholder like "x" doesn't and must never be fetched as a model.
    if not model_id or model_id == default or "/" not in model_id:
        return REGISTRY.get(default)
    try:
        return REGISTRY.get(model_id)
    except Exception as exc:
        log.warning(
            "requested model %r not loadable (%s) — serving default %s",
            model_id,
            type(exc).__name__,
            default,
        )
        return REGISTRY.get(default)


def _memos_for(model_id: str | None) -> dict:
    return REGISTRY.memos(model_id) if REGISTRY is not None else _memos


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, REGISTRY
    try:
        from scripts.banner import print_console

        print_console(model=MODEL_ID, extra="inference server")
    except Exception:
        pass
    # The registry picks the backend per model (KAS_BACKEND, else auto-detected)
    # — MLX today, llama.cpp/CUDA/ROCm later. Preload the default; `engine` points
    # at it so back-compat callers see a ready engine.
    REGISTRY = ModelRegistry(MODEL_ID)
    engine = REGISTRY.get(MODEL_ID)
    yield


app = FastAPI(title="kas", lifespan=lifespan)


def error_response(status: int, err_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": err_type, "message": message}},
    )


# Cap request body size — a localhost guard against a single client exhausting
# memory with a multi-GB messages array. Checks Content-Length (which the
# Anthropic SDK always sends); a chunked body with no such header is not bounded
# here, which is acceptable for a local-only server. Override via env.
MAX_BODY_BYTES = int(os.environ.get("KAS_MAX_BODY_BYTES", str(64 * 1024 * 1024)))


@app.middleware("http")
async def _limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
        return error_response(
            413, "request_too_large", f"request body exceeds {MAX_BODY_BYTES} bytes"
        )
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    return error_response(400, "invalid_request_error", str(exc.errors()[:3]))


@app.exception_handler(Exception)
async def fallback_handler(_: Request, exc: Exception):
    log.exception("internal error")
    return error_response(500, "api_error", f"{type(exc).__name__}: {exc}")


def _active_engine() -> EngineLike | None:
    """The engine to report live stats for: an actively-generating one if any,
    else the most-recently-used. Falls back to the global engine sans registry."""
    if REGISTRY is None:
        return engine
    for mid in reversed(REGISTRY.loaded()):  # most-recent first
        e = REGISTRY.peek(mid)
        if e is not None and getattr(e, "stats", {}).get("active"):
            return e
    return REGISTRY.most_recent()


def _available_models() -> list[dict[str, Any]]:
    """Every DOWNLOADED model — the single source of truth for client pickers
    (the agent's /model, specbuilder, …), built from the same on-disk scan the
    agent used to do locally, so local and remote clients agree. `loaded` marks
    the ones currently resident on this server."""
    try:
        from scripts.select_model import model_info
    except Exception:
        return []
    if REGISTRY is not None:
        resident = set(REGISTRY.loaded())
    else:
        resident = {engine.model_id} if engine else set()
    return [
        {
            "id": m["id"],
            "size_h": m.get("size_h"),
            "kind": m.get("kind"),
            "complete": m.get("complete", True),
            "loaded": m["id"] in resident,
        }
        for m in model_info()
    ]


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """`data` = models resident right now (Anthropic shape + dialect/GPU/default);
    `available` = every downloaded model (the picker source of truth)."""
    if REGISTRY is not None:
        data = [
            {
                "type": "model",
                "id": m["id"],
                "display_name": m["id"],
                "dialect": m["dialect"],
                "context_length": m["context_length"],
                "active": m["active"],
                "gpu_active_gb": m["gpu_active_gb"],
                "est_gb": m["est_gb"],
                "default": m["default"],
            }
            for m in REGISTRY.info()
        ]
    else:
        mid = engine.model_id if engine else MODEL_ID
        data = [{"type": "model", "id": mid, "display_name": mid}]
    return {"data": data, "available": _available_models()}


@app.post("/v1/models/select")
def select_model(payload: dict[str, Any]):
    """Preload a model and make it the default. With the registry, other loaded
    models stay resident (subject to the cap + GPU budget) rather than swapping."""
    model_id = (payload or {}).get("model")
    if not model_id or not isinstance(model_id, str):
        return error_response(400, "invalid_request_error", "body must include 'model'")
    if REGISTRY is None:
        return error_response(500, "api_error", "registry not ready")
    try:
        eng = REGISTRY.get(model_id)  # loads if needed; budget-guarded
    except RuntimeError as exc:  # GPU budget refused the load
        return error_response(503, "overloaded_error", str(exc))
    except Exception as exc:
        log.exception("model load failed")
        return error_response(500, "api_error", f"load failed: {type(exc).__name__}: {exc}")
    REGISTRY.default_model = model_id
    return {
        "ok": True,
        "model": eng.model_id,
        "dialect": eng.dialect.name,
        "loaded": REGISTRY.loaded(),
    }


@app.post("/v1/models/unload")
def unload_model(payload: dict[str, Any]):
    """Offload a model and free its GPU memory. Refuses one that's generating."""
    model_id = (payload or {}).get("model")
    if not model_id or not isinstance(model_id, str):
        return error_response(400, "invalid_request_error", "body must include 'model'")
    if REGISTRY is None:
        return error_response(400, "invalid_request_error", "no model registry")
    ok = REGISTRY.unload(model_id)
    return {
        "ok": ok,
        "loaded": REGISTRY.loaded(),
        "message": None if ok else "not loaded, or busy generating (refused)",
    }


@app.post("/v1/cancel")
def cancel_generation() -> dict[str, Any]:
    """Interrupt the in-flight generation NOW (including a long prefill). With the
    registry, cancels whichever loaded model is actively generating."""
    if REGISTRY is not None:
        active = False
        for mid in REGISTRY.loaded():
            e = REGISTRY.peek(mid)
            if e is not None and getattr(e, "stats", {}).get("active"):
                active = e.request_cancel() or active
        return {"ok": True, "active": active}
    if engine is None:
        return {"ok": False, "active": False}
    return {"ok": True, "active": engine.request_cancel()}


@app.get("/v1/stats")
def live_stats() -> dict[str, Any]:
    """Live generation progress (polled while a stream is quiet) + a `models`
    list so a client can show everything resident and offload from there."""
    eng = _active_engine()
    if eng is None:
        return {"model": MODEL_ID, "active": False}
    sysstats = getattr(eng, "system_stats", lambda: {})()
    out = {"model": eng.model_id, **eng.stats, **eng.ping_status(), **sysstats}
    if REGISTRY is not None:
        out["models"] = REGISTRY.info()
    return out


def _validate(req: MessagesRequest) -> JSONResponse | None:
    if not req.messages:
        return error_response(400, "invalid_request_error", "messages: must not be empty")
    if req.messages[0].role != "user":
        return error_response(400, "invalid_request_error", "first message must use the user role")
    if req.messages[-1].role == "assistant":
        return error_response(
            400, "invalid_request_error", "assistant-turn prefill is not supported"
        )
    return None


@app.post("/v1/messages")
def messages(req: MessagesRequest, request: Request):
    if (err := _validate(req)) is not None:
        return err
    # Each conversation thread (main agent + each subagent) gets its own KV
    # cache slot + continuation memo, keyed by this header.
    thread = request.headers.get("x-agent-thread", "main")
    # Diagnostic: two concurrent agents MUST show different threads here. If both
    # log thread=main they're sharing a KV slot + continuation memo (e.g. an agent
    # process running pre-fix code) — restart the agents.
    log.info("turn model=%s thread=%s stream=%s", req.model, thread, req.stream)
    # Route to the requested model — the registry loads/holds it (LRU + GPU
    # budget). A refused load (budget full) surfaces as 503, not a crash.
    try:
        eng = _resolve(req.model)
    except RuntimeError as exc:
        return error_response(503, "overloaded_error", str(exc))
    except Exception as exc:
        log.exception("model load failed")
        return error_response(500, "api_error", f"load failed: {type(exc).__name__}: {exc}")
    if eng is None:
        return error_response(500, "api_error", "engine not ready")
    memos = _memos_for(req.model)
    # /viz: when the client asks (any overlay on), the engine emits per-token
    # logprobs. Only then — the top-k+entropy compute isn't free.
    viz = request.headers.get("x-agent-viz", "").strip().lower() not in ("", "0", "false", "off")

    # Warm KV-resume: if persistence is on and the agent told us its session
    # dir, rehydrate this thread's KV cache + continuation memo from disk before
    # the first turn (no-op once warm). Best-effort; backends without rehydrate
    # (e.g. the VLM engine) simply cold-prefill.
    persist_dir = request.headers.get("x-agent-session-dir") if KV_PERSIST else None
    if persist_dir and hasattr(eng, "rehydrate"):
        try:
            status = eng.rehydrate(thread, persist_dir)
            if status.startswith("rehydrated") and thread not in memos:
                memo = kvpersist.read_json(
                    kvpersist.memo_path(kvpersist.thread_dir(persist_dir, thread))
                )
                if memo:
                    memos[thread] = memo
        except Exception:
            log.info("kv rehydrate trigger failed; cold prefill", exc_info=True)

    if req.stream:
        return StreamingResponse(
            stream_safe(req, eng, memos, thread, persist_dir, viz),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    return complete(req, eng, memos, thread, persist_dir)


# --------------------------------------------------------------------------
# Back-compat shims: bind the (now-pure) core functions to this module's shared
# engine/memo state. Kept so existing tests and any external callers that
# import these private names keep working unchanged.
# --------------------------------------------------------------------------

_req_key = req_key
_norm_blocks = norm_blocks
_echo_matches = echo_matches


def _try_continuation(req: MessagesRequest, key: str, thread: str) -> list[int] | None:
    return try_continuation(req, key, thread, engine, _memos)


def _run(req: MessagesRequest, thread: str = "main"):
    return run(req, engine, _memos, thread)
