# kas — K.A.S (Kasra's Agentic Shell)

Run open models **locally** (MLX on Apple Silicon, llama.cpp/GGUF on NVIDIA/CPU)
behind an **Anthropic Messages-compatible server**, driven by an agentic TUI.
Python ≥3.11, uv + hatchling. Two packages, strictly separated:

- `agent/` — the client: agentic loop + Textual TUI. Talks to any Anthropic-compatible
  server via the official `anthropic` SDK (`--base-url`, default `127.0.0.1:8765`).
- `server/` — the inference server: FastAPI, pluggable engine backends, per-model
  tool-call dialects, KV-cache continuation + persistence.

Console scripts: `kas = agent.cli:main`, `kas-server = server.cli:main`.

## Commands

```sh
make test        # the no-model test gate (runs each tests/test_*.py as a script)
make lint        # ruff check + format --check  (the CI gate; line-length 100)
make fmt         # auto-fix + format
make cov         # pytest with coverage floor (--cov-fail-under=45, via tests/test_scripts.py shim)
make typecheck   # mypy, permissive baseline (never blocking yet)
make serve       # foreground server (MODEL=... PORT=...)
make agent       # run the TUI (ARGS="--yolo")
uv run python tests/test_foo.py   # any single test, standalone
```

Tests are **standalone scripts** with bare asserts (run via `python tests/test_x.py`),
collected by pytest only through the `tests/test_scripts.py` runpy shim. `tests/*`
has an E402 lint exemption for its sys.path shims. Architecture rule enforced by
`tests/test_architecture.py`: `agent/core/` and `server/core/` must not import
from `adapters/` or `backends/` (hexagonal ports & adapters).

## Architecture: agent/

- **Entry**: `agent/cli.py` is the composition root — flag parsing, config mutation,
  server autostart (`_offer_to_start_server` → `_spawn_server` → `_wait_for_server`
  which tails `~/.kas/server.log` for HF download progress), then one of three modes:
  one-shot (positional task), TUI (`agent/tui/app.py`, Textual), or plain REPL.
- **`agent/core/loop.py`** — `agent_turn()`: stream → execute tool_use blocks →
  append tool_results → repeat. Reconnect-with-retry on transport errors, abort/steer
  handling (Esc, mid-task steering), truncation recovery, round soft-landing.
  Thread key = session id (`x-agent-thread` header) so each agent/subagent gets its
  own server KV slot; `x-agent-session-dir` enables server-side KV persistence.
- **`agent/core/compaction.py`** — hard (85% of context) vs soft (decode-tok/s valve)
  compaction. `compact_messages` reuses the same system+tools so the server's
  continuation memo key matches (cache-hit summarization). Originals archived to
  `compaction-NN.json` and indexed for `recall`.
- **`agent/core/subagent.py` + `loop.run_subagent`** — fresh message list, own KV
  thread (`{parent}-sub-{n}`), only the final report returns to the parent.
- **`agent/ports/`** — Protocols: `AgentIO` (ui), `ToolExecutor` (tools), 
  `SessionStorePort`, `MemoryBackend`/`Embedder`, `WorkspacePort`.
  NOTE: `ToolExecutor` understates reality — the loop/compaction touch ~13 runner
  attributes (`compact_*`, `tps_*`, `context_limit`, `persist_kv`, `net`, `rag`, `art`).
- **`agent/adapters/tools/`** — `executor.py:ToolRunner` dispatches `tool_<name>`
  from mixins: bash (PTY session, idle/wait/kill lifecycle), files (read/write/edit/
  apply_patch via `git apply`), image (async mflux render pool), web (ddgs/trafilatura,
  opt-in `--net`), memory (`recall` tool). Only **bash** asks for confirmation
  (y/n/a; `--yolo` skips); file writes and image gen run unprompted. Mutating tools
  trigger `GitWorkspace.ready()` + per-turn checkpoint commits.
- **`agent/adapters/retrieval/`** — BM25 (sqlite FTS5, `.agent/rag.db`) + optional
  vector store (sqlite-vec, `.agent/vec.db`); results fused by reciprocal-rank fusion.
  Embedder registry (`adapters/embeddings/`) is platform-gated: mlx → gguf →
  model2vec → hashing; degrades to BM25-only, never hard-errors (house rule:
  **all hardware-tied runtimes must degrade gracefully on the wrong platform**).
- **`agent/tui/`** — Textual app; `commands/` is an ordered registry of ~24 slash
  commands (`Command` base class; order matters for prefix matching). Streaming
  renders through `TuiIO` (`tui/io.py`) marshalled via `call_from_thread`. Esc =
  abort + `POST /v1/cancel` (cancels even mid-prefill). `/model` hot-swaps,
  `/theme`, `/viz` (token heatmap via `_viz` SSE extras), `/fx` ambient bar.
- **Sessions**: `<workdir>/.agent/sessions/<id>/transcript.json` (+ `compaction-NN.json`,
  `kvcache/<thread>/*.safetensors` written by the server). `--resume [ID]` rehydrates
  messages and the server KV (warm resume).

## Architecture: server/

- **Entry**: `server/cli.py` (port preflight → env vars → uvicorn `server.app:app`).
  Daemon management lives in the *agent* (`kas serve --stop/--status/--logs`).
- **`server/app.py`** — routes: `POST /v1/messages` (SSE or aggregate), `GET /v1/models`
  (resident + downloadable), `POST /v1/models/select|unload`, `POST /v1/cancel`,
  `GET /v1/stats`. `ModelRegistry` (`registry.py`) holds multiple resident engines
  (LRU eviction, `KAS_MAX_MODELS`/`KAS_GPU_BUDGET_GB`).
- **`server/core/`** — `ports.py` defines `EngineLike` (tokenize/encode/generate/
  cache_snapshot/swap/request_cancel + optional rehydrate) and `GenChunk`.
  `pipeline.py:run()` is the request use case: continuation check → tokenize →
  engine.generate → `StreamParser` → normalized events → memo write.
  `continuation.py:try_continuation()` verifies the client's echoed turn against the
  memo, then prefills **only the new tail** (KV continuation — the flagship feature).
  `kvpersist.py` lays out on-disk KV deltas per session/thread.
- **`server/backends/`** — registry pattern (`Backend(load, supported, installed,
  requires)`); `make_engine` checks platform/install *before* import so failures are
  readable. `mlx.py` (all MLX state confined to one `mlx-worker` thread; per-thread
  KV slots; 8-bit KV quantization past 8k on append-only threads), `llama_cpp.py`
  (GGUF header-based ctx sizing with halving back-off, quant auto-pick, three-layer
  EOG detection, BOS force-prepend), `mlx_vlm.py` (vision models; no continuation),
  `_gpu.py` (process-wide GPU lock — overlapping Metal command buffers abort).
- **`server/prompting/`** — the translation layer. `translate.py` (Anthropic → chat
  template), `dialects.py` (Gemma + Qwen, continuation-capable, byte-exact
  `continuation_tail`), `standard_dialects.py` (llama/mistral/hermes/deepseek/kimi/
  harmony re-render dialects), `parser.py` (4-state streaming marker parser),
  `recover.py` (fallback tool-call recovery ladder), `detect_dialect` (user config
  `~/.kascode/dialects.json` → template markers → model-id heuristics → Gemma default).
- **Model classification**: `scripts/select_model.py:model_kind()` → text / vision /
  embedding / stt / **image** (diffusers `model_index.json`) / other. Only text+vision
  are chat-servable; diffusion models (e.g. `mlx-community/qwen-image-edit-2511-8bit`)
  can never load under `kas serve` — they run through mflux (see below).

## Media generation (--art): images AND video

The `--art` flag gates ALL local generative media tools: `generate_image` /
`image_status` (mflux) and `generate_video` / `video_status` (mlx-video, LTX-2).
Both are async — they render on a shared bounded thread pool and return a task
id immediately. The validated "director" pattern: a small served chat model
(e.g. `mlx-community/Qwen3.5-9B-4bit`, 5 GB, qwen dialect, tool-calls verified)
fronts the heavy generators through these tools.

Video specifics (verified end-to-end 2026-08-15: 1.3 s MP4 + generated audio):
backend is `mlx_video.generate_av` from the **james-see/mlx-video-with-audio
fork** (the `video` extra — installed from git; the PyPI `mlx-video` name is an
unrelated placeholder, and the Blaizzy original expects a split model layout +
separate text-encoder repo and fails on our weights with KeyError 'text_config').
Default weights `notapalindrome/ltx2-mlx-av` (~45 GB unified MLX export,
self-contained incl. text encoder and audio; never default to
`Lightricks/LTX-2` — that repo is 314 GB). Frames must be 4n+1.
Env knobs: `KAS_VIDEO_BIN/MODEL/TEXT_ENCODER/FRAMES/FPS/SIZE/STEPS/TIMEOUT`.
Voice out (/say, /converse): Kokoro via mlx-audio HARD-REQUIRES `misaki[en]`
(G2P) at runtime — keep it in the tts extra; tts.py builds the synth command
with `sys.executable` (never bare `python`) and falls back to native `say` if
Kokoro fails mid-flight.
Qwen-Image-Edit (`mlx-community/qwen-image-edit-2511-8bit`, verified working)
runs via `mflux-generate-qwen-edit` and needs `--image-paths` — an `edit_image`
tool is a known TODO.

## Image generation (--art)

`generate_image` shells out to **mflux** (bundled in the installed uv tool).
`agent/config.py:_art_bin_for()` routes the model family to the right mflux entry
point (mflux refuses FLUX.2/Qwen on the generic `mflux-generate`):
flux2* → `mflux-generate-flux2`, qwen*edit → `mflux-generate-qwen-edit`, etc.
`agent/adapters/tools/image.py:find_bin()` resolves the binary from PATH **and from
the directory of `sys.executable`** — a uv tool install bundles mflux inside the kas
tool venv whose bin dir is not on the user's PATH. Env knobs: `KAS_ART_BIN/MODEL/
STEPS/QUANTIZE/STYLE/LORAS/OUTPUT_DIR`. Default model `flux2-klein-4b` (~20 GB in
HF cache, 4-step distilled, `-q 8`).

## Conventions & gotchas

- **Config is mutable module state** (`agent/config.py`, `server/config.py`): always
  `from ... import config` and read attributes late; importing names freezes them.
- Env vars mirror flags: `KAS_MODEL`, `KAS_BASE_URL`, `KAS_MAX_TOKENS`, `KAS_CTX`,
  `KAS_GPU_LAYERS`, `KAS_BACKEND`, `KAS_GGUF_QUANT`, `KAS_KV_PERSIST`, …
- Comment style: dense, first-person-plural rationale comments explaining *why*;
  keep that register when editing.
- `mlx-lm` and friends carry platform markers in `pyproject.toml`; optional heavy
  stacks (voice/vision/art/web/memory) are extras — **never** promote them to core
  deps. Tools return install hints instead of raising when an extra is missing.
- The user rejects `Co-Authored-By` trailers in commits.
- Committed-but-ignorable artifacts exist at repo root (`server.log`, `coverage.xml`,
  `htmlcov/`) — don't read them as live state.
- `--sandbox` is a hard exit by design (bash escaped the file jail; honesty over
  theater). Don't resurrect it casually.
- The `agent/main.py` façade re-exports legacy names — TUI imports go through it;
  keep it in sync when renaming core symbols.

## Known weak spots (as of 2026-08-15 evaluation)

Fixed already: mflux PATH resolution + per-family entry points; interactive server
autostart not tailing the download log; llama.cpp `ping_status` mutating on read;
`translate.py` list+=str on image-then-text turns; viz header truthiness.

Still open (verified by code reading, not yet fixed):
- `server/backends/llama_cpp.py` keeps **one KV slot for all threads** (ignores
  `cache_key` in `cache_snapshot`/persist) — cross-thread continuation corruption
  with subagents on the GGUF backend. MLX has proper per-thread slots.
- `server/backends/mlx.py`: `_slot()`/`cache_snapshot()` mutate slot LRU state from
  caller threads despite the "worker-thread-only" invariant; stop-sequence overshoot
  can leak partial stop strings; `finally` can desync slot tokens after mid-prefill
  exceptions.
- `server/backends/mlx_vlm.py`: single-slot `_pending` races concurrent requests;
  base64 temp images never deleted.
- Effective `max_tokens` default is 1024 (schema Field default), not the documented
  8192 (`DEFAULT_MAX_TOKENS` unreachable).
- `agent/core/compaction.py` sends `x-agent-session-dir` ignoring `/kv off`.
- `SessionStore.save_transcript` is not atomic (no tmp+rename); `resume()` has no
  corrupt-JSON guard.
- Bash "3 silent waits" detach leaks the PTY/process (`session = None` w/o kill).
- `recall()` refreshes the index twice per call (adapters/tools/memory.py).
