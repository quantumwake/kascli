"""llama.cpp / GGUF inference backend (cross-platform: CPU + CUDA + ROCm + Metal +
Vulkan, depending on how llama-cpp-python was built). This is the portable path
for non-Apple hardware — one runtime covers NVIDIA and AMD and CPU.

Conforms to server.core.ports.EngineLike: load + chat-template tokenize + streaming
generate + stats + management, plus KV warm-resume via llama.cpp's OWN state files
(in-memory prefix reuse between turns; llama_state_seq_save_file/load_file across a
restart, model-id guarded). The heavy import is deferred and the runtime glue is
kept standard + best-effort (any KV failure falls back to a cold prefill, never a
broken turn); the runtime path is validated on CPU CI + GPU CI (Modal/RunPod)
rather than on the Apple-only dev box.

Model resolution (model_id): a local *.gguf path, or a Hugging Face repo id —
then the quant file is chosen by `--quant`/KAS_GGUF_QUANT (e.g. Q4_K_M), or an
exact KAS_GGUF_FILE glob, or auto-picked (4-bit K-quants first), skipping
mmproj/MTP files. GPU offload via KAS_GPU_LAYERS (-1 = all); context is sized to
the model's trained length (KAS_CTX overrides, KAS_CTX_MAX caps for GPU memory).
"""

import json
import logging
import os
import pathlib
import shutil
import struct
import subprocess
import threading
import time
from collections.abc import Iterator
from typing import Any

from ..core.cache import longest_common_prefix
from ..core.ports import GenChunk

log = logging.getLogger("kas.llama_cpp")

# Resolved once: the nvidia-smi path on NVIDIA hosts, else None (cheap no-op on
# Metal/ROCm/CPU). Used for GPU memory in /stats — the MLX backend reports Metal
# memory, but llama.cpp on CUDA had no GPU figure at all.
_NVIDIA_SMI = shutil.which("nvidia-smi")
_GPU_CACHE: dict[str, Any] = {"t": 0.0, "val": None}


def _nvidia_gpu_mem() -> tuple[float, float, int] | None:
    """(used_gb, total_gb, util_pct) for GPU 0 from nvidia-smi, or None if not an
    NVIDIA host / unreadable. Cached ~1s — /stats polls once a second and a
    subprocess per poll is wasteful."""
    if not _NVIDIA_SMI:
        return None
    now = time.time()
    if now - _GPU_CACHE["t"] < 1.0:
        return _GPU_CACHE["val"]
    val = None
    try:
        out = subprocess.run(
            [
                _NVIDIA_SMI,
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        used, total, util = (x.strip() for x in out.splitlines()[0].split(","))
        val = (round(int(used) / 1024, 2), round(int(total) / 1024, 2), int(util))
    except Exception:
        val = None
    _GPU_CACHE["t"], _GPU_CACHE["val"] = now, val
    return val


# GGUF scalar value-type byte sizes (the enum from the GGUF spec). Strings (8) and
# arrays (9) are variable-length and handled separately.
_GGUF_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_INT = {2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}


def _gguf_meta_int(path: str, key_suffix: str) -> int:
    """Read an integer metadata value (by key suffix, e.g. '.context_length')
    straight from a GGUF file header — no model load, no context allocation. The
    interesting keys (`<arch>.context_length`) sit before the big tokenizer arrays,
    so this returns early and never parses them. 0 if absent/unreadable."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return 0
            struct.unpack("<I", f.read(4))  # version
            struct.unpack("<Q", f.read(8))  # tensor count
            (n_kv,) = struct.unpack("<Q", f.read(8))

            def skip(t: int) -> None:
                if t == 8:  # string
                    (ln,) = struct.unpack("<Q", f.read(8))
                    f.read(ln)
                elif t == 9:  # array: elem-type, count, then elements
                    (et,) = struct.unpack("<I", f.read(4))
                    (cnt,) = struct.unpack("<Q", f.read(8))
                    for _ in range(cnt):
                        skip(et)
                else:
                    f.read(_GGUF_SCALAR.get(t, 0))

            for _ in range(n_kv):
                (klen,) = struct.unpack("<Q", f.read(8))
                key = f.read(klen).decode("utf-8", "replace")
                (vtype,) = struct.unpack("<I", f.read(4))
                if key.endswith(key_suffix) and vtype in _GGUF_INT:
                    (val,) = struct.unpack(_GGUF_INT[vtype], f.read(_GGUF_SCALAR[vtype]))
                    return int(val)
                skip(vtype)
    except Exception as exc:
        log.warning("GGUF metadata read failed for %s (%s)", path, exc)
    return 0


def _kv_paths(persist_dir: str, thread: str) -> tuple[pathlib.Path, pathlib.Path]:
    d = pathlib.Path(persist_dir) / "kvcache" / thread
    return d / "kv.bin", d / "meta.json"


def kv_restore_plan(persist_dir: str, thread: str, model_id: str) -> tuple[bool, str]:
    """Pure pre-check for rehydrate (unit-testable without llama.cpp): is there a
    saved KV for this thread, and was it built with the SAME model? A KV cache is
    model-specific, so restoring across a model switch would be garbage — guard it."""
    kv, meta = _kv_paths(persist_dir, thread)
    if not kv.exists() or not meta.exists():
        return False, "no saved KV — cold prefill"
    try:
        saved = json.loads(meta.read_text()).get("model")
    except (OSError, json.JSONDecodeError):
        return False, "unreadable KV meta — cold prefill"
    if saved != model_id:
        return False, f"model changed ({saved} != {model_id}) — cold prefill"
    return True, "ok"


class LlamaCppEngine:
    def __init__(self, model_id: str) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # the registry already gates on this; guard direct use
            raise RuntimeError(
                "the llama.cpp backend needs llama-cpp-python — install it (build with "
                "CUDA/ROCm/Metal for GPU): pip install llama-cpp-python"
            ) from exc

        self.model_id = model_id
        self.stats: dict[str, Any] = {"active": False}
        self.ping_count = 0
        self.last_ping_ts = 0.0
        self._cancel = threading.Event()
        self._lock = threading.Lock()  # llama.cpp Llama is not reentrant (single-stream)
        self._Llama = Llama
        self._cached_tokens: list[int] = []  # tokens currently held in the KV cache
        self._load(model_id)

    # --- loading -------------------------------------------------------------

    @staticmethod
    def _pick_gguf_file(repo_id: str, glob: str | None, quant: str | None) -> str:
        """Choose a GGUF file from a multi-quant repo. Quant naming is wildly
        inconsistent (UD-Q4_K_M, Q4_K_XL, IQ4_XS, split shards, plus mmproj/MTP
        files mixed in), so a fixed glob fails. Precedence:
          1. KAS_GGUF_FILE — an exact filename/glob (power users, split BF16, …)
          2. `quant` (--quant / KAS_GGUF_QUANT) — a quant name like Q4_K_M; picks
             the file carrying it (case-insensitive), first shard.
          3. a sensible default preference order (4-bit K-quants first).
        Non-main files (mmproj vision projectors, MTP drafts) are excluded.
        """
        import fnmatch
        import re

        try:
            from huggingface_hub import HfApi

            files = [f for f in HfApi().list_repo_files(repo_id) if f.endswith(".gguf")]
        except Exception:
            return glob or (f"*{quant}*.gguf" if quant else "*Q4_K_M*.gguf")
        # drop vision projectors (mmproj) and multi-token-prediction draft files
        main = [f for f in files if not re.search(r"mmproj|mtp|-MTP\b", f, re.I)] or files

        def first_shard(f: str) -> bool:  # for split GGUF, take part 1 only
            m = re.search(r"(\d+)-of-\d+", f)
            return not m or int(m.group(1)) == 1

        if glob:  # 1. exact override
            hit = [f for f in main if fnmatch.fnmatch(f.rsplit("/", 1)[-1], glob)]
            if hit:
                return min(hit, key=len)
        if quant:  # 2. by quant name
            hit = [f for f in main if quant.lower() in f.lower() and first_shard(f)]
            if hit:
                return min(hit, key=len)
            log.warning("quant %r not in %s; falling back to default preference", quant, repo_id)
        # auto-picking: surface the options so the user can re-run with --quant.
        seen = {
            m.group(0)
            for f in main
            for m in [re.search(r"I?Q\d+(?:_[A-Z0-9]+)*|F16|BF16", f.rsplit("/", 1)[-1], re.I)]
            if m
        }
        if len(seen) > 1:
            log.info("quants in %s: %s  (pick one with --quant)", repo_id, ", ".join(sorted(seen)))
        # 3. default order — 4-bit K-quants first (agent sweet spot), then up/down
        order = [
            "Q4_K_M",
            "Q4_K_S",
            "Q4_K_XL",
            "Q4_K",
            "Q4_0",
            "IQ4_XS",
            "IQ4_NL",
            "Q5_K_M",
            "Q5_K_S",
            "Q5_K",
            "Q6_K",
            "Q8_0",
            "Q3_K_M",
            "Q3_K",
            "Q2_K",
            "IQ3",
            "IQ2",
            "F16",
            "BF16",
        ]
        for q in order:
            cands = [f for f in main if q.lower() in f.lower() and first_shard(f)]
            if cands:
                return min(cands, key=len)  # shortest name = least extra tagging
        shards = [f for f in main if first_shard(f)]
        return min(shards, key=len) if shards else (glob or "*.gguf")

    def _choose_ctx(self, src: dict) -> int:
        """Size the context window to the model — start as large as it was trained
        for, and let _load's backoff shrink it to what the GPU can actually hold.

        Precedence: explicit KAS_CTX wins; otherwise n_ctx = min(trained length,
        KAS_CTX_MAX). The KV cost per token is wildly model-specific — a dense 31B
        (gemma) has a big KV and tops out ~24k on a 40GB card, while a hybrid 27B
        (Qwen3.6: only 16 of 64 layers are full-attention, the rest are linear
        DeltaNet) fits 128k+ on the SAME card. There's no portable formula, so
        rather than a flat conservative cap (which throttled the big-context
        models) we aim high and back off on the actual allocation failure. 8192
        was far too small for agentic coding: a long prompt+output overran it and
        llama.cpp raised 'llama_decode returned 1' (no free KV slot)."""
        env = os.environ.get("KAS_CTX")
        if env:
            return int(env)
        # A generous ceiling, not a guess at what fits: _load backs off from here
        # if the KV won't allocate. Caps the 256k/1M-context models to a sane first
        # attempt; raise KAS_CTX_MAX on big GPUs, lower it to force a smaller window.
        cap = int(os.environ.get("KAS_CTX_MAX", "131072"))
        path = src.get("model_path")
        if not path:  # HF repo -> resolve the (cached) local file to read its header
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(repo_id=src["repo_id"], filename=src["filename"])
            except Exception as exc:
                log.warning("could not resolve GGUF for ctx peek (%s); using cap %d", exc, cap)
                path = None
        trained = _gguf_meta_int(path, ".context_length") if path else 0
        n = min(trained, cap) if trained else cap
        log.info("context: model trained=%s, cap=%s -> n_ctx=%s", trained or "?", cap, n)
        return n

    def _load(self, model_id: str) -> None:
        from ..prompting import detect_dialect

        n_gpu_layers = int(os.environ.get("KAS_GPU_LAYERS", "-1"))  # -1 = offload all
        # Resolve the GGUF source first (local path vs HF repo+file) so the context
        # window can be sized to *this* model rather than a magic constant.
        if model_id.endswith(".gguf") and os.path.exists(model_id):
            src = dict(model_path=model_id)
            ctor = self._Llama
        else:  # Hugging Face repo id -> pick the quant file
            filename = self._pick_gguf_file(
                model_id, os.environ.get("KAS_GGUF_FILE"), os.environ.get("KAS_GGUF_QUANT")
            )
            log.info("selected GGUF file: %s", filename)
            src = dict(repo_id=model_id, filename=filename)
            ctor = self._Llama.from_pretrained
        n_ctx = self._choose_ctx(src)
        # Flash Attention is a big win for the GGUF backend, and load-critical for
        # sliding-window models: gemma's interleaved-SWA layers have V embeddings of
        # different sizes, and WITHOUT FA llama.cpp pads every V cache to 4096 —
        # ballooning the KV until 'Failed to create llama_context' even on a clean
        # 40GB GPU. FA removes the padding (and speeds attention). On by default;
        # KAS_FLASH_ATTN=0 opts out for the rare build/quant that can't do it.
        flash_attn = os.environ.get("KAS_FLASH_ATTN", "1") not in ("0", "false", "False")
        t0 = time.time()
        # Allocating the KV at n_ctx can still exceed VRAM on smaller cards (or for
        # an unusually large model). Rather than hard-fail, back off — halve the
        # context and retry down to a floor — so kas loads with the largest window
        # the GPU can actually hold instead of refusing to start.
        floor = 2048
        while True:
            log.info(
                "loading %s (n_gpu_layers=%s, n_ctx=%s, flash_attn=%s) ...",
                model_id,
                n_gpu_layers,
                n_ctx,
                flash_attn,
            )
            try:
                self._llm = ctor(
                    **src,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    flash_attn=flash_attn,
                    verbose=False,
                )
                break
            except (ValueError, RuntimeError, MemoryError) as exc:
                if n_ctx <= floor:
                    raise
                n_ctx = max(floor, n_ctx // 2)
                log.warning("context alloc failed (%s); backing off to n_ctx=%s", exc, n_ctx)
        self.model_id = model_id
        self._cached_tokens = []  # a (re)load invalidates any prior KV
        meta = getattr(self._llm, "metadata", {}) or {}
        self._chat_template = meta.get("tokenizer.chat_template")
        self.context_length = int(getattr(self._llm, "n_ctx", lambda: n_ctx)())
        self.n_layers = n_gpu_layers if n_gpu_layers >= 0 else None
        self.dialect = detect_dialect(self._chat_template, self.model_id)
        self._stop_tokens = self._eog_tokens()
        self._is_eog = self._resolve_eog_fn()  # llama.cpp's authoritative EOG flag
        # Turn-end markers matched as STRINGS in the (special-rendered) output —
        # the reliable backstop when a GGUF's eot token id is missing/wrong. The
        # generic set covers most chat models; the active dialect adds its own
        # family-specific scaffolding (e.g. gemma-4's <turn|> / <|tool_response>),
        # which would otherwise leak into the visible answer.
        self._text_stops = (
            "<end_of_turn>",
            "<|im_end|>",
            "<|eot_id|>",
            "<|endoftext|>",
            "<eos>",
            "<|end|>",
            *getattr(self.dialect, "stop_strings", ()),
        )
        log.info(
            "loaded in %.1fs (dialect: %s, stop-tokens: %s)",
            time.time() - t0,
            self.dialect.name,
            sorted(self._stop_tokens),
        )

    def _eog_tokens(self) -> set[int]:
        """End-of-generation token ids. token_eos() alone isn't enough: a turn
        ends on a DIFFERENT token in most chat models (Gemma <end_of_turn>, ChatML
        <|im_end|>, Llama <|eot_id|>), so generation would otherwise run to
        max_tokens. Collect token_eos + token_eot (if exposed) + the ids of common
        turn-end strings, whichever the model actually defines."""
        stops: set[int] = set()
        # token_eos/token_eot live on the low-level _LlamaModel, not the Llama
        # wrapper — Gemma's turn-end <end_of_turn> is token_eot() there, not
        # token_eos(); missing it is exactly what made generation run to max_tokens.
        model = getattr(self._llm, "_model", None)
        for name in ("token_eos", "token_eot"):
            fn = getattr(model, name, None) or getattr(self._llm, name, None)
            if callable(fn):
                try:
                    t = fn()
                    if isinstance(t, int) and t >= 0:
                        stops.add(t)
                except Exception:
                    pass
        markers = (
            "<end_of_turn>",
            "<|im_end|>",
            "<|eot_id|>",
            "<|end|>",
            *getattr(self.dialect, "stop_strings", ()),  # dialect scaffolding (gemma <turn|> …)
        )
        for marker in markers:
            try:
                ids = self._llm.tokenize(marker.encode("utf-8"), add_bos=False, special=True)
                if len(ids) == 1 and ids[0] >= 0:  # a real single special token
                    stops.add(ids[0])
            except Exception:
                pass
        return stops or {self._llm.token_eos()}

    def _resolve_eog_fn(self):
        """A callable(token_id)->bool backed by llama.cpp's OWN end-of-generation
        flag — the authoritative answer to 'should generation stop here'. This is
        what catches a model's real end token when token_eos/eot are wrong or the
        marker isn't a single token we can pre-tokenize (e.g. <|im_end|> leaking
        through for some GGUFs). Best-effort and self-validating: it smoke-tests
        the binding against the known-EOG eos token at load and falls back to a
        no-op (current behaviour) if the binding/pointer shape doesn't match."""
        try:
            import llama_cpp

            model = self._llm._model  # low-level _LlamaModel wrapper
            eos = int(self._llm.token_eos())
            # Prefer the current vocab-based API; fall back to the model-based shim.
            get_vocab = getattr(llama_cpp, "llama_model_get_vocab", None)
            if get_vocab is not None and hasattr(llama_cpp, "llama_vocab_is_eog"):
                vocab = get_vocab(model.model)
                fn = llama_cpp.llama_vocab_is_eog
                if bool(fn(vocab, eos)):  # eos MUST be eog — proves the call works
                    return lambda t: bool(fn(vocab, t))
            fn = llama_cpp.llama_token_is_eog
            if bool(fn(model.model, eos)):
                return lambda t: bool(fn(model.model, t))
        except Exception as exc:
            log.warning("EOG flag unavailable (%s); relying on token/string stops", exc)
        return lambda t: False

    # --- tokenization --------------------------------------------------------

    def tokenize(
        self,
        chat_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
    ) -> list[int]:
        """Render the prompt via the GGUF's embedded chat template, then encode."""
        prompt = self._render_chat(chat_messages, tools, enable_thinking)
        ids = self._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        # Guarantee the BOS token. Many GGUFs leave `tokenizer.ggml.bos_token`
        # (the STRING) empty — only the id is stored — so the chat template emits
        # no <bos> and the model gets a head-less prompt -> garbage (this is what
        # broke Gemma). Prepend the real BOS id if it isn't already there.
        bos = self._llm.token_bos()
        if isinstance(bos, int) and bos >= 0 and (not ids or ids[0] != bos):
            ids = [bos, *ids]
        return ids

    def _render_chat(self, messages, tools, enable_thinking) -> str:
        if self._chat_template:
            try:
                from llama_cpp.llama_chat_format import Jinja2ChatFormatter

                meta = self._llm.metadata or {}
                fmt = Jinja2ChatFormatter(
                    template=self._chat_template,
                    bos_token=meta.get("tokenizer.ggml.bos_token", ""),
                    eos_token=meta.get("tokenizer.ggml.eos_token", ""),
                    add_generation_prompt=True,
                )
                kwargs = dict(getattr(self.dialect, "template_kwargs", {}))
                if tools:
                    kwargs["tools"] = tools
                return fmt(messages=messages, **kwargs).prompt
            except Exception as exc:  # bad/unknown template -> portable fallback
                log.warning("chat template render failed (%s); using plain fallback", exc)
        # Fallback: simple role-tagged concatenation (better than nothing).
        parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
        return "\n".join(parts) + "\nassistant:"

    def encode(self, text: str) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)

    def cache_snapshot(self, cache_key: str = "main") -> list[int]:
        """Tokens currently held in the KV cache (for prefix reuse + persistence)."""
        return list(self._cached_tokens)

    # --- generation ----------------------------------------------------------

    def _token_viz(self, tok: int) -> dict | None:
        """Per-token logprob summary for /viz from llama.cpp's last-step logits.
        Best-effort (numpy over the vocab); any failure -> None so generation is
        never affected. Same shape as the MLX backend, so the client is backend-
        agnostic."""
        try:
            import numpy as np

            scores = getattr(self._llm, "scores", None)
            if scores is None:
                scores = getattr(self._llm, "_scores", None)
            if scores is None:
                return None
            row = max(0, int(getattr(self._llm, "n_tokens", 1)) - 1)
            logits = np.asarray(scores[row], dtype="float64")
            if logits.ndim != 1 or logits.size == 0:
                return None
            p = np.exp(logits - logits.max())
            p /= p.sum()
            conf = float(p[tok])
            entropy = float(-(p * np.log(p + 1e-12)).sum())
            idx = np.argpartition(-p, 5)[:5]
            idx = idx[np.argsort(-p[idx])]
            top = [
                [self._llm.detokenize([int(i)]).decode("utf-8", "replace"), float(p[int(i)])]
                for i in idx
            ]
            return {"conf": conf, "entropy": entropy, "top": top}
        except Exception:
            return None

    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        stop_sequences: list[str],
        cache_key: str = "main",
        persist_dir: str | None = None,
        viz: bool = False,
    ) -> Iterator[GenChunk]:
        with self._lock:
            self._cancel.clear()
            t0 = time.time()
            gen_ids: list[int] = []
            text = ""
            finish: str | None = "length"
            full = list(prompt_tokens)
            # Reuse the KV prefix shared with the last request (append-only
            # transcripts share a long head): reset only when nothing overlaps, so
            # llama.cpp re-evals just the new suffix instead of the whole prompt.
            cached = longest_common_prefix(self._cached_tokens, full)

            def _drive(reset: bool):
                """Drive llama.cpp, yielding text-delta GenChunks. Raises on a
                llama.cpp decode error so the caller can retry from a clean KV."""
                nonlocal text, finish
                stream = self._llm.generate(
                    full,
                    temp=0.0 if temperature is None else float(temperature),
                    top_p=1.0 if top_p is None else float(top_p),
                    reset=reset,
                )
                for tok in stream:
                    if self._cancel.is_set():
                        finish = "stop"
                        return
                    if tok in self._stop_tokens or self._is_eog(tok):  # eos/eot/EOG flag
                        finish = "stop"
                        return
                    gen_ids.append(tok)
                    # special=True renders turn-end markers (<end_of_turn>, …) AS
                    # TEXT so we can stop on the STRING — the reliable signal when a
                    # GGUF's eot token id is wrong (Gemma here points eot at
                    # <start_of_turn>). We trim the marker off the user output below.
                    whole = self._llm.detokenize(gen_ids, special=True).decode("utf-8", "replace")
                    delta, text = whole[len(text) :], whole
                    tok_viz = self._token_viz(tok) if viz else None
                    hit = next(
                        (s for s in (*stop_sequences, *self._text_stops) if s and s in text), None
                    )
                    if hit:
                        cut = text.index(hit)
                        tail = text[len(text) - len(delta) : cut]
                        if tail:
                            yield GenChunk(text=tail, viz=tok_viz)
                        finish = "stop"
                        return
                    elapsed = max(1e-3, time.time() - t0)
                    self.stats = {
                        "active": True,
                        "phase": "generating",
                        "tps": round(len(gen_ids) / elapsed, 1),
                        "processed": len(gen_ids),
                        "total": max_tokens,
                    }
                    if delta:
                        yield GenChunk(text=delta, viz=tok_viz)
                    if len(gen_ids) >= max_tokens:
                        finish = "length"
                        return

            # llama.cpp decode failures split two ways, by whether we'd streamed
            # anything yet:
            #   * MID-GENERATION ('returned 1' = no free KV slot): the context
            #     filled — almost always a long agentic 'thinking' pass. Don't 500
            #     the turn; end it cleanly with what we have (finish='length') so
            #     the user can just continue — the next turn re-prefills with room.
            #   * ON PREFILL (before any token, typically 'returned -1'): KV
            #     prefix-reuse desynced. Clear the KV and re-prefill cold — a slower
            #     but reliable turn beats a broken one.
            yielded = False
            try:
                for chunk in _drive(reset=(cached == 0)):
                    yielded = True
                    yield chunk
            except RuntimeError as exc:
                if "llama_decode" not in str(exc):
                    raise
                if yielded:
                    log.warning(
                        "llama_decode failed mid-generation (KV full?); "
                        "ending turn early — continue to resume (%s)",
                        exc,
                    )
                    finish = "length"
                else:
                    log.warning(
                        "llama_decode failed on prefill; clearing KV, retrying cold (%s)", exc
                    )
                    gen_ids.clear()
                    text, finish, cached = "", "length", 0
                    try:
                        self._llm.reset()
                    except Exception:
                        pass
                    # The cold retry can ALSO fail when the prompt itself exceeds
                    # n_ctx (the conversation outgrew the window) — there's no KV
                    # slot for even the prefill. End the turn gracefully rather than
                    # 500-ing; the agent compacts and retries with a shorter prompt.
                    try:
                        for chunk in _drive(reset=True):
                            yielded = True
                            yield chunk
                    except RuntimeError as exc2:
                        if "llama_decode" not in str(exc2):
                            raise
                        log.warning(
                            "cold retry failed too (prompt > ctx of %s?); ending turn (%s)",
                            self.context_length,
                            exc2,
                        )
                        finish = "length"
            self.stats = {"active": False}
            # The cache now holds the prompt + every generated token except the
            # last (never fed back) — mirrors the MLX backend's accounting.
            self._cached_tokens = full + gen_ids[:-1] if gen_ids else full
            if persist_dir and not self._cancel.is_set():
                self._persist(cache_key, persist_dir)
            gen_tps = round(len(gen_ids) / max(1e-3, time.time() - t0), 1)
            yield GenChunk(
                text="",
                done=True,
                prompt_tokens=len(prompt_tokens),
                cached_tokens=cached,
                generation_tokens=len(gen_ids),
                generation_tps=gen_tps,
                finish_reason=finish,
            )

    # --- KV persistence (warm --resume; llama.cpp's own state files) ---------
    # Best-effort, like the MLX backend: any failure -> no persistence, never a
    # broken turn. llama.cpp serializes its OWN KV (ggml tensors in the context)
    # via llama_state_seq_save_file/load_file — a different format from MLX's
    # safetensors deltas, because the caches live in different memory. Full
    # snapshot per save (vs MLX's incremental deltas), but the restore is a fast
    # memcpy, so --resume fills the cache instantly instead of cold-prefilling.

    def _persist(self, thread: str, persist_dir: str) -> None:
        try:
            import llama_cpp

            kv, meta = _kv_paths(persist_dir, thread)
            kv.parent.mkdir(parents=True, exist_ok=True)
            toks = self._cached_tokens
            arr = (llama_cpp.llama_token * len(toks))(*toks)
            ctx = self._llm._ctx.ctx
            llama_cpp.llama_state_seq_save_file(ctx, str(kv).encode(), 0, arr, len(toks))
            meta.write_text(json.dumps({"model": self.model_id, "n": len(toks)}))
        except Exception:
            log.info("kv persist failed; next resume will cold-prefill", exc_info=True)

    def rehydrate(self, thread: str, persist_dir: str) -> str:
        """Restore this thread's KV from disk IF it was saved with the same model
        (a KV cache is model-specific). Returns a status string; the HTTP layer
        only treats a 'rehydrated …' prefix as a cache hit."""
        ok, reason = kv_restore_plan(persist_dir, thread, self.model_id)
        if not ok:
            return reason
        try:
            import ctypes

            import llama_cpp

            kv, meta = _kv_paths(persist_dir, thread)
            n = int(json.loads(meta.read_text())["n"])
            out = (llama_cpp.llama_token * n)()
            n_out = ctypes.c_size_t(0)
            ctx = self._llm._ctx.ctx
            llama_cpp.llama_state_seq_load_file(
                ctx, str(kv).encode(), 0, out, n, ctypes.byref(n_out)
            )
            self._cached_tokens = list(out[: n_out.value])
            self._llm.n_tokens = n_out.value  # tell the wrapper the cache is warm
            return f"rehydrated {n_out.value} KV tokens"
        except Exception:
            log.info("kv rehydrate failed; cold prefill", exc_info=True)
            self._cached_tokens = []
            return "rehydrate failed — cold prefill"

    # --- management ----------------------------------------------------------

    def swap(self, model_id: str) -> None:
        with self._lock:
            try:
                self._llm.close()
            except Exception:
                pass
            self._load(model_id)

    def request_cancel(self) -> bool:
        active = bool(self.stats.get("active"))
        self._cancel.set()
        return active

    def ping_status(self) -> dict[str, Any]:
        """Read-only, like the MLX engine's: this backend never emits engine-level
        pings itself (wall-clock keep-alives come from the pipeline), so the
        counters stay put unless generate() ever starts yielding pings."""
        age = round(time.monotonic() - self.last_ping_ts, 1) if self.last_ping_ts else None
        return {"pings": self.ping_count, "last_ping_age": age}

    def system_stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "layers": self.n_layers,
            "context_length": getattr(self, "context_length", None),
        }
        # GPU memory for the /stats panel. On NVIDIA, nvidia-smi gives device-wide
        # used/total; map used->gpu_active_gb, total->gpu_peak_gb so the panel
        # renders "used/total GB" with a fill gauge (same keys the MLX backend uses).
        mem = _nvidia_gpu_mem()
        if mem is not None:
            used_gb, total_gb, util = mem
            out["gpu_active_gb"] = used_gb
            out["gpu_peak_gb"] = total_gb
            out["gpu_util"] = util
        return out
