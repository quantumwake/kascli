"""Anthropic tool schemas advertised to the model. Pure data: the core loop
selects which sets to send (base TOOLS always; SUBAGENT_TOOL unless already a
subagent; RAG_TOOLS with --memory; WEB_TOOLS with --net), and the ToolRunner
adapter implements them.
"""

import os

# Default subagent round budget when the caller doesn't specify one, and the
# hard ceiling a caller-requested budget is clamped to (the volcano guard).
SUBAGENT_MAX_ROUNDS = int(os.environ.get("KAS_SUBAGENT_ROUNDS", "25"))
SUBAGENT_ROUNDS_CAP = int(os.environ.get("KAS_SUBAGENT_ROUNDS_CAP", "60"))

SUBAGENT_TOOL: dict = {
    "name": "subagent",
    "description": (
        "Delegate a self-contained subtask to a fresh agent with its own EMPTY "
        "context window. It has the same file/bash tools and working directory "
        "but sees NOTHING of this conversation — put every needed detail (file "
        "paths, requirements, conventions, constraints) into the task text. "
        "Only its final report returns to you, so use it to keep bulky work out "
        "of your context: analyzing large files or command output, building an "
        "isolated module, running a test-and-fix loop. Prefer it whenever a "
        "subtask would require reading lots of content you don't need to keep."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Complete, self-contained instructions for the subagent",
            },
            "report": {
                "type": "string",
                "description": "What the final report back to you must contain",
            },
            "max_rounds": {
                "type": "integer",
                "description": (
                    "How many tool-call rounds the subagent may use — scale to task "
                    f"complexity (default {SUBAGENT_MAX_ROUNDS}, max {SUBAGENT_ROUNDS_CAP}). "
                    "Give bigger/multi-step tasks more; trivial ones fewer."
                ),
            },
        },
        "required": ["task"],
    },
}

# Opt-in network tools (off unless --net / KAS_NET): web search + fetch. kas is
# offline by default — these are the only things that leave the machine.
WEB_TOOLS: list[dict] = [
    {
        "name": "web_search",
        "description": (
            "Search the web and return the top results (title, url, snippet). "
            "Use when the task needs current information or facts not in context, "
            "then web_fetch the most relevant url for full content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Default 5"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": (
            "Fetch a URL and return its main text content (article extraction, "
            "boilerplate stripped). Use after web_search, or on a URL the user gives."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to fetch"}},
            "required": ["url"],
        },
    },
]

# Opt-in local retrieval (--memory / KAS_MEMORY): ranked recall over the
# codebase, docs, and past session memory (including content compaction
# dropped). Complements grep — use it for "where/how is X" and to recall
# earlier decisions; use grep for exact strings.
RAG_TOOLS: list[dict] = [
    {
        "name": "recall",
        "description": (
            "Search a local index of this project's code + docs AND past session "
            "memory (decisions, summaries, content dropped by compaction), ranked "
            "by relevance. Use for 'where is X handled', 'how does Y work', or to "
            "remember earlier decisions — when you don't know the exact string to "
            "grep for. Returns the most relevant chunks with file:line."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for (natural language ok)",
                },
                "k": {"type": "integer", "description": "How many results (default 8)"},
            },
            "required": ["query"],
        },
    },
]

# Opt-in local image generation (--art / KAS_ART): render PNGs with a local
# diffusion model (FLUX via mflux on the Apple GPU). The image bytes are written
# to disk, not returned to the model.
IMAGE_TOOLS: list[dict] = [
    {
        "name": "generate_image",
        "description": (
            "Start generating an image with a LOCAL diffusion model — ASYNC: it renders "
            "in the BACKGROUND and returns IMMEDIATELY with a task id and the output path, "
            "so you keep working while it renders. Reference the returned path in your code "
            "right away; the PNG appears there when done (poll with image_status). Use for "
            "game sprites, textures, icons, concept art. Write a detailed prompt (subject, "
            "style, view/angle, background). For a CONSISTENT set of assets, keep the style "
            "wording identical across calls and pass a fixed integer `seed` per asset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description: subject, style, view/angle, background",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Output PNG path (relative to workdir; default assets/generated/<slug>.png)"
                    ),
                },
                "seed": {
                    "type": "integer",
                    "description": "Fix for reproducible / consistent results",
                },
                "steps": {
                    "type": "integer",
                    "description": "Inference steps (default suits distilled FLUX)",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "image_status",
        "description": (
            "Check background image-generation tasks started by generate_image: their "
            "status (running / done / error) and output paths. Pass a task_id for one, or "
            "omit to list all. Use it to confirm a sprite finished before relying on it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "A specific task id (omit to list all)",
                },
            },
        },
    },
    # Video generation rides the same --art gate as images: one opt-in for all
    # local generative media (the backends differ — mflux vs mlx-video — but
    # the capability the user grants is the same).
    {
        "name": "generate_video",
        "description": (
            "Start generating a short VIDEO (MP4) with a LOCAL model (LTX-2 via "
            "mlx-video) — ASYNC: it renders in the BACKGROUND and returns IMMEDIATELY "
            "with a task id and the output path. Rendering takes minutes; the FIRST use "
            "may also download the model (tens of GB — very slow once, cached after). "
            "Keep working and poll with video_status. Write a cinematic prompt (subject, "
            "motion, camera, lighting). Optionally pass `image` to animate an existing "
            "still (image-to-video). Frame count must be 4n+1 (default 33 ≈ 1.4 s)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description: subject, motion, camera, lighting",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Output MP4 path (relative to workdir; default assets/generated/<slug>.mp4)"
                    ),
                },
                "seed": {
                    "type": "integer",
                    "description": "Fix for reproducible results",
                },
                "frames": {
                    "type": "integer",
                    "description": "Frame count, must be 4n+1 (default 33; 24 fps)",
                },
                "image": {
                    "type": "string",
                    "description": "Optional input image path to animate (image-to-video)",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "video_status",
        "description": (
            "Check background video-generation tasks started by generate_video: their "
            "status (running / done / error) and output paths. Pass a task_id for one, "
            "or omit to list all."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "A specific task id (omit to list all)",
                },
            },
        },
    },
]

TOOLS: list[dict] = [
    {
        "name": "bash",
        "description": (
            "Run a shell command in the working directory (inside a pseudo-terminal) "
            "and return its output. Call this when you need to execute, build, test, "
            "search, or inspect anything not covered by the file tools. Prefer "
            "non-interactive flags (--yes, -y) where available; but if the command "
            "stops and waits for input (a prompt), you'll get the output so far and "
            "the process stays alive — answer it with bash_send_input, keep waiting "
            "with bash_wait, or stop it with bash_kill. Never re-run a command that "
            "is still running."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run"}},
            "required": ["command"],
        },
    },
    {
        "name": "bash_send_input",
        "description": (
            "Send a line of input to the still-running bash command (e.g. answer an "
            "interactive prompt like 'Ok to proceed? (y)'). A newline is appended. "
            "Returns the next chunk of output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Input to send (without trailing newline)",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "bash_wait",
        "description": (
            "Keep waiting for the still-running bash command and return its next "
            "output. Use when the command is doing slow work (installing, compiling) "
            "rather than waiting for input."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "bash_kill",
        "description": "Terminate the still-running bash command and return any final output.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": (
            "Read a text file. Call this before editing any file. For large files "
            "pass start_line/end_line (1-based, inclusive) to read only the relevant "
            "region — ranged reads are returned with line-number prefixes for "
            "navigation; full reads are returned verbatim (copy old_string for "
            "edit_file from a full or unprefixed read)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "start_line": {"type": "integer", "description": "First line (1-based)"},
                "end_line": {"type": "integer", "description": "Last line (inclusive)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or overwrite a file with the given content. Set append=true "
            "to append to the end of an existing file instead — use this to build "
            "large files in several smaller calls rather than one huge one (very "
            "large single calls risk hitting your output-token limit and being "
            "discarded)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {
                    "type": "boolean",
                    "description": "Append instead of overwrite (default false)",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file — for a SINGLE small spot. old_string "
            "must appear exactly once; read the file first to copy it verbatim. For "
            "several edits in one file, prefer apply_patch. Never rewrite the file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "apply_patch",
        "description": (
            "Apply a unified diff (git patch) to the workspace — the PREFERRED way to "
            "edit existing files. It changes only the affected lines (not the whole "
            "file) and lands several edits across a file in ONE call. Works even when "
            "the workspace isn't a git repo (no git init needed). Give a standard "
            "unified diff with @@ hunk headers; git-style 'a/path' and 'b/path' headers "
            "are fine, and context lines must match the current file. Use write_file "
            "only to create a brand-new file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "A unified diff to apply (one or more files/hunks)",
                },
            },
            "required": ["patch"],
        },
    },
    {
        "name": "list_dir",
        "description": "List the entries of a directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Defaults to ."}},
        },
    },
]
