"""Image-generation tool handlers, split out as a mixin on ToolRunner.

`generate_image` is ASYNC: it submits the render to a small thread pool and
returns immediately so the agent loop never blocks on the GPU; `image_status`
polls the task table. Both the pool and the task dict live on the ToolRunner
instance (self._art_pool / self._art_tasks), so this mixin holds no state of its
own. The mflux/render imports are deferred (the 'art' extra is optional).
"""


class ImageToolsMixin:
    def _art_executor(self):
        if self._art_pool is None:
            from concurrent.futures import ThreadPoolExecutor

            # cap concurrent renders — each is a separate GPU-using mflux process
            self._art_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="art")
        return self._art_pool

    def tool_generate_image(
        self,
        prompt: str,
        path: str | None = None,
        seed: int | None = None,
        steps: int | None = None,
    ) -> tuple[str, bool]:
        """ASYNC: kick off the render in the background and return immediately so
        the agent keeps working. The PNG appears at the returned path when done;
        poll with image_status."""
        from .image import render, resolve_out

        if not prompt or not prompt.strip():
            return "generate_image requires a non-empty 'prompt'", True
        # jail the output path under the sandbox (resolve_out passes absolute
        # paths through untouched — same policy as the file tools)
        out = self._paths.resolve(str(resolve_out(self.workdir, prompt, path)))
        self._art_seq += 1
        tid = self._art_seq
        self._art_tasks[tid] = {"status": "running", "prompt": prompt[:80], "path": str(out)}

        def work() -> None:
            try:
                output, err = render(prompt, out, seed=seed, steps=steps)
            except Exception as exc:  # a crash must never leave the task 'running' forever
                output, err = f"{type(exc).__name__}: {exc}", True
            self._art_tasks[tid].update(status="error" if err else "done", detail=output)

        self._art_executor().submit(work)
        return (
            f"image task #{tid} started in the background → {out}\n"
            "It renders while you keep working — reference that path now. Check progress with "
            f"image_status (task_id {tid}), or image_status() to list all tasks.",
            False,
        )

    def tool_image_status(self, task_id: int | None = None) -> tuple[str, bool]:
        if not self._art_tasks:
            return "no image tasks this session", False
        mark = {"running": "⏳", "done": "✓", "error": "✗"}

        def fmt(i: int, t: dict) -> str:
            line = f"#{i} {mark.get(t['status'], '?')} {t['status']}  {t['path']}"
            if t["status"] != "running" and t.get("detail"):
                line += f"\n    {t['detail'][:300]}"
            return line

        if task_id is not None:
            try:
                tid = int(task_id)
            except (TypeError, ValueError):
                return f"bad task_id {task_id!r}", True
            t = self._art_tasks.get(tid)
            if t is None:
                return f"no image task #{tid}", True
            return fmt(tid, t), t["status"] == "error"
        return "\n".join(fmt(i, t) for i, t in sorted(self._art_tasks.items())), False
