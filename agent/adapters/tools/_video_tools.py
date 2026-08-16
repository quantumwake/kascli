"""Video-generation tool handlers, split out as a mixin on ToolRunner.

Mirrors the image mixin, but on its OWN single-worker pool: one video can run
for minutes (up to VIDEO_TIMEOUT), so sharing the image pool would let two
videos starve every queued sprite render. `generate_video` is ASYNC — it
submits and returns immediately; tasks are 'queued' until a worker picks them
up, then 'running'; `video_status` polls the task table. Task state lives on
the ToolRunner instance (self._video_tasks); the mlx-video import is deferred
(the 'video' extra is optional).
"""


class VideoToolsMixin:
    def _video_executor(self):
        if self._video_pool is None:
            from concurrent.futures import ThreadPoolExecutor

            # one at a time: each render owns the GPU for minutes
            self._video_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video")
        return self._video_pool

    def tool_generate_video(
        self,
        prompt: str,
        path: str | None = None,
        seed: int | None = None,
        frames: int | None = None,
        image: str | None = None,
    ) -> tuple[str, bool]:
        """ASYNC: kick off the render in the background and return immediately.
        The MP4 appears at the returned path when done; poll with video_status."""
        from .video import render, resolve_out

        if not prompt or not prompt.strip():
            return "generate_video requires a non-empty 'prompt'", True
        # jail BOTH sides under the sandbox: the input image and the output
        # path (resolve_out passes absolute paths through untouched)
        out = self._paths.resolve(str(resolve_out(self.workdir, prompt, path)))
        img = str(self._paths.resolve(image)) if image else None
        self._video_seq += 1
        tid = self._video_seq
        self._video_tasks[tid] = {"status": "queued", "prompt": prompt[:80], "path": str(out)}

        def work() -> None:
            self._video_tasks[tid]["status"] = "running"
            try:
                output, err = render(prompt, out, seed=seed, frames=frames, image=img)
            except Exception as exc:  # a crash must never leave the task 'running' forever
                output, err = f"{type(exc).__name__}: {exc}", True
            self._video_tasks[tid].update(status="error" if err else "done", detail=output)

        self._video_executor().submit(work)
        return (
            f"video task #{tid} started in the background → {out}\n"
            "Rendering takes minutes (first use also downloads the model — much longer). "
            f"Keep working and check progress with video_status (task_id {tid}), or "
            "video_status() to list all tasks.",
            False,
        )

    def tool_video_status(self, task_id: int | None = None) -> tuple[str, bool]:
        if not self._video_tasks:
            return "no video tasks this session", False
        mark = {"queued": "…", "running": "⏳", "done": "✓", "error": "✗"}

        def fmt(i: int, t: dict) -> str:
            line = f"#{i} {mark.get(t['status'], '?')} {t['status']}  {t['path']}"
            if t["status"] not in ("queued", "running") and t.get("detail"):
                line += f"\n    {t['detail'][:300]}"
            return line

        if task_id is not None:
            try:
                tid = int(task_id)
            except (TypeError, ValueError):
                return f"bad task_id {task_id!r}", True
            t = self._video_tasks.get(tid)
            if t is None:
                return f"no video task #{tid}", True
            return fmt(tid, t), t["status"] == "error"
        return "\n".join(fmt(i, t) for i, t in sorted(self._video_tasks.items())), False
