"""FastAPI web server for AI Story Studio.

Run with:
    python -m src.server
    uvicorn src.server:app --reload

Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline_web import run_pipeline_for_web, STORIES_DIR
from src.web_builder import run_web_builder

log = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_WEB_DIR       = _PROJECT_ROOT / "web"
_WORKSPACE_DIR = _PROJECT_ROOT / "workspace"

# Ensure required directories exist before mounting
STORIES_DIR.mkdir(parents=True, exist_ok=True)
_WEB_DIR.mkdir(parents=True, exist_ok=True)
_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Story Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/web",       StaticFiles(directory=str(_WEB_DIR)),       name="web")
app.mount("/stories",   StaticFiles(directory=str(STORIES_DIR)),    name="stories")
app.mount("/workspace", StaticFiles(directory=str(_WORKSPACE_DIR)), name="workspace")

# ---------------------------------------------------------------------------
# In-memory job store + generation lock
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_is_generating: bool = False
_improve_jobs: dict[str, dict] = {}   # web-builder jobs


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str


class ImproveRequest(BaseModel):
    target: str = "all"        # "qa" | "design" | "frontend" | "backend" | "review" | "all"
    instruction: str = ""      # free-text, optional


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_WEB_DIR / "index.html")


@app.get("/api/stories")
async def list_stories() -> dict:
    """Return all archived stories sorted by newest first."""
    stories: list[dict] = []
    if STORIES_DIR.exists():
        entries = sorted(
            (d for d in STORIES_DIR.iterdir() if d.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for story_dir in entries:
            meta_file = story_dir / "meta.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    stories.append(meta)
                except Exception:
                    pass
    return {"stories": stories}


@app.get("/api/status")
async def server_status() -> dict:
    return {"generating": _is_generating, "story_count": len(list(STORIES_DIR.iterdir())) if STORIES_DIR.exists() else 0}


@app.post("/api/generate")
async def generate_story(request: GenerateRequest) -> dict:
    """Start a new story generation job. Returns a job_id for SSE streaming."""
    global _is_generating

    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if _is_generating:
        raise HTTPException(
            status_code=409,
            detail="A story is already being generated. Please wait for it to finish.",
        )

    job_id = str(uuid.uuid4())
    story_id = str(uuid.uuid4())[:8]
    queue: asyncio.Queue = asyncio.Queue()

    _jobs[job_id] = {
        "queue": queue,
        "status": "running",
        "story_id": story_id,
        "prompt": request.prompt,
    }

    async def _run() -> None:
        global _is_generating
        _is_generating = True
        try:
            await run_pipeline_for_web(
                user_prompt=request.prompt,
                story_id=story_id,
                progress_queue=queue,
            )
        except Exception as exc:
            log.error("server: pipeline error for job %s: %s", job_id, exc)
            # error event already emitted by pipeline_web
        finally:
            _is_generating = False
            _jobs[job_id]["status"] = "done"

    asyncio.create_task(_run())

    return {"job_id": job_id, "story_id": story_id}


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    """Server-Sent Events endpoint — streams progress for a generation job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    queue: asyncio.Queue = _jobs[job_id]["queue"]

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=120.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                # Keep-alive ping
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Web builder routes
# ---------------------------------------------------------------------------

@app.post("/api/improve")
async def start_improve(request: ImproveRequest) -> dict:
    """Start a web-builder improvement job. Returns job_id for SSE streaming."""
    valid_targets = {"qa", "design", "frontend", "backend", "review", "all"}
    if request.target not in valid_targets:
        raise HTTPException(status_code=400, detail=f"target must be one of {sorted(valid_targets)}")

    # Cap stored jobs to avoid memory growth
    if len(_improve_jobs) > 20:
        oldest = next(iter(_improve_jobs))
        del _improve_jobs[oldest]

    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _improve_jobs[job_id] = {"queue": queue, "status": "running"}

    async def _run() -> None:
        try:
            await run_web_builder(
                target=request.target,
                instruction=request.instruction,
                progress_queue=queue,
            )
        except Exception as exc:
            log.error("server: improve error for job %s: %s", job_id, exc)
            queue.put_nowait({"type": "error", "message": str(exc)})
        finally:
            _improve_jobs[job_id]["status"] = "done"

    asyncio.create_task(_run())
    return {"job_id": job_id, "target": request.target}


@app.get("/api/improve/{job_id}/stream")
async def stream_improve(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream for a web-builder job."""
    if job_id not in _improve_jobs:
        raise HTTPException(status_code=404, detail="Improve job not found.")

    queue: asyncio.Queue = _improve_jobs[job_id]["queue"]

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=120.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("web_complete", "error"):
                    break
            except asyncio.TimeoutError:
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start() -> None:
    """Start the server (used by project script entry point)."""
    import uvicorn
    import webbrowser
    import threading

    def _open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    start()
