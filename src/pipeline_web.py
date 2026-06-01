"""Web pipeline runner.

Wraps the full Phase 2-4 pipeline for use by the FastAPI server.
Progress events are emitted into an asyncio.Queue so the SSE endpoint
can stream them to the browser in real time.

Video naming
------------
Every generated video is saved as ``workspace/{title_slug}.mp4`` so
previously generated videos are never overwritten.  The stories/
directory keeps metadata + images; videos live in workspace/.

Workspace cleanup
-----------------
Before each run we remove only the transient per-run files
(voice.mp3, subtitles.srt, story.json, final.mp4, images/, tmp/).
Existing named .mp4 files in workspace/ are left untouched.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from src.config import get_config
from src.state import StoryState
from src.agents.story_writer import StoryWriterAgent
from src.agents.scene_director import SceneDirectorAgent
from src.agents.image_agent import ImageAgent
from src.agents.voice_agent import VoiceAgent
from src.agents.subtitle_agent import SubtitleAgent
from src.agents.compositor import CompositorAgent
from src.agents.reviewer import ReviewerAgent

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORIES_DIR   = _PROJECT_ROOT / "stories"

_TRANSIENT_FILES = ["voice.mp3", "subtitles.srt", "story.json", "final.mp4"]
_TRANSIENT_DIRS  = ["images", "tmp"]


def _title_slug(title: str) -> str:
    """Turn a story title into a safe filename stem.

    'The Little Dragon' -> 'the_little_dragon'
    "Brennan's Fire"    -> 'brennans_fire'
    """
    s = re.sub(r"[^\w\s-]", "", title.lower())
    s = re.sub(r"[\s_-]+", "_", s.strip())
    return s[:40] or "story"


def _clean_workspace(workspace: Path) -> None:
    """Remove only transient files — keep existing named .mp4 videos."""
    for name in _TRANSIENT_FILES:
        f = workspace / name
        if f.exists():
            f.unlink(missing_ok=True)
    for name in _TRANSIENT_DIRS:
        d = workspace / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


async def run_pipeline_for_web(
    user_prompt: str,
    story_id: str,
    progress_queue: asyncio.Queue,
) -> dict:
    """Run the full story pipeline and archive results.

    Emits progress dicts into *progress_queue* at each step.
    Returns a meta dict for the finished story.

    Event types emitted:
        {"type": "step",     "step": str, "status": "running"|"done", "label": str}
        {"type": "title",    "title": str}
        {"type": "complete", "story_id": str, ...}
        {"type": "error",    "message": str}
    """

    def emit(event: dict) -> None:
        progress_queue.put_nowait(event)

    config    = get_config()
    workspace = config.workspace_dir

    # Clean only transient files — preserve existing videos
    workspace.mkdir(parents=True, exist_ok=True)
    _clean_workspace(workspace)

    state = StoryState(user_prompt=user_prompt)
    comp_result_output: dict = {}

    try:
        # ---- Phase 2a: Story Writer ----------------------------------- #
        emit({"type": "step", "step": "story_writer", "status": "running",
              "label": "Writing your story..."})
        await StoryWriterAgent().run(state)
        emit({"type": "step", "step": "story_writer", "status": "done",
              "label": f'Story created: "{state.title}"'})
        emit({"type": "title", "title": state.title})

        # ---- Phase 2b: Scene Director --------------------------------- #
        emit({"type": "step", "step": "scene_director", "status": "running",
              "label": "Directing scenes..."})
        await SceneDirectorAgent().run(state)
        emit({"type": "step", "step": "scene_director", "status": "done",
              "label": f"Directed {len(state.scenes)} scenes"})

        # ---- Phase 3: Images + Voice (parallel) ---------------------- #
        emit({"type": "step", "step": "media", "status": "running",
              "label": "Generating images & voice..."})
        await asyncio.gather(ImageAgent().run(state), VoiceAgent().run(state))
        emit({"type": "step", "step": "media", "status": "done",
              "label": "Images and voice ready"})

        # ---- Phase 4a: Subtitles ------------------------------------- #
        emit({"type": "step", "step": "subtitles", "status": "running",
              "label": "Creating subtitles..."})
        await SubtitleAgent().run(state)
        emit({"type": "step", "step": "subtitles", "status": "done",
              "label": "Subtitles ready"})

        # ---- Phase 4b: Compositor ------------------------------------ #
        emit({"type": "step", "step": "compositor", "status": "running",
              "label": "Composing video..."})
        comp_result        = await CompositorAgent().run(state)
        comp_result_output = comp_result.output
        duration           = comp_result_output.get("duration_sec", 0)
        emit({"type": "step", "step": "compositor", "status": "done",
              "label": f"Video composed ({duration:.0f}s)"})

        # ---- Phase 4c: Reviewer ------------------------------------- #
        emit({"type": "step", "step": "reviewer", "status": "running",
              "label": "Reviewing quality..."})
        rev_result = await ReviewerAgent().run(state)
        emit({"type": "step", "step": "reviewer", "status": "done",
              "label": "Story ready!"})

    except Exception as exc:
        log.exception("pipeline_web: pipeline failed for story %s", story_id)
        emit({"type": "error", "message": str(exc)})
        raise

    # ------------------------------------------------------------------ #
    # Rename final.mp4 -> {slug}.mp4 in workspace/ (keep it there)        #
    # ------------------------------------------------------------------ #
    slug        = _title_slug(state.title)
    video_src   = workspace / "final.mp4"
    video_named = workspace / f"{slug}.mp4"

    # Handle duplicate slugs by appending the story_id suffix
    if video_named.exists() and video_src.exists():
        video_named = workspace / f"{slug}_{story_id}.mp4"

    if video_src.exists():
        video_src.rename(video_named)
        log.info("pipeline_web: renamed final.mp4 -> %s", video_named.name)

    video_url = f"/workspace/{video_named.name}"

    # ------------------------------------------------------------------ #
    # Archive metadata + images -> stories/{story_id}/                    #
    # (video stays in workspace/, no copy needed)                         #
    # ------------------------------------------------------------------ #
    story_dir = STORIES_DIR / story_id
    story_dir.mkdir(parents=True, exist_ok=True)

    images_src = workspace / "images"
    if images_src.exists():
        dest_images = story_dir / "images"
        if dest_images.exists():
            shutil.rmtree(dest_images)
        shutil.copytree(images_src, dest_images)

    if (workspace / "story.json").exists():
        shutil.copy2(workspace / "story.json", story_dir / "story.json")

    # Thumbnail
    thumbnail_url = f"/stories/{story_id}/images/scene_01.png"
    for ext in ("png", "jpg", "jpeg", "webp"):
        if (story_dir / "images" / f"scene_01.{ext}").exists():
            thumbnail_url = f"/stories/{story_id}/images/scene_01.{ext}"
            break

    meta = {
        "id":           story_id,
        "title":        state.title,
        "summary":      state.summary,
        "thumbnail":    thumbnail_url,
        "video":        video_url,
        "created_at":   datetime.now().isoformat(),
        "duration_sec": comp_result_output.get("duration_sec", 0),
        "scene_count":  len(state.scenes),
    }
    (story_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info("pipeline_web: story %s archived  video=%s", story_id, video_url)

    emit({
        "type":         "complete",
        "story_id":     story_id,
        "title":        state.title,
        "summary":      state.summary,
        "video":        video_url,
        "thumbnail":    thumbnail_url,
        "duration_sec": comp_result_output.get("duration_sec", 0),
    })

    return meta
