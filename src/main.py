"""CLI entry point for AI Story Studio.

Phases 1-4: story writing, scene direction, media generation, and video composition.

Usage::

    python -m src.main
    python -m src.main "a fairy tale about a wolf and a fox in a winter forest"
    python -m src.main "сказка про волка и лису"   # non-English prompts work too
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import get_config
from src.state import StoryState
from src.agents.story_writer import StoryWriterAgent
from src.agents.scene_director import SceneDirectorAgent
from src.agents.image_agent import ImageAgent
from src.agents.voice_agent import VoiceAgent
from src.agents.subtitle_agent import SubtitleAgent
from src.agents.compositor import CompositorAgent
from src.agents.reviewer import ReviewerAgent
from src.tools.file_tools import write_file

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORIES_DIR   = _PROJECT_ROOT / "stories"


async def run_pipeline(user_prompt: str) -> None:
    """Run Phases 1-4 of the pipeline.

    Phase 2: story writing + scene direction.
    Phase 3: image generation + voice narration (parallel).
    Phase 4: subtitles + video composition + reviewer quality gate.

    Args:
        user_prompt: Raw user input (any language treated as a topic hint).
    """
    console = Console()
    config = get_config()

    console.print(
        Panel.fit(
            f"[bold]Prompt:[/bold] {user_prompt}",
            title="[bold cyan]AI Story Studio[/bold cyan]",
        )
    )
    console.print(f"MOCK_MODE: {config.mock_mode}\n")

    state = StoryState(user_prompt=user_prompt)

    # ------------------------------------------------------------------ #
    # Phase 2a: Story Writer                                               #
    # ------------------------------------------------------------------ #
    console.print("[bold cyan]>> StoryWriter starting...[/bold cyan]")
    writer = StoryWriterAgent()
    result1 = await writer.run(state)
    console.print(
        f"[green][OK][/green] StoryWriter done: "
        f"'{state.title}' ({len(state.scenes)} scenes) "
        f"cost=${result1.cost_usd:.4f}\n"
    )

    # ------------------------------------------------------------------ #
    # Phase 2b: Scene Director (parallel per-scene internally)            #
    # ------------------------------------------------------------------ #
    console.print(
        "[bold cyan]>> SceneDirector starting (parallel per scene)...[/bold cyan]"
    )
    director = SceneDirectorAgent()
    result2 = await director.run(state)
    console.print(
        f"[green][OK][/green] SceneDirector done: "
        f"{len(state.scenes)} scenes enriched "
        f"cost=${result2.cost_usd:.4f}\n"
    )

    # ------------------------------------------------------------------ #
    # Phase 3: ImageAgent + VoiceAgent in PARALLEL                        #
    # They are fully independent — images don't depend on voice and       #
    # vice versa — so we launch both simultaneously via asyncio.gather.   #
    # ------------------------------------------------------------------ #
    console.print(
        "[bold cyan]>> Phase 3: ImageAgent + VoiceAgent starting in parallel...[/bold cyan]"
    )
    image_agent = ImageAgent()
    voice_agent = VoiceAgent()

    phase3_start = time.monotonic()
    image_result, voice_result = await asyncio.gather(
        image_agent.run(state),
        voice_agent.run(state),
    )
    phase3_elapsed = time.monotonic() - phase3_start

    console.print(
        f"[green][OK][/green] ImageAgent done: "
        f"{image_result.output.get('scenes_processed', '?')} images "
        f"cost=${image_result.cost_usd:.4f}"
    )
    console.print(
        f"[green][OK][/green] VoiceAgent done: "
        f"{voice_result.output.get('characters', '?')} chars "
        f"cost=${voice_result.cost_usd:.4f}"
    )
    console.print(
        f"[dim]  Phase 3 wall time: {phase3_elapsed:.2f}s "
        f"(sum of individual: {image_result.elapsed_sec + voice_result.elapsed_sec:.2f}s)[/dim]\n"
    )

    # ------------------------------------------------------------------ #
    # Phase 4a: SubtitleAgent — analyse voice.mp3, write subtitles.srt   #
    # ------------------------------------------------------------------ #
    console.print("[bold cyan]>> SubtitleAgent starting...[/bold cyan]")
    subtitle_agent = SubtitleAgent()
    sub_result = await subtitle_agent.run(state)
    console.print(
        f"[green][OK][/green] SubtitleAgent done: "
        f"{sub_result.output['segments']} segments "
        f"cost=${sub_result.cost_usd:.4f}\n"
    )

    # ------------------------------------------------------------------ #
    # Phase 4b: CompositorAgent — compose images + voice + subs → mp4    #
    # ------------------------------------------------------------------ #
    console.print(
        "[bold cyan]>> CompositorAgent starting (FFmpeg encoding)...[/bold cyan]"
    )
    compositor = CompositorAgent()
    comp_result = await compositor.run(state)
    file_size_kb = comp_result.output.get("file_size_bytes", 0) // 1024
    console.print(
        f"[green][OK][/green] CompositorAgent done: "
        f"{file_size_kb:,} KB "
        f"({comp_result.output.get('duration_sec', 0):.1f}s)\n"
    )

    # ------------------------------------------------------------------ #
    # Phase 4c: ReviewerAgent — quality gate                              #
    # ------------------------------------------------------------------ #
    console.print("[bold cyan]>> ReviewerAgent starting...[/bold cyan]")
    reviewer = ReviewerAgent()
    rev_result = await reviewer.run(state)
    decision = rev_result.output.get("decision", "UNKNOWN")
    console.print(
        f"[green][OK][/green] Reviewer decision: [bold]{decision}[/bold] "
        f"cost=${rev_result.cost_usd:.4f}"
    )
    for issue in rev_result.output.get("issues", []):
        console.print(f"  [yellow]![/yellow] {issue}")
    if rev_result.output.get("feedback"):
        console.print(f"[dim]{rev_result.output['feedback']}[/dim]")
    console.print()

    # ------------------------------------------------------------------ #
    # Save final story plan (all paths populated)                         #
    # ------------------------------------------------------------------ #
    write_file(
        "story.json",
        json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
    )

    # ------------------------------------------------------------------ #
    # Archive to stories/ so it appears in the web library                #
    # ------------------------------------------------------------------ #
    _archive_to_library(state, config, comp_result)

    # ------------------------------------------------------------------ #
    # Display story plan table                                             #
    # ------------------------------------------------------------------ #
    table = Table(title="Story Plan", show_lines=True)
    table.add_column("#", style="cyan", width=3)
    table.add_column("Mood", style="magenta", width=10)
    table.add_column("Narration", style="white", min_width=25, max_width=40)
    table.add_column("Image", style="green", width=8)
    table.add_column("Timing", style="yellow", width=12)

    for scene in state.scenes:
        narr = scene.narration
        narr_preview = narr[:45] + "..." if len(narr) > 45 else narr
        img = "[OK]" if scene.image_path else "[none]"
        timing = (
            f"{scene.voice_segment_start:.1f}-{scene.voice_segment_end:.1f}s"
            if scene.voice_segment_start is not None
            else "-"
        )
        table.add_row(str(scene.id), scene.mood, narr_preview, img, timing)

    console.print(table)

    # ------------------------------------------------------------------ #
    # Cost report                                                          #
    # ------------------------------------------------------------------ #
    total_cost = (
        result1.cost_usd + result2.cost_usd
        + image_result.cost_usd + voice_result.cost_usd
        + sub_result.cost_usd + comp_result.cost_usd + rev_result.cost_usd
    )

    COST_WARNING_THRESHOLD = 1.50

    console.print(f"\n[bold]Total cost:[/bold] ${total_cost:.4f}")
    if total_cost > COST_WARNING_THRESHOLD:
        console.print(
            f"\n[bold yellow]Cost ${total_cost:.4f} exceeds expected ${COST_WARNING_THRESHOLD}[/bold yellow]"
        )
        console.print("[yellow]Likely cause: StoryWriter produced too many scenes.[/yellow]")
    console.print(f"[bold]Review:[/bold] {decision}")
    console.print(f"[bold]Saved:[/bold] workspace/story.json")

    console.print("\n[bold]Generated artifacts:[/bold]")
    console.print(
        f"  Images:    workspace/images/scene_*.png ({len(state.scenes)} files)"
    )
    console.print(f"  Voice:     workspace/voice.mp3")
    console.print(f"  Subtitles: workspace/subtitles.srt")
    console.print(f"  Video:     {state.final_video_path}")

    console.print("\n[dim]Phase 4 complete. Next: Phase 5 - LangGraph + retry loop.[/dim]")


def _title_slug(title: str) -> str:
    import re
    s = re.sub(r"[^\w\s-]", "", title.lower())
    s = re.sub(r"[\s_-]+", "_", s.strip())
    return s[:40] or "story"


def _archive_to_library(state: StoryState, config, comp_result) -> None:
    """Rename video by story title and archive metadata into stories/."""
    try:
        story_id  = str(uuid.uuid4())[:8]
        workspace = config.workspace_dir
        slug      = _title_slug(state.title)

        # Rename final.mp4 -> {slug}.mp4 in workspace (keep it there)
        final_mp4   = workspace / "final.mp4"
        named_video = workspace / f"{slug}.mp4"
        if named_video.exists():
            named_video = workspace / f"{slug}_{story_id}.mp4"
        if final_mp4.exists():
            final_mp4.rename(named_video)

        video_url = f"/workspace/{named_video.name}"

        # Archive metadata + images to stories/{id}/
        story_dir   = STORIES_DIR / story_id
        story_dir.mkdir(parents=True, exist_ok=True)

        images_src = workspace / "images"
        if not images_src.exists():
            images_src = workspace
        dest_images = story_dir / "images"
        dest_images.mkdir(exist_ok=True)
        for img in images_src.glob("scene_*.png"):
            shutil.copy2(img, dest_images / img.name)

        if (workspace / "story.json").exists():
            shutil.copy2(workspace / "story.json", story_dir / "story.json")

        thumb    = f"/stories/{story_id}/images/scene_01.png"
        duration = comp_result.output.get("duration_sec", state.target_duration_sec or 0)

        meta = {
            "id":           story_id,
            "title":        state.title,
            "summary":      state.summary,
            "thumbnail":    thumb,
            "video":        video_url,
            "created_at":   datetime.now().isoformat(),
            "duration_sec": round(float(duration), 1),
            "scene_count":  len(state.scenes),
        }
        (story_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  Saved: workspace/{named_video.name}  library: stories/{story_id}/")
    except Exception as exc:
        print(f"  Warning: could not archive story — {exc}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "a fairy tale about a wolf and a fox in a winter forest"
    asyncio.run(run_pipeline(prompt))
