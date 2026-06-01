"""Animate existing workspace images with Luma, then rebuild final.mp4.

Reads workspace/images/scene_*.png + voice.mp3 + subtitles.srt,
sends images to Luma Dream Machine, downloads video clips, and
rebuilds final.mp4 — no image/story/voice API calls needed.

Usage::

    python -m src.animate_video
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from rich.console import Console

from src.config import get_config
from src.state import Scene, StoryState
from src.agents.video_animator import VideoAnimatorAgent
from src.agents.compositor import CompositorAgent
from src.agents.reviewer import ReviewerAgent


def _parse_srt_timing(srt_path: Path) -> list[tuple[float, float]]:
    content = srt_path.read_text(encoding="utf-8")
    timing: list[tuple[float, float]] = []
    for block in content.strip().split("\n\n"):
        for line in block.strip().splitlines():
            if " --> " in line:
                s, e = line.split(" --> ", 1)
                timing.append((_srt_to_sec(s.strip()), _srt_to_sec(e.strip())))
                break
    return timing


def _srt_to_sec(ts: str) -> float:
    time_part, ms = ts.split(",")
    h, m, s = time_part.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


async def animate_and_rebuild() -> None:
    console = Console()
    config = get_config()
    ws = config.workspace_dir

    # Validate artifacts
    images_dir = ws / "images"
    image_files = sorted(images_dir.glob("scene_*.png"))
    voice_path = ws / "voice.mp3"
    srt_path = ws / "subtitles.srt"

    for p in [voice_path, srt_path]:
        if not p.exists():
            console.print(f"[red]Missing: {p}[/red]")
            return
    if not image_files:
        console.print("[red]No scene_*.png found in workspace/images/[/red]")
        return

    console.print(
        f"[bold cyan]Animate + Rebuild: {len(image_files)} images found[/bold cyan]"
    )

    # Load or build state
    story_json = ws / "story.json"
    if story_json.exists():
        data = json.loads(story_json.read_text(encoding="utf-8"))
        state = StoryState(**data)
    else:
        state = StoryState(user_prompt="animated run")

    timing = _parse_srt_timing(srt_path)

    for i, img_file in enumerate(image_files):
        scene_id = i + 1
        scene = next((s for s in state.scenes if s.id == scene_id), None)
        if scene is None:
            scene = Scene(id=scene_id, narration=f"Scene {scene_id}.")
            state.scenes.append(scene)
        scene.image_path = str(img_file.relative_to(ws))
        if i < len(timing):
            scene.voice_segment_start = timing[i][0]
            scene.voice_segment_end = timing[i][1]
            scene.estimated_seconds = timing[i][1] - timing[i][0]

    state.voice_path = str(voice_path.relative_to(ws))
    state.subtitle_path = str(srt_path.relative_to(ws))

    # Run VideoAnimatorAgent
    console.print("[bold cyan]>> VideoAnimatorAgent (Luma Dream Machine)...[/bold cyan]")
    console.print("[dim]  Each clip takes ~90 seconds — please wait...[/dim]")
    animator = VideoAnimatorAgent()
    anim_result = await animator.run(state)
    console.print(
        f"[green][OK][/green] Animated {anim_result.output['animated']}/{anim_result.output['total']} scenes"
    )

    # Delete old mp4
    mp4 = ws / "final.mp4"
    if mp4.exists():
        mp4.unlink()
        console.print("[yellow]Deleted old final.mp4[/yellow]")

    # Run Compositor
    console.print("[bold cyan]>> CompositorAgent (FFmpeg encoding)...[/bold cyan]")
    compositor = CompositorAgent()
    comp_result = await compositor.run(state)
    file_size_kb = comp_result.output.get("file_size_bytes", 0) // 1024
    console.print(
        f"[green][OK][/green] Video: {file_size_kb:,} KB "
        f"({comp_result.output.get('duration_sec', 0):.1f}s)"
    )

    # Save state
    story_json.write_text(
        json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    console.print(f"\n[bold]Video:[/bold] {state.final_video_path}")
    console.print(
        f"[dim]Luma clips saved in workspace/videos/ — reuse with python -m src.recover_video[/dim]"
    )


if __name__ == "__main__":
    asyncio.run(animate_and_rebuild())
