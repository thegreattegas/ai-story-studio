"""Recovery script — rebuild final.mp4 from existing workspace artifacts.

Reconstructs the pipeline state from disk (images, voice.mp3, subtitles.srt)
and runs CompositorAgent + ReviewerAgent without re-calling any image/TTS APIs.

Usage::

    python -m src.recover_video
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from rich.console import Console

from src.config import get_config
from src.state import Scene, StoryState
from src.agents.compositor import CompositorAgent
from src.agents.reviewer import ReviewerAgent


def _parse_srt_timing(srt_path: Path) -> list[tuple[float, float]]:
    """Return list of (start_sec, end_sec) tuples from an SRT file."""
    content = srt_path.read_text(encoding="utf-8")
    timing: list[tuple[float, float]] = []
    for block in content.strip().split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # Find the line with '-->'
        for line in lines:
            if " --> " in line:
                start_str, end_str = line.split(" --> ", 1)
                timing.append((_srt_to_sec(start_str.strip()), _srt_to_sec(end_str.strip())))
                break
    return timing


def _srt_to_sec(ts: str) -> float:
    """Convert HH:MM:SS,mmm to float seconds."""
    time_part, ms = ts.split(",")
    h, m, s = time_part.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


async def recover() -> None:
    console = Console()
    config = get_config()
    ws = config.workspace_dir

    # --- Validate required artifacts ---
    images_dir = ws / "images"
    image_files = sorted(images_dir.glob("scene_*.png"))
    voice_path = ws / "voice.mp3"
    srt_path = ws / "subtitles.srt"

    missing = [p for p in [voice_path, srt_path] if not p.exists()]
    missing += [p for p in image_files if not p.exists()]
    if not image_files:
        missing.append(images_dir / "scene_*.png")
    if missing:
        console.print(f"[red]Missing artifacts:[/red] {missing}")
        return

    console.print(f"[bold cyan]Recovery: found {len(image_files)} images, voice.mp3, subtitles.srt[/bold cyan]")

    # --- Load base story from story.json ---
    story_json = ws / "story.json"
    if story_json.exists():
        data = json.loads(story_json.read_text(encoding="utf-8"))
        state = StoryState(**data)
    else:
        console.print("[yellow]story.json not found — building minimal state[/yellow]")
        state = StoryState(user_prompt="recovered run")

    # --- Rebuild scene list if enrichment fields are missing ---
    timing = _parse_srt_timing(srt_path)

    for i, img_file in enumerate(image_files):
        scene_id = i + 1
        # Find or create scene
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

    # --- Delete corrupt mp4 if it exists ---
    mp4 = ws / "final.mp4"
    if mp4.exists():
        mp4.unlink()
        console.print("[yellow]Deleted corrupt final.mp4[/yellow]")

    # --- Run Compositor ---
    console.print("[bold cyan]>> CompositorAgent (FFmpeg encoding)...[/bold cyan]")
    compositor = CompositorAgent()
    comp_result = await compositor.run(state)
    file_size_kb = comp_result.output.get("file_size_bytes", 0) // 1024
    console.print(
        f"[green][OK][/green] Video: {file_size_kb:,} KB "
        f"({comp_result.output.get('duration_sec', 0):.1f}s)"
    )

    # --- Run Reviewer ---
    console.print("[bold cyan]>> ReviewerAgent...[/bold cyan]")
    reviewer = ReviewerAgent()
    rev_result = await reviewer.run(state)
    decision = rev_result.output.get("decision", "UNKNOWN")
    console.print(
        f"[green][OK][/green] Reviewer: [bold]{decision}[/bold] "
        f"cost=${rev_result.cost_usd:.4f}"
    )
    for issue in rev_result.output.get("issues", []):
        console.print(f"  [yellow]![/yellow] {issue}")
    if rev_result.output.get("feedback"):
        console.print(f"[dim]{rev_result.output['feedback']}[/dim]")

    # --- Save updated story.json ---
    story_json.write_text(
        json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"\n[bold]Video:[/bold] {state.final_video_path}")
    console.print(f"[bold]Review:[/bold] {decision}")


if __name__ == "__main__":
    asyncio.run(recover())
