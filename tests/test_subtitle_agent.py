"""Tests for SubtitleAgent — Phase 4.

All tests run in mock mode (no real Whisper API calls).
The mock WhisperProvider returns a fixed 5-segment transcript.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from src.agents.image_agent import ImageAgent
from src.agents.scene_director import SceneDirectorAgent
from src.agents.story_writer import StoryWriterAgent
from src.agents.subtitle_agent import SubtitleAgent
from src.agents.voice_agent import VoiceAgent
from src.state import Scene, StoryState


# ---------------------------------------------------------------------------
# Fixture: state after Phase 3 (image_path + voice_path set)
# ---------------------------------------------------------------------------


@pytest.fixture
async def phase3_state() -> StoryState:
    """Return a StoryState that has been through Phases 2 + 3."""
    state = StoryState(user_prompt="a wolf and fox story")
    await StoryWriterAgent().run(state)
    await SceneDirectorAgent().run(state)
    await asyncio.gather(ImageAgent().run(state), VoiceAgent().run(state))
    return state


# ---------------------------------------------------------------------------
# Test 1 — Mock mode produces a subtitles.srt file on disk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_produces_srt_file(phase3_state: StoryState) -> None:
    """After SubtitleAgent.run(), workspace/subtitles.srt must exist."""
    from src.config import get_config

    state = phase3_state
    agent = SubtitleAgent()
    result = await agent.run(state)

    assert result.mocked is True
    assert state.subtitle_path, "state.subtitle_path must be set."

    srt_file = get_config().workspace_dir / "subtitles.srt"
    assert srt_file.exists(), f"subtitles.srt not found at {srt_file}"
    assert srt_file.stat().st_size > 0, "subtitles.srt is empty!"


# ---------------------------------------------------------------------------
# Test 2 — SRT file has valid format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_srt_format_is_valid(phase3_state: StoryState) -> None:
    """SRT file must have numbered blocks with HH:MM:SS,mmm --> HH:MM:SS,mmm timing."""
    from src.config import get_config

    state = phase3_state
    await SubtitleAgent().run(state)

    srt_file = get_config().workspace_dir / "subtitles.srt"
    content = srt_file.read_text(encoding="utf-8")

    # Split into non-empty blocks.
    blocks = [b.strip() for b in content.strip().split("\n\n") if b.strip()]
    assert len(blocks) >= 1, "SRT has no subtitle blocks."

    timing_re = re.compile(
        r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$"
    )

    for i, block in enumerate(blocks, 1):
        lines = block.splitlines()
        assert len(lines) >= 3, f"Block {i} has fewer than 3 lines: {block!r}"
        # Line 1: sequence number.
        assert lines[0].strip().isdigit(), (
            f"Block {i}: first line must be sequence number, got {lines[0]!r}"
        )
        # Line 2: timing.
        assert timing_re.match(lines[1].strip()), (
            f"Block {i}: timing line has wrong format: {lines[1]!r}"
        )
        # Line 3+: text.
        assert lines[2].strip(), f"Block {i}: text line is empty."


# ---------------------------------------------------------------------------
# Test 3 — _realign_to_scenes distributes to correct count
# ---------------------------------------------------------------------------


def test_realign_distributes_to_scene_count() -> None:
    """_realign_to_scenes must return exactly len(scenes) segments."""
    agent = SubtitleAgent.__new__(SubtitleAgent)
    # Minimal construction without calling __init__ (avoids get_config).
    from src.config import get_config

    agent.config = get_config()

    scenes = [
        Scene(id=i, narration=f"Scene {i} narration text here.", estimated_seconds=8.0)
        for i in range(1, 6)
    ]

    # Simulate a Whisper result with a different segment count (3 != 5).
    whisper_segments = [
        {"start": 0.0, "end": 15.0, "text": "Part one."},
        {"start": 15.0, "end": 30.0, "text": "Part two."},
        {"start": 30.0, "end": 50.0, "text": "Part three."},
    ]

    realigned = agent._realign_to_scenes(scenes, whisper_segments)

    assert len(realigned) == len(scenes), (
        f"Expected {len(scenes)} realigned segments, got {len(realigned)}"
    )
    # Segments must be consecutive (end of one == start of next).
    for j in range(len(realigned) - 1):
        assert abs(realigned[j]["end"] - realigned[j + 1]["start"]) < 0.01, (
            f"Gap between segment {j} and {j+1}: "
            f"{realigned[j]['end']:.3f} vs {realigned[j+1]['start']:.3f}"
        )
    # Total duration must equal whisper total.
    assert abs(realigned[-1]["end"] - whisper_segments[-1]["end"]) < 0.01
