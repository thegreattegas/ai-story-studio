"""Tests for CompositorAgent — Phase 4.

Test 1: SRT time conversion (pure function, no I/O).
Test 2: SRT parsing (pure function, reads a string).
Test 3: End-to-end video creation using mock workspace assets.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agents.compositor import CompositorAgent
from src.agents.image_agent import ImageAgent
from src.agents.scene_director import SceneDirectorAgent
from src.agents.story_writer import StoryWriterAgent
from src.agents.subtitle_agent import SubtitleAgent
from src.agents.voice_agent import VoiceAgent
from src.state import StoryState


# ---------------------------------------------------------------------------
# Fixture: state after Phase 3 + SubtitleAgent
# ---------------------------------------------------------------------------


@pytest.fixture
async def phase4_state() -> StoryState:
    """Return a StoryState that has been through Phases 2, 3, and SubtitleAgent."""
    state = StoryState(user_prompt="a wolf and fox story")
    await StoryWriterAgent().run(state)
    await SceneDirectorAgent().run(state)
    await asyncio.gather(ImageAgent().run(state), VoiceAgent().run(state))
    await SubtitleAgent().run(state)
    return state


# ---------------------------------------------------------------------------
# Test 1 — SRT time format conversion (pure function)
# ---------------------------------------------------------------------------


def test_srt_time_to_seconds() -> None:
    """_srt_time_to_seconds must correctly parse HH:MM:SS,mmm."""
    agent = CompositorAgent.__new__(CompositorAgent)

    assert agent._srt_time_to_seconds("00:00:00,000") == 0.0
    assert agent._srt_time_to_seconds("00:00:01,000") == 1.0
    assert agent._srt_time_to_seconds("00:01:00,000") == 60.0
    assert agent._srt_time_to_seconds("01:00:00,000") == 3600.0
    assert abs(agent._srt_time_to_seconds("00:00:01,500") - 1.5) < 0.001
    assert abs(agent._srt_time_to_seconds("00:02:30,250") - 150.25) < 0.001


# ---------------------------------------------------------------------------
# Test 2 — SRT parsing (pure function)
# ---------------------------------------------------------------------------


def test_parse_srt_extracts_segments(tmp_path) -> None:
    """_parse_srt must extract correct start/end/text for each block."""
    srt_content = (
        "1\n00:00:00,000 --> 00:00:10,000\nScene one text.\n\n"
        "2\n00:00:10,000 --> 00:00:20,500\nScene two text here.\n\n"
        "3\n00:00:20,500 --> 00:00:30,000\nScene three.\n"
    )
    srt_file = tmp_path / "test.srt"
    srt_file.write_text(srt_content, encoding="utf-8")

    agent = CompositorAgent.__new__(CompositorAgent)
    segments = agent._parse_srt(srt_file)

    assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"

    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 10.0
    assert segments[0]["text"] == "Scene one text."

    assert abs(segments[1]["end"] - 20.5) < 0.001
    assert segments[1]["text"] == "Scene two text here."

    assert abs(segments[2]["start"] - 20.5) < 0.001


# ---------------------------------------------------------------------------
# Test 3 — End-to-end video creation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_video_created(phase4_state: StoryState) -> None:
    """CompositorAgent must create workspace/final.mp4 that exists and is non-empty.

    This test runs real FFmpeg on mock (tiny) assets — no API calls.
    """
    from src.config import get_config

    state = phase4_state
    agent = CompositorAgent()
    result = await agent.run(state)

    assert state.final_video_path, "state.final_video_path must be set."

    video_file = get_config().workspace_dir / "final.mp4"
    assert video_file.exists(), f"final.mp4 not found at {video_file}"
    assert video_file.stat().st_size > 0, "final.mp4 is empty!"

    assert result.mocked is False, "CompositorAgent should never be mocked."
    assert result.output.get("file_size_bytes", 0) > 0
