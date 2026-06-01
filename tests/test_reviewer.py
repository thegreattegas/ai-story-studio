"""Tests for ReviewerAgent — Phase 4.

All tests run in mock mode (no Opus API calls).
"""

from __future__ import annotations

import asyncio

import pytest

from src.agents.compositor import CompositorAgent
from src.agents.image_agent import ImageAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.scene_director import SceneDirectorAgent
from src.agents.story_writer import StoryWriterAgent
from src.agents.subtitle_agent import SubtitleAgent
from src.agents.voice_agent import VoiceAgent
from src.state import StoryState


# ---------------------------------------------------------------------------
# Fixture: fully-composed state (all Phase 4 artifacts present)
# ---------------------------------------------------------------------------


@pytest.fixture
async def composed_state() -> StoryState:
    """Return a StoryState that has been through the full Phase 2-4 pipeline."""
    state = StoryState(user_prompt="a wolf and fox story")
    await StoryWriterAgent().run(state)
    await SceneDirectorAgent().run(state)
    await asyncio.gather(ImageAgent().run(state), VoiceAgent().run(state))
    await SubtitleAgent().run(state)
    await CompositorAgent().run(state)
    return state


# ---------------------------------------------------------------------------
# Test 1 — Mock mode returns APPROVED decision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_returns_approved(composed_state: StoryState) -> None:
    """ReviewerAgent in mock mode must return decision='APPROVED'."""
    state = composed_state
    agent = ReviewerAgent()
    result = await agent.run(state)

    assert result.mocked is True
    assert result.output["decision"] == "APPROVED", (
        f"Expected APPROVED, got {result.output['decision']!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Review output has all required fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_output_has_required_fields(composed_state: StoryState) -> None:
    """Review output dict must have 'decision', 'issues', and 'feedback' keys."""
    state = composed_state
    result = await ReviewerAgent().run(state)

    output = result.output
    assert "decision" in output, "Missing 'decision' key in reviewer output."
    assert "issues" in output, "Missing 'issues' key in reviewer output."
    assert "feedback" in output, "Missing 'feedback' key in reviewer output."

    assert output["decision"] in {"APPROVED", "NEEDS_FIXES"}, (
        f"decision must be APPROVED or NEEDS_FIXES, got {output['decision']!r}"
    )
    assert isinstance(output["issues"], list), "'issues' must be a list."
    assert isinstance(output["feedback"], str), "'feedback' must be a string."


# ---------------------------------------------------------------------------
# Test 3 — state.review_approved is set after run()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_review_approved_set(composed_state: StoryState) -> None:
    """state.review_approved must be True after a mock APPROVED review."""
    state = composed_state
    assert state.review_approved is False, (
        "review_approved should start False before ReviewerAgent runs."
    )

    await ReviewerAgent().run(state)

    assert state.review_approved is True, (
        "state.review_approved must be True after APPROVED review."
    )
    assert state.review_feedback, "state.review_feedback must be non-empty."
