"""Tests for StoryWriterAgent — 5 cases covering output shape and content."""

from __future__ import annotations

import pytest

from src.agents.story_writer import MOCK_STORY, StoryWriterAgent
from src.state import StoryState


@pytest.fixture
def state() -> StoryState:
    """Provide a fresh StoryState for each test."""
    return StoryState(user_prompt="a fairy tale about a wolf and a fox in a winter forest")


@pytest.fixture
async def writer_result(state: StoryState):
    """Run StoryWriterAgent in mock mode and return (result, state)."""
    agent = StoryWriterAgent()
    result = await agent.run(state)
    return result, state


# ---------------------------------------------------------------------------
# Test 1 — Mock mode returns valid StoryState with >= 3 scenes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_mode_produces_scenes(state: StoryState) -> None:
    """Mock mode should populate state.scenes with at least 3 entries."""
    agent = StoryWriterAgent()
    result = await agent.run(state)

    assert result.mocked is True, "Agent should be in mock mode during tests."
    assert len(state.scenes) >= 3, (
        f"Expected at least 3 scenes, got {len(state.scenes)}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Story JSON has all required top-level fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_required_top_level_fields(state: StoryState) -> None:
    """State should contain title, summary, language, target_duration_sec after run."""
    agent = StoryWriterAgent()
    await agent.run(state)

    assert state.title, "state.title must be non-empty."
    assert state.summary, "state.summary must be non-empty."
    assert state.language == "en", (
        f"language must always be 'en', got '{state.language}'"
    )
    assert isinstance(state.target_duration_sec, int), (
        "target_duration_sec must be an int."
    )
    assert state.target_duration_sec > 0, "target_duration_sec must be positive."


# ---------------------------------------------------------------------------
# Test 3 — Each scene has narration and mood
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_scene_has_narration_and_mood(state: StoryState) -> None:
    """Every Scene object must have non-empty narration and a valid mood."""
    agent = StoryWriterAgent()
    await agent.run(state)

    valid_moods = {
        "peaceful", "tense", "joyful", "sad", "mysterious",
        "exciting", "triumphant", "neutral",
    }

    for scene in state.scenes:
        assert scene.narration, f"Scene {scene.id}: narration must be non-empty."
        assert scene.mood in valid_moods, (
            f"Scene {scene.id}: mood '{scene.mood}' is not in valid set {valid_moods}."
        )


# ---------------------------------------------------------------------------
# Test 4 — language field equals "en"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_language_field_is_en() -> None:
    """Language must always equal 'en', even for non-English prompts."""
    # Simulate a Russian-language prompt.
    state_ru = StoryState(user_prompt="сказка про волка и лису")
    agent = StoryWriterAgent()
    await agent.run(state_ru)

    assert state_ru.language == "en", (
        f"language must be 'en' even for Russian prompts, got '{state_ru.language}'"
    )


# ---------------------------------------------------------------------------
# Test 5 — Narration text is English (>80% basic-Latin character ratio)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_narration_is_english(state: StoryState) -> None:
    """Scene narrations should be predominantly basic-Latin (English) text.

    Strategy: count ASCII printable characters vs. total non-whitespace.
    English text should be >80% basic Latin (a-z, A-Z, digits, punctuation).
    This is intentionally lenient to avoid false positives from special chars.
    """
    agent = StoryWriterAgent()
    await agent.run(state)

    for scene in state.scenes:
        non_ws = [c for c in scene.narration if not c.isspace()]
        if not non_ws:
            continue
        ascii_chars = [c for c in non_ws if ord(c) < 128]
        ratio = len(ascii_chars) / len(non_ws)
        assert ratio > 0.80, (
            f"Scene {scene.id}: narration appears non-English "
            f"(ASCII ratio={ratio:.2f}). "
            f"Narration: {scene.narration!r}"
        )


# ---------------------------------------------------------------------------
# Test 6 — Scene count is 4-5
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scene_count_is_4_to_5(state: StoryState) -> None:
    """StoryWriter must produce exactly 4-5 scenes (cost control)."""
    agent = StoryWriterAgent()
    await agent.run(state)

    assert len(state.scenes) <= 5, (
        f"Expected at most 5 scenes, got {len(state.scenes)} — risk of cost overrun."
    )
    assert len(state.scenes) >= 4, (
        f"Expected at least 4 scenes, got {len(state.scenes)}."
    )


# ---------------------------------------------------------------------------
# Test 7 — target_duration_sec is 45-60
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_target_duration_is_45_to_60(state: StoryState) -> None:
    """target_duration_sec must be within the 45-60s budget window."""
    agent = StoryWriterAgent()
    await agent.run(state)

    assert 45 <= state.target_duration_sec <= 60, (
        f"target_duration_sec={state.target_duration_sec} is outside [45, 60]."
    )


# ---------------------------------------------------------------------------
# Test 8 — Narration word count <= 30 per scene
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_narration_word_count(state: StoryState) -> None:
    """Each scene narration must be at most 30 words (budget: 15-25 target)."""
    agent = StoryWriterAgent()
    await agent.run(state)

    for scene in state.scenes:
        word_count = len(scene.narration.split())
        assert word_count <= 30, (
            f"Scene {scene.id}: narration is {word_count} words (max 30). "
            f"Narration: {scene.narration!r}"
        )


# ---------------------------------------------------------------------------
# Sanity: MOCK_STORY fixture itself is valid
# ---------------------------------------------------------------------------


def test_mock_story_fixture_is_valid() -> None:
    """MOCK_STORY dict must satisfy the expected shape (self-contained sanity check)."""
    assert "title" in MOCK_STORY
    assert "summary" in MOCK_STORY
    assert MOCK_STORY.get("language") == "en"
    assert "scenes" in MOCK_STORY
    assert len(MOCK_STORY["scenes"]) >= 3
    for s in MOCK_STORY["scenes"]:
        assert "id" in s
        assert "narration" in s
        assert "mood" in s
        assert "estimated_seconds" in s
