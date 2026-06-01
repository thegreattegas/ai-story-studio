"""Tests for SceneDirectorAgent — 4 cases covering enrichment output."""

from __future__ import annotations

import pytest

from src.agents.scene_director import SceneDirectorAgent, _STYLE_SUFFIX, MOCK_CHARACTERS
from src.agents.story_writer import MOCK_STORY, StoryWriterAgent
from src.state import Scene, StoryState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_pipeline(prompt: str = "a wolf and a fox story") -> StoryState:
    """Run StoryWriter then SceneDirector in mock mode, return final state."""
    state = StoryState(user_prompt=prompt)
    writer = StoryWriterAgent()
    await writer.run(state)
    director = SceneDirectorAgent()
    await director.run(state)
    return state


# ---------------------------------------------------------------------------
# Test 1 — Mock mode enriches all scenes with image_prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_scenes_get_image_prompt() -> None:
    """Every scene must have a non-empty image_prompt after SceneDirector runs."""
    state = await _run_pipeline()

    assert state.scenes, "State must have scenes after pipeline."
    for scene in state.scenes:
        assert scene.image_prompt, (
            f"Scene {scene.id}: image_prompt must be non-empty after SceneDirector."
        )


# ---------------------------------------------------------------------------
# Test 2 — Image prompts contain the style-consistency phrase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_prompts_contain_style_phrase() -> None:
    """Every image_prompt must contain the watercolor style suffix for visual consistency."""
    state = await _run_pipeline()

    style_keyword = "watercolor"
    for scene in state.scenes:
        prompt = scene.image_prompt or ""
        assert style_keyword in prompt.lower(), (
            f"Scene {scene.id}: image_prompt missing style keyword '{style_keyword}'.\n"
            f"Prompt: {prompt!r}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Image prompts are in English
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_prompts_are_english() -> None:
    """Image prompts should be predominantly ASCII/English text (>80% basic Latin)."""
    state = await _run_pipeline()

    for scene in state.scenes:
        prompt = scene.image_prompt or ""
        non_ws = [c for c in prompt if not c.isspace()]
        if not non_ws:
            continue
        ascii_chars = [c for c in non_ws if ord(c) < 128]
        ratio = len(ascii_chars) / len(non_ws)
        assert ratio > 0.80, (
            f"Scene {scene.id}: image_prompt appears non-English "
            f"(ASCII ratio={ratio:.2f}).\nPrompt: {prompt!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Image prompts mention character visual descriptions (consistency)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_prompts_mention_character_descriptions() -> None:
    """At least one character description substring must appear in each image_prompt.

    This enforces the character-consistency rule: the model must use the full
    visual description (e.g. 'young grey wolf with kind blue eyes') rather than
    a bare noun ('wolf'). We check for at least one recognisable description
    fragment per scene.
    """
    state = await _run_pipeline()

    # Extract partial tokens from each MOCK_CHARACTER description for matching.
    # E.g., "young grey wolf" -> partial match "grey wolf"
    char_fragments = [
        # Extract meaningful 2-word fragments from each description.
        desc.split(",")[0].strip().lower()
        for desc in MOCK_CHARACTERS
    ]

    for scene in state.scenes:
        prompt = (scene.image_prompt or "").lower()
        matched = any(frag in prompt for frag in char_fragments)
        assert matched, (
            f"Scene {scene.id}: image_prompt does not mention any character description.\n"
            f"Checked fragments: {char_fragments}\n"
            f"Prompt: {prompt!r}"
        )


# ---------------------------------------------------------------------------
# Test 5 — voice_tone and voice_pace fields are populated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voice_fields_are_populated() -> None:
    """Each scene must have voice_tone and voice_pace set after SceneDirector runs."""
    state = await _run_pipeline()

    valid_tones = {"warm", "dramatic", "mysterious", "cheerful", "gentle", "triumphant"}
    valid_paces = {"slow", "medium", "fast"}

    for scene in state.scenes:
        assert scene.voice_tone in valid_tones, (
            f"Scene {scene.id}: voice_tone '{scene.voice_tone}' not in {valid_tones}."
        )
        assert scene.voice_pace in valid_paces, (
            f"Scene {scene.id}: voice_pace '{scene.voice_pace}' not in {valid_paces}."
        )
