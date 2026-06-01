"""Tests for ImageAgent — Phase 3.

All tests run in mock mode (MOCK_MODE=true in .env) so no real API calls
are made.  The mock provider writes a valid 1x1 grey PNG to disk and
simulates a 50ms I/O delay so asyncio.gather parallelism is observable.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from src.agents.image_agent import ImageAgent
from src.agents.scene_director import SceneDirectorAgent
from src.agents.story_writer import StoryWriterAgent
from src.providers.google_provider import _MOCK_PNG_BYTES
from src.state import StoryState

# PNG magic bytes (first 4 bytes of every valid PNG file).
PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47])


# ---------------------------------------------------------------------------
# Fixture: pipeline state through Phase 2
# ---------------------------------------------------------------------------


@pytest.fixture
async def phase2_state() -> StoryState:
    """Return a StoryState that has been through StoryWriter + SceneDirector."""
    state = StoryState(user_prompt="a wolf and fox story")
    await StoryWriterAgent().run(state)
    await SceneDirectorAgent().run(state)
    return state


# ---------------------------------------------------------------------------
# Test 1 — Mock mode sets image_path on every scene
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_scenes_get_image_path(phase2_state: StoryState) -> None:
    """After ImageAgent.run(), every scene must have image_path set."""
    state = phase2_state
    agent = ImageAgent()
    result = await agent.run(state)

    assert result.mocked is True
    for scene in state.scenes:
        assert scene.image_path, (
            f"Scene {scene.id}: image_path must be set after ImageAgent runs."
        )


# ---------------------------------------------------------------------------
# Test 2 — PNG files exist on disk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_files_exist_on_disk(phase2_state: StoryState) -> None:
    """The PNG files referenced by scene.image_path must actually exist."""
    from src.config import get_config

    state = phase2_state
    await ImageAgent().run(state)

    workspace = get_config().workspace_dir
    for scene in state.scenes:
        full_path = workspace / scene.image_path  # type: ignore[arg-type]
        assert full_path.exists(), (
            f"Scene {scene.id}: expected PNG file at {full_path} but not found."
        )


# ---------------------------------------------------------------------------
# Test 3 — PNG files have valid magic bytes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_magic_bytes(phase2_state: StoryState) -> None:
    """Every generated image file must start with the PNG magic bytes 89 50 4E 47."""
    from src.config import get_config

    state = phase2_state
    await ImageAgent().run(state)

    workspace = get_config().workspace_dir
    for scene in state.scenes:
        full_path = workspace / scene.image_path  # type: ignore[arg-type]
        file_bytes = full_path.read_bytes()
        assert file_bytes[:4] == PNG_MAGIC, (
            f"Scene {scene.id}: {full_path.name} does not start with PNG magic bytes. "
            f"Got: {file_bytes[:4].hex()}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Parallel execution: wall time < sum of individual times
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_execution_faster_than_sequential(
    phase2_state: StoryState,
) -> None:
    """asyncio.gather should run scene tasks concurrently.

    The mock provider sleeps 50ms per image.  With N=5 scenes:
    - Sequential: 5 * 50ms = 250ms
    - Parallel:   ~50ms

    We assert that wall time < N * single_latency * 0.75 to prove
    parallelism (with generous margin for CI scheduling noise).
    """
    from src.providers.google_provider import GoogleImageProvider

    state = phase2_state
    n = len(state.scenes)
    single_latency = GoogleImageProvider.MOCK_LATENCY_SEC  # 0.05s

    t0 = time.monotonic()
    result = await ImageAgent().run(state)
    wall_time = time.monotonic() - t0

    # Wall time must be well below the sequential estimate.
    sequential_estimate = n * single_latency
    assert wall_time < sequential_estimate * 0.75, (
        f"Wall time {wall_time:.3f}s is not much less than sequential "
        f"estimate {sequential_estimate:.3f}s — gather may not be parallelising."
    )

    # Also assert that all scenes were processed.
    assert result.output["scenes_processed"] == n


# ---------------------------------------------------------------------------
# Test 5 — Mock PNG bytes are a valid minimal PNG (module-level check)
# ---------------------------------------------------------------------------


def test_mock_png_is_valid() -> None:
    """The pre-built mock PNG bytes must start with the PNG magic signature."""
    assert _MOCK_PNG_BYTES[:4] == PNG_MAGIC, (
        f"_MOCK_PNG_BYTES does not start with PNG magic. Got: {_MOCK_PNG_BYTES[:4].hex()}"
    )
    # Should be small but non-trivial (> 50 bytes for a minimal valid PNG).
    assert len(_MOCK_PNG_BYTES) > 50, "Mock PNG is suspiciously small."
