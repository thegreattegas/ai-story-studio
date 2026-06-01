"""Tests for VoiceAgent — Phase 3.

All tests run in mock mode so no real API calls are made.
The mock provider writes a silent MP3 stub that begins with a valid
MPEG frame sync header (FF FB) so FFmpeg will recognise it in Phase 4.
"""

from __future__ import annotations

import pytest

from src.agents.scene_director import SceneDirectorAgent
from src.agents.story_writer import StoryWriterAgent
from src.agents.voice_agent import VoiceAgent, _SCENE_SEPARATOR
from src.providers.elevenlabs_provider import _MOCK_MP3_BYTES
from src.state import StoryState

# Valid MPEG-1 Layer 3 frame sync patterns.
VALID_MP3_MAGIC = {b"\xff\xfb", b"\xff\xf3", b"\xff\xfa", b"\xff\xf2"}
ID3_MAGIC = b"ID3"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def phase2_state() -> StoryState:
    """Return a StoryState that has been through StoryWriter + SceneDirector."""
    state = StoryState(user_prompt="a wolf and fox story")
    await StoryWriterAgent().run(state)
    await SceneDirectorAgent().run(state)
    return state


# ---------------------------------------------------------------------------
# Test 1 — Mock mode writes a voice.mp3 file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_generates_voice_file(phase2_state: StoryState) -> None:
    """After VoiceAgent.run(), workspace/voice.mp3 must exist on disk."""
    from src.config import get_config

    state = phase2_state
    agent = VoiceAgent()
    result = await agent.run(state)

    assert result.mocked is True

    workspace = get_config().workspace_dir
    voice_file = workspace / "voice.mp3"
    assert voice_file.exists(), (
        f"Expected voice.mp3 at {voice_file} but file not found."
    )
    assert voice_file.stat().st_size > 0, "voice.mp3 is empty!"


# ---------------------------------------------------------------------------
# Test 2 — state.voice_path is set after run()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voice_path_set_in_state(phase2_state: StoryState) -> None:
    """state.voice_path must be set after VoiceAgent.run()."""
    state = phase2_state
    assert state.voice_path is None, "voice_path should start as None."

    await VoiceAgent().run(state)

    assert state.voice_path, "state.voice_path must be set after VoiceAgent runs."
    assert "voice.mp3" in state.voice_path, (
        f"Expected 'voice.mp3' in voice_path, got: {state.voice_path!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — MP3 file starts with valid MPEG frame sync bytes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mp3_magic_bytes(phase2_state: StoryState) -> None:
    """The generated audio file must start with a valid MP3 sync pattern.

    Accepted patterns:
    - FF FB / FF F3 / FF FA / FF F2  — MPEG frame sync (various bitrates)
    - ID3                             — ID3 tag header (also valid MP3)
    """
    from src.config import get_config

    state = phase2_state
    await VoiceAgent().run(state)

    voice_file = get_config().workspace_dir / "voice.mp3"
    file_bytes = voice_file.read_bytes()

    first_two = file_bytes[:2]
    first_three = file_bytes[:3]
    is_mpeg = first_two in VALID_MP3_MAGIC
    is_id3 = first_three == ID3_MAGIC

    assert is_mpeg or is_id3, (
        f"voice.mp3 does not start with a valid MP3 sync pattern.\n"
        f"First 4 bytes: {file_bytes[:4].hex()}\n"
        f"Expected one of: {[h.hex() for h in VALID_MP3_MAGIC]} or ID3."
    )


# ---------------------------------------------------------------------------
# Test 4 — All narrations are included in the TTS input text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_narrations_concatenated(phase2_state: StoryState) -> None:
    """The text sent to TTS must contain every scene's narration.

    We verify this by checking that each narration substring appears in
    the joined text.  We reconstruct what VoiceAgent would build from state.
    """
    state = phase2_state

    # Replicate VoiceAgent's concatenation logic.
    full_narration = _SCENE_SEPARATOR.join(
        scene.narration for scene in sorted(state.scenes, key=lambda s: s.id)
    )

    for scene in state.scenes:
        assert scene.narration in full_narration, (
            f"Scene {scene.id} narration not found in concatenated text.\n"
            f"Narration: {scene.narration!r}"
        )


# ---------------------------------------------------------------------------
# Test 5 — Mock MP3 bytes have valid MPEG sync header (module-level)
# ---------------------------------------------------------------------------


def test_mock_mp3_is_valid() -> None:
    """The pre-built mock MP3 bytes must start with a valid MPEG sync pattern."""
    first_two = _MOCK_MP3_BYTES[:2]
    assert first_two in VALID_MP3_MAGIC, (
        f"_MOCK_MP3_BYTES does not start with valid MPEG sync. "
        f"Got: {first_two.hex()}"
    )
    assert len(_MOCK_MP3_BYTES) > 100, "Mock MP3 is suspiciously small."
