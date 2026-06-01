"""VoiceAgent — Phase 3.

Generates story audio using a single ElevenLabs narrator voice (Rachel)
with mood-based tone settings.  Rather than switching voice IDs between
narrator and character, the same voice is used throughout — but with
different ``VoiceSettings`` to shift tone:

* Narrator lines  → calm, steady (high stability, no style)
* Child / excited → more animated (lower stability, higher style)
* Wise / solemn   → deliberate, low (very high stability, minimal style)
* Default char    → warm mid-range settings

This keeps the story sonically cohesive while making character voices
clearly distinguishable from narration.

Output
------
* ``workspace/voice.mp3`` — stitched narration + dialogue audio.
* ``state.voice_path`` set to the workspace-relative path.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from elevenlabs import VoiceSettings

from src.agents.base import AgentResult, BaseAgent
from src.providers.elevenlabs_provider import ElevenLabsProvider, COST_PER_1K_CHARS_USD
from src.state import StoryState

log = logging.getLogger(__name__)

# Single narrator voice — custom voice from ElevenLabs library
NARRATOR_VOICE_ID = "eUdJpUEN3EslrgE24PKx"

# ---------------------------------------------------------------------------
# Tone presets (all use the same voice, just different VoiceSettings)
# ---------------------------------------------------------------------------

_NARRATOR_SETTINGS = VoiceSettings(
    stability=0.60,        # lower = more natural variation / emotional warmth
    similarity_boost=0.85,
    style=0.30,            # adds gentle expressiveness to the narrator voice
    use_speaker_boost=True,
)

# Child / young / excited character
_CHILD_SETTINGS = VoiceSettings(
    stability=0.52,
    similarity_boost=0.75,
    style=0.38,
    use_speaker_boost=True,
)

# Wise / elder / solemn character
_WISE_SETTINGS = VoiceSettings(
    stability=0.90,
    similarity_boost=0.85,
    style=0.04,
    use_speaker_boost=True,
)

# Default character (tense / joyful / standard)
_DEFAULT_CHAR_SETTINGS = VoiceSettings(
    stability=0.65,
    similarity_boost=0.80,
    style=0.22,
    use_speaker_boost=True,
)

# Map scene mood → character VoiceSettings
_MOOD_TO_SETTINGS: dict[str, VoiceSettings] = {
    "exciting":    _CHILD_SETTINGS,
    "joyful":      _CHILD_SETTINGS,
    "triumphant":  _DEFAULT_CHAR_SETTINGS,
    "tense":       _DEFAULT_CHAR_SETTINGS,
    "mysterious":  _WISE_SETTINGS,
    "peaceful":    _WISE_SETTINGS,
    "sad":         _NARRATOR_SETTINGS,
    "neutral":     _DEFAULT_CHAR_SETTINGS,
}


class VoiceAgent(BaseAgent):
    """Generates story audio with a single voice and mood-shifted tone.

    Every scene produces:
    1. Narrator segment — calm, steady reading of ``scene.narration``.
    2. Character segment — same voice but mood-tuned settings reading
       ``scene.character_dialogue`` (skipped when no dialogue exists).

    Segments are stitched by FFmpeg into a single ``voice.mp3``.
    """

    name: str = "VoiceAgent"
    default_model: str = "eleven_multilingual_v2"
    system_prompt: str = ""

    def __init__(self) -> None:
        super().__init__()
        self.provider = ElevenLabsProvider()
        self.voice_output_path = self.config.workspace_dir / "voice.mp3"

    def mock_response(self, state: StoryState) -> dict[str, Any]:
        return {"voice_path": str(self.voice_output_path), "characters": 0, "mocked": True}

    async def run(self, state: StoryState) -> AgentResult:
        start = time.monotonic()

        if not state.scenes:
            log.warning("VoiceAgent: no scenes — skipping TTS.")
            return AgentResult(
                agent_name=self.name,
                model_used=self.default_model,
                output={"characters": 0, "voice_path": ""},
                mocked=self.config.mock_mode,
            )

        if self.config.mock_mode:
            result = self._write_mock(state)
        else:
            result = await self._generate_audio(state)

        state.voice_path = str(
            self.voice_output_path.relative_to(self.config.workspace_dir)
        )

        cost = result.get("cost_usd", 0.0)
        char_count = result.get("characters_used", 0)
        elapsed = time.monotonic() - start

        state.add_cost(cost)
        log.info(
            "VoiceAgent: DONE  chars=%d  cost=$%.4f  elapsed=%.2fs  mock=%s",
            char_count, cost, elapsed, self.config.mock_mode,
        )

        return AgentResult(
            agent_name=self.name,
            model_used=self.default_model if not self.config.mock_mode else "MOCK",
            output={"characters": char_count, "voice_path": state.voice_path},
            cost_usd=cost,
            elapsed_sec=elapsed,
            mocked=self.config.mock_mode,
        )

    # ------------------------------------------------------------------
    # Audio generation
    # ------------------------------------------------------------------

    async def _generate_audio(self, state: StoryState) -> dict[str, Any]:
        scenes = sorted(state.scenes, key=lambda s: s.id)
        total_chars = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            temp_files: list[Path] = []

            for scene in scenes:
                # 1. Narrator reads the narration
                narr_path = tmp / f"narr_{scene.id:03d}.mp3"
                log.info(
                    "VoiceAgent: scene %d narrator  mood=%-12s chars=%d",
                    scene.id, scene.mood, len(scene.narration),
                )
                await self.provider.text_to_speech(
                    scene.narration, narr_path,
                    voice_id=NARRATOR_VOICE_ID,
                    voice_settings=_NARRATOR_SETTINGS,
                    speed=0.88,  # gentle storytelling pace — warm, not rushed
                )
                temp_files.append(narr_path)
                total_chars += len(scene.narration)

                # 2. Character speaks dialogue with mood-shifted tone
                if scene.character_dialogue:
                    char_path = tmp / f"char_{scene.id:03d}.mp3"
                    settings = _MOOD_TO_SETTINGS.get(scene.mood, _DEFAULT_CHAR_SETTINGS)
                    speaker = scene.character_dialogue_speaker or "character"
                    log.info(
                        "VoiceAgent: scene %d character  speaker=%-10s mood=%-12s chars=%d",
                        scene.id, speaker, scene.mood, len(scene.character_dialogue),
                    )
                    await self.provider.text_to_speech(
                        scene.character_dialogue, char_path,
                        voice_id=NARRATOR_VOICE_ID,
                        voice_settings=settings,
                        speed=0.92,  # characters slightly livelier than narrator
                    )
                    temp_files.append(char_path)
                    total_chars += len(scene.character_dialogue)

            self._concat_segments(temp_files, self.voice_output_path)

        cost = (total_chars / 1_000) * COST_PER_1K_CHARS_USD
        return {"characters_used": total_chars, "cost_usd": cost}

    def _concat_segments(self, audio_files: list[Path], output_path: Path) -> None:
        """Stitch MP3 segments using FFmpeg concat filter."""
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        inputs: list[str] = []
        for f in audio_files:
            inputs += ["-i", str(f)]

        cmd = (
            [ffmpeg, "-y"]
            + inputs
            + [
                "-filter_complex",
                f"concat=n={len(audio_files)}:v=0:a=1[out]",
                "-map", "[out]",
                str(output_path),
            ]
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"VoiceAgent: FFmpeg concat failed:\n{result.stderr[-600:]}"
            )
        log.info(
            "VoiceAgent: stitched %d segments -> %s (%.1f KB)",
            len(audio_files), output_path.name,
            output_path.stat().st_size / 1024,
        )

    # ------------------------------------------------------------------
    # Mock
    # ------------------------------------------------------------------

    def _write_mock(self, state: StoryState) -> dict[str, Any]:
        from src.providers.elevenlabs_provider import _MOCK_MP3_BYTES  # noqa: PLC0415
        n = sum(2 if s.character_dialogue else 1 for s in state.scenes)
        self.voice_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.voice_output_path.write_bytes(_MOCK_MP3_BYTES * max(n, 3))
        return {"characters_used": 0, "cost_usd": 0.0, "mocked": True}
