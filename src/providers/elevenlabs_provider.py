"""ElevenLabs TTS provider — wraps the elevenlabs Python SDK.

Mock mode
---------
Writes a minimal silent MP3 stub (valid MPEG-1 Layer 3 frame header so
FFmpeg can parse it in Phase 4) without any API call.

Real API mode
-------------
Uses ``ElevenLabs.text_to_speech.convert()`` from elevenlabs>=1.0.0.
The SDK call is synchronous, so we wrap it in ``asyncio.to_thread``.

Cost: ElevenLabs free tier = 10,000 chars/month.
      Paid tier = $0.30 per 1,000 characters.
      We track character count and estimate cost per run.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost constants
# ---------------------------------------------------------------------------

COST_PER_1K_CHARS_USD: float = 0.30
FREE_TIER_CHARS_PER_MONTH: int = 10_000

# ---------------------------------------------------------------------------
# Minimal silent MP3 stub (valid MPEG frame header for FFmpeg compatibility)
# ---------------------------------------------------------------------------

# MPEG-1 Layer 3 frame sync pattern:
#   FF FB = sync + MPEG-1, Audio Layer 3
#   90    = bitrate index 9 (128 kbps), sample rate 0 (44100 Hz)
#   00    = stereo mode 0, extension 0, not copyrighted, not original, no emphasis
# Followed by zero-filled frame data.  This gives us a technically valid
# (if silent and incomplete) MP3 frame that FFmpeg will recognise.
_SILENT_FRAME_HEADER = bytes([0xFF, 0xFB, 0x90, 0x00])

# At 128kbps / 44100Hz, one MP3 frame = 417 bytes.
# Pad with zeros to approximate a full frame.
_MOCK_MP3_BYTES: bytes = (_SILENT_FRAME_HEADER + b"\x00" * 413) * 3


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ElevenLabsProvider:
    """Wrapper around the ElevenLabs SDK for TTS audio synthesis.

    Usage::

        provider = ElevenLabsProvider()
        result = await provider.text_to_speech(full_text, output_path)
        # result = {"path": Path, "characters_used": int, "cost_usd": float}
    """

    # Rachel — warm English narrator voice available on free tier.
    DEFAULT_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"
    DEFAULT_MODEL: str = "eleven_multilingual_v2"

    def __init__(self) -> None:
        from src.config import get_config  # noqa: PLC0415

        self.config = get_config()
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazy-init ElevenLabs client on first live use."""
        if self._client is None:
            if not self.config.elevenlabs_api_key:
                raise ValueError(
                    "ELEVENLABS_API_KEY is not set. Add it to .env before running in live mode."
                )
            try:
                from elevenlabs.client import ElevenLabs  # noqa: PLC0415

                self._client = ElevenLabs(api_key=self.config.elevenlabs_api_key)
            except ImportError as exc:
                raise ImportError(
                    "elevenlabs package not installed. Run: pip install elevenlabs"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def text_to_speech(
        self,
        text: str,
        output_path: Path,
        voice_id: str | None = None,
        voice_settings: Any | None = None,
        speed: float = 0.85,
    ) -> dict[str, Any]:
        """Synthesise ``text`` to speech and write MP3 to ``output_path``.

        Args:
            text:        Full narration text to synthesise.
            output_path: Absolute path where the MP3 will be written.
            voice_id:    ElevenLabs voice ID (defaults to Rachel).
            speed:       Playback speed multiplier (0.7=slow … 1.2=fast, default 0.85).

        Returns:
            Dict with keys: ``path``, ``characters_used``, ``cost_usd``, ``mocked``.
        """
        char_count = len(text)

        if self.config.mock_mode:
            return self._write_mock_mp3(output_path, char_count)

        return await asyncio.to_thread(
            self._tts_sync,
            text,
            output_path,
            voice_id or self.DEFAULT_VOICE_ID,
            voice_settings,
            speed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_mock_mp3(self, output_path: Path, char_count: int) -> dict[str, Any]:
        """Write the pre-built silent MP3 stub to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_MOCK_MP3_BYTES)
        logger.debug(
            "ElevenLabsProvider: wrote mock MP3 (%d bytes) -> %s",
            len(_MOCK_MP3_BYTES),
            output_path,
        )
        return {
            "path": output_path,
            "characters_used": char_count,
            "cost_usd": 0.0,
            "mocked": True,
        }

    def _tts_sync(
        self, text: str, output_path: Path, voice_id: str, voice_settings: Any | None = None, speed: float = 0.85
    ) -> dict[str, Any]:
        """Synchronous ElevenLabs TTS call — run inside a thread.

        Args:
            text:        Full narration text.
            output_path: Destination MP3 file path.
            voice_id:    ElevenLabs voice ID.

        Returns:
            Dict with path, characters_used, cost_usd, mocked=False.
        """
        from elevenlabs import VoiceSettings  # noqa: PLC0415

        client = self._get_client()
        char_count = len(text)

        logger.debug(
            "ElevenLabsProvider: synthesising %d chars with voice=%s model=%s",
            char_count,
            voice_id,
            self.DEFAULT_MODEL,
        )

        # Use caller-provided settings or fall back to a safe default.
        if voice_settings is None:
            from elevenlabs import VoiceSettings as VS  # noqa: PLC0415
            voice_settings = VS(stability=0.85, similarity_boost=0.80, style=0.0, use_speaker_boost=True)

        # elevenlabs>=1.0.0 SDK: text_to_speech.convert returns a generator of bytes.
        audio_chunks = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=self.DEFAULT_MODEL,
            output_format="mp3_44100_128",
            voice_settings=voice_settings,
        )
        audio_bytes = b"".join(audio_chunks)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_bytes)

        # Apply speed adjustment via FFmpeg atempo (more reliable than SDK param).
        if abs(speed - 1.0) > 0.01:
            self._apply_atempo(output_path, speed)

        cost = (char_count / 1_000) * COST_PER_1K_CHARS_USD

        logger.info(
            "ElevenLabsProvider: audio written (%d bytes, %d chars, speed=%.2f) -> %s cost=$%.4f",
            output_path.stat().st_size,
            char_count,
            speed,
            output_path,
            cost,
        )

        return {
            "path": output_path,
            "characters_used": char_count,
            "cost_usd": cost,
            "mocked": False,
        }

    @staticmethod
    def _apply_atempo(audio_path: Path, speed: float) -> None:
        """Slow down or speed up an MP3 in-place using FFmpeg atempo filter.

        atempo range: 0.5–2.0. Values like 0.75 (25% slower) are fine directly.

        Args:
            audio_path: MP3 file to modify in-place.
            speed:      Target speed multiplier (e.g. 0.75 = 25% slower).
        """
        import subprocess  # noqa: PLC0415
        import imageio_ffmpeg  # noqa: PLC0415

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        tmp = audio_path.with_suffix(".atempo_tmp.mp3")
        cmd = [
            ffmpeg, "-y",
            "-i", str(audio_path),
            "-filter:a", f"atempo={speed}",
            str(tmp),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            tmp.replace(audio_path)
        else:
            tmp.unlink(missing_ok=True)
            logger.warning(
                "ElevenLabsProvider: atempo filter failed (exit %d) — keeping original speed.",
                result.returncode,
            )
