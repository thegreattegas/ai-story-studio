"""OpenAI Whisper API provider for subtitle timing — Phase 4.

Mock mode
---------
Returns a fixed 5-segment transcript with 10-second intervals.
No API call is made.

Real API mode
-------------
Calls ``whisper-1`` via OpenAI with ``verbose_json`` + ``segment`` timestamps.
Cost: ~$0.006 per minute of audio.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Cost per minute of audio for Whisper API (USD).
COST_PER_MINUTE_USD: float = 0.006


class WhisperProvider:
    """OpenAI Whisper wrapper for audio transcription with timing.

    Usage::

        provider = WhisperProvider()
        result = await provider.transcribe_with_timing(audio_path)
        # result = {"segments": [...], "cost_usd": float, "mocked": bool}
        # segment = {"start": float, "end": float, "text": str}
    """

    def __init__(self) -> None:
        from src.config import get_config  # noqa: PLC0415

        self.config = get_config()
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazy-init OpenAI client on first live use."""
        if self._client is None:
            if not self.config.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add it to .env before running in live mode."
                )
            try:
                from openai import OpenAI  # noqa: PLC0415

                self._client = OpenAI(api_key=self.config.openai_api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def transcribe_with_timing(self, audio_path: Path) -> dict[str, Any]:
        """Transcribe audio and return segment-level timing.

        Args:
            audio_path: Absolute path to the MP3/WAV file.

        Returns:
            Dict with ``segments`` (list of start/end/text dicts),
            ``cost_usd`` (float), and ``mocked`` (bool).
        """
        if self.config.mock_mode or not self.config.openai_api_key:
            log.debug("WhisperProvider: mock mode — returning mock transcript.")
            return self._mock_transcript(audio_path)

        return await asyncio.to_thread(self._transcribe_sync, audio_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transcribe_sync(self, audio_path: Path) -> dict[str, Any]:
        """Synchronous Whisper API call — run inside a thread.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Dict with segments, cost_usd, mocked=False.
        """
        client = self._get_client()

        log.info(
            "WhisperProvider: transcribing %s (%.1f KB)",
            audio_path.name,
            audio_path.stat().st_size / 1024,
        )

        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            }
            for seg in transcript.segments
        ]

        duration_min = (segments[-1]["end"] / 60.0) if segments else 0.0
        cost = round(duration_min * COST_PER_MINUTE_USD, 4)

        log.info(
            "WhisperProvider: %d segments, %.1fs audio, cost=$%.4f",
            len(segments),
            segments[-1]["end"] if segments else 0,
            cost,
        )

        return {"segments": segments, "cost_usd": cost, "mocked": False}

    def _mock_transcript(self, audio_path: Path) -> dict[str, Any]:
        """Fixed mock transcript — 5 segments × 10 s each."""
        return {
            "segments": [
                {"start": 0.0,  "end": 10.0, "text": "Scene one mock narration."},
                {"start": 10.0, "end": 20.0, "text": "Scene two mock narration."},
                {"start": 20.0, "end": 30.0, "text": "Scene three mock narration."},
                {"start": 30.0, "end": 40.0, "text": "Scene four mock narration."},
                {"start": 40.0, "end": 50.0, "text": "Scene five mock narration."},
            ],
            "cost_usd": 0.0,
            "mocked": True,
        }
