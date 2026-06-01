"""SubtitleAgent — Phase 4.

Generates a synchronised .srt subtitle file from the story's voice track.

Strategy
--------
1. Mock / no OpenAI key  → use a fixed 5-segment mock transcript,
   then realign to scene boundaries.
2. Live mode            → call OpenAI Whisper for word-level timestamps,
   realign if segment count != scene count.

Output
------
* ``workspace/subtitles.srt`` — SRT file usable by any video player.
* ``state.subtitle_path`` set to workspace-relative path.
* ``scene.voice_segment_start`` / ``scene.voice_segment_end`` populated
  on every scene so the Compositor knows per-scene audio timing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.providers.whisper_provider import WhisperProvider
from src.state import Scene, StoryState

log = logging.getLogger(__name__)


class SubtitleAgent(BaseAgent):
    """Transcribes voice.mp3 and writes workspace/subtitles.srt.

    In mock mode the provider returns a pre-built transcript; the SRT file
    is still written to disk so downstream tests can assert on its content.
    """

    name: str = "SubtitleAgent"
    default_model: str = "whisper-1"
    system_prompt: str = ""

    def __init__(self) -> None:
        super().__init__()
        self.provider = WhisperProvider()
        self.subtitle_path: Path = self.config.workspace_dir / "subtitles.srt"

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def mock_response(self, state: StoryState) -> dict[str, Any]:
        """Return canned summary — actual mock logic lives in the provider."""
        return {
            "subtitle_path": str(
                self.subtitle_path.relative_to(self.config.workspace_dir)
            ),
            "segments": 5,
            "mocked": True,
        }

    async def run(self, state: StoryState) -> AgentResult:
        """Generate subtitles and write workspace/subtitles.srt.

        Args:
            state: Pipeline state with ``voice_path`` and ``scenes`` set.

        Returns:
            :class:`AgentResult` with segment count and cost.
        """
        start = time.monotonic()

        # Resolve workspace-relative voice path to absolute.
        if not state.voice_path:
            raise FileNotFoundError("SubtitleAgent: state.voice_path is not set.")
        voice_full = self.config.workspace_dir / state.voice_path
        if not voice_full.exists():
            raise FileNotFoundError(
                f"SubtitleAgent: voice file not found at {voice_full}"
            )

        log.info(
            "SubtitleAgent: START — transcribing %s (%.1f KB)",
            voice_full.name,
            voice_full.stat().st_size / 1024,
        )

        result = await self.provider.transcribe_with_timing(voice_full)
        segments: list[dict[str, Any]] = result["segments"]

        # Realign if Whisper returned different segment count than scene count.
        if len(segments) != len(state.scenes):
            log.warning(
                "SubtitleAgent: %d Whisper segments vs %d scenes — realigning.",
                len(segments),
                len(state.scenes),
            )
            segments = self._realign_to_scenes(state.scenes, segments)

        # Write SRT file.
        self._write_srt(segments, self.subtitle_path)

        # Update state.
        state.subtitle_path = str(
            self.subtitle_path.relative_to(self.config.workspace_dir)
        )
        for scene, seg in zip(
            sorted(state.scenes, key=lambda s: s.id), segments
        ):
            scene.voice_segment_start = seg["start"]
            scene.voice_segment_end = seg["end"]

        cost = result.get("cost_usd", 0.0)
        elapsed = time.monotonic() - start
        is_mock = result.get("mocked", False)

        state.add_cost(cost)
        log.info(
            "SubtitleAgent: DONE — %d segments, cost=$%.4f, elapsed=%.2fs, mock=%s",
            len(segments),
            cost,
            elapsed,
            is_mock,
        )

        return AgentResult(
            agent_name=self.name,
            model_used="whisper-1" if not is_mock else "MOCK",
            output={
                "segments": len(segments),
                "subtitle_path": state.subtitle_path,
            },
            cost_usd=cost,
            elapsed_sec=elapsed,
            mocked=is_mock,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _realign_to_scenes(
        self,
        scenes: list[Scene],
        whisper_segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Distribute audio duration proportionally across scenes.

        Uses narration character counts as weights so longer scenes get
        more on-screen time.

        Args:
            scenes:           Ordered list of :class:`~src.state.Scene`.
            whisper_segments: Original Whisper segments (any count).

        Returns:
            Realigned list with exactly ``len(scenes)`` entries.
        """
        total_duration = (
            whisper_segments[-1]["end"] if whisper_segments else 50.0
        )
        total_chars = sum(len(s.narration) for s in scenes) or 1

        realigned: list[dict[str, Any]] = []
        current_time = 0.0
        for scene in sorted(scenes, key=lambda s: s.id):
            scene_duration = (len(scene.narration) / total_chars) * total_duration
            realigned.append(
                {
                    "start": round(current_time, 3),
                    "end": round(current_time + scene_duration, 3),
                    "text": scene.narration,
                }
            )
            current_time += scene_duration

        return realigned

    def _write_srt(
        self,
        segments: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Write segments to an SRT file.

        Args:
            segments:    List of ``{start, end, text}`` dicts.
            output_path: Destination path for the .srt file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for i, seg in enumerate(segments, 1):
            start_ts = self._seconds_to_srt_time(seg["start"])
            end_ts = self._seconds_to_srt_time(seg["end"])
            lines.append(f"{i}\n{start_ts} --> {end_ts}\n{seg['text']}\n")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        log.debug(
            "SubtitleAgent: wrote %d-block SRT to %s", len(segments), output_path
        )

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert a float second offset to ``HH:MM:SS,mmm`` SRT format."""
        total_ms = int(round(seconds * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        h = total_s // 3600
        m = (total_s % 3600) // 60
        s = total_s % 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
