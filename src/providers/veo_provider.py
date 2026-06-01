"""Veo 3 provider — image-to-video with ambient sound effects via Google AI.

Flow per scene
--------------
1. Submit image + mood-based prompt to Veo 3 (generate_audio=True for SFX).
2. Poll the Long Running Operation until completed (~60-120 seconds).
3. Download the MP4 (video + ambient audio, no AI speech) to workspace/videos/.

Audio strategy
--------------
Veo 3 generates ambient sound effects (wind, snow, footsteps, etc.) but NOT
voice narration — the prompt explicitly says "no dialogue, no narration".
The compositor mixes this ambient audio at 20% volume under the ElevenLabs
narration at 100%, creating a layered cinematic sound design.

Cost
----
Google AI Pro plan includes Veo 3 quota — check your usage at aistudio.google.com.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Motion + sound prompt per mood.  Explicitly suppress AI speech.
_MOOD_PROMPTS: dict[str, str] = {
    "tense":       "slow cinematic zoom in, dramatic cold winter forest, wind howling softly, ambient sound effects only no dialogue no narration",
    "sad":         "gentle slow zoom in, soft melancholic winter light, quiet wind through bare branches, ambient sound effects only no dialogue no narration",
    "mysterious":  "slow dolly forward through misty snowy forest, ethereal silence broken by distant wind, ambient sound effects only no dialogue no narration",
    "determined":  "steady camera push forward, crisp winter air, subtle snowfall sounds, ambient sound effects only no dialogue no narration",
    "triumphant":  "slow cinematic zoom out, golden winter sunset light, joyful wind and sparkling snow sounds, ambient sound effects only no dialogue no narration",
    "peaceful":    "gentle slow pan across snowy landscape, soft breeze, distant birds, peaceful ambient sound effects only no dialogue no narration",
    "hopeful":     "soft upward tilt toward light through pine trees, gentle wind and soft snow crunch, ambient sound effects only no dialogue no narration",
    "joyful":      "warm gentle pan, playful snow sounds and light wind chimes, ambient sound effects only no dialogue no narration",
    "exciting":    "slow dramatic pan right, crisp snow crunch and energetic wind, ambient sound effects only no dialogue no narration",
    "grateful":    "warm gentle zoom in, soft heartfelt winter ambience, light snowfall sounds, ambient sound effects only no dialogue no narration",
    "curious":     "slow exploratory pan, curious quiet winter forest sounds, soft footsteps in snow, ambient sound effects only no dialogue no narration",
    "neutral":     "slow gentle zoom in, soft winter forest ambience, ambient sound effects only no dialogue no narration",
}

_POLL_INTERVAL_SEC = 8
_POLL_TIMEOUT_SEC = 600   # 10 minutes max per clip
_VIDEO_DURATION_SEC = 8   # generate 8s (trimmed to scene duration in compositor)


class VeoProvider:
    """Wraps Google's Veo 3 image-to-video API via google-genai SDK.

    Usage::

        provider = VeoProvider()
        result = await provider.image_to_video(img_path, mood, output_path)
        # result = {"path": Path, "operation_name": str, "mocked": bool}
    """

    MODEL = "veo-3.0-generate-preview"

    def __init__(self) -> None:
        from src.config import get_config  # noqa: PLC0415

        self.config = get_config()
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.config.google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY is not set. Add it to .env."
                )
            try:
                from google import genai  # noqa: PLC0415

                self._client = genai.Client(api_key=self.config.google_api_key)
            except ImportError as exc:
                raise ImportError(
                    "google-genai not installed. Run: pip install google-genai"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def image_to_video(
        self,
        img_path: Path,
        mood: str,
        output_path: Path,
    ) -> dict[str, Any]:
        """Generate a video clip from a static image with ambient SFX.

        Args:
            img_path:    Local path to the source PNG/JPEG.
            mood:        Scene mood — selects the motion + audio prompt.
            output_path: Where to write the downloaded MP4.

        Returns:
            Dict with ``path``, ``operation_name``, ``mocked``.
        """
        import asyncio  # noqa: PLC0415

        return await asyncio.to_thread(self._generate_sync, img_path, mood, output_path)

    # ------------------------------------------------------------------
    # Synchronous internals (run in thread pool)
    # ------------------------------------------------------------------

    def _generate_sync(
        self, img_path: Path, mood: str, output_path: Path
    ) -> dict[str, Any]:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        client = self._get_client()
        prompt = _MOOD_PROMPTS.get(mood, _MOOD_PROMPTS["neutral"])

        logger.info(
            "VeoProvider: submitting scene  mood=%s  model=%s", mood, self.MODEL
        )

        # Read image bytes.
        image_bytes = img_path.read_bytes()
        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"

        # Submit generation.
        operation = client.models.generate_videos(
            model=self.MODEL,
            prompt=prompt,
            image=types.Image(image_bytes=image_bytes, mime_type=mime),
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                duration_seconds=_VIDEO_DURATION_SEC,
                number_of_videos=1,
                generate_audio=True,   # ambient SFX — no AI speech (controlled via prompt)
            ),
        )

        logger.info("VeoProvider: operation started — polling ...")

        # Poll until completed or failed.
        deadline = time.monotonic() + _POLL_TIMEOUT_SEC
        while time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL_SEC)
            operation = client.operations.get(operation)
            logger.debug("VeoProvider: operation done=%s", operation.done)
            if operation.done:
                break
        else:
            raise RuntimeError(f"Veo 3 operation timed out after {_POLL_TIMEOUT_SEC}s")

        if operation.error.code != 0:
            raise RuntimeError(
                f"Veo 3 generation failed: {operation.error.message}"
            )

        # Download generated video.
        generated = operation.result.generated_videos[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("VeoProvider: downloading video ...")
        with client.files.download(file=generated.video) as response:
            output_path.write_bytes(response.read())

        size_kb = output_path.stat().st_size // 1024
        logger.info(
            "VeoProvider: saved %s (%d KB)  mood=%s",
            output_path.name,
            size_kb,
            mood,
        )

        return {
            "path": output_path,
            "operation_name": getattr(operation, "name", ""),
            "mocked": False,
        }
