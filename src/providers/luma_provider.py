"""Luma Dream Machine provider — image-to-video via lumaai SDK.

Flow per scene
--------------
1. Upload the scene PNG to 0x0.st (free, no account needed) to get a public HTTPS URL.
2. Submit an image-to-video generation to Luma with a mood-based motion prompt.
3. Poll until the generation is ``completed`` (typically 60-120 seconds).
4. Download the resulting MP4 and save it to ``workspace/videos/scene_XX.mp4``.

Cost
----
Luma free tier: 30 generations (enough for 10 scenes × 3 runs).
Paid: ~$0.014 per second of generated video.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Motion prompt per scene mood — guides Luma on what camera move to apply.
_MOOD_PROMPTS: dict[str, str] = {
    "tense":       "slow cinematic zoom in, dramatic tension, cold winter forest light",
    "sad":         "gentle slow zoom in, soft melancholic atmosphere, muted tones",
    "mysterious":  "slow dolly forward through misty winter forest, ethereal",
    "determined":  "steady camera push forward, strong resolute energy",
    "triumphant":  "slow cinematic zoom out, triumphant golden light reveal",
    "peaceful":    "gentle slow pan across peaceful snowy scene, soft warm light",
    "hopeful":     "soft upward tilt, warm hopeful light filters through pine trees",
    "joyful":      "gentle pan with warm light, playful joyful atmosphere",
    "exciting":    "slow dramatic pan right, exciting energy, sparkling snow",
    "grateful":    "warm gentle zoom in, heartfelt tender atmosphere",
    "curious":     "slow exploratory pan left, curious gentle movement",
    "neutral":     "slow gentle zoom in, cinematic children's book illustration style",
}

_POLL_INTERVAL_SEC = 6
_POLL_TIMEOUT_SEC = 360   # 6 minutes max per clip


class LumaProvider:
    """Wraps the Luma Dream Machine image-to-video API.

    Usage::

        provider = LumaProvider()
        result = await provider.image_to_video(img_path, mood, output_path)
        # result = {"path": Path, "generation_id": str, "mocked": bool}
    """

    def __init__(self) -> None:
        from src.config import get_config  # noqa: PLC0415

        self.config = get_config()
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.config.luma_api_key:
                raise ValueError(
                    "LUMA_API_KEY is not set. Add it to .env — get one free at lumalabs.ai/dream-machine/api"
                )
            try:
                from lumaai import LumaAI  # noqa: PLC0415

                self._client = LumaAI(auth_token=self.config.luma_api_key)
            except ImportError as exc:
                raise ImportError(
                    "lumaai package not installed. Run: pip install lumaai httpx"
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
        """Generate a 5-second video clip from a static image.

        Args:
            img_path:    Local path to the source PNG/JPEG.
            mood:        Scene mood — selects the motion prompt.
            output_path: Where to write the downloaded MP4.

        Returns:
            Dict with ``path``, ``generation_id``, ``mocked``.
        """
        import asyncio  # noqa: PLC0415

        return await asyncio.to_thread(self._generate_sync, img_path, mood, output_path)

    # ------------------------------------------------------------------
    # Synchronous internals (run in thread pool)
    # ------------------------------------------------------------------

    def _upload_image(self, img_path: Path) -> str:
        """Upload image to 0x0.st and return the public HTTPS URL."""
        import httpx  # noqa: PLC0415

        logger.info("LumaProvider: uploading %s to 0x0.st ...", img_path.name)
        with open(img_path, "rb") as fh:
            response = httpx.post(
                "https://0x0.st",
                files={"file": (img_path.name, fh, "image/png")},
                timeout=60.0,
            )
        response.raise_for_status()
        url = response.text.strip()
        logger.info("LumaProvider: image URL = %s", url)
        return url

    def _generate_sync(
        self, img_path: Path, mood: str, output_path: Path
    ) -> dict[str, Any]:
        import httpx  # noqa: PLC0415

        client = self._get_client()
        motion_prompt = _MOOD_PROMPTS.get(mood, _MOOD_PROMPTS["neutral"])

        # Step 1: upload image to get public URL.
        image_url = self._upload_image(img_path)

        # Step 2: submit generation.
        logger.info(
            "LumaProvider: submitting image-to-video  mood=%s  prompt='%s'",
            mood,
            motion_prompt,
        )
        generation = client.generations.create(
            prompt=motion_prompt,
            keyframes={
                "frame0": {
                    "type": "image",
                    "url": image_url,
                }
            },
            aspect_ratio="16:9",
            loop=False,
        )
        gen_id: str = generation.id
        logger.info("LumaProvider: generation id=%s  state=%s", gen_id, generation.state)

        # Step 3: poll until completed or failed.
        deadline = time.monotonic() + _POLL_TIMEOUT_SEC
        while time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL_SEC)
            gen = client.generations.get(gen_id)
            logger.debug("LumaProvider: id=%s  state=%s", gen_id, gen.state)
            if gen.state == "completed":
                break
            if gen.state == "failed":
                raise RuntimeError(
                    f"Luma generation {gen_id} failed: {getattr(gen, 'failure_reason', 'unknown')}"
                )
        else:
            raise RuntimeError(f"Luma generation {gen_id} timed out after {_POLL_TIMEOUT_SEC}s")

        # Step 4: download video.
        video_url: str = gen.assets.video
        logger.info("LumaProvider: downloading video from %s", video_url)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        video_bytes = httpx.get(video_url, timeout=120.0, follow_redirects=True).content
        output_path.write_bytes(video_bytes)

        logger.info(
            "LumaProvider: saved %s (%.1f KB)",
            output_path.name,
            len(video_bytes) / 1024,
        )
        return {"path": output_path, "generation_id": gen_id, "mocked": False}
