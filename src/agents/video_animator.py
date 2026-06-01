"""VideoAnimatorAgent — animates scene images using Veo 3 or Luma Dream Machine.

Sends each scene's PNG to the configured video provider and saves the resulting
MP4 clips to ``workspace/videos/scene_XX.mp4``.

Provider selection (set VIDEO_PROVIDER in .env):
  veo3  — Google Veo 3 (Google AI Pro key, best quality, ambient SFX)
  luma  — Luma Dream Machine (LUMA_API_KEY, 30 free generations)

Scenes are processed with limited concurrency (2 at a time) to respect
rate limits on both providers.

Output
------
* ``workspace/videos/scene_XX.mp4`` — one clip per scene, with audio.
* ``scene.video_path`` set to the workspace-relative path of each clip.
* ``state.has_ambient_audio`` set to True so compositor mixes audio correctly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.state import Scene, StoryState

log = logging.getLogger(__name__)

_MAX_CONCURRENCY = 2   # conservative — both providers have rate limits


class VideoAnimatorAgent(BaseAgent):
    """Animates each scene image with Veo 3 or Luma Dream Machine.

    Falls back gracefully per scene: if animation fails, ``scene.video_path``
    stays None and the compositor uses Ken Burns instead.
    """

    name: str = "VideoAnimator"
    system_prompt: str = ""

    def __init__(self) -> None:
        super().__init__()
        self.videos_dir = self.config.workspace_dir / "videos"
        provider_name = self.config.video_provider.lower()

        if provider_name == "veo3":
            from src.providers.veo_provider import VeoProvider  # noqa: PLC0415
            self._provider = VeoProvider()
            self.default_model = "veo-3.0-generate-preview"
        elif provider_name == "luma":
            from src.providers.luma_provider import LumaProvider  # noqa: PLC0415
            self._provider = LumaProvider()
            self.default_model = "luma-dream-machine"
        else:
            self._provider = None
            self.default_model = "none"

    def mock_response(self, state: StoryState) -> dict[str, Any]:
        return {"animated": 0, "mocked": True}

    async def run(self, state: StoryState) -> AgentResult:
        start = time.monotonic()

        if self._provider is None:
            log.info("VideoAnimatorAgent: VIDEO_PROVIDER=none — skipping animation.")
            return AgentResult(
                agent_name=self.name,
                model_used="none",
                output={"animated": 0, "total": len(state.scenes), "skipped": True},
                cost_usd=0.0,
                elapsed_sec=0.0,
                mocked=False,
            )

        scenes = sorted(state.scenes, key=lambda s: s.id)
        log.info(
            "VideoAnimatorAgent: START — animating %d scenes via %s",
            len(scenes), self.default_model,
        )

        sem = asyncio.Semaphore(_MAX_CONCURRENCY)
        tasks = [self._animate_scene(scene, sem) for scene in scenes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        animated = 0
        for scene, result in zip(scenes, results):
            if isinstance(result, Exception):
                log.warning(
                    "VideoAnimatorAgent: scene %d failed (%s) — Ken Burns fallback.",
                    scene.id, result,
                )
            else:
                animated += 1

        # Signal compositor to mix ambient audio when using Veo 3.
        if animated > 0 and self.config.video_provider.lower() == "veo3":
            state.has_ambient_audio = True

        elapsed = time.monotonic() - start
        log.info(
            "VideoAnimatorAgent: DONE — %d/%d animated in %.1fs",
            animated, len(scenes), elapsed,
        )

        return AgentResult(
            agent_name=self.name,
            model_used=self.default_model,
            output={"animated": animated, "total": len(scenes)},
            cost_usd=0.0,
            elapsed_sec=elapsed,
            mocked=False,
        )

    async def _animate_scene(self, scene: Scene, sem: asyncio.Semaphore) -> None:
        """Submit one scene to the provider and set scene.video_path on success."""
        if not scene.image_path:
            raise ValueError(f"Scene {scene.id} has no image_path")

        img_full = self.config.workspace_dir / scene.image_path
        if not img_full.exists():
            raise FileNotFoundError(f"Image not found: {img_full}")

        output_path = self.videos_dir / f"scene_{scene.id:02d}.mp4"

        # Resume-friendly: skip if clip already exists and is large enough.
        if output_path.exists() and output_path.stat().st_size > 50_000:
            log.info("VideoAnimatorAgent: scene %d — clip exists, skipping.", scene.id)
            scene.video_path = str(output_path.relative_to(self.config.workspace_dir))
            return

        async with sem:
            log.info(
                "VideoAnimatorAgent: scene %d — submitting (mood=%s, provider=%s) ...",
                scene.id, scene.mood, self.default_model,
            )
            result = await self._provider.image_to_video(
                img_path=img_full,
                mood=scene.mood,
                output_path=output_path,
            )

        scene.video_path = str(
            Path(result["path"]).relative_to(self.config.workspace_dir)
        )
        log.info("VideoAnimatorAgent: scene %d — saved %s", scene.id, output_path.name)
