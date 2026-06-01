"""ImageAgent — Phase 3.

Generates one illustration per scene using the OpenAIImageProvider (gpt-image-1).
All scenes are processed in parallel via ``asyncio.gather`` — start/end
timestamps are logged per scene to prove concurrent execution.

Output
------
* ``workspace/images/scene_<id:02d>.png`` for each scene.
* ``scene.image_path`` set on every :class:`~src.state.Scene` in state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.providers.openai_image_provider import OpenAIImageProvider
from src.providers.google_provider import GoogleImageProvider
from src.state import Scene, StoryState

log = logging.getLogger(__name__)


class ImageAgent(BaseAgent):
    """Generates scene illustrations in parallel via Google Gemini image gen.

    Each scene's :attr:`~src.state.Scene.image_prompt` is sent to the
    :class:`~src.providers.google_provider.GoogleImageProvider`.  All N
    provider calls fire simultaneously via ``asyncio.gather`` — the total
    wall time should be roughly equal to a single image's latency, not N
    times it.
    """

    name: str = "ImageAgent"
    system_prompt: str = ""  # Image models don't use system prompts.

    def __init__(self) -> None:
        super().__init__()
        if self.config.image_provider == "google":
            self.provider = GoogleImageProvider()
            self.default_model = "gemini-2.5-flash-image"
        else:  # openai (default)
            self.provider = OpenAIImageProvider()
            self.default_model = "gpt-image-1"
        self.images_dir: Path = self.config.workspace_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def mock_response(self, state: StoryState) -> dict[str, Any]:
        """Canned summary for logging — actual mock logic is in the provider."""
        return {
            "scenes_processed": len(state.scenes),
            "mocked": True,
        }

    async def run(self, state: StoryState) -> AgentResult:
        """Generate images for all scenes in parallel.

        Logs a START and DONE line per scene with wall-clock timestamps so
        that parallel execution is visible in the output.

        Args:
            state: Pipeline state with ``scenes`` populated and enriched.

        Returns:
            :class:`AgentResult` with aggregate cost and elapsed time.
        """
        overall_start = time.monotonic()
        n = len(state.scenes)

        if n == 0:
            log.warning("ImageAgent: no scenes in state — nothing to generate.")
            return AgentResult(
                agent_name=self.name,
                model_used=self.default_model,
                output={"scenes_processed": 0},
                mocked=self.config.mock_mode,
            )

        log.info(
            "ImageAgent: launching %d parallel image generations at t=%.3fs",
            n,
            time.monotonic() - overall_start,
        )

        # Fire all scene generation tasks at the same time.
        tasks = [
            self._generate_one(scene, overall_start)
            for scene in state.scenes
        ]
        scene_results: list[dict[str, Any]] = await asyncio.gather(*tasks)

        total_cost = sum(r.get("cost_usd", 0.0) for r in scene_results)
        total_elapsed = time.monotonic() - overall_start

        state.add_cost(total_cost)
        log_line = (
            f"[{self.name}] scenes={n} "
            f"cost=${total_cost:.6f} "
            f"elapsed={total_elapsed:.2f}s "
            f"mock={self.config.mock_mode}"
        )
        state.add_log(log_line)
        log.info(
            "ImageAgent: all %d images done in %.2fs total (vs %.2fs sequential estimate)",
            n,
            total_elapsed,
            n * 0.05,
        )

        return AgentResult(
            agent_name=self.name,
            model_used=self.default_model if not self.config.mock_mode else "MOCK",
            output={"scenes_processed": n, "elapsed_sec": total_elapsed},
            cost_usd=total_cost,
            elapsed_sec=total_elapsed,
            mocked=self.config.mock_mode,
        )

    # ------------------------------------------------------------------
    # Per-scene helper
    # ------------------------------------------------------------------

    async def _generate_one(
        self, scene: Scene, pipeline_start: float
    ) -> dict[str, Any]:
        """Generate one image, update ``scene.image_path``, and log timing.

        Args:
            scene:          The scene to illustrate.
            pipeline_start: ``time.monotonic()`` when the overall run started,
                            used to compute relative timestamps.

        Returns:
            Provider result dict ``{"path", "tokens_used", "cost_usd", ...}``.
        """
        t_start = time.monotonic() - pipeline_start
        log.info(
            "ImageAgent: scene %02d START @ pipeline_t=%.3fs  prompt='%s...'",
            scene.id,
            t_start,
            (scene.image_prompt or "")[:50],
        )

        output_path = self.images_dir / f"scene_{scene.id:02d}.png"
        prompt = scene.image_prompt or f"A fairy tale scene: {scene.narration[:80]}"

        result = await self.provider.generate_image(prompt, output_path)

        # Write back into state so downstream agents can find the file.
        scene.image_path = str(output_path.relative_to(self.config.workspace_dir))

        t_end = time.monotonic() - pipeline_start
        log.info(
            "ImageAgent: scene %02d DONE  @ pipeline_t=%.3fs  path=%s",
            scene.id,
            t_end,
            scene.image_path,
        )
        return result
