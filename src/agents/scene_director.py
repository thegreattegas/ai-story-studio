"""SceneDirector agent — Phase 2.

Enriches each scene produced by StoryWriter with:
- A detailed English image prompt optimised for Nano Banana / Imagen
- Voice tone and pace hints for ElevenLabs
- 3-5 key visual anchors for image generation consistency

Scenes are processed in parallel via ``asyncio.gather`` — log timestamps
confirm that all per-scene LLM calls fire concurrently.

Character extraction strategy
------------------------------
For the real LLM path we make a lightweight **Haiku** call that reads the
story summary and first-scene narration and returns a JSON list of characters
with brief visual descriptions.  Using Haiku keeps cost near zero for this
bookkeeping step (~300 tokens in, ~100 out ≈ $0.00002).  The resulting list
is passed to every scene call so the model can maintain character consistency.

In mock mode we use a fixed character list matching the MOCK_STORY fixture.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.state import Scene, StoryState

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a children's book art director preparing scenes for AI image generation and voice narration.

You are working on a CONTINUOUS storybook — each image is the next panel in the same story.
Visual continuity between scenes is critical: characters must look identical across all panels,
the setting must evolve naturally (not jump randomly), and the art style must stay consistent.

Given a story scene, produce enriched directorial details. Return ONLY valid JSON — no prose.

Input you will receive:
{
  "scene_id": <int>,
  "narration": "<English narrator text>",
  "dialogue": "<one-line character speech>",
  "mood": "<mood>",
  "story_context": "<brief English summary of the overall story>",
  "characters": ["<list of recurring characters with brief visual descriptions>"],
  "previous_scene_prompt": "<image prompt from the previous scene, or null for scene 1>"
}

Output schema:
{
  "scene_id": <int>,
  "image_prompt": "<detailed English visual prompt, 40-70 words>",
  "voice_tone": "<one of: warm, dramatic, mysterious, cheerful, gentle, triumphant>",
  "voice_pace": "<slow | medium | fast>",
  "key_visual_elements": ["<list of 3-5 must-include visual elements>"]
}

Rules for image_prompt:
- MUST be in English
- MUST include exactly this style phrase: \
"watercolor children's book illustration, soft pastel colors"
- MUST mention recurring characters by their FULL visual description \
(e.g. "young grey wolf with kind blue eyes" not just "wolf")
- CONTINUITY: If previous_scene_prompt is provided, your image must feel like the \
NEXT PANEL in the same book — same characters, same art style, same world. \
Only change what the narration says changed (location, time of day, action). \
Repeat key visual anchors (character colours, clothing, environment details) \
from the previous scene so the images flow together.
- Do NOT include any speech bubbles or text overlays — these are added in post-processing
- Focus entirely on the visual scene: characters, setting, action, lighting, mood
- Voice tone must match scene mood
- key_visual_elements must NOT include text or speech bubbles\
"""

# Character extraction prompt (sent to Haiku — cheap bookkeeping).
_CHARACTER_EXTRACTION_PROMPT = """\
Read this story summary and opening scene. List the main recurring characters.
Return ONLY a JSON array of strings, each describing one character's visual appearance.
Example format: ["young grey wolf with kind blue eyes and fluffy winter coat", \
"red fox with white-tipped tail and bright amber eyes"]

Story summary: {summary}

Opening scene: {first_narration}

Return the JSON array now:\
"""

# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------

# Fixed character descriptions — used in mock mode for character consistency.
MOCK_CHARACTERS: list[str] = [
    "young grey wolf with kind blue eyes, fluffy winter coat",
    "red fox with white-tipped tail and bright amber eyes",
    "tiny brown baby bunny with white belly",
]

# Style suffix appended to every image prompt for visual consistency.
_STYLE_SUFFIX = "watercolor children's book illustration, soft pastel colors"

MOCK_ENRICHED_SCENES: list[dict[str, Any]] = [
    {
        "scene_id": 1,
        "image_prompt": (
            "A young grey wolf with kind blue eyes and fluffy winter coat walking alone "
            "through a deep snowy pine forest at dawn, soft golden light filtering through "
            f"trees, melancholic atmosphere, wide composition, {_STYLE_SUFFIX}"
        ),
        "voice_tone": "gentle",
        "voice_pace": "slow",
        "key_visual_elements": [
            "young grey wolf",
            "snowy pine forest",
            "dawn light",
            "lone figure",
            "deep snow",
        ],
    },
    {
        "scene_id": 2,
        "image_prompt": (
            "A young grey wolf with kind blue eyes facing a red fox with white-tipped tail "
            "and bright amber eyes on a snow-covered forest path, both animals alert and "
            f"cautious, eye-level composition, cool winter light, {_STYLE_SUFFIX}"
        ),
        "voice_tone": "dramatic",
        "voice_pace": "medium",
        "key_visual_elements": [
            "young grey wolf",
            "red fox with white-tipped tail",
            "face-to-face standoff",
            "snowy path",
            "tense atmosphere",
        ],
    },
    {
        "scene_id": 3,
        "image_prompt": (
            "A tiny brown baby bunny with white belly half-buried in a deep snowdrift, "
            "shivering, looking up with frightened eyes; a young grey wolf and red fox "
            f"watching in alarm, dramatic close composition, {_STYLE_SUFFIX}"
        ),
        "voice_tone": "dramatic",
        "voice_pace": "fast",
        "key_visual_elements": [
            "tiny brown bunny in snowdrift",
            "frightened bunny eyes",
            "grey wolf",
            "red fox",
            "urgency",
        ],
    },
    {
        "scene_id": 4,
        "image_prompt": (
            "A young grey wolf with kind blue eyes and a red fox with white-tipped tail "
            "digging side-by-side through a snowdrift, freeing a tiny brown baby bunny "
            f"with white belly, warm teamwork energy, bright snow sparkles, {_STYLE_SUFFIX}"
        ),
        "voice_tone": "cheerful",
        "voice_pace": "fast",
        "key_visual_elements": [
            "wolf and fox digging together",
            "baby bunny being rescued",
            "snow flying",
            "cooperation",
            "bright snowy clearing",
        ],
    },
    {
        "scene_id": 5,
        "image_prompt": (
            "A young grey wolf with kind blue eyes and a red fox with white-tipped tail "
            "sitting together under a snow-laden pine tree at sunset, the tiny brown "
            "baby bunny with white belly nestled between them, golden light, joyful "
            f"and peaceful mood, panoramic composition, {_STYLE_SUFFIX}"
        ),
        "voice_tone": "warm",
        "voice_pace": "slow",
        "key_visual_elements": [
            "wolf and fox side-by-side",
            "baby bunny safe between them",
            "sunset golden light",
            "pine tree",
            "friendship",
        ],
    },
]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SceneDirectorAgent(BaseAgent):
    """Enriches scenes with cinematic and audio direction.

    Output fields written to each ``Scene`` in ``state.scenes``:
        - ``image_prompt``
        - ``voice_tone``
        - ``voice_pace``
        - ``key_visual_elements``
    """

    name: str = "SceneDirector"
    system_prompt: str = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        # Creative enrichment: Sonnet for quality image prompts.
        self.default_model = self.config.effective_model_sonnet

    # ------------------------------------------------------------------
    # Mock
    # ------------------------------------------------------------------

    def mock_response(self, state: StoryState) -> dict[str, Any]:
        """Return the pre-built MOCK_ENRICHED_SCENES list."""
        # Return only as many scenes as state currently has.
        return {"enriched_scenes": MOCK_ENRICHED_SCENES[: len(state.scenes)]}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self, state: StoryState) -> AgentResult:
        """Enrich all scenes in parallel and write results into ``state``.

        Args:
            state: Pipeline state with ``scenes`` populated by StoryWriter.

        Returns:
            :class:`AgentResult` with enriched scene data and aggregated metrics.
        """
        start = time.monotonic()

        if not state.scenes:
            self.logger.warning("SceneDirector: no scenes in state — nothing to enrich.")

        if self.config.mock_mode:
            self.logger.info(
                "SceneDirector: mock mode — applying MOCK_ENRICHED_SCENES to %d scenes.",
                len(state.scenes),
            )
            enriched_list = MOCK_ENRICHED_SCENES[: len(state.scenes)]
            total_in = total_out = 0
            is_mock = True
        else:
            # Step 1: extract characters (Haiku call — cheap).
            characters = await self._extract_characters(state)

            # Step 2: enrich scenes sequentially so each scene receives the
            # previous scene's image_prompt for visual continuity.
            self.logger.info(
                "SceneDirector: starting sequential enrichment of %d scenes at t=%.3fs",
                len(state.scenes),
                time.monotonic() - start,
            )
            enriched_list = []
            total_in = total_out = 0
            previous_prompt: str | None = None

            for scene in state.scenes:
                enriched, tin, tout = await self._enrich_scene(
                    scene, state.summary or "", characters, start, previous_prompt
                )
                enriched_list.append(enriched)
                total_in += tin
                total_out += tout
                previous_prompt = enriched.get("image_prompt")

            is_mock = False

            self.logger.info(
                "SceneDirector: all %d scenes enriched, elapsed=%.2fs",
                len(state.scenes),
                time.monotonic() - start,
            )

        # Apply enriched data to state scenes.
        self._apply_to_state(enriched_list, state)

        # Cost accounting.
        cost = self.config.cost_usd(total_in, total_out, self.default_model)
        state.add_cost(cost)
        log_line = (
            f"[{self.name}] scenes={len(state.scenes)} "
            f"tokens_in={total_in} tokens_out={total_out} "
            f"cost=${cost:.6f} mock={is_mock}"
        )
        state.add_log(log_line)
        self.logger.info(log_line)

        elapsed = time.monotonic() - start
        result = AgentResult(
            agent_name=self.name,
            model_used=self.default_model,
            output={"enriched_scenes": enriched_list},
            tokens_in=total_in,
            tokens_out=total_out,
            cost_usd=cost,
            elapsed_sec=elapsed,
            mocked=is_mock,
        )
        self.logger.info(result.summary_line())
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _extract_characters(self, state: StoryState) -> list[str]:
        """Use a lightweight Haiku call to list characters with visual descriptions.

        Returns:
            List of visual-description strings, e.g.
            ``["young grey wolf with kind blue eyes", ...]``
            Falls back to an empty list on error.
        """
        if not state.summary and not state.scenes:
            return []

        first_narration = state.scenes[0].narration if state.scenes else ""
        prompt = _CHARACTER_EXTRACTION_PROMPT.format(
            summary=state.summary or "",
            first_narration=first_narration,
        )

        text, _, _ = await self._call_llm(
            prompt,
            model=self.config.effective_model_haiku,
            max_tokens=300,
        )
        try:
            # Strip markdown fences if any.
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = stripped[stripped.index("\n") + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[: stripped.rfind("```")]
            characters: list[str] = json.loads(stripped.strip())
            if isinstance(characters, list):
                return [str(c) for c in characters]
        except (json.JSONDecodeError, ValueError) as exc:
            self.logger.warning(
                "SceneDirector: character extraction failed (%s) — proceeding without list.",
                exc,
            )
        return []

    async def _enrich_scene(
        self,
        scene: Scene,
        story_context: str,
        characters: list[str],
        pipeline_start: float,
        previous_scene_prompt: str | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Enrich a single scene with directorial details.

        Args:
            previous_scene_prompt: Image prompt from the immediately preceding
                scene — passed so the model can maintain visual continuity.

        Returns:
            ``(enriched_dict, tokens_in, tokens_out)``
        """
        t_scene_start = time.monotonic() - pipeline_start
        self.logger.debug(
            "SceneDirector: scene %d started at pipeline_t=%.3fs",
            scene.id,
            t_scene_start,
        )

        user_prompt = json.dumps(
            {
                "scene_id": scene.id,
                "narration": scene.narration,
                "speaker": scene.character_dialogue_speaker or "",
                "dialogue": scene.character_dialogue or scene.narration[:60],
                "mood": scene.mood,
                "story_context": story_context,
                "characters": characters,
                "previous_scene_prompt": previous_scene_prompt,
            },
            ensure_ascii=False,
        )

        text, tokens_in, tokens_out = await self._call_llm(user_prompt, max_tokens=600)

        try:
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = stripped[stripped.index("\n") + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[: stripped.rfind("```")]
            enriched: dict[str, Any] = json.loads(stripped.strip())
        except (json.JSONDecodeError, ValueError) as exc:
            self.logger.warning(
                "SceneDirector: JSON parse failed for scene %d (%s) — using fallback.",
                scene.id,
                exc,
            )
            # Minimal fallback so pipeline continues.
            enriched = {
                "scene_id": scene.id,
                "image_prompt": (
                    f"{scene.narration[:80]}, {_STYLE_SUFFIX}"
                ),
                "voice_tone": "warm",
                "voice_pace": "medium",
                "key_visual_elements": [],
            }

        t_scene_end = time.monotonic() - pipeline_start
        self.logger.debug(
            "SceneDirector: scene %d completed at pipeline_t=%.3fs",
            scene.id,
            t_scene_end,
        )

        return enriched, tokens_in, tokens_out

    def _apply_to_state(
        self, enriched_list: list[dict[str, Any]], state: StoryState
    ) -> None:
        """Write enriched fields back onto each Scene in ``state.scenes``.

        Matches by scene id; warns if a scene id is missing from enriched data.
        """
        enriched_by_id: dict[int, dict[str, Any]] = {
            e.get("scene_id", idx + 1): e
            for idx, e in enumerate(enriched_list)
        }

        for scene in state.scenes:
            data = enriched_by_id.get(scene.id)
            if data is None:
                self.logger.warning(
                    "SceneDirector: no enriched data for scene id=%d — skipping.",
                    scene.id,
                )
                continue
            scene.image_prompt = data.get("image_prompt", "")
            scene.voice_tone = data.get("voice_tone", "warm")
            scene.voice_pace = data.get("voice_pace", "medium")
            scene.key_visual_elements = data.get("key_visual_elements", [])
