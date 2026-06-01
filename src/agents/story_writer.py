"""StoryWriter agent — Phase 2.

Converts a user prompt into a structured English fairy tale plan:
    {title, summary, language, target_duration_sec, scenes: [...]}

Always uses Sonnet via force_model override (creative work needs quality).
Saves the story plan to workspace/story.json for inspection and downstream use.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.state import Scene, StoryState
from src.tools.file_tools import write_file

# ---------------------------------------------------------------------------
# System prompt (constant — easy to tune without touching logic)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an award-winning children's book author. Your fairy tales feel complete, \
emotionally moving, and leave readers with a warm heart.

A great story is NOT a list of events — it is a SINGLE CONTINUOUS JOURNEY.
Every scene is a direct consequence of the scene before it.
There are no "meanwhile" cuts, no sudden unrelated situations, no random episodes.
The story has ONE clear problem (Point A) and ONE clear resolution (Point B).
Every scene moves the protagonist closer to or further from that resolution.

Return ONLY valid JSON — no prose, no markdown fences.

Schema:
{
  "title": "<short, memorable English title — evokes wonder>",
  "summary": "<3-4 sentence English summary covering the full arc: who, what problem, how resolved, what changed>",
  "language": "en",
  "target_duration_sec": <integer, 45-55>,
  "scenes": [
    {
      "id": <integer starting at 1>,
      "narration": "<one vivid English sentence — paint the scene, carry the emotion, DIRECTLY reference what just happened>",
      "speaker": "<the character who speaks this scene's dialogue line>",
      "dialogue": "<one line of spoken words ONLY — no attribution — must reveal character personality>",
      "mood": "<one of: peaceful, tense, joyful, sad, mysterious, exciting, triumphant, determined, hopeful, curious>",
      "estimated_seconds": <float, 4-6>
    }
  ]
}

━━━ NARRATIVE SPINE (define this BEFORE writing scenes) ━━━
Every story must have a spine you commit to and never break:
  POINT A: What is the protagonist's SPECIFIC problem right now? (not a vague situation — a concrete obstacle)
  JOURNEY: What THREE things must they do or overcome to solve it?
  POINT B: What does the world look like after the problem is solved? What has the protagonist learned?

Bad spine: "Dragon goes on adventures."
Good spine: "Ember can't breathe fire → tries alone and fails → asks mentor → practices at midnight → breathes fire and saves her friend."

━━━ CAUSAL CHAIN LAW ━━━
⚠ CRITICAL: Scene N+1 must be a DIRECT CONSEQUENCE of scene N.
Each narration must begin with a reference to what just happened:
  Good: "Shocked by her failure, Ember ran deeper into the cave."
  Good: "But the fire fizzled again — Ember had forgotten to believe in herself."
  Bad:  "The next day, something surprising happened." (no connection)
  Bad:  "Meanwhile, in the forest..." (unrelated cut)

If you find yourself writing a new scene without connecting to the previous one — STOP and rewrite.

━━━ CHARACTER RULES ━━━
- Maximum 3 named characters. Give each a FULL NAME and one personality trait in scene 1.
  Good: "Ember the curious little dragon" or "Sage the patient old tortoise"
  Bad: just "Dragon" or "Tortoise" with no personality
- Every character must want something and fear something — this drives the story.
- Dialogue must sound like THAT character — brave ones speak boldly, shy ones speak quietly.

━━━ NARRATION RULES ━━━
- Each narration: 10-16 words MAXIMUM — one vivid sentence, emotion + consequence.
- EVERY narration connects back to the previous scene (what happened → what now).
  Good: "Ember's heart sank — the flame had gone out before reaching the ice."
  Bad:  "Ember looked at the mountain and felt brave."
- Pack emotion into few words — every word must earn its place.
- Simple words a 6-year-old knows — but written with the heart of a poet.

━━━ DIALOGUE RULES ━━━
- ONLY the spoken words — NO attribution prefix.
  Wrong: "Ember said: I will try again."
  Right: "I'm not giving up — not ever."
- Dialogue must reveal personality AND emotion in the context of THIS scene's struggle.

━━━ STORY ARC (10 scenes — strict) ━━━
Scene 1  [peaceful/curious]     — Protagonist in their ordinary world. Show their personality AND hint at what they lack or want.
Scene 2  [tense/curious]        — The specific problem appears or is discovered. Point A is established.
Scene 3  [sad/tense]            — First attempt fails OR the true scale of the problem is revealed. Stakes hurt.
Scene 4  [determined/hopeful]   — BECAUSE of that failure, protagonist makes a conscious choice to keep trying.
Scene 5  [exciting/hopeful]     — New approach or helper found AS A RESULT of scene 4's choice. Some progress.
Scene 6  [tense/sad]            — Worst moment — a direct setback from scene 5's progress. Doubt is real.
Scene 7  [determined/exciting]  — BECAUSE of scene 6's darkness, a new insight or act of courage emerges.
Scene 8  [exciting/joyful]      — The breakthrough — CAUSED by scene 7's insight. The problem is solved!
Scene 9  [joyful/triumphant]    — The immediate consequence of success: celebration, reunion, relief.
Scene 10 [peaceful/triumphant]  — The world after the journey. What has changed? A quiet, earned reflection.

━━━ TECHNICAL RULES ━━━
- EXACTLY 10 scenes — always 10, never fewer, never more
- "speaker" field: REQUIRED in every scene
- "dialogue" field: REQUIRED in every scene — spoken words only, max 10 words
- estimated_seconds: 4-6 per scene; total ≈ target_duration_sec
- target_duration_sec: 45-55
- Language: always "en"\
"""

# ---------------------------------------------------------------------------
# Fixed mock fixture — used in mock mode AND as test reference
# ---------------------------------------------------------------------------

MOCK_STORY: dict[str, Any] = {
    "title": "The Forest Friendship",
    "summary": (
        "A lonely young wolf and a clever fox learn to trust each other "
        "when they team up to rescue a tiny bunny trapped in the snow."
    ),
    "language": "en",
    "target_duration_sec": 50,
    "scenes": [
        {
            "id": 1,
            "narration": (
                "Deep in a snowy winter forest lived a young wolf named Shadow. "
                "He was lonely and always hunted alone."
            ),
            "mood": "peaceful",
            "estimated_seconds": 8.0,
        },
        {
            "id": 2,
            "narration": (
                "One quiet morning, he met a red fox named Ruby. "
                "At first, they watched each other with deep distrust."
            ),
            "mood": "tense",
            "estimated_seconds": 9.0,
        },
        {
            "id": 3,
            "narration": (
                "Suddenly, they heard a tiny squeak — a small bunny was trapped "
                "in a deep snowdrift, shivering with cold!"
            ),
            "mood": "tense",
            "estimated_seconds": 10.0,
        },
        {
            "id": 4,
            "narration": (
                "Shadow and Ruby dug through the snow together and freed the little bunny. "
                "The baby trembled, but smiled with hope."
            ),
            "mood": "exciting",
            "estimated_seconds": 11.0,
        },
        {
            "id": 5,
            "narration": (
                "From that day on, the wolf and the fox became the very best of friends. "
                "A new legend of friendship was born in the forest."
            ),
            "mood": "triumphant",
            "estimated_seconds": 10.0,
        },
    ],
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class StoryWriterAgent(BaseAgent):
    """Generates the full story structure from the user's raw prompt.

    Output fields written to ``StoryState``:
        - ``title``
        - ``summary``
        - ``language``
        - ``target_duration_sec``
        - ``scenes`` (list of :class:`~src.state.Scene`)

    Also saves ``workspace/story.json`` for inspection.
    """

    name: str = "StoryWriter"
    system_prompt: str = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        # Always force Sonnet — creative writing quality over cost.
        self.default_model = self.config.effective_model_sonnet

    # ------------------------------------------------------------------
    # Mock
    # ------------------------------------------------------------------

    def mock_response(self, state: StoryState) -> dict[str, Any]:
        """Return the fixed MOCK_STORY fixture."""
        return MOCK_STORY

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self, state: StoryState) -> AgentResult:
        """Generate a structured story and write it into ``state``.

        In mock mode: applies MOCK_STORY to state with no API call.
        In live mode: calls Sonnet, parses JSON (with one retry on failure),
        applies parsed data to state.

        Args:
            state: Pipeline state to mutate.

        Returns:
            :class:`AgentResult` with story payload and usage metrics.
        """
        start = time.monotonic()
        tokens_in = tokens_out = 0

        if self.config.mock_mode:
            self.logger.info("StoryWriter: mock mode — using MOCK_STORY fixture.")
            story_data = self.mock_response(state)
            is_mock = True
        else:
            user_prompt = self._build_user_prompt(state)
            story_data, tokens_in, tokens_out = await self._call_llm_json(
                user_prompt, max_tokens=3500
            )
            is_mock = False

        # Apply parsed story to state.
        self._apply_to_state(story_data, state)

        # Persist story plan to workspace.
        self._save_to_workspace(state)

        # Cost accounting.
        cost = self.config.cost_usd(tokens_in, tokens_out, self.default_model)
        state.add_cost(cost)
        log_line = (
            f"[{self.name}] title='{state.title}' scenes={len(state.scenes)} "
            f"tokens_in={tokens_in} tokens_out={tokens_out} "
            f"cost=${cost:.6f} mock={is_mock}"
        )
        state.add_log(log_line)
        self.logger.info(log_line)

        elapsed = time.monotonic() - start
        result = AgentResult(
            agent_name=self.name,
            model_used=self.default_model,
            output=story_data,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            elapsed_sec=elapsed,
            mocked=is_mock,
        )
        self.logger.info(result.summary_line())
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_prompt(self, state: StoryState) -> str:
        """Construct the user-turn message sent to the model."""
        return (
            f"Write a complete fairy tale based on this idea:\n\n"
            f"  \"{state.user_prompt}\"\n\n"
            f"STEP 1 — Before writing the JSON, silently define your narrative spine "
            f"(do NOT include this in the output):\n"
            f"  • POINT A: What is the protagonist's SPECIFIC problem right now?\n"
            f"  • POINT B: What does the world look like after it is solved?\n"
            f"  • CAUSAL CHAIN: Scene 1 leads to 2 leads to 3 … leads to 10.\n"
            f"    Write one sentence per scene showing how each scene CAUSES the next.\n"
            f"  • DARKEST MOMENT: What goes wrong in scene 6 that nearly breaks hope?\n"
            f"  • BREAKTHROUGH: What specific thing happens in scene 7-8 that solves it?\n\n"
            f"STEP 2 — Write the JSON. CRITICAL requirements:\n"
            f"  — EXACTLY 10 scenes\n"
            f"  — Each scene narration MUST start by referencing what just happened\n"
            f"  — NO sudden new situations that aren't caused by the previous scene\n"
            f"  — NO 'meanwhile', 'one day', 'suddenly' as scene openers without cause\n"
            f"  — Narrations: 10-16 words MAX each\n"
            f"  — Dialogue reveals character personality, not just facts\n"
            f"  — Every scene has 'speaker' and 'dialogue' fields\n"
            f"  — Total duration 45-55 seconds MAX\n"
            f"  — ALL text in English\n\n"
            f"Return ONLY the JSON. No prose, no markdown fences."
        )

    async def _call_llm_json(
        self,
        user_prompt: str,
        max_tokens: int,
    ) -> tuple[dict[str, Any], int, int]:
        """Call LLM and parse JSON. Retries once if the first response is invalid.

        Args:
            user_prompt: Initial user-turn prompt.
            max_tokens:  Max output tokens.

        Returns:
            ``(parsed_dict, total_tokens_in, total_tokens_out)``

        Raises:
            ValueError: If both attempts produce unparseable JSON.
        """
        text, tin, tout = await self._call_llm(user_prompt, max_tokens=max_tokens)

        try:
            return json.loads(self._strip_markdown_fences(text)), tin, tout
        except json.JSONDecodeError as first_err:
            self.logger.warning(
                "StoryWriter: JSON parse failed on first attempt (%s). Retrying...",
                first_err,
            )

        retry_prompt = (
            "Your previous response was not valid JSON. "
            "Return ONLY the JSON object with no additional text or markdown fences.\n\n"
            f"Original request:\n{user_prompt}"
        )
        text2, tin2, tout2 = await self._call_llm(retry_prompt, max_tokens=max_tokens)

        try:
            return (
                json.loads(self._strip_markdown_fences(text2)),
                tin + tin2,
                tout + tout2,
            )
        except json.JSONDecodeError as second_err:
            raise ValueError(
                f"StoryWriter failed to produce valid JSON after 2 attempts. "
                f"Last error: {second_err}\nLast response snippet: {text2[:200]}"
            ) from second_err

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove ```json ... ``` fences that some models add despite instructions."""
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove opening fence line.
            stripped = stripped[stripped.index("\n") + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
        return stripped.strip()

    def _apply_to_state(self, story_data: dict[str, Any], state: StoryState) -> None:
        """Write story_data fields into ``state``.

        Validates required fields and builds :class:`~src.state.Scene` objects.
        Unknown extra fields in ``story_data`` are silently ignored.
        """
        state.title = story_data.get("title", "Untitled Story")
        state.summary = story_data.get("summary", "")
        state.language = story_data.get("language", "en")
        duration = story_data.get("target_duration_sec")
        if isinstance(duration, int) and duration > 0:
            state.target_duration_sec = duration

        raw_scenes: list[dict] = story_data.get("scenes", [])
        if not raw_scenes:
            self.logger.warning("StoryWriter: story_data contains no scenes!")

        state.scenes = [
            Scene(
                id=s.get("id", idx + 1),
                narration=s.get("narration", ""),
                character_dialogue=self._strip_speaker_prefix(s.get("dialogue")),
                character_dialogue_speaker=self._extract_speaker(
                    s.get("speaker"), s.get("dialogue")
                ),
                mood=s.get("mood", "neutral"),
                estimated_seconds=float(s.get("estimated_seconds", 6.0)),
            )
            for idx, s in enumerate(raw_scenes)
        ]

    @staticmethod
    def _extract_speaker(speaker: str | None, dialogue: str | None) -> str | None:
        """Return speaker name — from explicit field or extracted from attribution."""
        if speaker:
            return speaker.strip()
        if not dialogue:
            return None
        # Try to extract from "Name verb: words" pattern
        m = re.match(r'^([A-Z][a-zA-Z]+)\s+\w+\s*:', dialogue)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _strip_speaker_prefix(dialogue: str | None) -> str | None:
        """Remove 'Character verb: ' attribution, keeping only the spoken words.

        Handles patterns like:
            'Owl said: Face your fear!'       → 'Face your fear!'
            'Animals cheered: Hooray!'        → 'Hooray!'
            'Wolf growled: "I am hungry"'     → 'I am hungry'
        """
        if not dialogue:
            return None
        # Match "Word(s) AnyVerb: " at the start — e.g. "Owl advised:", "Animals cheered:"
        stripped = re.sub(
            r'^[A-Z][a-zA-Z ]{0,25}:\s*',
            '',
            dialogue,
        ).strip().strip('"\'')
        return stripped or dialogue

    def _save_to_workspace(self, state: StoryState) -> None:
        """Persist the current state as workspace/story.json."""
        try:
            write_file(
                "story.json",
                json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
            )
            self.logger.debug("StoryWriter: saved story plan to workspace/story.json")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("StoryWriter: could not save story.json — %s", exc)
