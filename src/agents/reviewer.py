"""ReviewerAgent — Phase 4.

Quality-gates the full pipeline output. Returns APPROVED or NEEDS_FIXES
with structured feedback so the pipeline can decide whether to retry.

Model
-----
Always uses Claude Opus — the highest-quality model for judgment tasks.

Output
------
* ``state.review_approved`` — True when the review decision is APPROVED.
* ``state.review_feedback`` — reviewer's summary sentence(s).
* Returns :class:`AgentResult` with ``decision``, ``issues``, ``feedback``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.state import StoryState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a quality control reviewer for an AI-generated children's fairy tale video (ages 6-16).

Given the story plan and pipeline metadata, decide if the result meets standards.

Return ONLY valid JSON — no prose, no markdown fences:
{
  "decision": "APPROVED" | "NEEDS_FIXES",
  "issues": ["<specific problem>"],
  "feedback": "<2-3 sentence summary>"
}

Check ALL of these:
1. Scene count: must be exactly 10
2. Total estimated duration: must be 55-70 seconds
3. Every scene must have image_path set
4. Every scene must have voice timing (voice_segment_start/end) set
5. Every scene must have a character_dialogue_speaker (character voice assigned)
6. Every scene must have a character_dialogue line (non-empty)
7. Story arc: clear setup → rising action → climax → resolution across 10 scenes
8. Narration language: simple words, short sentences, suitable for age 6-16
9. subtitle_path and final_video_path must be present
10. Narration should be emotionally engaging — not flat or generic

Mark NEEDS_FIXES for ANY failed check above. APPROVED only when all 10 checks pass.\
"""


class ReviewerAgent(BaseAgent):
    """Reviews the completed pipeline and returns APPROVED or NEEDS_FIXES.

    In mock mode returns an instant APPROVED without any API call.
    In live mode calls Opus with a JSON summary of the full pipeline state.
    """

    name: str = "Reviewer"
    system_prompt: str = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        # Always use Opus — complex judgment task.
        self.default_model = self.config.effective_model_opus

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def mock_response(self, state: StoryState) -> dict[str, Any]:
        """Return an instant APPROVED decision for mock mode."""
        return {
            "decision": "APPROVED",
            "issues": [],
            "feedback": "Mock review: structure looks good, 5 scenes, timing correct.",
        }

    async def run(self, state: StoryState) -> AgentResult:
        """Review the pipeline and set ``state.review_approved``.

        Args:
            state: Fully-populated pipeline state (post Phase 4).

        Returns:
            :class:`AgentResult` with review decision and cost.
        """
        start = time.monotonic()
        log.info("Reviewer: START — evaluating pipeline output")

        if self.config.mock_mode:
            review = self.mock_response(state)
            state.review_approved = review["decision"] == "APPROVED"
            state.review_feedback = review["feedback"]
            elapsed = time.monotonic() - start
            log.info("Reviewer: DONE (mock) — %s", review["decision"])
            return AgentResult(
                agent_name=self.name,
                model_used="MOCK",
                output=review,
                cost_usd=0.0,
                elapsed_sec=elapsed,
                mocked=True,
            )

        # Build a concise JSON summary of the pipeline for the reviewer.
        story_summary = json.dumps(
            {
                "title": state.title,
                "summary": state.summary,
                "scene_count": len(state.scenes),
                "total_estimated_sec": sum(
                    s.estimated_seconds for s in state.scenes
                ),
                "scenes": [
                    {
                        "id": s.id,
                        "narration": s.narration,
                        "mood": s.mood,
                        "has_image": bool(s.image_path),
                        "has_voice_timing": s.voice_segment_start is not None,
                        "speaker": s.character_dialogue_speaker,
                        "dialogue": s.character_dialogue,
                    }
                    for s in state.scenes
                ],
                "has_voice": bool(state.voice_path),
                "has_subtitles": bool(state.subtitle_path),
                "has_final_video": bool(state.final_video_path),
            },
            indent=2,
        )

        user_prompt = (
            f"Review this video production pipeline:\n\n"
            f"```json\n{story_summary}\n```\n\n"
            f"Return your decision as JSON."
        )

        text, tokens_in, tokens_out = await self._call_llm(
            user_prompt,
            model=self.config.effective_model_opus,
            max_tokens=800,
        )

        review = self._parse_review(text)
        state.review_approved = review["decision"] == "APPROVED"
        state.review_feedback = review.get("feedback", "")

        cost = self.config.cost_usd(tokens_in, tokens_out, self.config.effective_model_opus)
        elapsed = time.monotonic() - start

        state.add_cost(cost)
        log.info(
            "Reviewer: DONE — %s  cost=$%.4f  elapsed=%.2fs",
            review["decision"],
            cost,
            elapsed,
        )

        return AgentResult(
            agent_name=self.name,
            model_used=self.config.effective_model_opus,
            output=review,
            cost_usd=cost,
            elapsed_sec=elapsed,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            mocked=False,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_review(self, text: str) -> dict[str, Any]:
        """Parse the LLM's JSON response, falling back to APPROVED on error.

        Args:
            text: Raw text from the LLM (may contain markdown fences).

        Returns:
            Dict with ``decision``, ``issues``, ``feedback``.
        """
        cleaned = self._strip_fences(text)
        try:
            review = json.loads(cleaned)
            # Ensure required fields exist.
            if "decision" not in review:
                raise ValueError("Missing 'decision' key")
            if review["decision"] not in {"APPROVED", "NEEDS_FIXES"}:
                raise ValueError(f"Unknown decision: {review['decision']!r}")
            review.setdefault("issues", [])
            review.setdefault("feedback", "")
            return review
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning(
                "Reviewer: could not parse LLM response (%s) — defaulting to APPROVED",
                exc,
            )
            return {
                "decision": "APPROVED",
                "issues": [],
                "feedback": "Could not parse reviewer response — defaulting to approved.",
            }

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove ``` fences that some models add despite instructions."""
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped[stripped.index("\n") + 1:]
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
        return stripped.strip()
