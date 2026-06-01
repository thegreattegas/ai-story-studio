"""Shared state dataclasses for the AI Story Studio pipeline.

All agents read from and write into `StoryState`. LangGraph passes this
object as the graph's state between nodes.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Scene(BaseModel):
    """A single story scene — unit of work for Image and Voice agents.

    Attributes:
        id:                  Scene index (1-based, as produced by StoryWriter).
        narration:           The English text read aloud by the narrator.
        mood:                Emotional tone label (used by SceneDirector for voice/image).
        estimated_seconds:   Target duration for this scene in seconds.
        image_prompt:        Detailed English prompt for image generation (set by SceneDirector).
        voice_tone:          Narrator voice quality hint, e.g. "warm", "dramatic" (SceneDirector).
        voice_pace:          Narration speed hint: "slow" | "medium" | "fast" (SceneDirector).
        key_visual_elements: 3-5 must-include visual anchors for the image model (SceneDirector).
        character_dialogue:         Exact spoken words for the manga speech bubble (StoryWriter).
        character_dialogue_speaker: Name of the character who speaks (StoryWriter).
        image_path:                 Workspace-relative path to the generated image (ImageAgent, Phase 3).
        voice_segment_start: Start timestamp (sec) within the voice track (VoiceAgent, Phase 3).
        voice_segment_end:   End timestamp (sec) within the voice track (VoiceAgent, Phase 3).
    """

    id: int
    narration: str
    mood: str = "neutral"
    estimated_seconds: float = 8.0
    character_dialogue: Optional[str] = None
    character_dialogue_speaker: Optional[str] = None
    image_prompt: Optional[str] = None
    voice_tone: Optional[str] = None
    voice_pace: Optional[str] = None
    key_visual_elements: list[str] = Field(default_factory=list)
    image_path: Optional[str] = None
    video_path: Optional[str] = None          # workspace-relative path to Luma-generated clip
    voice_segment_start: Optional[float] = None
    voice_segment_end: Optional[float] = None


class StoryState(BaseModel):
    """Central state object shared across all pipeline agents.

    Agents receive the full state, mutate it in place, and return an
    :class:`~src.agents.base.AgentResult` with usage metrics. The LangGraph
    graph (Phase 5) will merge results back into state as an immutable update;
    for Phases 1-4 we mutate directly for simplicity.

    Attributes:
        user_prompt:         The raw user prompt (any language — treated as topic hint).
        language:            Always "en" — all generated content is English.
        target_duration_sec: Desired total video length in seconds.

        title:               Story title set by StoryWriter.
        summary:             2-3 sentence English summary set by StoryWriter.
        scenes:              Scene list: set by StoryWriter, enriched by SceneDirector.

        voice_path:          Workspace-relative path to the generated voice MP3 (Phase 3).
        subtitle_path:       Workspace-relative path to the .srt subtitle file (Phase 4).
        final_video_path:    Workspace-relative path to the composed .mp4 file (Phase 4).

        review_feedback:     Reviewer's text feedback when not approved (Phase 5).
        review_approved:     True once the Reviewer approves the final video.
        retry_count:         Regeneration attempts so far; max 2 enforced by graph.

        total_cost:          Accumulated API cost in USD across all agents.
        log_entries:         Ordered log strings for UI streaming (Phase 6).
    """

    # --- User input ---
    user_prompt: str
    language: str = "en"
    target_duration_sec: int = 60

    # --- StoryWriter output ---
    title: Optional[str] = None
    summary: Optional[str] = None
    scenes: list[Scene] = Field(default_factory=list)

    # --- Media outputs ---
    voice_path: Optional[str] = None
    subtitle_path: Optional[str] = None
    final_video_path: Optional[str] = None

    # --- Pipeline control ---
    review_feedback: Optional[str] = None
    review_approved: bool = False
    retry_count: int = 0
    has_ambient_audio: bool = False   # True when Veo 3 clips carry SFX audio

    # --- Tracking ---
    total_cost: float = 0.0
    log_entries: list[str] = Field(default_factory=list)

    def add_log(self, entry: str) -> None:
        """Append a log entry (mutates in place for convenience)."""
        self.log_entries.append(entry)

    def add_cost(self, amount: float) -> None:
        """Accumulate cost and warn if ceiling is exceeded."""
        self.total_cost += amount
        from src.config import get_config  # noqa: PLC0415

        cfg = get_config()
        if self.total_cost > cfg.cost_ceiling_usd:
            warning = (
                f"[WARNING] Total cost ${self.total_cost:.4f} exceeds "
                f"ceiling ${cfg.cost_ceiling_usd:.2f}"
            )
            self.log_entries.append(warning)
