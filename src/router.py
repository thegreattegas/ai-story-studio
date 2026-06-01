"""Model router — selects the right Claude model for a given task.

Routing logic
-------------
Tasks are scored by keyword matches and length:

* Trivial keywords  → favour Haiku  (cheap, fast)
* Complex keywords  → favour Opus   (expensive, best quality)
* Everything else   → Sonnet        (balanced default)

A ``force_model`` override bypasses scoring entirely — used by creative
agents (StoryWriter) that always need Sonnet regardless of prompt content.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

TRIVIAL_KEYWORDS: list[str] = [
    "rename",
    "format",
    "list",
    "read",
    "summarize briefly",
    "count",
    "extract",
    "convert",
    "translate briefly",
    "simple",
]

COMPLEX_KEYWORDS: list[str] = [
    "architecture",
    "design",
    "review",
    "story",
    "creative",
    "narrative",
    "write a",
    "generate",
    "compose",
    "complex",
    "analyze",
    "evaluate",
    "critique",
    "plan",
]

# Score thresholds
_TRIVIAL_THRESHOLD = 1
_COMPLEX_THRESHOLD = 2

# Task length (words) thresholds
_SHORT_TASK_WORDS = 8   # ≤ this → lean trivial
_LONG_TASK_WORDS = 25   # ≥ this → lean medium


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """Result of a model routing call.

    Attributes:
        model:        The chosen model ID string.
        complexity:   Human-readable complexity label: "trivial", "medium", or "complex".
        score:        Raw routing score (positive = complex, negative = trivial).
        keyword_hits: List of keywords that influenced the decision.
        forced:       True if the model was set via ``force_model`` override.
    """

    model: str
    complexity: str
    score: int
    keyword_hits: list[str] = field(default_factory=list)
    forced: bool = False

    def __str__(self) -> str:  # noqa: D105
        forced_tag = " [FORCED]" if self.forced else ""
        hits = ", ".join(self.keyword_hits) if self.keyword_hits else "none"
        return (
            f"RoutingDecision(model={self.model}, complexity={self.complexity}, "
            f"score={self.score}, hits=[{hits}]{forced_tag})"
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ModelRouter:
    """Routes tasks to the appropriate Claude model tier.

    Example::

        router = ModelRouter()
        decision = router.route("write a creative story about a fox")
        print(decision.model)  # claude-sonnet-4-6  (complex → Opus, but story → Sonnet by force)
    """

    def __init__(self) -> None:
        from src.config import get_config  # noqa: PLC0415

        cfg = get_config()
        self._haiku = cfg.model_haiku
        self._sonnet = cfg.model_sonnet
        self._opus = cfg.model_opus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, task: str, *, force_model: str | None = None) -> RoutingDecision:
        """Select a model for ``task``.

        Args:
            task:        Natural-language description of what the agent needs to do.
            force_model: If provided, skip scoring and return this model directly.
                         Useful for agents with a fixed tier requirement (e.g.
                         StoryWriter always on Sonnet, Reviewer always on Opus).

        Returns:
            A :class:`RoutingDecision` with the chosen model and diagnostic info.
        """
        if force_model is not None:
            complexity = self._model_to_complexity(force_model)
            return RoutingDecision(
                model=force_model,
                complexity=complexity,
                score=0,
                keyword_hits=[],
                forced=True,
            )

        task_lower = task.lower()
        score, hits = self._score(task_lower, task)
        model, complexity = self._score_to_model(score)

        return RoutingDecision(
            model=model,
            complexity=complexity,
            score=score,
            keyword_hits=hits,
            forced=False,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score(self, task_lower: str, task_original: str) -> tuple[int, list[str]]:
        """Compute routing score and return (score, keyword_hits).

        Positive score → complex (Opus direction).
        Negative score → trivial (Haiku direction).
        """
        score = 0
        hits: list[str] = []

        for kw in COMPLEX_KEYWORDS:
            if kw in task_lower:
                score += 1
                hits.append(kw)

        for kw in TRIVIAL_KEYWORDS:
            if kw in task_lower:
                score -= 1
                hits.append(f"-{kw}")

        # Adjust for task length
        word_count = len(task_original.split())
        if word_count <= _SHORT_TASK_WORDS:
            score -= 1
        elif word_count >= _LONG_TASK_WORDS:
            score += 1

        return score, hits

    def _score_to_model(self, score: int) -> tuple[str, str]:
        """Map numeric score to (model_id, complexity_label)."""
        if score >= _COMPLEX_THRESHOLD:
            return self._opus, "complex"
        if score <= -_TRIVIAL_THRESHOLD:
            return self._haiku, "trivial"
        return self._sonnet, "medium"

    def _model_to_complexity(self, model: str) -> str:
        """Return a human-readable complexity label for a model ID."""
        if model == self._haiku:
            return "trivial"
        if model == self._opus:
            return "complex"
        return "medium"
