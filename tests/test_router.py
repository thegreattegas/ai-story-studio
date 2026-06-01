"""Tests for ModelRouter — 6 cases covering the full routing decision space."""

from __future__ import annotations

import pytest

from src.router import ModelRouter, RoutingDecision


@pytest.fixture
def router() -> ModelRouter:
    """Provide a ModelRouter instance for each test."""
    return ModelRouter()


# ---------------------------------------------------------------------------
# Test 1 — Trivial keyword routes to Haiku
# ---------------------------------------------------------------------------


def test_trivial_keyword_routes_to_haiku(router: ModelRouter) -> None:
    """A task containing a trivial keyword (e.g. 'rename') should route to Haiku."""
    decision = router.route("rename the file to snake_case")
    assert decision.complexity == "trivial", f"Expected trivial, got: {decision}"
    assert "haiku" in decision.model.lower(), f"Expected Haiku model, got: {decision.model}"
    assert decision.forced is False


# ---------------------------------------------------------------------------
# Test 2 — Complex keyword routes to Opus
# ---------------------------------------------------------------------------


def test_complex_keyword_routes_to_opus(router: ModelRouter) -> None:
    """A task with multiple complex keywords should route to Opus."""
    decision = router.route(
        "design the architecture for a distributed narrative generation system "
        "and review all the creative components for quality and consistency"
    )
    assert decision.complexity == "complex", f"Expected complex, got: {decision}"
    assert "opus" in decision.model.lower(), f"Expected Opus model, got: {decision.model}"
    assert decision.forced is False


# ---------------------------------------------------------------------------
# Test 3 — Long neutral task routes to Sonnet
# ---------------------------------------------------------------------------


def test_long_neutral_task_routes_to_sonnet(router: ModelRouter) -> None:
    """A long task with no strong keywords should default to Sonnet (medium)."""
    # 30+ words, no trivial or complex keywords
    decision = router.route(
        "process the input data and transform each record into the output format "
        "according to the mapping table provided in the configuration file and "
        "write the results to the destination location"
    )
    assert decision.complexity == "medium", f"Expected medium, got: {decision}"
    assert "sonnet" in decision.model.lower(), f"Expected Sonnet model, got: {decision.model}"


# ---------------------------------------------------------------------------
# Test 4 — Mixed keywords — complex wins over trivial
# ---------------------------------------------------------------------------


def test_mixed_keywords_complex_wins(router: ModelRouter) -> None:
    """When both trivial and complex keywords appear, complex keywords should dominate."""
    decision = router.route(
        "read the existing story and review its narrative architecture for quality"
    )
    # 'read' is trivial (-1), but 'story', 'review', 'narrative', 'architecture' are complex (+4)
    assert decision.complexity in ("complex", "medium"), (
        f"Expected complex or medium (complex keywords dominate), got: {decision}"
    )
    # Score should be positive — NOT haiku
    assert "haiku" not in decision.model.lower(), (
        f"Should NOT route to Haiku when complex keywords dominate, got: {decision.model}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Short neutral task routes to Haiku
# ---------------------------------------------------------------------------


def test_short_neutral_task_routes_to_haiku(router: ModelRouter) -> None:
    """A very short task with no keywords should lean toward Haiku."""
    decision = router.route("do it")
    # Short task gets -1 for length; no complex keywords → score ≤ -2 → Haiku
    assert decision.complexity == "trivial", f"Expected trivial for very short task, got: {decision}"
    assert "haiku" in decision.model.lower(), f"Expected Haiku, got: {decision.model}"


# ---------------------------------------------------------------------------
# Test 6 — force_model override works
# ---------------------------------------------------------------------------


def test_force_model_override(router: ModelRouter) -> None:
    """force_model should bypass scoring entirely and mark the decision as forced."""
    from src.config import get_config

    sonnet_id = get_config().model_sonnet
    # Even a trivial task should route to Sonnet when forced.
    decision = router.route("rename the file", force_model=sonnet_id)

    assert decision.model == sonnet_id, f"Expected forced model {sonnet_id}, got: {decision.model}"
    assert decision.complexity == "medium", f"Expected medium complexity for Sonnet, got: {decision}"
    assert decision.forced is True, "Expected forced=True when force_model is used"
    assert decision.keyword_hits == [], "Forced decisions should have no keyword hits"
