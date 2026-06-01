"""WebReviewerAgent — quality gate for web file changes.

Reads all managed web files, asks Claude Opus to review them for
issues, and writes findings to state.review_notes.
Does NOT modify any files.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.web_agents.state import MANAGED_FILES, WebBuildState

log = logging.getLogger(__name__)

_SYSTEM = """\
You are a senior code reviewer specialising in web applications.

You receive the HTML, CSS, JS, and Python server files for a web app.
Produce a concise structured review covering:

## Critical Issues   (anything broken, insecure, or inaccessible)
## UX Improvements   (specific, actionable — cite file and line)
## Accessibility     (ARIA, keyboard nav, colour contrast)
## Performance       (assets, SSE handling, canvas)
## Quick Wins        (5 specific changes, each under 5 lines of code)

Be specific. No generic advice. Cite file names and approximate line numbers.
"""


class WebReviewerAgent(BaseAgent):
    """Reviews all web files and produces a quality report."""

    name: str = "WebReviewerAgent"
    system_prompt: str = _SYSTEM

    def __init__(self) -> None:
        super().__init__()
        self.default_model = self.config.effective_model_opus

    def mock_response(self, state: Any) -> dict:  # type: ignore[override]
        return {"summary": "MOCK: no issues found"}

    async def run(self, state: Any) -> AgentResult:  # type: ignore[override]
        start = time.monotonic()

        parts: list[str] = []
        for key, path in MANAGED_FILES.items():
            if path.exists():
                content = path.read_text(encoding="utf-8")
                parts.append(f"--- FILE: {path.name} ---\n{content}")

        user_prompt = "\n\n".join(parts)

        review, tin, tout = await self._call_llm(user_prompt, max_tokens=3000)
        state.review_notes = review

        cost = self.config.cost_usd(tin, tout, self.default_model)
        state.add_cost(cost)
        elapsed = time.monotonic() - start

        summary = "Review complete — see review_notes for details"
        state.add_log(summary)
        log.info("WebReviewerAgent done  cost=$%.4f", cost)

        return AgentResult(
            agent_name=self.name,
            model_used=self.default_model,
            output={"summary": summary, "review_notes": review},
            tokens_in=tin, tokens_out=tout, cost_usd=cost, elapsed_sec=elapsed,
        )
