"""WebDesignerAgent — improves style.css on demand.

Reads the current CSS, sends it to Claude with the user's specific
instruction, writes the improved CSS back (with a timestamped backup).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.web_agents.state import MANAGED_FILES, WebBuildState

log = logging.getLogger(__name__)

_CSS_PATH = MANAGED_FILES["frontend_css"]

_SYSTEM = """\
You are a senior CSS engineer specialising in premium dark-theme streaming UIs \
(Netflix, Disney+, Apple TV+).

You receive the current style.css and a specific improvement instruction.
Apply ONLY what the instruction asks for — preserve everything else unchanged.

Rules:
- Return ONLY the complete improved CSS. No prose, no markdown fences, no explanations.
- Keep all existing CSS variables, class names, and media queries.
- New rules go at the END of the file, after all existing content.
- Valid CSS only — no SCSS, no nesting (unless the file already uses it).
"""


class WebDesignerAgent(BaseAgent):
    """Applies targeted CSS/visual improvements to web/style.css."""

    name: str = "WebDesignerAgent"
    system_prompt: str = _SYSTEM

    def __init__(self) -> None:
        super().__init__()
        self.default_model = self.config.effective_model_sonnet

    def mock_response(self, state: Any) -> dict:  # type: ignore[override]
        return {"summary": "MOCK: CSS not modified"}

    async def run(self, state: Any) -> AgentResult:  # type: ignore[override]
        start = time.monotonic()

        current_css = _CSS_PATH.read_text(encoding="utf-8")
        chars_before = len(current_css)

        instruction = state.instruction or "Polish animations, improve hover effects, and add focus-visible outlines for accessibility."

        user_prompt = (
            f"INSTRUCTION: {instruction}\n\n"
            f"CURRENT style.css ({chars_before} chars):\n\n"
            f"{current_css}"
        )

        new_css, tin, tout = await self._call_llm(user_prompt, max_tokens=8000)
        new_css = _strip_code_fence(new_css)

        # Sanity check: result must look like CSS
        if "{" not in new_css or len(new_css) < chars_before * 0.5:
            raise ValueError(
                "WebDesignerAgent: response does not look like valid CSS. "
                f"Got {len(new_css)} chars starting with: {new_css[:120]!r}"
            )

        # Backup then write
        backup = _backup(_CSS_PATH)
        _CSS_PATH.write_text(new_css, encoding="utf-8")
        state.record_file(_CSS_PATH)

        cost = self.config.cost_usd(tin, tout, self.default_model)
        state.add_cost(cost)
        elapsed = time.monotonic() - start

        summary = f"style.css updated ({chars_before} -> {len(new_css)} chars)"
        state.add_log(summary)
        log.info("WebDesignerAgent: %s  backup=%s", summary, backup.name)

        return AgentResult(
            agent_name=self.name,
            model_used=self.default_model,
            output={"summary": summary, "backup": str(backup), "chars_before": chars_before, "chars_after": len(new_css)},
            tokens_in=tin, tokens_out=tout, cost_usd=cost, elapsed_sec=elapsed,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(f".bak.{ts}")
    import shutil
    shutil.copy2(path, backup)
    return backup


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s[s.index("\n") + 1:]
    if s.endswith("```"):
        s = s[: s.rfind("```")]
    return s.strip()
