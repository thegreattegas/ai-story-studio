"""WebFrontendAgent — improves app.js and/or index.html on demand.

Reads the current HTML + JS, applies the user's instruction via Claude,
writes back the improved app.js (HTML changes are reported as suggestions).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.web_agents.designer_agent import _backup, _strip_code_fence
from src.web_agents.state import MANAGED_FILES, WebBuildState

log = logging.getLogger(__name__)

_JS_PATH   = MANAGED_FILES["frontend_js"]
_HTML_PATH = MANAGED_FILES["frontend_html"]

_SYSTEM = """\
You are a senior JavaScript engineer specialising in vanilla JS web apps.

You receive the current app.js (and HTML for context) plus a specific \
improvement instruction.

Rules:
- Return ONLY the complete improved app.js. No prose, no markdown fences.
- Preserve ALL existing functions — add new code at the bottom.
- Do NOT use any frameworks (no React, Vue, jQuery).
- Replace any alert() / confirm() calls with a toast or modal pattern.
- All new features must degrade gracefully if their DOM element is missing.
"""


class WebFrontendAgent(BaseAgent):
    """Applies targeted JavaScript/HTML improvements to web/app.js."""

    name: str = "WebFrontendAgent"
    system_prompt: str = _SYSTEM

    def __init__(self) -> None:
        super().__init__()
        self.default_model = self.config.effective_model_sonnet

    def mock_response(self, state: Any) -> dict:  # type: ignore[override]
        return {"summary": "MOCK: JS not modified"}

    async def run(self, state: Any) -> AgentResult:  # type: ignore[override]
        start = time.monotonic()

        current_js   = _JS_PATH.read_text(encoding="utf-8")
        current_html = _HTML_PATH.read_text(encoding="utf-8")
        chars_before = len(current_js)

        instruction = state.instruction or "Add story search/filter, story deletion with confirmation, and toast notifications."

        user_prompt = (
            f"INSTRUCTION: {instruction}\n\n"
            f"--- CURRENT app.js ---\n{current_js}\n\n"
            f"--- index.html (context only, do not rewrite) ---\n{current_html}"
        )

        new_js, tin, tout = await self._call_llm(user_prompt, max_tokens=8000)
        new_js = _strip_code_fence(new_js)

        # Sanity check
        if "function" not in new_js and "=>" not in new_js:
            raise ValueError(
                "WebFrontendAgent: response does not look like valid JavaScript. "
                f"Got {len(new_js)} chars starting with: {new_js[:120]!r}"
            )

        backup = _backup(_JS_PATH)
        _JS_PATH.write_text(new_js, encoding="utf-8")
        state.record_file(_JS_PATH)

        cost = self.config.cost_usd(tin, tout, self.default_model)
        state.add_cost(cost)
        elapsed = time.monotonic() - start

        summary = f"app.js updated ({chars_before} -> {len(new_js)} chars)"
        state.add_log(summary)
        log.info("WebFrontendAgent: %s  backup=%s", summary, backup.name)

        return AgentResult(
            agent_name=self.name,
            model_used=self.default_model,
            output={"summary": summary, "backup": str(backup), "chars_before": chars_before, "chars_after": len(new_js)},
            tokens_in=tin, tokens_out=tout, cost_usd=cost, elapsed_sec=elapsed,
        )
