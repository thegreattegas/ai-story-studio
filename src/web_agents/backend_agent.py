"""WebBackendAgent — improves src/server.py on demand.

Reads the current server.py, applies the user's instruction via Claude,
writes the improved version back with a timestamped backup.

Note: Changes require a server restart to take effect.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.web_agents.designer_agent import _backup, _strip_code_fence
from src.web_agents.state import MANAGED_FILES, WebBuildState

log = logging.getLogger(__name__)

_SERVER_PATH = MANAGED_FILES["backend"]

_SYSTEM = """\
You are a senior Python/FastAPI engineer.

You receive the current server.py and a specific improvement instruction.
Apply ONLY what the instruction asks for — preserve all existing routes,
middleware, and helper functions unchanged.

Rules:
- Return ONLY the complete improved server.py. No prose, no markdown fences.
- Preserve the module docstring and all existing imports.
- New routes go BEFORE the `start()` function at the bottom.
- Use the same code style and patterns as the existing file.
- All new endpoints must have proper error handling (HTTPException).
"""


class WebBackendAgent(BaseAgent):
    """Applies targeted API improvements to src/server.py."""

    name: str = "WebBackendAgent"
    system_prompt: str = _SYSTEM

    def __init__(self) -> None:
        super().__init__()
        self.default_model = self.config.effective_model_sonnet

    def mock_response(self, state: Any) -> dict:  # type: ignore[override]
        return {"summary": "MOCK: server.py not modified"}

    async def run(self, state: Any) -> AgentResult:  # type: ignore[override]
        start = time.monotonic()

        current_py = _SERVER_PATH.read_text(encoding="utf-8")
        chars_before = len(current_py)

        instruction = state.instruction or (
            "Add DELETE /api/stories/{story_id} and GET /api/stories/{story_id} endpoints."
        )

        user_prompt = (
            f"INSTRUCTION: {instruction}\n\n"
            f"CURRENT server.py ({chars_before} chars):\n\n"
            f"{current_py}"
        )

        new_py, tin, tout = await self._call_llm(user_prompt, max_tokens=8000)
        new_py = _strip_code_fence(new_py)

        # Sanity check
        if "from fastapi" not in new_py or "app = " not in new_py:
            raise ValueError(
                "WebBackendAgent: response does not look like a FastAPI server. "
                f"Got {len(new_py)} chars starting with: {new_py[:120]!r}"
            )

        backup = _backup(_SERVER_PATH)
        _SERVER_PATH.write_text(new_py, encoding="utf-8")
        state.record_file(_SERVER_PATH)

        cost = self.config.cost_usd(tin, tout, self.default_model)
        state.add_cost(cost)
        elapsed = time.monotonic() - start

        summary = f"server.py updated ({chars_before} -> {len(new_py)} chars) — restart required"
        state.add_log(summary)
        log.info("WebBackendAgent: %s  backup=%s", summary, backup.name)

        return AgentResult(
            agent_name=self.name,
            model_used=self.default_model,
            output={"summary": summary, "backup": str(backup), "restart_required": True},
            tokens_in=tin, tokens_out=tout, cost_usd=cost, elapsed_sec=elapsed,
        )
