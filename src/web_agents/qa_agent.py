"""WebQAAgent — tests all API endpoints of the running server.

No LLM required. Uses httpx to hit the live server and reports
pass/fail for every endpoint. Results go into state.qa_report.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from src.agents.base import AgentResult, BaseAgent
from src.web_agents.state import WebBuildState

log = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

# Each entry: (method, path, expected_status, body)
_TESTS: list[tuple[str, str, int, Any]] = [
    ("GET",  "/",                        200, None),
    ("GET",  "/api/stories",             200, None),
    ("GET",  "/api/status",              200, None),
    ("POST", "/api/generate",            422, None),           # missing body → 422
    ("POST", "/api/generate",            400, {"prompt": ""}), # empty prompt → 400
    ("GET",  "/api/jobs/bad-id/stream",  404, None),
]


class WebQAAgent(BaseAgent):
    """Smoke-tests the running FastAPI server."""

    name: str = "WebQAAgent"
    default_model: str = ""
    system_prompt: str = ""

    def mock_response(self, state: Any) -> dict:  # type: ignore[override]
        return {"passed": 6, "failed": 0, "tests": []}

    async def run(self, state: Any) -> AgentResult:  # type: ignore[override]
        start = time.monotonic()
        results: list[dict] = []
        passed = failed = 0

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            for method, path, expected, body in _TESTS:
                t0 = time.monotonic()
                try:
                    if method == "GET":
                        r = await client.get(path)
                    else:
                        r = await client.post(path, json=body)

                    ok      = r.status_code == expected
                    latency = round((time.monotonic() - t0) * 1000)
                    results.append({
                        "method": method, "path": path,
                        "expected": expected, "got": r.status_code,
                        "ok": ok, "latency_ms": latency,
                    })
                    if ok:
                        passed += 1
                    else:
                        failed += 1
                        log.warning("QA FAIL %s %s — expected %d got %d", method, path, expected, r.status_code)

                except Exception as exc:
                    failed += 1
                    results.append({
                        "method": method, "path": path,
                        "expected": expected, "got": None,
                        "ok": False, "error": str(exc),
                    })

        report = {"passed": passed, "failed": failed, "tests": results}
        state.qa_report = report

        summary = f"QA: {passed} passed, {failed} failed"
        state.add_log(summary)
        log.info(summary)

        return AgentResult(
            agent_name=self.name,
            model_used="httpx",
            output={"summary": summary, **report},
            elapsed_sec=time.monotonic() - start,
        )
