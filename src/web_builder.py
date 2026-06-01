"""Web builder orchestrator.

Runs web agents to improve the Story Studio website on demand.
Each agent is independent — only the agent(s) matching the target run.

CLI usage
---------
    # Run QA only (no LLM, just tests the server)
    python -m src.web_builder qa

    # Improve the CSS/design
    python -m src.web_builder design "add toast notification styles"

    # Improve the JavaScript frontend
    python -m src.web_builder frontend "add story search and deletion"

    # Improve the FastAPI backend (requires server restart after)
    python -m src.web_builder backend "add DELETE /api/stories/{id} endpoint"

    # Full quality review (reads all files, no changes)
    python -m src.web_builder review

    # Run everything in sequence: QA → design → frontend → backend → review
    python -m src.web_builder all "improve overall quality"

API usage (called from src/server.py)
--------------------------------------
    from src.web_builder import run_web_builder
    await run_web_builder(target="frontend", instruction="add search", queue=asyncio.Queue())
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Callable

from src.web_agents.state import WebBuildState
from src.web_agents.qa_agent import WebQAAgent
from src.web_agents.designer_agent import WebDesignerAgent
from src.web_agents.frontend_agent import WebFrontendAgent
from src.web_agents.backend_agent import WebBackendAgent
from src.web_agents.reviewer_agent import WebReviewerAgent

log = logging.getLogger(__name__)

# Map target name → (agent class, step label)
_AGENT_MAP = {
    "qa":       (WebQAAgent,       "Running API health checks…"),
    "design":   (WebDesignerAgent, "Improving style.css…"),
    "frontend": (WebFrontendAgent, "Improving app.js…"),
    "backend":  (WebBackendAgent,  "Improving server.py…"),
    "review":   (WebReviewerAgent, "Reviewing all web files…"),
}

# Ordered sequence for "all"
_ALL_TARGETS = ["qa", "design", "frontend", "backend", "review"]


async def run_web_builder(
    target: str,
    instruction: str = "",
    progress_queue: asyncio.Queue | None = None,
) -> WebBuildState:
    """Run one or all web builder agents.

    Args:
        target:      One of "qa", "design", "frontend", "backend", "review", "all".
        instruction: Free-text instruction passed to LLM agents.
        progress_queue: Optional asyncio.Queue for SSE progress events.

    Returns:
        The populated WebBuildState after all agents have run.
    """
    def emit(event: dict) -> None:
        if progress_queue is not None:
            progress_queue.put_nowait(event)

    state = WebBuildState(target=target, instruction=instruction)

    targets = _ALL_TARGETS if target == "all" else [target]

    total_start = time.monotonic()

    for t in targets:
        if t not in _AGENT_MAP:
            log.warning("web_builder: unknown target %r — skipping", t)
            continue

        agent_cls, label = _AGENT_MAP[t]
        agent = agent_cls()

        emit({"type": "web_step", "step": t, "status": "running", "label": label})
        try:
            result = await agent.run(state)
            summary = result.output.get("summary", label)
            emit({"type": "web_step", "step": t, "status": "done", "label": summary})
            log.info("web_builder: [%s] done — %s  cost=$%.4f", t, summary, result.cost_usd)
        except Exception as exc:
            log.exception("web_builder: [%s] failed — %s", t, exc)
            emit({"type": "web_step", "step": t, "status": "error", "label": str(exc)})
            # Continue with remaining agents even if one fails

    elapsed = time.monotonic() - total_start
    emit({
        "type": "web_complete",
        "files_modified": state.files_modified,
        "total_cost": round(state.total_cost, 6),
        "elapsed_sec": round(elapsed, 2),
        "review_notes": state.review_notes,
        "qa_report": state.qa_report,
    })

    log.info(
        "web_builder: all done  target=%s  files=%s  cost=$%.4f  elapsed=%.2fs",
        target, state.files_modified, state.total_cost, elapsed,
    )
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _usage() -> None:
    print(
        "Usage:\n"
        "  python -m src.web_builder qa\n"
        "  python -m src.web_builder design   \"<instruction>\"\n"
        "  python -m src.web_builder frontend \"<instruction>\"\n"
        "  python -m src.web_builder backend  \"<instruction>\"\n"
        "  python -m src.web_builder review\n"
        "  python -m src.web_builder all      [\"<instruction>\"]\n"
    )


def _cli_main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    args = sys.argv[1:]
    if not args:
        _usage()
        sys.exit(1)

    target      = args[0].lower()
    instruction = " ".join(args[1:]) if len(args) > 1 else ""

    valid = set(_AGENT_MAP) | {"all"}
    if target not in valid:
        print(f"Unknown target: {target!r}\nValid: {sorted(valid)}")
        _usage()
        sys.exit(1)

    print(f"\nWeb Builder — target={target!r}  instruction={instruction!r}\n")

    state = asyncio.run(run_web_builder(target=target, instruction=instruction))

    print("\n--- Results ---")
    print(f"Files modified : {state.files_modified or 'none'}")
    print(f"Total cost     : ${state.total_cost:.4f}")

    if state.qa_report:
        r = state.qa_report
        print(f"QA             : {r.get('passed', 0)} passed, {r.get('failed', 0)} failed")

    if state.review_notes:
        print("\n--- Review Notes ---")
        print(state.review_notes[:1200])
        if len(state.review_notes) > 1200:
            print("... (truncated)")

    print("\nDone.")


if __name__ == "__main__":
    _cli_main()
