"""State model for the web builder pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Absolute paths to the files these agents manage
WEB_DIR     = _PROJECT_ROOT / "web"
BACKEND_DIR = _PROJECT_ROOT / "src"

MANAGED_FILES = {
    "frontend_html": WEB_DIR / "index.html",
    "frontend_css":  WEB_DIR / "style.css",
    "frontend_js":   WEB_DIR / "app.js",
    "backend":       BACKEND_DIR / "server.py",
}


class WebBuildState(BaseModel):
    """Shared state passed between web builder agents."""

    # What the user wants done
    target: str = ""             # "frontend" | "backend" | "design" | "qa" | "review"
    instruction: str = ""        # Free-text improvement instruction

    # Agent outputs (populated as pipeline runs)
    qa_report: dict[str, Any] = Field(default_factory=dict)
    review_notes: str = ""
    files_modified: list[str] = Field(default_factory=list)
    suggestions: dict[str, str] = Field(default_factory=dict)  # agent_name → suggestion text

    # Cost / logs
    total_cost: float = 0.0
    logs: list[str] = Field(default_factory=list)

    def add_cost(self, cost: float) -> None:
        self.total_cost += cost

    def add_log(self, msg: str) -> None:
        self.logs.append(msg)

    def record_file(self, path: Path) -> None:
        self.files_modified.append(path.name)
