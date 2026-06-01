"""Base agent abstract class and AgentResult model.

All pipeline agents inherit from :class:`BaseAgent` and implement two methods:

* :meth:`BaseAgent.mock_response` — returns a canned ``dict`` for mock mode.
* :meth:`BaseAgent.run` — performs real work and returns an :class:`AgentResult`.

Logging
-------
Each agent logs to:

1. **Console** — via a ``rich``-formatted handler on the root logger.
2. **In-memory list** — via :class:`ListHandler`, so the Streamlit UI can
   stream log lines without reading files.

Access in-memory logs via :data:`log_handler.records`.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from src.config import get_config
from src.router import ModelRouter
from src.state import StoryState

# ---------------------------------------------------------------------------
# In-memory log handler (for Streamlit streaming — Phase 6)
# ---------------------------------------------------------------------------


class ListHandler(logging.Handler):
    """A logging handler that appends formatted records to an in-memory list."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        self.records.append(self.format(record))


# Module-level handler — shared across all agents in a process.
log_handler = ListHandler()
log_handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))

# Configure root logger once (idempotent).
_root_logger = logging.getLogger()
if not any(isinstance(h, ListHandler) for h in _root_logger.handlers):
    _root_logger.addHandler(log_handler)

# Plain stream handler for debug output.
# We intentionally avoid RichHandler here: Rich's legacy Windows console
# renderer cannot encode non-ASCII characters (arrows, em-dashes, etc.)
# that may appear in log messages or content strings.  The user-facing
# Rich output is produced by main.py's Console object, which is separate.
if not any(isinstance(h, logging.StreamHandler) for h in _root_logger.handlers):
    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(
        logging.Formatter("%(levelname)s [%(name)s] %(message)s")
    )
    _root_logger.addHandler(_stream_handler)

_root_logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------


class AgentResult(BaseModel):
    """Structured result returned by every agent.

    Attributes:
        agent_name:  Human-readable agent identifier.
        model_used:  Model ID string that produced the response.
        output:      Agent-specific payload dict (varies by agent type).
        tokens_in:   Input (prompt) tokens consumed.
        tokens_out:  Output (completion) tokens generated.
        cost_usd:    Estimated API call cost in USD.
        elapsed_sec: Wall-clock time for the agent's ``run()`` call in seconds.
        mocked:      True when the response came from mock mode (no API call).
        cached:      True when the response was served from the disk cache.
    """

    agent_name: str
    model_used: str
    output: dict[str, Any]
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    elapsed_sec: float = 0.0
    mocked: bool = False
    cached: bool = False

    def summary_line(self) -> str:
        """Return a single-line log summary suitable for console output."""
        tags = []
        if self.mocked:
            tags.append("MOCK")
        if self.cached:
            tags.append("CACHED")
        tag_str = f" [{','.join(tags)}]" if tags else ""
        return (
            f"[{self.agent_name}] [{self.model_used}] "
            f"tokens_in={self.tokens_in} tokens_out={self.tokens_out} "
            f"cost=${self.cost_usd:.6f} elapsed={self.elapsed_sec:.2f}s"
            f"{tag_str}"
        )


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base class for all AI Story Studio pipeline agents.

    Subclasses must set class-level ``name``, ``default_model``, and
    ``system_prompt``, then implement :meth:`mock_response` and :meth:`run`.
    """

    name: str = "BaseAgent"
    default_model: str = ""
    system_prompt: str = ""

    def __init__(self) -> None:
        self.config = get_config()
        self.router = ModelRouter()
        self.logger = logging.getLogger(self.name)

        self._client: Any | None = None        # OpenAI async client
        self._anthropic: Any | None = None     # Anthropic async client

        if self.config.mock_mode:
            return

        if self.config.llm_provider == "anthropic":
            if self.config.anthropic_api_key:
                try:
                    import anthropic  # noqa: PLC0415
                    self._anthropic = anthropic.AsyncAnthropic(
                        api_key=self.config.anthropic_api_key
                    )
                except ImportError:
                    self.logger.warning(
                        "anthropic package not installed — falling back to mock mode."
                    )
        else:  # openai (default)
            if self.config.openai_api_key:
                try:
                    from openai import AsyncOpenAI  # noqa: PLC0415
                    self._client = AsyncOpenAI(api_key=self.config.openai_api_key)
                except ImportError:
                    self.logger.warning(
                        "openai package not installed — agent will run in mock mode."
                    )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def mock_response(self, state: StoryState) -> dict[str, Any]:
        """Return a canned response dict for mock mode.

        Must return the same *shape* of dict that the real ``run()`` produces
        so downstream agents behave identically in mock vs. live mode.
        """
        ...

    @abstractmethod
    async def run(self, state: StoryState) -> AgentResult:
        """Execute the agent, mutate ``state`` in place, return metrics.

        Args:
            state: Current pipeline state — mutate fields owned by this agent.

        Returns:
            :class:`AgentResult` with output payload and usage metrics.
        """
        ...

    # ------------------------------------------------------------------
    # LLM call helper — Phase 2: proper token tracking + structured cache
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 2000,
    ) -> tuple[str, int, int]:
        """Call the Anthropic API with caching and mock-mode support.

        Cache format (Phase 2+): each entry is a dict
        ``{"text": str, "tokens_in": int, "tokens_out": int}``.
        Legacy string entries (Phase 1) are handled transparently.

        Args:
            user_prompt: The user-turn message to send.
            model:       Model ID override; defaults to ``self.default_model``.
            max_tokens:  Maximum output tokens allowed.

        Returns:
            ``(response_text, tokens_in, tokens_out)`` — token counts are
            zero in mock mode and on cache hits (no billing event occurred).
        """
        resolved_model = model or self.default_model or self.config.effective_model_sonnet

        # --- Mock mode ---
        no_client = (self._client is None and self._anthropic is None)
        if self.config.mock_mode or no_client:
            self.logger.debug("_call_llm: mock mode — skipping API call.")
            return "MOCK_RESPONSE", 0, 0

        # --- Cache lookup ---
        from src.cache import _cache_key, _load_cache, _save_cache  # noqa: PLC0415

        key = _cache_key(user_prompt, resolved_model)
        store = _load_cache(self.config.cache_file)

        if key in store:
            entry = store[key]
            if isinstance(entry, dict) and "text" in entry:
                self.logger.debug("_call_llm: cache HIT key=%s...", key[:12])
                return (
                    entry["text"],
                    entry.get("tokens_in", 0),
                    entry.get("tokens_out", 0),
                )
            elif isinstance(entry, str):
                return entry, 0, 0

        # --- Live API call ---
        self.logger.debug(
            "_call_llm: cache MISS — provider=%s model=%s",
            self.config.llm_provider, resolved_model,
        )
        t0 = time.monotonic()

        if self.config.llm_provider == "anthropic" and self._anthropic is not None:
            text, tokens_in, tokens_out = await self._call_anthropic(
                user_prompt, resolved_model, max_tokens
            )
        else:
            text, tokens_in, tokens_out = await self._call_openai(
                user_prompt, resolved_model, max_tokens
            )

        elapsed = time.monotonic() - t0
        self.logger.debug(
            "_call_llm: done model=%s tokens_in=%d tokens_out=%d elapsed=%.2fs",
            resolved_model, tokens_in, tokens_out, elapsed,
        )

        store[key] = {"text": text, "tokens_in": tokens_in, "tokens_out": tokens_out}
        _save_cache(self.config.cache_file, store)

        return text, tokens_in, tokens_out

    async def _call_openai(
        self, user_prompt: str, model: str, max_tokens: int
    ) -> tuple[str, int, int]:
        """Call OpenAI chat completions API."""
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = await self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        text = response.choices[0].message.content or ""
        return text, response.usage.prompt_tokens, response.usage.completion_tokens

    async def _call_anthropic(
        self, user_prompt: str, model: str, max_tokens: int
    ) -> tuple[str, int, int]:
        """Call Anthropic messages API."""
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        response = await self._anthropic.messages.create(**kwargs)
        text = response.content[0].text if response.content else ""
        return text, response.usage.input_tokens, response.usage.output_tokens

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _mock_result(self, state: StoryState, elapsed: float = 0.01) -> AgentResult:
        """Build an AgentResult from mock_response() — for simple Phase 1 stubs only.

        Note: Phase 2+ agents should NOT use this helper because they need to
        mutate ``state`` fields *before* building the result. Use the inline
        pattern in each agent's ``run()`` instead.
        """
        payload = self.mock_response(state)
        result = AgentResult(
            agent_name=self.name,
            model_used=self.default_model or self.config.effective_model_sonnet,
            output=payload,
            mocked=True,
            elapsed_sec=elapsed,
        )
        self.logger.info(result.summary_line())
        return result
