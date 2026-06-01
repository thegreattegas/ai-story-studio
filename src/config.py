"""Application configuration — API keys, model names, pricing, and workspace paths.

Loaded once from environment variables via `get_config()` singleton.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Pricing table  (USD per 1 million tokens)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    # Claude (kept for reference)
    "claude-haiku-4-5": {
        "input": 0.25,
        "output": 1.25,
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
    },
    "claude-opus-4-6": {
        "input": 15.00,
        "output": 75.00,
    },
}

# Fallback for unknown / future model IDs
_DEFAULT_PRICING = {"input": 2.50, "output": 10.00}


def cost_usd(tokens_in: int, tokens_out: int, model: str) -> float:
    """Calculate API call cost in USD.

    Args:
        tokens_in:  Number of input (prompt) tokens.
        tokens_out: Number of output (completion) tokens.
        model:      Model ID string — must match a key in PRICING.

    Returns:
        Cost in USD as a float.
    """
    rates = PRICING.get(model, _DEFAULT_PRICING)
    return (tokens_in * rates["input"] + tokens_out * rates["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------

# Resolve project root so relative paths work regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppConfig(BaseSettings):
    """Pydantic settings model — values populated from environment / .env file.

    All keys are optional at the pydantic level; validation errors only
    surface at runtime when a provider is actually used (not in mock mode).
    """

    # --- API Keys ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    luma_api_key: str = Field(default="", alias="LUMA_API_KEY")

    # --- Pipeline control ---
    mock_mode: bool = Field(default=True, alias="MOCK_MODE")
    # "veo3" | "luma" | "none"  (none = Ken Burns only)
    video_provider: str = Field(default="none", alias="VIDEO_PROVIDER")

    # --- Provider selection ---
    # "openai" | "anthropic"
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    # "openai" | "google"
    image_provider: str = Field(default="openai", alias="IMAGE_PROVIDER")

    # --- Model names — auto-resolved from llm_provider if not overridden ---
    # Leave as empty string to use provider defaults below.
    model_haiku: str = ""
    model_sonnet: str = ""
    model_opus: str = ""

    @property
    def effective_model_haiku(self) -> str:
        if self.model_haiku:
            return self.model_haiku
        return "gpt-4o" if self.llm_provider == "openai" else "claude-haiku-4-5"

    @property
    def effective_model_sonnet(self) -> str:
        if self.model_sonnet:
            return self.model_sonnet
        return "gpt-4o" if self.llm_provider == "openai" else "claude-sonnet-4-6"

    @property
    def effective_model_opus(self) -> str:
        if self.model_opus:
            return self.model_opus
        return "gpt-4o" if self.llm_provider == "openai" else "claude-opus-4-6"

    # --- Paths ---
    workspace_dir: Path = _PROJECT_ROOT / "workspace"
    cache_file: Path = _PROJECT_ROOT / "cache.json"

    # --- Cost ceiling ---
    cost_ceiling_usd: float = 1.50

    model_config = {
        "populate_by_name": True,
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("workspace_dir", mode="after")
    @classmethod
    def _ensure_workspace(cls, v: Path) -> Path:
        """Create workspace directory if it does not exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def cost_usd(self, tokens_in: int, tokens_out: int, model: str) -> float:
        """Convenience instance method — delegates to module-level helper."""
        return cost_usd(tokens_in, tokens_out, model)

    def __str__(self) -> str:  # noqa: D105
        return (
            f"AppConfig("
            f"mock_mode={self.mock_mode}, "
            f"anthropic_key={'***' if self.anthropic_api_key else 'NOT SET'}, "
            f"workspace={self.workspace_dir}, "
            f"cache={self.cache_file}"
            f")"
        )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the singleton AppConfig instance.

    Loads `.env` on first call; subsequent calls return the cached instance.
    """
    # Ensure .env is loaded before pydantic reads env vars.
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    return AppConfig()
