"""File-based SHA-256 keyed response cache.

Usage::

    result = await cached_call(prompt="tell me a story", model="claude-sonnet-4-6", fn=my_api_fn)

The cache is stored as a flat JSON file (`cache.json` at the project root).
Keys are ``SHA-256(prompt + "|" + model)``.  On a cache hit the coroutine
``fn`` is never called, saving API cost.

Thread / async safety: the cache file is read and written with a simple
read-modify-write pattern.  For a single-process pipeline this is fine.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


def _cache_key(prompt: str, model: str) -> str:
    """Return the SHA-256 hex digest for a (prompt, model) pair."""
    raw = f"{prompt}|{model}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache(cache_file: Path) -> dict[str, Any]:
    """Load the cache JSON file, returning an empty dict if missing or corrupt."""
    if not cache_file.exists():
        return {}
    try:
        with cache_file.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cache file unreadable (%s) — starting fresh.", exc)
        return {}


def _save_cache(cache_file: Path, data: dict[str, Any]) -> None:
    """Persist the cache dict to disk."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("Failed to save cache: %s", exc)


async def cached_call(
    prompt: str,
    model: str,
    fn: Callable[[], Coroutine[Any, Any, str]],
    *,
    cache_file: Path | None = None,
) -> tuple[str, bool]:
    """Check the cache for a (prompt, model) pair; call ``fn`` on a miss.

    Args:
        prompt:     The full prompt string sent to the model.
        model:      The model identifier (used as part of the cache key).
        fn:         Async callable that performs the real API call and returns
                    the response text.
        cache_file: Override the cache file path (defaults to config value).

    Returns:
        A ``(response_text, was_cached)`` tuple.
        ``was_cached`` is ``True`` when the result came from the cache.
    """
    if cache_file is None:
        from src.config import get_config  # noqa: PLC0415

        cache_file = get_config().cache_file

    key = _cache_key(prompt, model)
    store = _load_cache(cache_file)

    if key in store:
        logger.debug("Cache HIT  key=%s...", key[:12])
        return store[key], True

    logger.debug("Cache MISS key=%s...", key[:12])
    response = await fn()

    store[key] = response
    _save_cache(cache_file, store)
    logger.debug("Cache SET  key=%s...", key[:12])

    return response, False


def clear_cache(cache_file: Path | None = None) -> int:
    """Delete all cached entries and return the number cleared.

    Useful for testing or forcing a fresh pipeline run.
    """
    if cache_file is None:
        from src.config import get_config  # noqa: PLC0415

        cache_file = get_config().cache_file

    store = _load_cache(cache_file)
    count = len(store)
    _save_cache(cache_file, {})
    logger.info("Cache cleared (%d entries removed).", count)
    return count
