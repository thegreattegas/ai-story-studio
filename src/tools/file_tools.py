"""Workspace-scoped file I/O tools for agents.

All paths are sandboxed to the ``workspace/`` directory defined in
:class:`~src.config.AppConfig`.  Attempts to escape the workspace via
``../`` path traversal raise a :class:`ValueError`.

Usage::

    from src.tools.file_tools import write_file, read_file, list_files

    write_file("images/scene_01.txt", "image prompt text")
    content = read_file("images/scene_01.txt")
    files = list_files("images/*.txt")
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _workspace() -> Path:
    """Return the absolute workspace directory path from config."""
    from src.config import get_config  # noqa: PLC0415

    return get_config().workspace_dir


def _safe_path(relative_path: str) -> Path:
    """Resolve ``relative_path`` inside workspace and guard against traversal.

    Args:
        relative_path: A path string relative to the workspace directory.

    Returns:
        Absolute :class:`~pathlib.Path` inside the workspace.

    Raises:
        ValueError: If the resolved path escapes the workspace directory.
    """
    workspace = _workspace().resolve()
    target = (workspace / relative_path).resolve()

    # Guard: the resolved target must be *inside* the workspace.
    try:
        target.relative_to(workspace)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: '{relative_path}' resolves outside workspace."
        ) from None

    return target


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_file(relative_path: str, content: str, encoding: str = "utf-8") -> Path:
    """Write a text file into the workspace.

    Args:
        relative_path: Destination path relative to workspace root.
        content:       Text content to write.
        encoding:      File encoding (default UTF-8).

    Returns:
        Absolute path to the written file.
    """
    target = _safe_path(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding=encoding)
    logger.debug("write_file -> %s (%d chars)", target, len(content))
    return target


def write_binary(relative_path: str, data: bytes) -> Path:
    """Write binary data (image, audio, video) into the workspace.

    Args:
        relative_path: Destination path relative to workspace root.
        data:          Raw bytes to write.

    Returns:
        Absolute path to the written file.
    """
    target = _safe_path(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    logger.debug("write_binary -> %s (%d bytes)", target, len(data))
    return target


def read_file(relative_path: str, encoding: str = "utf-8") -> str:
    """Read a text file from the workspace.

    Args:
        relative_path: Source path relative to workspace root.
        encoding:      File encoding (default UTF-8).

    Returns:
        File content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    target = _safe_path(relative_path)
    if not target.exists():
        raise FileNotFoundError(f"Workspace file not found: {relative_path!r}")
    content = target.read_text(encoding=encoding)
    logger.debug("read_file <- %s (%d chars)", target, len(content))
    return content


def read_binary(relative_path: str) -> bytes:
    """Read binary data from a workspace file.

    Args:
        relative_path: Source path relative to workspace root.

    Returns:
        Raw bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    target = _safe_path(relative_path)
    if not target.exists():
        raise FileNotFoundError(f"Workspace file not found: {relative_path!r}")
    data = target.read_bytes()
    logger.debug("read_binary <- %s (%d bytes)", target, len(data))
    return data


def list_files(pattern: str = "*") -> list[str]:
    """List workspace files matching a glob pattern.

    Args:
        pattern: Glob pattern relative to workspace root (default ``"*"``).
                 Supports ``**`` for recursive matching.

    Returns:
        Sorted list of relative path strings (relative to workspace root).
    """
    workspace = _workspace().resolve()
    matched = sorted(workspace.glob(pattern))
    relative = [str(p.relative_to(workspace)) for p in matched if p.is_file()]
    logger.debug("list_files(pattern=%r) → %d files", pattern, len(relative))
    return relative
