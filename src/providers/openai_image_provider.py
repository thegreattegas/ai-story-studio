"""OpenAI image generation provider — wraps openai SDK (gpt-image-1).

Mock mode
---------
Returns a valid 1x1 grey PNG after a simulated 50ms delay so that
``asyncio.gather`` can demonstrate true parallel execution.

Real API mode
-------------
Uses ``gpt-image-1`` via the OpenAI images.generate endpoint.
The response is returned as base64 which is decoded and written to disk.

Cost estimate: ~$0.04 per generated image (1024x1024 standard quality).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import struct
import zlib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost constants
# ---------------------------------------------------------------------------

COST_PER_IMAGE_USD: float = 0.04

# ---------------------------------------------------------------------------
# Minimal valid PNG builder (no external deps) — reused from google_provider
# ---------------------------------------------------------------------------


def _make_1x1_grey_png() -> bytes:
    def _chunk(name: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)
    raw_scanline = b"\x00\x80"
    idat = _chunk(b"IDAT", zlib.compress(raw_scanline, level=9))
    iend = _chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


_MOCK_PNG_BYTES: bytes = _make_1x1_grey_png()

MOCK_LATENCY_SEC: float = 0.05


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageProvider:
    """Wrapper around OpenAI images.generate for gpt-image-1.

    Usage::

        provider = OpenAIImageProvider()
        result = await provider.generate_image(prompt, output_path)
        # result = {"path": Path, "cost_usd": float, "mocked": bool}
    """

    OPENAI_IMAGE_MODEL = "gpt-image-1"

    def __init__(self) -> None:
        from src.config import get_config  # noqa: PLC0415

        self.config = get_config()
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazy-init: create AsyncOpenAI client on first use."""
        if self._client is None:
            if not self.config.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add it to .env before running in live mode."
                )
            try:
                from openai import AsyncOpenAI  # noqa: PLC0415

                self._client = AsyncOpenAI(api_key=self.config.openai_api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def generate_image(self, prompt: str, output_path: Path) -> dict[str, Any]:
        """Generate an image from ``prompt`` and write it to ``output_path``.

        Args:
            prompt:      English image generation prompt.
            output_path: Absolute path where the PNG will be written.

        Returns:
            Dict with keys: ``path``, ``cost_usd``, ``mocked``.
        """
        if self.config.mock_mode:
            await asyncio.sleep(MOCK_LATENCY_SEC)
            return self._write_mock_png(output_path)

        return await self._generate_image_async(prompt, output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_mock_png(self, output_path: Path) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_MOCK_PNG_BYTES)
        logger.debug(
            "OpenAIImageProvider: wrote mock PNG (%d bytes) -> %s",
            len(_MOCK_PNG_BYTES),
            output_path,
        )
        return {"path": output_path, "cost_usd": 0.0, "mocked": True}

    async def _generate_image_async(
        self, prompt: str, output_path: Path
    ) -> dict[str, Any]:
        """Call OpenAI images.generate and save the result to disk.

        Args:
            prompt:      Image prompt string.
            output_path: Destination file path.

        Returns:
            Dict with path, cost_usd, mocked=False.

        Raises:
            RuntimeError: If the API response contains no image data.
        """
        client = self._get_client()

        logger.debug(
            "OpenAIImageProvider: calling %s for prompt '%s...'",
            self.OPENAI_IMAGE_MODEL,
            prompt[:60],
        )

        response = await client.images.generate(
            model=self.OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )

        b64_data = response.data[0].b64_json if response.data else None
        if not b64_data:
            raise RuntimeError(
                f"OpenAI returned no image data for prompt: '{prompt[:80]}...'"
            )

        image_bytes = base64.b64decode(b64_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        logger.info(
            "OpenAIImageProvider: image written (%d bytes) -> %s",
            len(image_bytes),
            output_path,
        )

        return {
            "path": output_path,
            "cost_usd": COST_PER_IMAGE_USD,
            "mocked": False,
        }
