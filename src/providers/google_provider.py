"""Google AI image generation provider — wraps google-genai SDK.

Mock mode
---------
Returns a valid 1x1 grey PNG (constructed with Python stdlib ``zlib`` +
``struct`` — no Pillow required) after a simulated 50ms delay so that
``asyncio.gather`` can demonstrate true parallel execution.

Real API mode
-------------
Uses ``gemini-2.5-flash-image`` with ``response_modalities=["IMAGE"]``.
The google-genai SDK call is synchronous, so we wrap it in
``asyncio.to_thread`` to keep the async interface non-blocking.

Cost estimate: ~$0.045 per generated image (Gemini 2.5 Flash image tier).
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
import zlib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost constants
# ---------------------------------------------------------------------------

# Approximate cost per generated image (USD).
COST_PER_IMAGE_USD: float = 0.045

# ---------------------------------------------------------------------------
# Minimal valid PNG builder (no external deps)
# ---------------------------------------------------------------------------


def _make_1x1_grey_png() -> bytes:
    """Return bytes of a valid 1x1 greyscale PNG (8-bit, 128/255 grey).

    Built using Python stdlib ``zlib`` + ``struct`` — no Pillow required.
    The CRC32 checksums are computed correctly so any PNG reader accepts it.
    """

    def _chunk(name: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)

    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR: width=1, height=1, bit_depth=8, color_type=0 (greyscale),
    #       compression=0, filter=0, interlace=0
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # IDAT: filter byte 0 (None) + pixel value 128 (mid-grey)
    raw_scanline = b"\x00\x80"
    idat = _chunk(b"IDAT", zlib.compress(raw_scanline, level=9))

    iend = _chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


# Pre-compute once at module load — used by every mock call.
_MOCK_PNG_BYTES: bytes = _make_1x1_grey_png()

# Validate our own output at import time (sanity check).
assert _MOCK_PNG_BYTES[:4] == bytes([0x89, 0x50, 0x4E, 0x47]), \
    "BUG: _make_1x1_grey_png() did not produce a valid PNG signature!"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GoogleImageProvider:
    """Wrapper around google-genai for Gemini image generation.

    Usage::

        provider = GoogleImageProvider()
        result = await provider.generate_image(prompt, output_path)
        # result = {"path": Path, "tokens_used": int, "cost_usd": float, "mocked": bool}
    """

    GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"

    # Simulated I/O latency in mock mode — makes asyncio.gather parallelism visible.
    MOCK_LATENCY_SEC: float = 0.05

    def __init__(self) -> None:
        from src.config import get_config  # noqa: PLC0415

        self.config = get_config()
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazy-init: create google-genai Client on first use."""
        if self._client is None:
            if not self.config.google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY is not set. Add it to .env before running in live mode."
                )
            try:
                from google import genai  # noqa: PLC0415

                self._client = genai.Client(api_key=self.config.google_api_key)
            except ImportError as exc:
                raise ImportError(
                    "google-genai package not installed. "
                    "Run: pip install google-genai"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def generate_image(self, prompt: str, output_path: Path) -> dict[str, Any]:
        """Generate an image from ``prompt`` and write it to ``output_path``.

        Args:
            prompt:      English image generation prompt (30-60 words).
            output_path: Absolute path where the PNG will be written.

        Returns:
            Dict with keys: ``path``, ``tokens_used``, ``cost_usd``, ``mocked``.
        """
        if self.config.mock_mode:
            # Small delay so parallel gather calls visibly interleave.
            await asyncio.sleep(self.MOCK_LATENCY_SEC)
            return self._write_mock_png(output_path)

        # Wrap the synchronous SDK call in a thread so we don't block the
        # event loop — allows other async tasks to run concurrently.
        return await asyncio.to_thread(self._generate_image_sync, prompt, output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_mock_png(self, output_path: Path) -> dict[str, Any]:
        """Write the pre-computed mock PNG to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_MOCK_PNG_BYTES)
        logger.debug(
            "GoogleImageProvider: wrote mock PNG (%d bytes) -> %s",
            len(_MOCK_PNG_BYTES),
            output_path,
        )
        return {
            "path": output_path,
            "tokens_used": 0,
            "cost_usd": 0.0,
            "mocked": True,
        }

    def _generate_image_sync(self, prompt: str, output_path: Path) -> dict[str, Any]:
        """Synchronous Gemini image generation call (run inside a thread).

        Args:
            prompt:      Image prompt string.
            output_path: Destination file path.

        Returns:
            Dict with path, tokens_used, cost_usd, mocked=False.

        Raises:
            RuntimeError: If the API response contains no image data.
        """
        from google.genai import types  # noqa: PLC0415

        client = self._get_client()

        logger.debug(
            "GoogleImageProvider: calling %s for prompt '%s...'",
            self.GEMINI_IMAGE_MODEL,
            prompt[:60],
        )

        response = client.models.generate_content(
            model=self.GEMINI_IMAGE_MODEL,
            contents=f"Generate a children's book illustration: {prompt}",
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        # Extract image bytes from the response parts.
        image_bytes: bytes | None = None
        tokens_used: int = 0

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data is not None and part.inline_data.data:
                    image_bytes = part.inline_data.data
                    break
            if image_bytes:
                break

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = getattr(response.usage_metadata, "total_token_count", 0)

        if image_bytes is None:
            raise RuntimeError(
                f"Gemini returned no image data for prompt: '{prompt[:80]}...'. "
                "Ensure your API key has access to Gemini image generation."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        logger.info(
            "GoogleImageProvider: image written (%d bytes) -> %s",
            len(image_bytes),
            output_path,
        )

        return {
            "path": output_path,
            "tokens_used": tokens_used,
            "cost_usd": COST_PER_IMAGE_USD,
            "mocked": False,
        }
