"""Abstract interface for LLM provider request/response parsing."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator

    from khazad._models import ParsedRequest


class ProviderParser(ABC):
    """Port defining the contract for parsing provider-specific HTTP schemas."""

    @abstractmethod
    def can_handle(self, url: httpx.URL) -> bool:
        """Return True if this parser handles requests to the given URL."""

    @abstractmethod
    def parse_request(self, request: httpx.Request) -> ParsedRequest:
        """Extract prompt, model and stream flag from a request.

        Raises on bodies this parser cannot understand — the request is
        then passed through uncached.
        """

    def build_response(self, cached_data: bytes) -> httpx.Response:
        """Reconstruct a plain JSON response from cached bytes."""
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=cached_data,
        )

    def stream_chunks(self, cached_data: bytes) -> Iterator[bytes]:
        """Yield SSE frames that replay cached data as a streaming response."""
        yield cached_data

    def response_from_stream(self, sse_data: bytes) -> bytes | None:
        """Rebuild the canonical JSON response from captured SSE bytes.

        Returns None when reconstruction is unsupported — the streamed
        response is then simply not cached.
        """
        return None

    # ------------------------------------------------------------------
    # SSE helpers shared by concrete parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _sse(payload: dict, event: str | None = None) -> bytes:
        """Encode a single SSE frame, optionally with an event name."""
        data = f"data: {json.dumps(payload)}\n\n"
        if event is not None:
            data = f"event: {event}\n{data}"
        return data.encode()

    @staticmethod
    def _flatten_text(content: object) -> str:
        """Flatten a message content field (string or list of typed parts) to text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part["text"]
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        return ""

    @staticmethod
    def _iter_sse_payloads(sse_data: bytes) -> Iterator[dict]:
        """Yield decoded JSON payloads from raw SSE bytes."""
        for line in sse_data.decode("utf-8", errors="replace").splitlines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except ValueError:
                continue
            if isinstance(payload, dict):
                yield payload
