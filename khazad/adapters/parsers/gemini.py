"""Google Gemini parser."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from khazad._models import ParsedRequest
from khazad.ports.parser import ProviderParser

if TYPE_CHECKING:
    import httpx

_GEMINI_HOSTS = {"generativelanguage.googleapis.com"}
_MODEL_PATTERN = re.compile(r"/models/([^:/]+):generateContent$")


class GeminiParser(ProviderParser):
    """Parses Google Gemini generateContent requests and responses.

    Streaming (``:streamGenerateContent``) is deliberately not matched —
    those requests pass through uncached. Cached hits inherit the default
    ``stream_chunks``/``response_from_stream`` no-ops from the base class.
    """

    def can_handle(self, url: httpx.URL) -> bool:
        return url.host in _GEMINI_HOSTS and _MODEL_PATTERN.search(url.path) is not None

    def parse_request(self, request: httpx.Request) -> ParsedRequest:
        """Build the semantic prompt from systemInstruction plus all turns."""
        data = json.loads(request.content)
        contents = data.get("contents", [])
        if not any(c.get("role", "user") == "user" for c in contents):
            raise ValueError("No user content found in Gemini request")

        lines = []
        system = data.get("systemInstruction") or data.get("system_instruction")
        if system:
            lines.append(f"system: {self._flatten_text(system.get('parts'))}")
        lines.extend(
            f"{c.get('role', 'user')}: {self._flatten_text(c.get('parts'))}" for c in contents
        )

        match = _MODEL_PATTERN.search(request.url.path)
        return ParsedRequest(
            prompt="\n".join(lines),
            model=match.group(1) if match else None,
            stream=False,
        )
