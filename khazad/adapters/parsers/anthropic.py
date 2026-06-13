"""Anthropic messages parser."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from khazad._models import ParsedRequest
from khazad.ports.parser import ProviderParser

if TYPE_CHECKING:
    from collections.abc import Iterator

    import httpx

_ANTHROPIC_HOSTS = {"api.anthropic.com"}
_MESSAGES_PATH = "/v1/messages"
_STREAM_CHUNK_CHARS = 16


class AnthropicParser(ProviderParser):
    """Parses Anthropic /v1/messages requests and responses."""

    def can_handle(self, url: httpx.URL) -> bool:
        return url.host in _ANTHROPIC_HOSTS and url.path == _MESSAGES_PATH

    def parse_request(self, request: httpx.Request) -> ParsedRequest:
        """Build the semantic prompt from the system field plus all messages."""
        data = json.loads(request.content)
        messages = data.get("messages", [])
        if not any(m.get("role") == "user" for m in messages):
            raise ValueError("No user message found in Anthropic request")

        lines = []
        if data.get("system"):
            lines.append(f"system: {self._flatten_text(data['system'])}")
        lines.extend(
            f"{m.get('role', 'user')}: {self._flatten_text(m.get('content'))}" for m in messages
        )
        return ParsedRequest(
            prompt="\n".join(lines),
            model=data.get("model"),
            stream=bool(data.get("stream", False)),
        )

    def stream_chunks(self, cached_data: bytes) -> Iterator[bytes]:
        """Replay a cached Anthropic message as its SSE event sequence."""
        data = json.loads(cached_data)
        text = next(
            (b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"),
            "",
        )

        message = {
            "id": data.get("id", f"msg_{uuid.uuid4().hex[:12]}"),
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": data.get("model", "unknown"),
            "usage": data.get("usage", {"input_tokens": 0, "output_tokens": 0}),
        }
        yield self._sse({"type": "message_start", "message": message}, event="message_start")
        yield self._sse(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            event="content_block_start",
        )

        for i in range(0, len(text), _STREAM_CHUNK_CHARS):
            yield self._sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text[i : i + _STREAM_CHUNK_CHARS]},
                },
                event="content_block_delta",
            )

        yield self._sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")
        yield self._sse(
            {
                "type": "message_delta",
                "delta": {"stop_reason": data.get("stop_reason", "end_turn")},
                "usage": data.get("usage", {}),
            },
            event="message_delta",
        )
        yield self._sse({"type": "message_stop"}, event="message_stop")

    def response_from_stream(self, sse_data: bytes) -> bytes | None:
        """Reassemble a complete Anthropic message from captured SSE events."""
        message: dict | None = None
        parts: list[str] = []
        stop_reason = None
        usage_delta: dict = {}

        for payload in self._iter_sse_payloads(sse_data):
            kind = payload.get("type")
            if kind == "message_start":
                message = payload.get("message", {})
            elif kind == "content_block_delta":
                delta = payload.get("delta", {})
                if delta.get("type") == "text_delta":
                    parts.append(delta.get("text", ""))
            elif kind == "message_delta":
                stop_reason = payload.get("delta", {}).get("stop_reason")
                usage_delta = payload.get("usage", {})

        if message is None or not parts:
            return None

        message["content"] = [{"type": "text", "text": "".join(parts)}]
        message["stop_reason"] = stop_reason or "end_turn"
        message["usage"] = {**message.get("usage", {}), **usage_delta}
        return json.dumps(message).encode()
