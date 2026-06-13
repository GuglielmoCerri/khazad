"""OpenAI Responses API parser (POST /v1/responses)."""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING

from khazad._models import ParsedRequest
from khazad.ports.parser import ProviderParser

if TYPE_CHECKING:
    from collections.abc import Iterator

    import httpx

_RESPONSES_PATH_SUFFIX = "/responses"
_STREAM_CHUNK_CHARS = 16


class OpenAIResponsesParser(ProviderParser):
    """Parses OpenAI-compatible Responses API requests (POST .../responses).

    Matches any host whose URL path ends with ``/responses``. Covers
    api.openai.com plus OpenAI-compatible proxies exposing the Responses API.
    """

    def can_handle(self, url: httpx.URL) -> bool:
        return url.path.endswith(_RESPONSES_PATH_SUFFIX)

    def parse_request(self, request: httpx.Request) -> ParsedRequest:
        """Build the semantic prompt from instructions plus the full input.

        ``input`` may be a plain string or a list of message objects whose
        content is a string or a list of typed parts.
        """
        data = json.loads(request.content)
        raw_input = data.get("input")
        if raw_input is None:
            raise ValueError("No 'input' field found in Responses API request")

        lines = []
        if data.get("instructions"):
            lines.append(f"instructions: {data['instructions']}")

        if isinstance(raw_input, str):
            lines.append(f"user: {raw_input}")
        else:
            if not any(item.get("role") == "user" for item in raw_input):
                raise ValueError("No user message found in Responses API request")
            lines.extend(
                f"{item.get('role', 'user')}: {self._flatten_text(item.get('content'))}"
                for item in raw_input
            )

        return ParsedRequest(
            prompt="\n".join(lines),
            model=data.get("model"),
            stream=bool(data.get("stream", False)),
        )

    def stream_chunks(self, cached_data: bytes) -> Iterator[bytes]:
        """Replay a cached Responses API object as its SSE event sequence."""
        data = json.loads(cached_data)
        text = _extract_output_text(data)
        msg_item_id = f"msg_{uuid.uuid4().hex[:24]}"
        response_head = {
            "id": data.get("id", f"resp_{uuid.uuid4().hex}"),
            "object": "response",
            "created_at": data.get("created_at", int(time.time())),
            "model": data.get("model", "unknown"),
        }
        item = {"id": msg_item_id, "type": "message", "role": "assistant"}
        part = {"type": "output_text", "text": "", "annotations": []}

        yield self._sse(
            {
                "type": "response.created",
                "response": {**response_head, "status": "in_progress", "output": []},
            },
            event="response.created",
        )
        yield self._sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**item, "status": "in_progress", "content": []},
            },
            event="response.output_item.added",
        )
        yield self._sse(
            {
                "type": "response.content_part.added",
                "item_id": msg_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": part,
            },
            event="response.content_part.added",
        )

        for i in range(0, len(text), _STREAM_CHUNK_CHARS):
            yield self._sse(
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text[i : i + _STREAM_CHUNK_CHARS],
                },
                event="response.output_text.delta",
            )

        done_part = {**part, "text": text}
        yield self._sse(
            {
                "type": "response.output_text.done",
                "item_id": msg_item_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            },
            event="response.output_text.done",
        )
        yield self._sse(
            {
                "type": "response.content_part.done",
                "item_id": msg_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": done_part,
            },
            event="response.content_part.done",
        )
        completed_item = {**item, "status": "completed", "content": [done_part]}
        yield self._sse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": completed_item,
            },
            event="response.output_item.done",
        )
        yield self._sse(
            {
                "type": "response.completed",
                "response": {**response_head, "status": "completed", "output": [completed_item]},
            },
            event="response.completed",
        )

    def response_from_stream(self, sse_data: bytes) -> bytes | None:
        """Take the final response object from the ``response.completed`` event."""
        for payload in self._iter_sse_payloads(sse_data):
            if payload.get("type") == "response.completed" and "response" in payload:
                return json.dumps(payload["response"]).encode()
        return None


def _extract_output_text(data: dict) -> str:
    """Pull the assistant text out of a Responses API response object."""
    for item in data.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part.get("text", "")
    return ""
