"""OpenAI-compatible Chat Completions parser."""

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

_CHAT_PATH_SUFFIX = "/chat/completions"
_STREAM_CHUNK_CHARS = 16


class OpenAIParser(ProviderParser):
    """Parses OpenAI-compatible chat completion requests and responses.

    Matches any host whose URL path ends with ``/chat/completions``. Covers
    api.openai.com, Azure OpenAI deployments, and OpenAI-compatible proxies
    (LiteLLM, vLLM, Ollama, custom gateways).
    """

    def can_handle(self, url: httpx.URL) -> bool:
        return url.path.endswith(_CHAT_PATH_SUFFIX)

    def parse_request(self, request: httpx.Request) -> ParsedRequest:
        """Build the semantic prompt from the full conversation.

        All messages (system, user, assistant) participate so that two
        conversations sharing only their last user turn never collide.
        """
        data = json.loads(request.content)
        messages = data.get("messages", [])
        if not any(m.get("role") == "user" for m in messages):
            raise ValueError("No user message found in request")
        prompt = "\n".join(
            f"{m.get('role', 'user')}: {self._flatten_text(m.get('content'))}" for m in messages
        )
        return ParsedRequest(
            prompt=prompt,
            model=data.get("model"),
            stream=bool(data.get("stream", False)),
        )

    def stream_chunks(self, cached_data: bytes) -> Iterator[bytes]:
        """Replay a cached chat completion as SSE chunks."""
        data = json.loads(cached_data)
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content") or ""
        base = {
            "id": data.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": data.get("created") or int(time.time()),
            "model": data.get("model", "unknown"),
        }

        first = {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
        yield self._sse({**base, "choices": [first]})

        for i in range(0, len(content), _STREAM_CHUNK_CHARS):
            piece = content[i : i + _STREAM_CHUNK_CHARS]
            delta = {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            yield self._sse({**base, "choices": [delta]})

        finish = choice.get("finish_reason") or "stop"
        yield self._sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        yield b"data: [DONE]\n\n"

    def response_from_stream(self, sse_data: bytes) -> bytes | None:
        """Reassemble a complete chat completion from captured SSE chunks.

        Returns None unless the stream carried its terminal ``[DONE]``
        sentinel — a partial/aborted stream must never be cached.
        """
        if b"[DONE]" not in sse_data:
            return None

        meta: dict = {}
        parts: list[str] = []
        finish_reason = None
        usage = None

        for payload in self._iter_sse_payloads(sse_data):
            if payload.get("object") != "chat.completion.chunk":
                continue
            for field in ("id", "created", "model"):
                if payload.get(field) is not None:
                    meta.setdefault(field, payload[field])
            if payload.get("usage"):
                usage = payload["usage"]
            for choice in payload.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                content = choice.get("delta", {}).get("content")
                if content:
                    parts.append(content)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

        if not parts:
            return None

        response = {
            "id": meta.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            "object": "chat.completion",
            "created": meta.get("created", int(time.time())),
            "model": meta.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(parts)},
                    "finish_reason": finish_reason or "stop",
                }
            ],
        }
        if usage:
            response["usage"] = usage
        return json.dumps(response).encode()
