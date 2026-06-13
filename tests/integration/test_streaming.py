"""Integration tests for streaming support.

Verifies that:
- Streaming cache misses are captured from real SSE traffic and cached as JSON
- Streaming cache hits replay valid SSE for both sync and async clients
- A streaming miss can later serve a non-streaming hit (and vice versa)
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from khazad._transport import install, uninstall
from khazad.adapters.parsers.openai import OpenAIParser
from khazad.khazad import Khazad

CHAT_URL = "https://api.openai.com/v1/chat/completions"


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    uninstall()


@pytest.fixture
def engine(fake_embedder, memory_store):
    return Khazad(
        threshold=0.99,
        _vector_store=memory_store,
        _embedder_instance=fake_embedder,
    )


@pytest.fixture
def plain_chat_body() -> bytes:
    """Non-streaming body with the same messages as openai_streaming_body."""
    return json.dumps(
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }
    ).encode()


def sse_handler(openai_chat_response: bytes):
    """MockTransport handler that streams the canned response as SSE."""
    parser = OpenAIParser()
    chunks = list(parser.stream_chunks(openai_chat_response))

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"".join(chunks),
        )

    return handler


class TestStreamingMissCapture:
    """A streamed upstream response must be reconstructed and cached."""

    @pytest.mark.asyncio
    async def test_async_streaming_miss_then_hit(
        self, engine, memory_store, openai_streaming_body, openai_chat_response
    ):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return sse_handler(openai_chat_response)(req)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            # Miss — consume the SSE stream fully so it gets cached.
            async with client.stream(
                "POST",
                CHAT_URL,
                content=openai_streaming_body,
                headers={"content-type": "application/json"},
            ) as resp:
                first = [chunk async for chunk in resp.aiter_bytes()]
            assert api_calls == 1
            assert b"".join(first)  # stream produced data

            # The store happens on a worker thread — wait for it to land.
            for _ in range(100):
                if memory_store._responses:
                    break
                await asyncio.sleep(0.02)
            assert memory_store._responses

            # Hit — replayed from cache, no upstream call.
            async with client.stream(
                "POST",
                CHAT_URL,
                content=openai_streaming_body,
                headers={"content-type": "application/json"},
            ) as resp:
                replayed = b"".join([chunk async for chunk in resp.aiter_bytes()])
            assert api_calls == 1
            assert b"data: [DONE]" in replayed

    def test_sync_streaming_miss_then_hit(
        self, engine, openai_streaming_body, openai_chat_response
    ):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return sse_handler(openai_chat_response)(req)

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as client:
            with client.stream(
                "POST",
                CHAT_URL,
                content=openai_streaming_body,
                headers={"content-type": "application/json"},
            ) as resp:
                list(resp.iter_bytes())
            assert api_calls == 1

            with client.stream(
                "POST",
                CHAT_URL,
                content=openai_streaming_body,
                headers={"content-type": "application/json"},
            ) as resp:
                replayed = b"".join(resp.iter_bytes())
            assert api_calls == 1  # served from cache
            assert b"data: [DONE]" in replayed

    def test_streaming_miss_serves_non_streaming_hit(
        self, engine, openai_streaming_body, plain_chat_body, openai_chat_response
    ):
        """The cached entry is canonical JSON — usable by non-streaming requests."""
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return sse_handler(openai_chat_response)(req)

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as client:
            with client.stream(
                "POST",
                CHAT_URL,
                content=openai_streaming_body,
                headers={"content-type": "application/json"},
            ) as resp:
                list(resp.iter_bytes())
            assert api_calls == 1

            resp = client.post(
                CHAT_URL,
                content=plain_chat_body,
                headers={"content-type": "application/json"},
            )
            assert api_calls == 1
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == "Paris is the capital of France."

    def test_aborted_stream_not_cached(self, engine, openai_streaming_body, openai_chat_response):
        """A stream closed before exhaustion must not poison the cache."""
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return sse_handler(openai_chat_response)(req)

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as client:
            with client.stream(
                "POST",
                CHAT_URL,
                content=openai_streaming_body,
                headers={"content-type": "application/json"},
            ) as resp:
                next(resp.iter_bytes())  # read one chunk, then abort

            client.post(
                CHAT_URL,
                content=openai_streaming_body,
                headers={"content-type": "application/json"},
            )
            assert api_calls == 2  # second call still reached the API


class TestStreamingHitReplay:
    """Cached non-streaming responses must replay as SSE for streaming requests."""

    def test_sync_client_streaming_hit(
        self, engine, plain_chat_body, openai_streaming_body, openai_chat_response
    ):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as client:
            # Warm the cache with a non-streaming request.
            client.post(
                CHAT_URL, content=plain_chat_body, headers={"content-type": "application/json"}
            )
            assert api_calls == 1

            # Streaming request — must be replayed as SSE by a *sync* client.
            with client.stream(
                "POST",
                CHAT_URL,
                content=openai_streaming_body,
                headers={"content-type": "application/json"},
            ) as resp:
                assert resp.headers["content-type"] == "text/event-stream"
                raw = b"".join(resp.iter_bytes())
            assert api_calls == 1
            assert b"data: [DONE]" in raw

    @pytest.mark.asyncio
    async def test_async_client_streaming_hit(
        self, engine, plain_chat_body, openai_streaming_body, openai_chat_response
    ):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                CHAT_URL, content=plain_chat_body, headers={"content-type": "application/json"}
            )
            assert api_calls == 1

            async with client.stream(
                "POST",
                CHAT_URL,
                content=openai_streaming_body,
                headers={"content-type": "application/json"},
            ) as resp:
                raw = b"".join([chunk async for chunk in resp.aiter_bytes()])
            assert api_calls == 1

            # Reassemble the deltas and verify content round-trips.
            full = ""
            for line in raw.decode().splitlines():
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    payload = json.loads(line[6:])
                    full += payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
            assert full == "Paris is the capital of France."
