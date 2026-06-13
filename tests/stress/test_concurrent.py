"""Stress tests — concurrent access and high-throughput scenarios."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from khazad._transport import install, uninstall
from khazad.khazad import Khazad


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


@pytest.mark.stress
class TestConcurrentRequests:
    """Verify thread-safety under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_cache_misses(self, engine, openai_chat_response):
        """Multiple concurrent requests for different prompts should all be misses."""
        install(engine)
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)

        async def make_request(i: int) -> httpx.Response:
            async with httpx.AsyncClient(transport=transport) as client:
                body = json.dumps(
                    {
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": f"Concurrent request #{i}"}],
                    }
                ).encode()
                return await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    content=body,
                    headers={"content-type": "application/json"},
                )

        results = await asyncio.gather(*[make_request(i) for i in range(20)])
        assert all(r.status_code == 200 for r in results)
        # All should be unique prompts → all misses → all API calls
        assert call_count == 20

    @pytest.mark.asyncio
    async def test_concurrent_cache_hits(self, engine, openai_chat_body, openai_chat_response):
        """After warming the cache, concurrent identical requests should all be hits."""
        install(engine)

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)

        # Warm cache
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )

        hit_count = 0

        def counting_handler(req: httpx.Request) -> httpx.Response:
            nonlocal hit_count
            hit_count += 1
            return httpx.Response(200, content=openai_chat_response)

        counting_transport = httpx.MockTransport(counting_handler)

        async def make_hit_request() -> httpx.Response:
            async with httpx.AsyncClient(transport=counting_transport) as client:
                return await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    content=openai_chat_body,
                    headers={"content-type": "application/json"},
                )

        results = await asyncio.gather(*[make_hit_request() for _ in range(50)])
        assert all(r.status_code == 200 for r in results)
        # All should be cache hits → handler should not be called
        assert hit_count == 0

    @pytest.mark.asyncio
    async def test_stats_thread_safety(self, engine, openai_chat_body, openai_chat_response):
        """Stats must be consistent after concurrent requests."""
        install(engine)

        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, content=openai_chat_response)
        )

        # Warm cache
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )

        async def make_request() -> None:
            async with httpx.AsyncClient(transport=transport) as client:
                await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    content=openai_chat_body,
                    headers={"content-type": "application/json"},
                )

        await asyncio.gather(*[make_request() for _ in range(100)])

        stats = engine.get_stats()
        assert stats.total_requests == 101  # 1 warm-up + 100 concurrent
        assert stats.cache_hits == 100
        assert stats.cache_misses == 1
        assert stats.cache_hits + stats.cache_misses == stats.total_requests
