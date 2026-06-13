"""End-to-end integration tests — full init → request → cache → stop flow.

Tests the complete lifecycle using in-memory fakes for Redis and embeddings.
"""

from __future__ import annotations

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


class TestFullCacheLifecycle:
    """Test the complete cache lifecycle: miss → store → hit → stop."""

    @pytest.mark.asyncio
    async def test_miss_then_hit_then_stop(self, engine, openai_chat_body, openai_chat_response):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            # First call — cache miss
            resp1 = await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            assert api_calls == 1
            assert resp1.status_code == 200

            # Second call — cache hit
            resp2 = await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            assert api_calls == 1  # No additional API call
            assert resp2.json() == resp1.json()

        # Stop and verify stats
        stats = engine.get_stats()
        assert stats.total_requests == 2
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1
        assert stats.hit_rate == 0.5

        # After stop, new clients should go directly to API
        uninstall()
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            assert api_calls == 2  # This one went through


class TestMultiProviderSession:
    """Test using multiple LLM providers in the same session."""

    @pytest.mark.asyncio
    async def test_openai_and_anthropic_cached_independently(
        self,
        engine,
        openai_chat_body,
        openai_chat_response,
        anthropic_chat_body,
        anthropic_chat_response,
    ):
        install(engine)
        openai_calls = 0
        anthropic_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal openai_calls, anthropic_calls
            if "openai" in req.url.host:
                openai_calls += 1
                return httpx.Response(200, content=openai_chat_response)
            if "anthropic" in req.url.host:
                anthropic_calls += 1
                return httpx.Response(200, content=anthropic_chat_response)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            # OpenAI miss
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            # Anthropic miss
            await client.post(
                "https://api.anthropic.com/v1/messages",
                content=anthropic_chat_body,
                headers={"content-type": "application/json"},
            )
            # OpenAI hit
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            # Anthropic hit
            await client.post(
                "https://api.anthropic.com/v1/messages",
                content=anthropic_chat_body,
                headers={"content-type": "application/json"},
            )

        assert openai_calls == 1
        assert anthropic_calls == 1
        stats = engine.get_stats()
        assert stats.total_requests == 4
        assert stats.cache_hits == 2


class TestStatsAccuracy:
    """Verify stats reflect actual cache behavior correctly."""

    @pytest.mark.asyncio
    async def test_stats_after_multiple_operations(self, engine, openai_chat_response):
        install(engine)

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            # 3 unique requests (misses) — use very distinct prompts
            prompts = [
                "Explain quantum entanglement in particle physics",
                "Write a Python function to sort a binary tree",
                "What are the ingredients in traditional Japanese ramen",
            ]
            for prompt in prompts:
                body = json.dumps(
                    {
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": prompt}],
                    }
                ).encode()
                await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    content=body,
                    headers={"content-type": "application/json"},
                )

            # Repeat each (hits)
            for prompt in prompts:
                body = json.dumps(
                    {
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": prompt}],
                    }
                ).encode()
                await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    content=body,
                    headers={"content-type": "application/json"},
                )

        stats = engine.get_stats()
        assert stats.total_requests == 6
        assert stats.cache_misses == 3
        assert stats.cache_hits == 3
        assert stats.hit_rate == 0.5

    @pytest.mark.asyncio
    async def test_non_llm_requests_not_counted(self, engine):
        install(engine)

        transport = httpx.MockTransport(lambda req: httpx.Response(200, text="OK"))
        async with httpx.AsyncClient(transport=transport) as client:
            for _ in range(10):
                await client.get("https://example.com/api/data")

        stats = engine.get_stats()
        assert stats.total_requests == 0
