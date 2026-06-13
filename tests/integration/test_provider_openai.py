"""Provider integration tests — OpenAI Chat Completions + Responses API."""

from __future__ import annotations

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


class TestOpenAIChatCompletions:
    """Verify caching against api.openai.com /v1/chat/completions."""

    @pytest.mark.asyncio
    async def test_miss_then_hit(self, engine, openai_chat_body, openai_chat_response):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            assert req.url.host == "api.openai.com"
            assert req.url.path == "/v1/chat/completions"
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )

        assert api_calls == 1
        stats = engine.get_stats()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1


class TestOpenAIResponsesAPI:
    """Verify caching against api.openai.com /v1/responses."""

    @pytest.mark.asyncio
    async def test_miss_then_hit(self, engine, openai_responses_body, openai_responses_response):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            assert req.url.path == "/v1/responses"
            return httpx.Response(200, content=openai_responses_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                "https://api.openai.com/v1/responses",
                content=openai_responses_body,
                headers={"content-type": "application/json"},
            )
            await client.post(
                "https://api.openai.com/v1/responses",
                content=openai_responses_body,
                headers={"content-type": "application/json"},
            )

        assert api_calls == 1
        stats = engine.get_stats()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1


class TestOpenAINonCachedEndpoints:
    """Other OpenAI endpoints (embeddings, models) pass through uncached."""

    @pytest.mark.asyncio
    async def test_embeddings_endpoint_not_cached(self, engine):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return httpx.Response(200, json={"data": []})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            for _ in range(3):
                await client.post(
                    "https://api.openai.com/v1/embeddings",
                    json={"model": "text-embedding-3-small", "input": "hi"},
                )

        assert api_calls == 3
        stats = engine.get_stats()
        assert stats.total_requests == 0
