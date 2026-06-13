"""Provider integration tests — Google Gemini generateContent."""

from __future__ import annotations

import httpx
import pytest

from khazad._transport import install, uninstall
from khazad.khazad import Khazad

GEMINI_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-2.0-flash:generateContent?key=fake-key"
)


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


class TestGeminiGenerateContent:
    """Verify caching against generativelanguage.googleapis.com /*:generateContent."""

    @pytest.mark.asyncio
    async def test_miss_then_hit(self, engine, gemini_chat_body, gemini_chat_response):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            assert req.url.host == "generativelanguage.googleapis.com"
            assert ":generateContent" in req.url.path
            return httpx.Response(200, content=gemini_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                GEMINI_URL,
                content=gemini_chat_body,
                headers={"content-type": "application/json"},
            )
            await client.post(
                GEMINI_URL,
                content=gemini_chat_body,
                headers={"content-type": "application/json"},
            )

        assert api_calls == 1
        stats = engine.get_stats()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1


class TestGeminiNonGenerateContentNotCached:
    """Other Gemini paths (e.g. listModels) are not cached."""

    @pytest.mark.asyncio
    async def test_list_models_passes_through(self, engine):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return httpx.Response(200, json={"models": []})

        transport = httpx.MockTransport(handler)
        url = "https://generativelanguage.googleapis.com/v1beta/models?key=fake-key"
        async with httpx.AsyncClient(transport=transport) as client:
            for _ in range(3):
                await client.get(url)

        assert api_calls == 3
        stats = engine.get_stats()
        assert stats.total_requests == 0
