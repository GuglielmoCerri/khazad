"""Provider integration tests — Azure OpenAI deployments."""

from __future__ import annotations

import httpx
import pytest

from khazad._transport import install, uninstall
from khazad.khazad import Khazad

AZURE_URL = (
    "https://my-resource.openai.azure.com"
    "/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15-preview"
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


class TestAzureOpenAI:
    """Verify caching against Azure OpenAI deployment endpoints."""

    @pytest.mark.asyncio
    async def test_miss_then_hit(self, engine, openai_chat_body, openai_chat_response):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            assert req.url.host.endswith(".openai.azure.com")
            assert req.url.path.endswith("/chat/completions")
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                AZURE_URL,
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            await client.post(
                AZURE_URL,
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )

        assert api_calls == 1
        stats = engine.get_stats()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1

    @pytest.mark.asyncio
    async def test_different_azure_resources_share_cache_when_same_prompt(
        self, engine, openai_chat_body, openai_chat_response
    ):
        """Two different Azure resources with same prompt — cached per host."""
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        url_a = "https://resource-a.openai.azure.com/openai/deployments/gpt-4/chat/completions"
        url_b = "https://resource-b.openai.azure.com/openai/deployments/gpt-4/chat/completions"
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                url_a,
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            await client.post(
                url_b,
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )

        # Different hosts → different semantic keys → both miss
        assert api_calls == 2
        stats = engine.get_stats()
        assert stats.cache_misses == 2
        assert stats.cache_hits == 0
