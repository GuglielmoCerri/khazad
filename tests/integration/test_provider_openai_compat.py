"""Provider integration tests — OpenAI-compatible proxies (LiteLLM, vLLM, Ollama, etc)."""

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


class TestLiteLLMProxy:
    """LiteLLM exposes /v1/chat/completions on a custom host."""

    @pytest.mark.asyncio
    async def test_miss_then_hit(self, engine, openai_chat_body, openai_chat_response):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        url = "http://litellm-proxy.internal:4000/v1/chat/completions"
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                url,
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            await client.post(
                url,
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )

        assert api_calls == 1
        stats = engine.get_stats()
        assert stats.cache_hits == 1


class TestVLLMServer:
    """vLLM exposes OpenAI-compatible endpoints on a self-hosted server."""

    @pytest.mark.asyncio
    async def test_miss_then_hit(self, engine, openai_chat_body, openai_chat_response):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        url = "http://vllm-server:8000/v1/chat/completions"
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                url,
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            await client.post(
                url,
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )

        assert api_calls == 1


class TestOllamaCompatEndpoint:
    """Ollama exposes /v1/chat/completions in OpenAI-compatible mode."""

    @pytest.mark.asyncio
    async def test_localhost_ollama_cached(self, engine, openai_chat_body, openai_chat_response):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        url = "http://localhost:11434/v1/chat/completions"
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                url,
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            await client.post(
                url,
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )

        assert api_calls == 1
        stats = engine.get_stats()
        assert stats.cache_hits == 1
