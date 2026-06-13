"""Provider integration tests — Anthropic Claude API."""

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


class TestAnthropicMessages:
    """Verify caching against api.anthropic.com /v1/messages."""

    @pytest.mark.asyncio
    async def test_miss_then_hit(self, engine, anthropic_chat_body, anthropic_chat_response):
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            assert req.url.host == "api.anthropic.com"
            assert req.url.path == "/v1/messages"
            return httpx.Response(200, content=anthropic_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                "https://api.anthropic.com/v1/messages",
                content=anthropic_chat_body,
                headers={
                    "content-type": "application/json",
                    "x-api-key": "sk-ant-test",
                    "anthropic-version": "2023-06-01",
                },
            )
            await client.post(
                "https://api.anthropic.com/v1/messages",
                content=anthropic_chat_body,
                headers={
                    "content-type": "application/json",
                    "x-api-key": "sk-ant-test",
                    "anthropic-version": "2023-06-01",
                },
            )

        assert api_calls == 1
        stats = engine.get_stats()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1

    @pytest.mark.asyncio
    async def test_content_blocks_extracted(self, engine, anthropic_chat_response):
        """Anthropic content can be a list of blocks — must extract text correctly."""
        install(engine)
        api_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal api_calls
            api_calls += 1
            return httpx.Response(200, content=anthropic_chat_response)

        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-latest",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello Claude"},
                        ],
                    }
                ],
            }
        ).encode()

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                "https://api.anthropic.com/v1/messages",
                content=body,
                headers={"content-type": "application/json"},
            )
            await client.post(
                "https://api.anthropic.com/v1/messages",
                content=body,
                headers={"content-type": "application/json"},
            )

        assert api_calls == 1
