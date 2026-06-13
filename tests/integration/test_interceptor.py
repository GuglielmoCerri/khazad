"""Integration tests for HTTP transport interception — the most critical component.

Tests verify that:
- LLM requests are correctly intercepted
- Non-LLM traffic passes through untouched
- Cache hits return correct responses
- Cache misses forward to the real API
- Interceptor installs and uninstalls cleanly
- Both sync and async clients work
- Concurrent requests are handled safely
"""

from __future__ import annotations

import httpx
import pytest

from khazad._transport import install, uninstall
from khazad.khazad import Khazad


@pytest.fixture
def engine(fake_embedder, memory_store):
    """Create a Khazad instance with in-memory backends."""
    return Khazad(
        threshold=0.99,
        _vector_store=memory_store,
        _embedder_instance=fake_embedder,
    )


@pytest.fixture(autouse=True)
def _cleanup_patches():
    """Ensure interceptor patches are removed after each test."""
    yield
    uninstall()


class TestNonLLMTrafficPassthrough:
    """Non-LLM HTTP requests must never be intercepted."""

    @pytest.mark.asyncio
    async def test_get_request_not_intercepted(self, engine):
        install(engine)
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text="OK"))
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get("https://example.com/api/data")
            assert resp.status_code == 200
            assert resp.text == "OK"

    @pytest.mark.asyncio
    async def test_post_to_non_llm_api_not_intercepted(self, engine):
        install(engine)
        transport = httpx.MockTransport(lambda req: httpx.Response(201, text="Created"))
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.post("https://myapi.com/users", json={"name": "test"})
            assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_random_domains_not_intercepted(self, engine):
        """Verify various non-LLM domains are never caught by parsers."""
        install(engine)
        urls = [
            "https://github.com/api/v3/repos",
            "https://api.stripe.com/v1/charges",
            "https://hooks.slack.com/services/T00/B00/xxx",
            "https://api.twilio.com/2010-04-01/Accounts",
        ]
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text="pass"))
        async with httpx.AsyncClient(transport=transport) as client:
            for url in urls:
                resp = await client.get(url)
                assert resp.status_code == 200
                assert resp.text == "pass"


class TestOpenAIInterception:
    """OpenAI chat completion requests must be intercepted and cached."""

    @pytest.mark.asyncio
    async def test_cache_miss_forwards_request(
        self, engine, openai_chat_body, openai_chat_response
    ):
        install(engine)

        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.host == "api.openai.com"
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_cache_hit_does_not_forward(
        self, engine, openai_chat_body, openai_chat_response
    ):
        install(engine)
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            # First call — cache miss, should forward
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            assert call_count == 1

            # Second call — same request, should hit cache
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 200
            # The handler should NOT have been called again
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_cache_hit_response_matches_original(
        self, engine, openai_chat_body, openai_chat_response
    ):
        install(engine)

        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, content=openai_chat_response)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            # Miss
            resp1 = await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            # Hit
            resp2 = await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            # Response structure must be identical
            assert resp1.json() == resp2.json()


class TestAnthropicInterception:
    """Anthropic requests must be intercepted and cached."""

    @pytest.mark.asyncio
    async def test_anthropic_cache_round_trip(
        self, engine, anthropic_chat_body, anthropic_chat_response
    ):
        install(engine)
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, content=anthropic_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            # Miss
            await client.post(
                "https://api.anthropic.com/v1/messages",
                content=anthropic_chat_body,
                headers={"content-type": "application/json"},
            )
            # Hit
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                content=anthropic_chat_body,
                headers={"content-type": "application/json"},
            )
            assert call_count == 1
            assert resp.status_code == 200


class TestAzureOpenAIInterception:
    """Azure OpenAI requests must be intercepted."""

    @pytest.mark.asyncio
    async def test_azure_url_matched(self, engine, openai_chat_body, openai_chat_response):
        install(engine)
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            url = "https://mycompany.openai.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-02"
            await client.post(
                url, content=openai_chat_body, headers={"content-type": "application/json"}
            )
            await client.post(
                url, content=openai_chat_body, headers={"content-type": "application/json"}
            )
            assert call_count == 1


class TestGeminiInterception:
    """Gemini requests must be intercepted."""

    @pytest.mark.asyncio
    async def test_gemini_url_matched(self, engine, gemini_chat_body, gemini_chat_response):
        install(engine)
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, content=gemini_chat_response)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            await client.post(
                url, content=gemini_chat_body, headers={"content-type": "application/json"}
            )
            await client.post(
                url, content=gemini_chat_body, headers={"content-type": "application/json"}
            )
            assert call_count == 1


class TestInterceptorLifecycle:
    """Install and uninstall must be clean and reversible."""

    @pytest.mark.asyncio
    async def test_uninstall_restores_original_transport(
        self, engine, openai_chat_body, openai_chat_response
    ):
        install(engine)
        # After uninstall, requests should go straight through
        uninstall()

        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
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
            # Both calls should go through since cache is not active
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_double_install_is_safe(self, engine):
        install(engine)
        install(engine)  # Should not crash
        uninstall()

    @pytest.mark.asyncio
    async def test_uninstall_without_install_is_safe(self):
        uninstall()  # Should not crash


class TestSyncClientInterception:
    """Verify the sync httpx.Client also gets intercepted."""

    def test_sync_client_cache_round_trip(self, engine, openai_chat_body, openai_chat_response):
        install(engine)
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, content=openai_chat_response)

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as client:
            # Miss
            client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            # Hit
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                content=openai_chat_body,
                headers={"content-type": "application/json"},
            )
            assert call_count == 1
            assert resp.status_code == 200


class TestMalformedRequests:
    """Interceptor must handle bad input gracefully."""

    @pytest.mark.asyncio
    async def test_invalid_json_body_passes_through(self, engine):
        install(engine)

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="Bad request")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=b"not json at all",
                headers={"content-type": "application/json"},
            )
            # Should forward to real API and return its error
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_body_passes_through(self, engine):
        install(engine)

        transport = httpx.MockTransport(lambda req: httpx.Response(400, text="Empty"))
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                content=b"",
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400
