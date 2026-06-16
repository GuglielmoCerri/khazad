"""Unit tests for the Khazad cache operations."""

from __future__ import annotations

import json

import httpx
import pytest

from khazad._models import CacheScope
from khazad.khazad import Khazad

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class TestKeyGeneration:
    """Verify deterministic key derivation."""

    def test_same_text_same_key(self):
        assert Khazad._make_key("Hello world") == Khazad._make_key("Hello world")

    def test_different_text_different_key(self):
        assert Khazad._make_key("Hello world") != Khazad._make_key("Goodbye world")

    def test_key_length(self):
        assert len(Khazad._make_key("test")) == 32


class TestPrepare:
    """Verify request preparation (parser matching + parsing)."""

    def test_none_when_no_parser_matches(self, make_engine, openai_chat_body):
        engine = make_engine()
        req = httpx.Request("POST", "https://unknown.api.com/v1/chat", content=openai_chat_body)
        assert engine.prepare(req) is None

    def test_none_when_body_unparseable(self, make_engine):
        engine = make_engine()
        req = httpx.Request("POST", OPENAI_URL, content=b"not json")
        assert engine.prepare(req) is None

    def test_scope_includes_host_and_model(self, make_engine, openai_chat_body):
        engine = make_engine()
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        prepared = engine.prepare(req)
        assert prepared is not None
        assert prepared.scope == "api.openai.com/gpt-4"

    def test_cache_scope_host_is_host_only(self, make_engine, openai_chat_body):
        engine = make_engine(cache_scope=CacheScope.HOST)
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        prepared = engine.prepare(req)
        assert prepared is not None
        assert prepared.scope == "api.openai.com"

    def test_cache_scope_host_same_host_different_model_share_scope(
        self, make_engine, openai_chat_body
    ):
        engine = make_engine(cache_scope=CacheScope.HOST)
        first = engine.prepare(httpx.Request("POST", OPENAI_URL, content=openai_chat_body))
        other = json.loads(openai_chat_body)
        other["model"] = "gpt-4o-mini"
        second = engine.prepare(
            httpx.Request("POST", OPENAI_URL, content=json.dumps(other).encode())
        )
        assert first is not None
        assert second is not None
        assert first.scope == second.scope == "api.openai.com"

    def test_cache_scope_host_different_host_stay_isolated(self, make_engine, openai_chat_body):
        engine = make_engine(cache_scope=CacheScope.HOST)
        openai = engine.prepare(httpx.Request("POST", OPENAI_URL, content=openai_chat_body))
        azure_url = (
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions"
        )
        azure = engine.prepare(httpx.Request("POST", azure_url, content=openai_chat_body))
        assert openai is not None
        assert azure is not None
        assert openai.scope != azure.scope

    def test_unparseable_requests_not_counted(self, make_engine):
        engine = make_engine()
        req = httpx.Request("GET", "https://example.com/api/data", content=b"")
        assert engine.prepare(req) is None
        assert engine.get_stats().total_requests == 0


class TestHostAllowlist:
    """Verify the opt-in hosts allowlist."""

    def _engine(self, fake_embedder, memory_store, hosts):
        return Khazad(hosts=hosts, _vector_store=memory_store, _embedder_instance=fake_embedder)

    def test_allowed_host_is_prepared(self, fake_embedder, memory_store, openai_chat_body):
        engine = self._engine(fake_embedder, memory_store, ["api.openai.com"])
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        assert engine.prepare(req) is not None

    def test_other_host_passes_through(self, fake_embedder, memory_store, openai_chat_body):
        engine = self._engine(fake_embedder, memory_store, ["api.openai.com"])
        req = httpx.Request(
            "POST", "https://my-proxy.example.com/v1/chat/completions", content=openai_chat_body
        )
        assert engine.prepare(req) is None
        assert engine.get_stats().total_requests == 0

    def test_wildcard_subdomain(self, fake_embedder, memory_store, openai_chat_body):
        engine = self._engine(fake_embedder, memory_store, ["*.openai.azure.com"])
        url = "https://my-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions"
        req = httpx.Request("POST", url, content=openai_chat_body)
        assert engine.prepare(req) is not None
        other = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        assert engine.prepare(other) is None

    def test_host_match_is_case_insensitive(self, fake_embedder, memory_store, openai_chat_body):
        engine = self._engine(fake_embedder, memory_store, ["API.OpenAI.com"])
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        assert engine.prepare(req) is not None

    def test_empty_allowlist_rejected(self, fake_embedder, memory_store):
        with pytest.raises(ValueError, match="hosts"):
            self._engine(fake_embedder, memory_store, [])


class TestLookup:
    """Verify cache lookup logic."""

    def test_miss_on_empty_store(self, make_engine, openai_chat_body):
        engine = make_engine()
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        assert engine.lookup(engine.prepare(req)) is None

    def test_hit_after_store(self, make_engine, openai_chat_body, openai_chat_response):
        engine = make_engine(threshold=0.5)
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        engine.store(engine.prepare(req), openai_chat_response)

        result = engine.lookup(engine.prepare(req))
        assert result is not None
        assert result.similarity >= 0.5
        assert result.response_data == openai_chat_response

    def test_miss_when_below_threshold(self, make_engine, openai_chat_body, openai_chat_response):
        engine = make_engine(threshold=1.0)
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        engine.store(engine.prepare(req), openai_chat_response)

        different_body = json.dumps(
            {
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Completely unrelated topic about quantum physics"}
                ],
            }
        ).encode()
        req2 = httpx.Request("POST", OPENAI_URL, content=different_body)
        assert engine.lookup(engine.prepare(req2)) is None

    def test_different_model_never_hits(self, make_engine, openai_chat_body, openai_chat_response):
        """Same prompt sent to a different model must not reuse the cache."""
        engine = make_engine(threshold=0.5)
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        engine.store(engine.prepare(req), openai_chat_response)

        other_model = json.loads(openai_chat_body)
        other_model["model"] = "gpt-3.5-turbo"
        req2 = httpx.Request("POST", OPENAI_URL, content=json.dumps(other_model).encode())
        assert engine.lookup(engine.prepare(req2)) is None

    def test_cache_scope_host_lets_different_model_hit(
        self, make_engine, openai_chat_body, openai_chat_response
    ):
        """With cache_scope=HOST, the same prompt to another model reuses the cache."""
        engine = make_engine(threshold=0.5, cache_scope=CacheScope.HOST)
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        engine.store(engine.prepare(req), openai_chat_response)

        other_model = json.loads(openai_chat_body)
        other_model["model"] = "gpt-4o-mini"
        req2 = httpx.Request("POST", OPENAI_URL, content=json.dumps(other_model).encode())
        result = engine.lookup(engine.prepare(req2))
        assert result is not None
        assert result.response_data == openai_chat_response

    def test_different_conversation_history_never_hits(self, make_engine, openai_chat_response):
        """Identical last user turn in different conversations must not collide."""
        engine = make_engine(threshold=0.99)

        def body(history):
            return json.dumps({"model": "gpt-4", "messages": history}).encode()

        conv_a = body(
            [
                {"role": "user", "content": "Tell me about Rome"},
                {"role": "assistant", "content": "Rome is the capital of Italy..."},
                {"role": "user", "content": "What about its population?"},
            ]
        )
        conv_b = body(
            [
                {"role": "user", "content": "Tell me about Tokyo"},
                {"role": "assistant", "content": "Tokyo is the capital of Japan..."},
                {"role": "user", "content": "What about its population?"},
            ]
        )
        engine.store(
            engine.prepare(httpx.Request("POST", OPENAI_URL, content=conv_a)), openai_chat_response
        )
        assert (
            engine.lookup(engine.prepare(httpx.Request("POST", OPENAI_URL, content=conv_b)))
            is None
        )

    def test_stale_vector_pruned_when_response_missing(
        self, make_engine, memory_store, openai_chat_body, openai_chat_response
    ):
        """A vector whose response expired must be deleted and count as a miss."""
        engine = make_engine(threshold=0.5)
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        prepared = engine.prepare(req)
        engine.store(prepared, openai_chat_response)

        # Simulate TTL expiry of the response body only.
        memory_store._responses.clear()

        assert engine.lookup(engine.prepare(req)) is None
        # The orphaned vector must be gone too.
        assert all(not v for v in memory_store._vectors.values())


class TestEmbeddingReuse:
    """The embedding must be computed once per request."""

    def test_lookup_then_store_embeds_once(
        self, fake_embedder, memory_store, openai_chat_body, openai_chat_response
    ):
        calls = 0
        original = fake_embedder.embed

        def counting_embed(text):
            nonlocal calls
            calls += 1
            return original(text)

        fake_embedder.embed = counting_embed
        engine = Khazad(
            threshold=0.9, _vector_store=memory_store, _embedder_instance=fake_embedder
        )
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        prepared = engine.prepare(req)
        engine.lookup(prepared)
        engine.store(prepared, openai_chat_response)
        assert calls == 1


class TestStats:
    """Verify stats tracking."""

    def test_stats_after_miss(self, make_engine, openai_chat_body):
        engine = make_engine()
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        engine.lookup(engine.prepare(req))
        stats = engine.get_stats()
        assert stats.total_requests == 1
        assert stats.cache_misses == 1
        assert stats.cache_hits == 0

    def test_stats_after_hit(self, make_engine, openai_chat_body, openai_chat_response):
        engine = make_engine(threshold=0.5)
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        engine.store(engine.prepare(req), openai_chat_response)
        engine.lookup(engine.prepare(req))
        stats = engine.get_stats()
        assert stats.total_requests == 1
        assert stats.cache_hits == 1
        assert stats.cache_misses == 0
        assert stats.avg_hit_similarity > 0

    def test_stats_accumulate(self, make_engine, openai_chat_body, openai_chat_response):
        engine = make_engine(threshold=0.5)
        req = httpx.Request("POST", OPENAI_URL, content=openai_chat_body)
        engine.store(engine.prepare(req), openai_chat_response)
        engine.lookup(engine.prepare(req))
        engine.lookup(engine.prepare(req))
        stats = engine.get_stats()
        assert stats.total_requests == 2
        assert stats.cache_hits == 2
