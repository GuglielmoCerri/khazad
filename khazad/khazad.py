"""Khazad — the single class for transparent semantic caching of LLM API calls."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from khazad._models import CacheHit, CacheScope, Stats
from khazad.adapters.parsers.anthropic import AnthropicParser
from khazad.adapters.parsers.gemini import GeminiParser
from khazad.adapters.parsers.openai import OpenAIParser
from khazad.adapters.parsers.openai_responses import OpenAIResponsesParser

if TYPE_CHECKING:
    import httpx

    from khazad.ports.embedder import Embedder
    from khazad.ports.parser import ProviderParser
    from khazad.ports.store import VectorStore


@dataclass(slots=True)
class PreparedRequest:
    """A parsed LLM request plus everything needed to look it up and store it.

    The embedding is computed lazily and memoized, so a miss that later
    stores its response embeds the prompt exactly once.
    """

    parser: ProviderParser
    prompt: str
    scope: str
    stream: bool
    embedding: list[float] | None = field(default=None, repr=False)


class Khazad:
    """Transparent semantic cache for LLM API calls.

    Instantiating this class activates the HTTP transport patch and wires all
    internal components. Use :func:`khazad.init` for the module-level singleton
    API, or create a ``Khazad`` instance directly for explicit lifecycle control::

        cache = Khazad(redis_url="redis://localhost:6379", threshold=0.92)
        # ... run your app ...
        cache.stop()

    By default each ``(host, model)`` pair gets its own cache scope. Pass
    ``cache_scope=CacheScope.HOST`` (or the string ``"host"``) to scope by host
    only, so every model or deployment on the same provider shares one vector
    set. This is safe only because the response format is identical within a
    provider; different providers stay isolated because the scope always
    includes the host.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        threshold: float = 0.90,
        ttl: int | None = None,
        namespace: str = "khazad",
        embedder: str = "huggingface",
        embedding_model: str = "redis/langcache-embed-v2",
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
        hosts: list[str] | None = None,
        *,
        cache_scope: CacheScope | Literal["model", "host"] = CacheScope.MODEL,
        _vector_store: VectorStore | None = None,
        _embedder_instance: Embedder | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if ttl is not None and ttl < 1:
            raise ValueError("ttl must be >= 1 or None")
        if hosts is not None and not hosts:
            raise ValueError("hosts must be a non-empty list or None (None = all hosts)")
        if (_vector_store is None) != (_embedder_instance is None):
            raise ValueError("_vector_store and _embedder_instance must be provided together")
        try:
            cache_scope = CacheScope(cache_scope)
        except ValueError:
            raise ValueError("cache_scope must be 'model' or 'host'") from None

        self._threshold = threshold
        self._ttl = ttl
        self._hosts = None if hosts is None else {h.lower() for h in hosts}
        self._cache_scope = cache_scope
        self._stats = Stats()
        self._lock = threading.Lock()

        self.logger = logging.getLogger("khazad")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
            self.logger.propagate = False

        self._parsers: list[ProviderParser] = [
            OpenAIParser(),
            OpenAIResponsesParser(),
            AnthropicParser(),
            GeminiParser(),
        ]

        if _vector_store is not None and _embedder_instance is not None:
            # Test seam — injected fakes, no Redis and no transport patch.
            self._store: VectorStore = _vector_store
            self._embedder: Embedder = _embedder_instance
        else:
            self._activate(redis_url, namespace, embedder, embedding_model)

        self._active = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Deactivate Khazad — restores original HTTP transports."""
        if not self._active:
            self.logger.warning("[Khazad] Not currently active")
            return

        from khazad._transport import uninstall

        uninstall()
        self._active = False
        self._store.close()
        self.logger.info("[Khazad] Stopped — original HTTP transports restored")

    def get_stats(self) -> Stats:
        """Return a snapshot of the current cache statistics."""
        with self._lock:
            return Stats(
                total_requests=self._stats.total_requests,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                total_hit_similarity=self._stats.total_hit_similarity,
            )

    def flush(self) -> None:
        """Clear all cached entries and reset stats."""
        self._store.flush()
        with self._lock:
            self._stats = Stats()
        self.logger.info("[Khazad] Cache flushed")

    def is_active(self) -> bool:
        """Return True if this cache is running (not stopped)."""
        return self._active

    # ------------------------------------------------------------------
    # Cache operations (used by the transport layer)
    # ------------------------------------------------------------------

    def prepare(self, request: httpx.Request) -> PreparedRequest | None:
        """Parse an outgoing request once; None means pass through uncached."""
        if not self._host_allowed(request.url.host):
            return None
        parser = next((p for p in self._parsers if p.can_handle(request.url)), None)
        if parser is None:
            return None
        try:
            parsed = parser.parse_request(request)
        except Exception:
            self.logger.debug("[Khazad] Unparseable request to %s — passing through", request.url)
            return None
        scope = (
            request.url.host
            if self._cache_scope is CacheScope.HOST
            else f"{request.url.host}/{parsed.model or 'default'}"
        )
        return PreparedRequest(
            parser=parser,
            prompt=parsed.prompt,
            scope=scope,
            stream=parsed.stream,
        )

    def lookup(self, prepared: PreparedRequest) -> CacheHit | None:
        """Attempt to serve a prepared request from the cache."""
        with self._lock:
            self._stats.total_requests += 1

        start = time.perf_counter()
        result = self._store.search(prepared.scope, self._embed(prepared), self._threshold)

        if result is not None:
            key, similarity = result
            response_data = self._store.get_response(key)
            if response_data is not None:
                elapsed = (time.perf_counter() - start) * 1000
                self.logger.info(
                    "[Khazad] CACHE HIT - Similarity: %.2f - Latency: %.0fms", similarity, elapsed
                )
                with self._lock:
                    self._stats.cache_hits += 1
                    self._stats.total_hit_similarity += similarity
                return CacheHit(
                    key=key,
                    similarity=similarity,
                    response_data=response_data,
                    latency_ms=elapsed,
                )
            # Response expired (TTL) but its vector survived — prune it.
            self._store.delete(prepared.scope, key)

        self.logger.info("[Khazad] CACHE MISS - Forwarding to API")
        with self._lock:
            self._stats.cache_misses += 1
        return None

    def store(self, prepared: PreparedRequest, response_data: bytes) -> None:
        """Cache a response for future semantic lookups."""
        key = self._make_key(f"{prepared.scope}:{prepared.prompt}")
        self._store.store(prepared.scope, key, self._embed(prepared), response_data, ttl=self._ttl)
        self.logger.debug("[Khazad] Stored response under key %s", key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _host_allowed(self, host: str | None) -> bool:
        """Check the host against the opt-in allowlist (None = everything)."""
        if self._hosts is None:
            return True
        if not host:
            return False
        host = host.lower()
        return any(
            host == allowed or (allowed.startswith("*.") and host.endswith(allowed[1:]))
            for allowed in self._hosts
        )

    def _embed(self, prepared: PreparedRequest) -> list[float]:
        """Embed the prompt once per request, caching the vector on it."""
        if prepared.embedding is None:
            prepared.embedding = self._embedder.embed(prepared.prompt)
        return prepared.embedding

    @staticmethod
    def _make_key(text: str) -> str:
        """Derive a deterministic cache key from scoped prompt text."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _activate(
        self,
        redis_url: str,
        namespace: str,
        embedder: str,
        embedding_model: str,
    ) -> None:
        """Build all components and install the HTTP transport patch."""
        from khazad._transport import install
        from khazad.adapters.redis.store import RedisVectorStore

        self._embedder = self._build_embedder(embedder, embedding_model)
        self._store = RedisVectorStore(redis_url=redis_url, namespace=namespace)

        try:
            self._store.ping()
        except Exception as exc:
            raise ConnectionError(
                f"[Khazad] Cannot connect to Redis at {redis_url!r}: {exc}"
            ) from exc

        install(self)
        self.logger.info(
            "[Khazad] Initialized — threshold=%.2f, embedder=%s", self._threshold, embedder
        )

    @staticmethod
    def _build_embedder(embedder: str, embedding_model: str) -> Embedder:
        """Construct the configured embedding backend."""
        if embedder == "huggingface":
            from khazad.adapters.embedders.huggingface import HuggingFaceEmbedder

            return HuggingFaceEmbedder(model_name=embedding_model)

        if embedder == "openai":
            from khazad.adapters.embedders.openai import OpenAIEmbedder

            return OpenAIEmbedder(model=embedding_model)

        raise ValueError(f"Unknown embedder: {embedder!r}. Use 'huggingface' or 'openai'.")
