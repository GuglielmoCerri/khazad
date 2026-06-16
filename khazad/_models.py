"""Domain models for parsed requests, cache hits and observability stats."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CacheScope(str, Enum):
    """How cache entries are partitioned within a provider host.

    The provider host is always part of the scope, so a cached response is
    never replayed across providers. This enum only controls whether the
    **model** is also part of the key.

    - :attr:`MODEL` (default) — each ``(host, model)`` pair gets its own vector
      set, so a ``gpt-4o`` answer is never served to a ``gpt-4o-mini`` call.
    - :attr:`HOST` — every model or deployment on the same provider shares one
      vector set. Safe only for format-compatible pools (e.g. several Azure
      OpenAI deployments, or treating ``gpt-4o`` and ``gpt-4o-mini`` as
      interchangeable).

    Members are also plain strings, so ``"model"`` / ``"host"`` are accepted
    anywhere a :class:`CacheScope` is expected.
    """

    MODEL = "model"
    HOST = "host"


@dataclass(frozen=True, slots=True)
class ParsedRequest:
    """Semantic content extracted from a provider request body."""

    prompt: str
    model: str | None = None
    stream: bool = False


@dataclass(frozen=True, slots=True)
class CacheHit:
    """Result of a successful cache lookup."""

    key: str
    similarity: float
    response_data: bytes
    latency_ms: float


@dataclass(slots=True)
class Stats:
    """Observable cache performance metrics."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_hit_similarity: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Return the cache hit ratio as a value between 0 and 1."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def avg_hit_similarity(self) -> float:
        """Return the average cosine similarity of cache hits."""
        if self.cache_hits == 0:
            return 0.0
        return self.total_hit_similarity / self.cache_hits

    def to_dict(self) -> dict:
        """Serialize stats to a plain dictionary."""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
            "avg_hit_similarity": round(self.avg_hit_similarity, 4),
        }
