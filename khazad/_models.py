"""Domain models for parsed requests, cache hits and observability stats."""

from __future__ import annotations

from dataclasses import dataclass


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
