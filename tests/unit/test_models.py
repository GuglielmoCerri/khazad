"""Unit tests for domain models."""

from __future__ import annotations

from khazad._models import CacheHit, Stats


class TestStats:
    """Verify Stats computation and serialization."""

    def test_initial_stats_are_zero(self):
        s = Stats()
        assert s.total_requests == 0
        assert s.cache_hits == 0
        assert s.cache_misses == 0

    def test_hit_rate_zero_when_no_requests(self):
        s = Stats()
        assert s.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        s = Stats(total_requests=100, cache_hits=75, cache_misses=25)
        assert s.hit_rate == 0.75

    def test_avg_similarity_zero_when_no_hits(self):
        s = Stats(total_requests=10, cache_misses=10)
        assert s.avg_hit_similarity == 0.0

    def test_avg_similarity_calculation(self):
        s = Stats(cache_hits=4, total_hit_similarity=3.6)
        assert s.avg_hit_similarity == 0.9

    def test_to_dict_keys(self):
        s = Stats(total_requests=10, cache_hits=7, cache_misses=3, total_hit_similarity=6.3)
        d = s.to_dict()
        assert set(d.keys()) == {
            "total_requests",
            "cache_hits",
            "cache_misses",
            "hit_rate",
            "avg_hit_similarity",
        }
        assert d["hit_rate"] == 0.7
        assert d["avg_hit_similarity"] == 0.9

    def test_to_dict_rounds_values(self):
        s = Stats(total_requests=3, cache_hits=1, cache_misses=2, total_hit_similarity=0.333)
        d = s.to_dict()
        assert d["hit_rate"] == 0.3333
        assert d["avg_hit_similarity"] == 0.333

    def test_all_hits(self):
        s = Stats(total_requests=50, cache_hits=50, cache_misses=0, total_hit_similarity=47.5)
        assert s.hit_rate == 1.0
        assert s.avg_hit_similarity == 0.95


class TestCacheHit:
    """Verify CacheHit is a frozen data container."""

    def test_immutable(self):
        hit = CacheHit(key="abc", similarity=0.95, response_data=b"data", latency_ms=2.5)
        assert hit.key == "abc"
        assert hit.similarity == 0.95

    def test_fields(self):
        hit = CacheHit(key="k", similarity=0.91, response_data=b"resp", latency_ms=1.0)
        assert hit.response_data == b"resp"
        assert hit.latency_ms == 1.0
