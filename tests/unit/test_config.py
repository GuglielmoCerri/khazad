"""Unit tests for Khazad initialization validation."""

from __future__ import annotations

import pytest

from khazad._models import CacheScope
from khazad.khazad import Khazad


# Minimal fakes to construct Khazad without Redis
class _FakeEmbedder:
    def embed(self, text):
        return [0.0]

    @property
    def dimension(self):
        return 1


class _FakeStore:
    def search(self, *a, **kw):
        return None

    def store(self, *a, **kw):
        pass

    def get_response(self, *a, **kw):
        return None

    def delete(self, *a, **kw):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _make(**kwargs) -> Khazad:
    defaults = {
        "_vector_store": _FakeStore(),
        "_embedder_instance": _FakeEmbedder(),
    }
    defaults.update(kwargs)
    return Khazad(**defaults)


class TestKhazadDefaults:
    """Verify default values when creating a Khazad instance."""

    def test_default_threshold(self):
        k = _make()
        assert k._threshold == 0.90

    def test_default_ttl_is_none(self):
        k = _make()
        assert k._ttl is None

    def test_default_cache_scope_is_model(self):
        k = _make()
        assert k._cache_scope is CacheScope.MODEL


class TestKhazadValidation:
    """Verify initialization validation rules."""

    def test_threshold_must_be_between_0_and_1(self):
        with pytest.raises(ValueError):
            _make(threshold=1.5)

    def test_threshold_negative_rejected(self):
        with pytest.raises(ValueError):
            _make(threshold=-0.1)

    def test_threshold_zero_allowed(self):
        k = _make(threshold=0.0)
        assert k._threshold == 0.0

    def test_threshold_one_allowed(self):
        k = _make(threshold=1.0)
        assert k._threshold == 1.0

    def test_ttl_must_be_positive(self):
        with pytest.raises(ValueError):
            _make(ttl=0)

    def test_ttl_negative_rejected(self):
        with pytest.raises(ValueError):
            _make(ttl=-10)

    def test_cache_scope_accepts_enum(self):
        k = _make(cache_scope=CacheScope.HOST)
        assert k._cache_scope is CacheScope.HOST

    def test_cache_scope_accepts_string(self):
        k = _make(cache_scope="host")
        assert k._cache_scope is CacheScope.HOST

    def test_invalid_cache_scope_rejected(self):
        with pytest.raises(ValueError):
            _make(cache_scope="banana")

    def test_custom_values(self):
        k = _make(threshold=0.95, ttl=3600)
        assert k._threshold == 0.95
        assert k._ttl == 3600
