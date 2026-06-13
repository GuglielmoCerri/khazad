"""Tests for the Redis Vector Store adapter.

The redis-py client is replaced with a Mock — no Redis server required.
Real end-to-end behavior against Redis 8 is exercised manually (see README).
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from khazad.adapters.redis.store import RedisVectorStore, _parse_vsim_response

SCOPE = "api.openai.com/gpt-4"


@pytest.fixture
def store() -> RedisVectorStore:
    s = RedisVectorStore(redis_url="redis://fake:6379")
    s._client = MagicMock()
    return s


class TestStore:
    def test_store_pipelines_vadd_and_set(self, store):
        pipe = store._client.pipeline.return_value
        store.store(SCOPE, "key1", [0.1, 0.2, 0.3], b"data", ttl=300)

        vadd_args = pipe.execute_command.call_args[0]
        assert vadd_args[0] == "VADD"
        assert vadd_args[1] == f"khazad:vset:{SCOPE}"
        assert vadd_args[-1] == "key1"

        pipe.set.assert_called_once_with("khazad:resp:key1", b"data", ex=300)
        pipe.execute.assert_called_once()

    def test_store_without_ttl(self, store):
        pipe = store._client.pipeline.return_value
        store.store(SCOPE, "key1", [0.1], b"data", ttl=None)
        pipe.set.assert_called_once_with("khazad:resp:key1", b"data", ex=None)


class TestSearch:
    def test_returns_none_on_empty(self, store):
        store._client.execute_command = Mock(return_value=[])
        assert store.search(SCOPE, [0.1, 0.2], threshold=0.9) is None

    def test_returns_hit_above_threshold_resp2(self, store):
        store._client.execute_command = Mock(return_value=[b"key1", b"0.95"])
        assert store.search(SCOPE, [0.1, 0.2], threshold=0.9) == ("key1", 0.95)

    def test_returns_hit_above_threshold_resp3(self, store):
        store._client.execute_command = Mock(return_value={b"key1": 0.95})
        assert store.search(SCOPE, [0.1, 0.2], threshold=0.9) == ("key1", 0.95)

    def test_returns_none_below_threshold(self, store):
        store._client.execute_command = Mock(return_value=[b"key1", b"0.5"])
        assert store.search(SCOPE, [0.1, 0.2], threshold=0.9) is None

    def test_searches_scoped_vset(self, store):
        store._client.execute_command = Mock(return_value=[])
        store.search(SCOPE, [0.1], threshold=0.9)
        assert store._client.execute_command.call_args[0][1] == f"khazad:vset:{SCOPE}"

    def test_handles_redis_error(self, store):
        store._client.execute_command = Mock(side_effect=ConnectionError("Redis down"))
        assert store.search(SCOPE, [0.1, 0.2], threshold=0.9) is None


class TestGetResponse:
    def test_reads_response_key(self, store):
        store._client.get = Mock(return_value=b"cached")
        assert store.get_response("key1") == b"cached"
        store._client.get.assert_called_once_with("khazad:resp:key1")


class TestDelete:
    def test_removes_vector_and_response(self, store):
        pipe = store._client.pipeline.return_value
        store.delete(SCOPE, "key1")
        pipe.execute_command.assert_called_once_with("VREM", f"khazad:vset:{SCOPE}", "key1")
        pipe.delete.assert_called_once_with("khazad:resp:key1")
        pipe.execute.assert_called_once()


class TestFlush:
    def test_scans_and_deletes_namespace(self, store):
        store._client.scan = Mock(return_value=(0, [b"khazad:resp:key1", b"khazad:vset:s"]))
        store.flush()
        store._client.scan.assert_called_once_with(cursor=0, match="khazad:*", count=500)
        store._client.delete.assert_called_once_with(b"khazad:resp:key1", b"khazad:vset:s")


class TestClose:
    def test_releases_client(self, store):
        client = store._client
        store.close()
        assert store._client is None
        client.close.assert_called_once()

    def test_close_without_client_is_safe(self):
        RedisVectorStore(redis_url="redis://fake:6379").close()


class TestNamespace:
    def test_custom_namespace_in_keys(self):
        s = RedisVectorStore(redis_url="redis://fake:6379", namespace="myapp")
        assert s._vset_key("scope") == "myapp:vset:scope"
        assert s._resp_key("k") == "myapp:resp:k"


class TestParseVsimResponse:
    def test_empty(self):
        assert _parse_vsim_response(None) == (None, 0.0)
        assert _parse_vsim_response([]) == (None, 0.0)

    def test_resp2_flat_list(self):
        assert _parse_vsim_response([b"k", b"0.9"]) == ("k", 0.9)

    def test_resp3_dict(self):
        assert _parse_vsim_response({"k": 0.9}) == ("k", 0.9)

    def test_short_list(self):
        assert _parse_vsim_response([b"k"]) == (None, 0.0)

    def test_unexpected_type(self):
        assert _parse_vsim_response(42) == (None, 0.0)
