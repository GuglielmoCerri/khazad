"""Redis 8 Vector Set adapter using raw Vector Set commands."""

from __future__ import annotations

import logging

import redis

from khazad.ports.store import VectorStore

logger = logging.getLogger("khazad")


class RedisVectorStore(VectorStore):
    """VectorStore backed by Redis 8 Vector Sets via redis-py ≥ 8.0.

    One vector set per scope (provider host + model) under
    ``{namespace}:vset:{scope}``; response bodies under
    ``{namespace}:resp:{key}``.
    """

    def __init__(self, redis_url: str, namespace: str = "khazad") -> None:
        self._redis_url = redis_url
        self._namespace = namespace
        self._client: redis.Redis | None = None

    def _get_client(self) -> redis.Redis:
        """Lazily create and return the Redis client."""
        if self._client is None:
            self._client = redis.from_url(self._redis_url, decode_responses=False)
        return self._client

    def _vset_key(self, scope: str) -> str:
        return f"{self._namespace}:vset:{scope}"

    def _resp_key(self, key: str) -> str:
        return f"{self._namespace}:resp:{key}"

    def ping(self) -> None:
        """Verify the Redis connection is alive. Raises on failure."""
        self._get_client().ping()

    def search(
        self, scope: str, embedding: list[float], threshold: float
    ) -> tuple[str, float] | None:
        """Search the scope's Vector Set for the nearest neighbor above threshold.

        Uses raw ``execute_command`` instead of ``client.vset().vsim()`` to stay
        independent of redis-py's ``parse_vsim_result`` callback, which misparses
        RESP3 dict responses when the WITHSCORES option flag is not propagated
        (present in 8.0.0b2; callback unchanged in 8.0.0 GA).
        ``_parse_vsim_response`` handles both RESP2 and RESP3 shapes directly.
        """
        try:
            response = self._get_client().execute_command(
                "VSIM",
                self._vset_key(scope),
                "VALUES",
                len(embedding),
                *embedding,
                "WITHSCORES",
                "COUNT",
                1,
            )
            element, score = _parse_vsim_response(response)
            if element is not None and score >= threshold:
                return element, score
            return None
        except Exception:
            logger.exception("Redis VSIM failed")
            return None

    def store(
        self,
        scope: str,
        key: str,
        embedding: list[float],
        response_data: bytes,
        ttl: int | None = None,
    ) -> None:
        """Store the embedding and response body atomically."""
        pipe = self._get_client().pipeline(transaction=False)
        pipe.execute_command(
            "VADD", self._vset_key(scope), "VALUES", len(embedding), *embedding, key
        )
        pipe.set(self._resp_key(key), response_data, ex=ttl)
        pipe.execute()

    def get_response(self, key: str) -> bytes | None:
        """Retrieve the cached response bytes for the given key."""
        return self._get_client().get(self._resp_key(key))  # type: ignore[return-value]

    def delete(self, scope: str, key: str) -> None:
        """Remove a cached entry from both the Vector Set and response store."""
        pipe = self._get_client().pipeline(transaction=False)
        pipe.execute_command("VREM", self._vset_key(scope), key)
        pipe.delete(self._resp_key(key))
        pipe.execute()

    def flush(self) -> None:
        """Remove all entries (vector sets and responses) in this namespace."""
        client = self._get_client()
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor=cursor, match=f"{self._namespace}:*", count=500)
            if keys:
                client.delete(*keys)
            if cursor == 0:
                break

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            self._client.close()
            self._client = None


def _parse_vsim_response(response) -> tuple[str | None, float]:
    """Extract (element, score) from raw VSIM WITHSCORES response.

    Redis 8 returns either:
    - RESP3: dict ``{element: score, ...}``
    - RESP2: flat list ``[element, score, element, score, ...]``
    Element may be bytes or str; score may be bytes, str, float, or int.
    """
    if not response:
        return None, 0.0

    if isinstance(response, dict):
        element, score = next(iter(response.items()))
    elif isinstance(response, (list, tuple)):
        if len(response) < 2:
            return None, 0.0
        element, score = response[0], response[1]
    else:
        return None, 0.0

    if isinstance(element, bytes):
        element = element.decode()
    if isinstance(score, bytes):
        score = score.decode()
    return element, float(score)
