"""Khazad — transparent semantic cache for LLM API calls.

"You shall not pass" — Khazad stands between your application and
expensive LLM API calls, turning semantically equivalent requests
away at the bridge.

Usage::

    # Functional singleton API
    import khazad
    khazad.init(redis_url="redis://localhost:6379", threshold=0.92)
    khazad.stop()

    # Or manage the instance explicitly
    from khazad import Khazad
    cache = Khazad(redis_url="redis://localhost:6379", threshold=0.92)
    cache.stop()
"""

from __future__ import annotations

import logging
from typing import Literal

from khazad._models import CacheHit, CacheScope, ParsedRequest, Stats
from khazad.khazad import Khazad

__version__ = "0.1.2"
__all__ = [
    "CacheHit",
    "CacheScope",
    "Khazad",
    "ParsedRequest",
    "Stats",
    "flush",
    "get_stats",
    "init",
    "is_active",
    "stop",
]

# ---------------------------------------------------------------------------
# Module-level singleton — functional interface over a single Khazad instance
# ---------------------------------------------------------------------------

_instance: Khazad | None = None


def init(
    redis_url: str = "redis://localhost:6379",
    threshold: float = 0.90,
    ttl: int | None = None,
    namespace: str = "khazad",
    embedder: str = "huggingface",
    embedding_model: str = "redis/langcache-embed-v2",
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    hosts: list[str] | None = None,
    cache_scope: CacheScope | Literal["model", "host"] = CacheScope.MODEL,
) -> None:
    """Activate the global Khazad singleton."""
    global _instance

    if _instance is not None and _instance.is_active():
        logging.getLogger("khazad").warning("[Khazad] Already initialized, call stop() first")
        return

    _instance = Khazad(
        redis_url=redis_url,
        threshold=threshold,
        ttl=ttl,
        namespace=namespace,
        embedder=embedder,
        embedding_model=embedding_model,
        log_level=log_level,
        hosts=hosts,
        cache_scope=cache_scope,
    )


def stop() -> None:
    """Deactivate the global Khazad singleton."""
    global _instance

    if _instance is None or not _instance.is_active():
        logging.getLogger("khazad").warning("[Khazad] Not currently active")
        return

    _instance.stop()
    _instance = None


def get_stats() -> dict:
    """Return current cache performance metrics as a dictionary."""
    if _instance is None:
        return Stats().to_dict()
    return _instance.get_stats().to_dict()


def flush() -> None:
    """Clear all cached entries from Redis."""
    if _instance is None:
        logging.getLogger("khazad").warning("[Khazad] Not initialized, nothing to flush")
        return
    _instance.flush()


def is_active() -> bool:
    """Return True if Khazad is currently intercepting HTTP traffic."""
    return _instance is not None and _instance.is_active()
