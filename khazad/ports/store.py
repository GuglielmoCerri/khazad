"""Abstract interface for vector storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Port defining the contract for vector similarity storage.

    Every entry lives in a *scope* (one vector set per provider host and
    model), so semantically similar prompts sent to different models can
    never serve each other's responses.
    """

    @abstractmethod
    def search(
        self, scope: str, embedding: list[float], threshold: float
    ) -> tuple[str, float] | None:
        """Search the scope for the nearest vector above the similarity threshold.

        Returns a tuple of (cache_key, similarity_score) or None if no match.
        """

    @abstractmethod
    def store(
        self,
        scope: str,
        key: str,
        embedding: list[float],
        response_data: bytes,
        ttl: int | None = None,
    ) -> None:
        """Store an embedding and its associated response data."""

    @abstractmethod
    def get_response(self, key: str) -> bytes | None:
        """Retrieve the cached response data for a given key."""

    @abstractmethod
    def delete(self, scope: str, key: str) -> None:
        """Remove a specific cached entry by key."""

    @abstractmethod
    def flush(self) -> None:
        """Remove all cached entries."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the store."""
