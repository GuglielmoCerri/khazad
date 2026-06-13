"""Abstract interface for text embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Port defining the contract for text-to-vector embedding."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate a vector embedding for the given text."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
