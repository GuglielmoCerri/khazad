"""HuggingFace embedding adapter using sentence-transformers."""

from __future__ import annotations

import logging
import threading
from functools import cached_property

from sentence_transformers import SentenceTransformer

from khazad.ports.embedder import Embedder

logger = logging.getLogger("khazad")


class HuggingFaceEmbedder(Embedder):
    """Embedder backed by a local sentence-transformers model."""

    def __init__(self, model_name: str = "redis/langcache-embed-v2") -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._load_lock = threading.Lock()

    def _get_model(self) -> SentenceTransformer:
        """Lazily load the model, guarding against concurrent double-loads."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    logger.info("[Khazad] Loading embedding model: %s", self._model_name)
                    self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Generate a normalized embedding vector for the given text."""
        vector = self._get_model().encode(text, normalize_embeddings=True)
        return vector.tolist()

    @cached_property
    def dimension(self) -> int:
        """Return the dimensionality of this model's embeddings."""
        return self._get_model().get_sentence_embedding_dimension()  # type: ignore[return-value]
