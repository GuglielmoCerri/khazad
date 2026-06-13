"""OpenAI embedding adapter (optional paid backend)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from khazad.ports.embedder import Embedder

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger("khazad")


class OpenAIEmbedder(Embedder):
    """Embedder backed by the OpenAI Embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimension: int = 1536,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._dimension = dimension
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Lazily create the OpenAI client; fails clearly if the extra is missing."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "Install the 'openai' package: pip install khazad[openai-embeddings]"
                ) from exc
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, text: str) -> list[float]:
        """Generate an embedding via the OpenAI API."""
        response = self._get_client().embeddings.create(input=text, model=self._model)
        return response.data[0].embedding

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self._dimension
