"""Shared test fixtures for Khazad."""

from __future__ import annotations

import hashlib
import json

import pytest

from khazad.khazad import Khazad
from khazad.ports.embedder import Embedder
from khazad.ports.store import VectorStore

# ---------------------------------------------------------------------------
# Fake Embedder
# ---------------------------------------------------------------------------


class FakeEmbedder(Embedder):
    """Deterministic embedder for testing — returns a fixed-dimension vector."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        """Hash the text into a reproducible unit vector (stable across processes)."""
        digest = hashlib.sha256(text.encode()).digest()
        raw = [digest[i] for i in range(self._dim)]
        magnitude = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / magnitude for x in raw]

    @property
    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# In-memory VectorStore
# ---------------------------------------------------------------------------


class InMemoryVectorStore(VectorStore):
    """VectorStore that keeps everything in memory — no Redis needed."""

    def __init__(self) -> None:
        self._vectors: dict[str, dict[str, list[float]]] = {}
        self._responses: dict[str, bytes] = {}

    def search(
        self, scope: str, embedding: list[float], threshold: float
    ) -> tuple[str, float] | None:
        best_key: str | None = None
        best_sim = -1.0
        for key, stored in self._vectors.get(scope, {}).items():
            sim = self._cosine(embedding, stored)
            if sim > best_sim:
                best_sim = sim
                best_key = key
        if best_key is not None and best_sim >= threshold:
            return best_key, best_sim
        return None

    def store(
        self,
        scope: str,
        key: str,
        embedding: list[float],
        response_data: bytes,
        ttl: int | None = None,
    ) -> None:
        self._vectors.setdefault(scope, {})[key] = embedding
        self._responses[key] = response_data

    def get_response(self, key: str) -> bytes | None:
        return self._responses.get(key)

    def delete(self, scope: str, key: str) -> None:
        self._vectors.get(scope, {}).pop(key, None)
        self._responses.pop(key, None)

    def flush(self) -> None:
        self._vectors.clear()
        self._responses.clear()

    def close(self) -> None:
        pass

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def memory_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def openai_chat_body() -> bytes:
    """Standard OpenAI chat completion request body."""
    return json.dumps(
        {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        }
    ).encode()


@pytest.fixture
def openai_chat_response() -> bytes:
    """Standard OpenAI chat completion response body."""
    return json.dumps(
        {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Paris is the capital of France."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        }
    ).encode()


@pytest.fixture
def openai_streaming_body() -> bytes:
    """OpenAI chat completion request with stream=true."""
    return json.dumps(
        {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "stream": True,
        }
    ).encode()


@pytest.fixture
def anthropic_chat_body() -> bytes:
    """Standard Anthropic messages request body."""
    return json.dumps(
        {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ],
        }
    ).encode()


@pytest.fixture
def anthropic_chat_response() -> bytes:
    """Standard Anthropic messages response body."""
    return json.dumps(
        {
            "id": "msg_abc123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Paris is the capital of France."}],
            "model": "claude-3-sonnet-20240229",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 15, "output_tokens": 10},
        }
    ).encode()


@pytest.fixture
def gemini_chat_body() -> bytes:
    """Standard Gemini generateContent request body."""
    return json.dumps(
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "What is the capital of France?"}],
                }
            ]
        }
    ).encode()


@pytest.fixture
def openai_responses_body() -> bytes:
    """Standard OpenAI Responses API request body (string input)."""
    return json.dumps(
        {
            "model": "gpt-4o",
            "input": "What is the capital of France?",
        }
    ).encode()


@pytest.fixture
def openai_responses_body_list() -> bytes:
    """OpenAI Responses API request body with a list-of-messages input."""
    return json.dumps(
        {
            "model": "gpt-4o",
            "input": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        }
    ).encode()


@pytest.fixture
def openai_responses_streaming_body() -> bytes:
    """OpenAI Responses API streaming request body."""
    return json.dumps(
        {
            "model": "gpt-4o",
            "input": "What is the capital of France?",
            "stream": True,
        }
    ).encode()


@pytest.fixture
def openai_responses_response() -> bytes:
    """Standard OpenAI Responses API response body."""
    return json.dumps(
        {
            "id": "resp_abc123",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "id": "msg_abc123",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Paris is the capital of France.",
                            "annotations": [],
                        }
                    ],
                }
            ],
        }
    ).encode()


@pytest.fixture
def gemini_chat_response() -> bytes:
    """Standard Gemini generateContent response body."""
    return json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Paris is the capital of France."}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ]
        }
    ).encode()


@pytest.fixture
def make_engine(fake_embedder, memory_store):
    """Factory fixture to create a Khazad instance with fake deps."""

    def _make(threshold: float = 0.90, ttl: int | None = None) -> Khazad:
        return Khazad(
            threshold=threshold,
            ttl=ttl,
            _vector_store=memory_store,
            _embedder_instance=fake_embedder,
        )

    return _make
