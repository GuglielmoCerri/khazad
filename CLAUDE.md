# CLAUDE.md - Khazad Project Context

## Frequently Used Commands

```bash
# Setup
uv sync --group dev        # Install all dependencies (creates .venv automatically)

# Testing (no Redis or API keys needed — fakes and mock transports)
uv run python -m pytest tests/ -q                              # Full suite
uv run python -m pytest tests/unit/ -q                         # Unit tests only
uv run python -m pytest -m "not stress"                        # Skip stress tests

# Lint / format
uv run python -m ruff check . --fix    # Lint with auto-fix
uv run python -m ruff format .         # Format code

# Quick smoke test (requires Redis 8 + endpoint credentials)
uv run python examples/azure_openai.py
```

## Important Architectural Patterns

### Single Entry Point — `Khazad` Class
Everything goes through one class. There is no separate engine, config model, or orchestrator.
`Khazad` owns the embedder, vector store, parsers, stats, and cache logic directly.

```python
from khazad import Khazad

cache = Khazad(redis_url="redis://localhost:6379", threshold=0.92)
# ... all LLM HTTP traffic is now cached ...
cache.stop()
```

A module-level singleton API (`khazad.init()` / `khazad.stop()`) wraps `Khazad` for convenience.

### Request lifecycle: prepare → lookup → store
The transport calls `Khazad.prepare(request)` exactly once per request. It returns a
`PreparedRequest` (parser, prompt, scope, stream flag) or `None` for pass-through.
The request body is JSON-parsed **once**; the embedding is computed lazily and memoized
on the `PreparedRequest`, so a miss that later stores its response never re-embeds.

- `prepare(request) -> PreparedRequest | None` — parser matching + body parsing
- `lookup(prepared) -> CacheHit | None` — embed, VSIM search, stats
- `store(prepared, response_bytes)` — reuses the memoized embedding

`prepare()` also applies the opt-in `hosts=[...]` allowlist (exact match or `*.suffix`
wildcard, case-insensitive) — non-allowed hosts pass through untouched.

A temperature-based gate (`cache_only_deterministic`) was evaluated and deliberately
**rejected**: GPT-5-family and o-series models hard-reject any `temperature` other than
the default 1.0 (400 error), so gating on `temperature=0` would make flagship models
permanently uncacheable. Do not reintroduce it.

Unparseable, unmatched, or non-allowlisted requests are **not counted** in stats.

### Cache scope: host + model
`scope = f"{host}/{model or 'default'}"`. Each scope gets its own Redis vector set,
so the same prompt sent to `gpt-4o` and `gpt-4o-mini` can never cross-serve.
The prompt text embedded is the **full conversation** (`role: text` lines, including
system), not just the last user message — prevents multi-turn collisions.

The opt-in `cache_scope` parameter (a `CacheScope` enum — `MODEL` by default, `HOST`
to opt in; keyword-only on `Khazad`, also on `khazad.init`) controls this. Pass
`cache_scope=CacheScope.HOST` (or the string `"host"`) to collapse the scope to `host`
only, so every model/deployment on the same provider shares one vector set. The host
always stays in the scope, so different providers (Azure OpenAI vs Gemini) remain
isolated and a response is never replayed to a client expecting a different wire
format. Use it only for format-compatible pools (e.g. several Azure OpenAI
deployments, or `gpt-4o` + `gpt-4o-mini`).

### Hexagonal Architecture (Ports & Adapters)
- **Ports** (`khazad/ports/`) — abstract interfaces: `Embedder`, `ProviderParser`, `VectorStore`
- **Adapters** (`khazad/adapters/`) — concrete implementations (Redis, HuggingFace, OpenAI, parsers)
- `ProviderParser` is an ABC with shared concrete helpers (`build_response`, `_sse`,
  `_iter_sse_payloads`, `_flatten_text`) — subclasses implement `can_handle` and
  `parse_request`, and optionally override `stream_chunks` / `response_from_stream`.
- There is **no Azure parser** — Azure OpenAI is covered by `OpenAIParser`'s
  path-suffix matching (`/chat/completions`).

### httpx Transport Monkey-Patching
Khazad intercepts LLM traffic by patching `httpx.Client.__init__` and `httpx.AsyncClient.__init__`
to wrap their transports (`khazad/_transport.py`, `install(cache)` / `uninstall()`).

`install()` is **idempotent for the original-init capture**: only the first call records the
pristine `httpx.*Client.__init__` references. Subsequent calls swap the active cache without
overwriting the originals, so `uninstall()` always restores real httpx.

Transports check `cache.is_active()` on every request — clients created while the patch
was installed stop serving from cache immediately after `stop()`.

### Streaming
- **Hit**: `parser.stream_chunks(cached_json)` is a *sync* generator of SSE frames;
  `_ReplayStream` implements both `SyncByteStream` and `AsyncByteStream`, so sync and
  async clients both replay correctly.
- **Miss**: the upstream SSE body is tee'd through `_SyncTeeStream` / `_AsyncTeeStream`
  with zero added latency. The collected bytes are passed to `parser.response_from_stream(sse)`
  when the stream ends — on natural exhaustion **or** on `close()`/`aclose()`. The latter
  matters because SDKs (e.g. the OpenAI client) break their read loop on the terminal SSE
  sentinel and close the response without driving the byte stream to EOF, so caching on
  natural exhaustion alone would never fire. `response_from_stream` reconstructs the
  **canonical JSON response** only when the capture is complete (OpenAI Chat requires the
  `[DONE]` sentinel, Anthropic requires `message_stop`, Responses requires
  `response.completed`); a partial/aborted stream reconstructs to `None` and is never cached.
- Compressed SSE bodies (`content-encoding != identity`) are passed through uncached.
- Gemini streaming (`:streamGenerateContent`) is not matched at all — pass-through.

### Redis adapter (`khazad/adapters/redis/store.py`)
- One vector set per scope: `{namespace}:vset:{scope}`; bodies at `{namespace}:resp:{key}`.
- `store()` pipelines `VADD` + `SET ex=ttl` (single round-trip).
- VSIM workaround: redis-py's `parse_vsim_result` callback misparses RESP3 dict responses
  when the WITHSCORES option flag isn't propagated (found in 8.0.0b2, callback unchanged
  in 8.0.0 GA — dependency is now `redis>=8.0.0,<9`). `search()` issues a raw
  `execute_command("VSIM", ...)` and `_parse_vsim_response` handles both RESP3 dict and
  RESP2 flat-list shapes.
- TTL: only the response body expires. `Khazad.lookup` prunes the orphaned vector
  (`store.delete(scope, key)`) when the body is gone, then counts a miss.

### Testing with Dependency Injection
For tests, `Khazad` accepts `_vector_store` and `_embedder_instance` keyword args (both or
neither) to bypass Redis and real embedding models:

```python
cache = Khazad(
    threshold=0.99,
    _vector_store=InMemoryVectorStore(),
    _embedder_instance=FakeEmbedder(),
)
```

This skips Redis connection and transport patching entirely (tests call
`install(cache)` / `uninstall()` themselves).

## Critical Rules

### Git Operations
**CRITICAL**: NEVER use `git push` or attempt to push to remote repositories. The user will handle all git push operations.

### Code Quality
**IMPORTANT**: Always run `uv run python -m ruff check . --fix && uv run python -m ruff format .` before committing.

### README.md Maintenance
**IMPORTANT**: DO NOT modify README.md unless explicitly requested.

### No Pydantic
The project deliberately removed `pydantic` as a dependency. Validation is done inline in `Khazad.__init__`.
Do not reintroduce pydantic.

### No Separate Engine Class
All cache logic (lookup, store, stats, key generation) lives in the `Khazad` class.
Do not create a separate `CacheEngine` or orchestrator class.

### Python 3.10 Compatibility
`requires-python = ">=3.10"` (ruff target py310). No 3.11+ stdlib APIs (`tomllib`,
`StrEnum`, `asyncio.timeout`, exception groups). Every module starts with
`from __future__ import annotations`. Dev venv is pinned to 3.13 via `.python-version`.

## Testing Notes

- **Unit tests** use `FakeEmbedder` and `InMemoryVectorStore` from `tests/conftest.py` — no external services needed
- **Integration tests** use `httpx.MockTransport` to simulate LLM APIs — no real API keys needed
- **Redis store tests** (`test_redis_store.py`) mock the redis-py client with plain `Mock`/`MagicMock` (the store is sync — never use `AsyncMock` there)
- **Stress tests** are marked with `@pytest.mark.stress`
- `pytest-asyncio` runs in auto mode (`asyncio_mode = "auto"`)
- `conftest.py` provides fixtures for all provider request/response bodies
- `FakeEmbedder` hashes with sha256 (deterministic across processes — `hash()` is salted)

## Project Structure

```
khazad/
├── __init__.py           # Public API + module-level singleton (init/stop/get_stats/flush)
├── khazad.py             # Khazad class + PreparedRequest — all cache logic
├── _models.py            # Domain models: ParsedRequest, CacheHit, Stats
├── _transport.py         # httpx patch, cached transports, tee/replay streams
├── ports/                # Abstract interfaces (Hexagonal Architecture boundaries)
│   ├── embedder.py       # Embedder ABC (embed, dimension)
│   ├── parser.py         # ProviderParser ABC + shared SSE/text helpers
│   └── store.py          # VectorStore ABC (scope-aware search/store/delete)
└── adapters/             # Concrete implementations
    ├── embedders/
    │   ├── huggingface.py    # HuggingFaceEmbedder (sentence-transformers, free)
    │   └── openai.py         # OpenAIEmbedder (OpenAI API, paid)
    ├── parsers/
    │   ├── openai.py             # OpenAI Chat Completions (+ Azure, proxies)
    │   ├── openai_responses.py   # OpenAI Responses API
    │   ├── anthropic.py          # Anthropic Messages
    │   └── gemini.py             # Google Gemini
    └── redis/
        └── store.py          # RedisVectorStore (Redis 8 Vector Sets)

tests/
├── conftest.py           # FakeEmbedder, InMemoryVectorStore, provider fixtures
├── unit/                 # Pure logic tests (no I/O)
│   ├── test_config.py    # Khazad init validation
│   ├── test_engine.py    # prepare/lookup/store, scoping, stats, embedding reuse
│   ├── test_models.py    # Stats, CacheHit
│   └── test_parsers/     # Per-provider parser tests (incl. SSE round-trips)
├── integration/          # Full lifecycle with mock transports
│   ├── test_end_to_end.py
│   ├── test_interceptor.py
│   ├── test_redis_store.py              # Mocked redis-py client
│   ├── test_streaming.py                # SSE capture + replay, sync & async
│   ├── test_provider_openai.py          # OpenAI Chat + Responses
│   ├── test_provider_azure_openai.py    # Azure deployments
│   ├── test_provider_openai_compat.py   # LiteLLM, vLLM, Ollama
│   ├── test_provider_anthropic.py       # Anthropic Claude
│   └── test_provider_gemini.py          # Google Gemini
└── stress/               # Concurrent access, thread safety
    └── test_concurrent.py

examples/
└── azure_openai.py       # Smoke test against a real endpoint

docs/
└── _static/              # Logos (logo-light.png / logo-dark.png are transparent)
```

## Supported Providers

| Provider | Parser | URL pattern matched |
|---|---|---|
| OpenAI Chat | `OpenAIParser` | any path ending `/chat/completions` |
| OpenAI Responses | `OpenAIResponsesParser` | any path ending `/responses` |
| Azure OpenAI | (covered by `OpenAIParser`) | any path ending `/chat/completions` |
| OpenAI-compat proxies | (covered by `OpenAIParser`) | any path ending `/chat/completions` |
| Anthropic | `AnthropicParser` | `api.anthropic.com/v1/messages` |
| Google Gemini | `GeminiParser` | `generativelanguage.googleapis.com/*:generateContent` |

## Cache Flow

```
Request → prepare(request)
  None → pass-through to real API (not counted in stats)
  PreparedRequest → embed(conversation) → VSIM in scope {host}/{model}
      HIT  → replay cached JSON (or synthesize SSE stream)
      MISS → forward to API
             non-streaming 200 → store JSON body
             SSE 200 → tee stream to client, reconstruct JSON at end, store
```
