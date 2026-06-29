# Agent Notes

`khazad` is a semantic HTTP cache for LLM API traffic. It is not a generic
proxy or a client SDK. It transparently intercepts `httpx` requests to LLM
providers, embeds the conversation, and serves a cached response when a
sufficiently similar prompt has already been answered. The goal is a small,
readable Python codebase with a single entry point, hexagonal boundaries, and
zero added latency on the miss path.

## Goals

- Keep the production path as transparent `httpx` interception: install once,
  and all LLM HTTP traffic is cached without changing call sites.
- Always make sure that the OpenAI Chat, OpenAI Responses, Azure OpenAI,
  Anthropic, and Gemini provider paths are not affected by fixes to other parts
  of the code. A response must never be replayed to a client expecting a
  different wire format.
- Keep the request lifecycle `prepare → lookup → store`: parse the body once,
  embed lazily and memoize on the `PreparedRequest`, and reuse that embedding on
  store so a miss never re-embeds.
- Keep streaming zero-latency on a miss: tee the upstream SSE body through to the
  client and only reconstruct the canonical JSON when the capture is provably
  complete. A partial or aborted stream must never be cached.
- Keep the cache scope `host + model` by default so the same prompt sent to two
  models can never cross-serve; only collapse to `host` when the caller opts in
  for a format-compatible pool.
- Preserve correctness before hit-rate. Do not widen matching with an unexplained
  threshold, scope, or embedding change that could replay a wrong response.

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Try to write elegant
  code in a state of grace. Don't settle for the first thing that comes to mind,
  find the most minimal and best working design. Don't introduce slop: fragile
  code that patches specific cases, dead code, useless code, and code far more
  complicated than it needs to be.
- Comment cache code where the lifecycle, scoping, embedding reuse, TTL/pruning
  policy, or stream reconstruction is not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a scope, threshold, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. The `Khazad` class and the module-level singleton
  (`init`/`stop`/`get_stats`/`flush`) are the surface; CLI/example code should not
  know parser or store internals.
- Do not add permanent semantic variants behind flags. A temperature gate
  (`cache_only_deterministic`) was evaluated and deliberately rejected — GPT-5 and
  o-series models hard-reject non-default temperatures, so it would make flagship
  models permanently uncacheable. Do not reintroduce it.
- **No separate engine class.** All cache logic (lookup, store, stats, key
  generation) lives in the `Khazad` class. Do not create a `CacheEngine` or
  orchestrator.
- **No pydantic.** It was deliberately removed; validation is inline in
  `Khazad.__init__`. Do not reintroduce it.
- **Python 3.10 compatibility.** `requires-python = ">=3.10"` (ruff target py310).
  No 3.11+ stdlib APIs (`tomllib`, `StrEnum`, `asyncio.timeout`, exception groups).
  Every module starts with `from __future__ import annotations`.
- Always run `uv run python -m ruff check . --fix && uv run python -m ruff format .`
  before committing.

## Safety

- **NEVER** use `git push` or attempt to push to remote repositories. The user
  handles all push operations.
- **DO NOT** modify `README.md` unless explicitly requested.
- The httpx patch must stay reversible: `install()` records the pristine
  `__init__` references only on the first call, so `uninstall()` always restores
  real httpx. Do not let a re-install overwrite the captured originals.
- Compressed SSE bodies (`content-encoding != identity`) are passed through
  uncached. Do not try to cache a body you cannot deterministically reconstruct.

## Layout

- `khazad/khazad.py`: the `Khazad` class and `PreparedRequest` — all cache logic
  (prepare/lookup/store, scoping, stats, key generation, validation).
- `khazad/__init__.py`: public API and the module-level singleton wrapper
  (`init`/`stop`/`get_stats`/`flush`).
- `khazad/_models.py`: domain models (`ParsedRequest`, `CacheHit`, `Stats`).
- `khazad/_transport.py`: httpx monkey-patch, cached sync/async transports, and
  the tee/replay byte streams.
- `khazad/ports/`: abstract boundaries — `Embedder`, `ProviderParser`,
  `VectorStore`.
- `khazad/adapters/`: concrete implementations — Redis vector store (Redis 8
  Vector Sets), HuggingFace and OpenAI embedders, and the per-provider parsers
  (OpenAI Chat, OpenAI Responses, Anthropic, Gemini). There is no Azure parser;
  `OpenAIParser` covers it via `/chat/completions` path-suffix matching.
- `tests/`: `unit/` (pure logic, no I/O), `integration/` (full lifecycle with
  `httpx.MockTransport`), and `stress/` (concurrency, thread safety).
- `examples/`: runnable smoke tests against real endpoints.
- `docs/_static/`: logos.

This list is not complete, check the files for more info.

## Testing

Set up with `uv sync --group dev`. The full suite needs no Redis or API keys —
it uses `FakeEmbedder` and `InMemoryVectorStore` from `tests/conftest.py` and
`httpx.MockTransport` to simulate providers. `Khazad` accepts `_vector_store` and
`_embedder_instance` keyword args (both or neither) to inject those fakes and
bypass Redis/transport patching entirely.

```bash
uv run python -m pytest tests/ -q          # Full suite
uv run python -m pytest tests/unit/ -q     # Unit tests only
uv run python -m pytest -m "not stress"    # Skip stress tests
```

At every major change where one of the following could be affected, make sure to:

1. Test the non-streaming path for every provider parser (OpenAI Chat, OpenAI
   Responses, Anthropic, Gemini) and that scoping still isolates models/hosts.
2. Test the streaming path: SSE capture and replay must work for both sync and
   async clients, and a partial/aborted stream must reconstruct to `None`.
3. Test the Redis store against the mocked redis-py client (the store is sync —
   never use `AsyncMock` there), including the RESP3/RESP2 VSIM parsing and TTL
   pruning of orphaned vectors.
4. Run the live smoke test only when intentionally testing against a real
   endpoint; it requires Redis 8 and endpoint credentials:
   `uv run --group examples python -P examples/azure_openai.py`.
