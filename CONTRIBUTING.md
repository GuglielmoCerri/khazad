# Contributing

Khazad changes should be tested against the failure mode they can realistically
affect. The project has two regression tracks: correctness and concurrency.
Please include the commands you ran, the Python version, the provider SDK and
endpoint, and any notable failures in the PR or commit notes.

Do not send PRs affecting one or more provider parsers, the cache lifecycle, or
the transport without checking that the cache still replays the *right* response
for the *right* scope. The only acceptable change that lowers hit-rate is when an
important correctness bug is fixed (e.g. a cross-serve between two models) and
the narrower matching is the cost of correctness.

## Correctness Regression Tests

Set up the dev environment first (Python >= 3.10 and
[uv](https://docs.astral.sh/uv/)):

```sh
git clone https://github.com/GuglielmoCerri/khazad.git
cd khazad
uv sync --group dev
```

`uv sync` creates `.venv` and installs the project editable — nothing else to do.
No Redis or API keys are needed: the suite uses `FakeEmbedder`,
`InMemoryVectorStore`, and `httpx.MockTransport` everywhere.

Running the suite without a selector is the full check:

```sh
uv run python -m pytest tests/ -q
```

Useful narrower checks:

```sh
uv run python -m pytest tests/unit/ -q                 # pure logic, no I/O
uv run python -m pytest tests/unit/test_parsers/ -q    # per-provider parsers
uv run python -m pytest tests/integration/ -q          # full lifecycle
uv run python -m pytest -m "not stress"                # skip stress tests
uv run python -m pytest -m "not integration"           # skip Redis-marked tests
```

What they cover:

- `tests/unit/test_engine.py`: the `prepare → lookup → store` lifecycle, host+model
  scoping, stats accounting, and embedding reuse. This is the best quick check for
  cache-logic changes.
- `tests/unit/test_parsers/`: per-provider request parsing and SSE round-trips —
  `stream_chunks` → `response_from_stream` must preserve content. This catches
  parser, conversation-flattening, and stream-reconstruction regressions.
- `tests/integration/test_streaming.py`: SSE capture and replay for both sync and
  async clients; a partial/aborted stream must reconstruct to `None`.
- `tests/integration/test_redis_store.py`: the Redis adapter against a mocked
  redis-py client (the store is sync — never `AsyncMock`), including RESP3/RESP2
  VSIM parsing and TTL pruning of orphaned vectors.
- `tests/integration/` (providers): full interception lifecycle per provider with
  `httpx.MockTransport`. This is the best check for transport and end-to-end changes.

New behavior needs new tests. For tests, `Khazad` accepts `_vector_store` and
`_embedder_instance` keyword args (both or neither) to inject the fakes from
`tests/conftest.py` and bypass Redis/transport patching.

## Quality Checks For Parser And Transport Changes

CI rejects unformatted or unlinted code. Run both before every PR (config lives in
`pyproject.toml`: line length 99, target py310):

```sh
uv run python -m ruff check . --fix
uv run python -m ruff format .
```

The non-negotiable architecture rules (full picture in `AGENT.md`):

1. **One entry point.** All cache logic lives in the `Khazad` class — no separate
   engine, orchestrator, or config object.
2. **No pydantic.** Validation is inline in `Khazad.__init__`; plain dataclasses for models.
3. **Ports & Adapters.** New providers implement `ProviderParser`
   (`khazad/ports/parser.py`); new backends implement `VectorStore`; new embedders
   implement `Embedder`. Adapters never import other adapters.
4. **Parse once, embed once.** The body is JSON-parsed exactly once
   (`parse_request`) and the embedding is computed at most once per request. Don't
   add paths that re-parse or re-embed.
5. **Canonical JSON only.** Streamed responses must be reconstructed via
   `response_from_stream` before storing — never cache raw SSE bytes.
6. **Python 3.10 compatibility.** No 3.11+ stdlib APIs (`tomllib`, `StrEnum`,
   `asyncio.timeout`, exception groups...); `from __future__ import annotations` in
   every module.

To add a new provider parser:

1. Create `khazad/adapters/parsers/<provider>.py` implementing `ProviderParser`:
   - `can_handle(url)` — match by URL path suffix when the API is host-agnostic
     (proxies!), by host only when the schema is unique to one vendor.
   - `parse_request(request)` — return a `ParsedRequest(prompt, model, stream)`. The
     prompt must include the **full conversation** (`role: text` lines); raise
     `ValueError` for bodies you can't understand.
   - Override `stream_chunks` / `response_from_stream` only if the provider streams
     over SSE.
2. Register it in the `_parsers` list in `khazad/khazad.py`.
3. Add fixtures in `tests/conftest.py`, unit tests in `tests/unit/test_parsers/`,
   and an interception test in `tests/integration/`.
4. Document the URL pattern in the README "Supported Providers" table.

## Concurrency Regression Tests

Thread-safety and concurrent-access tests live under `tests/stress/` behind the
`stress` marker (they're skipped by `-m "not stress"`). Run them explicitly when
touching the transport, stats counters, or any shared state:

```sh
uv run python -m pytest tests/stress/ -q
```

## Live Provider Checks

For changes that need a real endpoint (a new provider, a wire-format fix, an
embedder), use the scripts in `examples/`. They call real provider SDKs, so they
live in a separate `examples` dependency group (`openai`, `anthropic`,
`google-genai`, `azure-identity`). Install it only when needed and run against a
live endpoint (Redis 8 + provider credentials required):

```sh
docker run -d --name redis8 -p 6379:6379 redis:8
uv run --group examples python -P examples/azure_openai.py
```

## Pull requests

- Branch from `main`; one logical change per PR.
- Subject line in imperative mood ("Add Mistral parser", not "Added...").
- Explain *why* in the body if it isn't obvious from the diff.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Don't bump the version — maintainers handle releases.

## Reporting bugs

Open an issue with: Python version, khazad version, the SDK and provider you're
calling, a minimal reproduction, and (if relevant) `log_level="DEBUG"` output. For
suspected cache-correctness issues, include the two prompts involved and your
`threshold`.

Don't open public issues for security problems — see the contact in `pyproject.toml`.