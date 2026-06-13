# Contributing to Khazad

Thanks for considering a contribution! This document covers everything you need to get a change merged.

## Development setup

Requirements: Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/GuglielmoCerri/khazad.git
cd khazad
uv sync --group dev
```

`uv sync` creates `.venv` and installs the project in editable mode — nothing else to do.

A running Redis is **not** required for the test suite (fakes and mock transports are used everywhere). For manual end-to-end testing against a real instance:

```bash
docker run -d --name redis8 -p 6379:6379 redis:8
```

## Running tests

```bash
uv run python -m pytest tests/ -q          # full suite
uv run python -m pytest tests/unit/ -q     # unit tests only
uv run python -m pytest -m "not stress"    # skip stress tests
```

The suite must be green before any PR. New behavior needs new tests:

- **Parsers** → `tests/unit/test_parsers/` (include an SSE round-trip test if the parser supports streaming: `stream_chunks` → `response_from_stream` must preserve content)
- **Cache logic** → `tests/unit/test_engine.py`
- **Transport / interception** → `tests/integration/`

## Lint and format

```bash
uv run python -m ruff check . --fix
uv run python -m ruff format .
```

CI rejects unformatted code. Configuration lives in `pyproject.toml` (line length 99, target py310).

## Architecture ground rules

Read `CLAUDE.md` for the full picture. The non-negotiables:

1. **One entry point.** All cache logic lives in the `Khazad` class — do not introduce a separate engine, orchestrator, or config object.
2. **No pydantic.** Validation is inline in `Khazad.__init__`. Plain dataclasses for models.
3. **Ports & Adapters.** New providers implement `ProviderParser` (`khazad/ports/parser.py`); new storage backends implement `VectorStore`; new embedders implement `Embedder`. Adapters never import other adapters.
4. **Parse once.** A request body is JSON-parsed exactly once (`parse_request`); the embedding is computed at most once per request. Don't add code paths that re-parse or re-embed.
5. **The cache stores canonical JSON only.** Streamed responses must be reconstructed via `response_from_stream` before storing — never cache raw SSE bytes.
6. **Python 3.10 compatibility.** No 3.11+ stdlib APIs (`tomllib`, `StrEnum`, `asyncio.timeout`, exception groups...). Use `from __future__ import annotations` in every module.

## Adding a new provider parser

1. Create `khazad/adapters/parsers/<provider>.py` implementing `ProviderParser`:
   - `can_handle(url)` — match by URL path suffix when the API is host-agnostic (proxies!), by host only when the schema is unique to one vendor.
   - `parse_request(request)` — return a `ParsedRequest(prompt, model, stream)`. The prompt must include the **full conversation** (`role: text` lines), and raise `ValueError` for bodies you can't understand.
   - Override `stream_chunks` / `response_from_stream` only if the provider streams over SSE.
2. Register it in the `_parsers` list in `khazad/khazad.py`.
3. Add request/response fixtures in `tests/conftest.py`, unit tests in `tests/unit/test_parsers/`, and an interception test in `tests/integration/`.
4. Document the URL pattern in the README "Supported Providers" table.

## Pull requests

- Branch from `main`; one logical change per PR.
- Subject line in imperative mood ("Add Mistral parser", not "Added...").
- Explain *why* in the body if it isn't obvious from the diff.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Don't bump the version — maintainers handle releases.

## Reporting bugs

Open an issue with: Python version, khazad version, the SDK and provider you're calling, a minimal reproduction, and (if relevant) `log_level="DEBUG"` output. For suspected cache-correctness issues, include the two prompts involved and your `threshold`.

## Security

Don't open public issues for security problems — see the contact in `pyproject.toml`.
