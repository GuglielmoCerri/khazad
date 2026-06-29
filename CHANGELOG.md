# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-06-29

### Added

- `examples` dependency group in `pyproject.toml` (`OpenAI`, `Anthropic`, `google-genai`, `azure-identity`) so the provider example scripts run with `uv run --group examples`.
- Extended and expanded the per-provider example scripts.

### Changed

- README images now use absolute `raw.githubusercontent.com` URLs so they render on the PyPI project page.

## [0.1.1] - 2026-06-24

### Changed

- Replaced the `shared_models` parameter with `cache_scope` (a `CacheScope` enum — `MODEL` by default, `HOST` to opt in) to control cache partitioning.
- Rewrote the README and added a flow diagram illustrating the request lifecycle.

### Added

- Standalone example scripts for each supported provider, including streaming usage.

### Fixed

- Streaming cache misses are now correctly tee'd, reconstructed into canonical JSON at stream end, and cached.

## [0.1.0] - 2026-06-13

First public release.

### Added

- Transparent semantic cache for LLM API calls via `httpx` transport patching — zero changes to application code.
- Module-level singleton API (`khazad.init()` / `stop()` / `get_stats()` / `flush()` / `is_active()`) and explicit `Khazad` class with the same surface.
- Redis 8 Vector Sets backend (`VADD` / `VSIM`), one vector set per `(provider host, model)` scope so different models never cross-serve.
- Provider parsers: OpenAI Chat Completions (incl. Azure OpenAI and any OpenAI-compatible proxy), OpenAI Responses API, Anthropic Messages, Google Gemini `generateContent`.
- Conversation-aware matching: the full message list (system, user, assistant) is embedded, not just the last user turn.
- Streaming support both ways:
  - cache hits replay as SSE streams for sync and async clients;
  - streaming cache misses are tee'd to the client with no added latency, reconstructed into canonical JSON at stream end, and cached (aborted streams are never cached).
- Embedding backends: HuggingFace `sentence-transformers` (default, local) and OpenAI Embeddings (optional extra `khazad[openai-embeddings]`).
- Configurable similarity threshold, TTL with automatic pruning of orphaned vectors, Redis key namespace, log level.
- Thread-safe hit/miss statistics (`total_requests`, `cache_hits`, `cache_misses`, `hit_rate`, `avg_hit_similarity`).
- `hosts` opt-in allowlist (exact hosts and `*.` wildcard subdomains) — restricts interception to explicitly listed endpoints.

[0.1.2]: https://github.com/GuglielmoCerri/khazad/releases/tag/v0.1.2
[0.1.1]: https://github.com/GuglielmoCerri/khazad/releases/tag/v0.1.1
[0.1.0]: https://github.com/GuglielmoCerri/khazad/releases/tag/v0.1.0
