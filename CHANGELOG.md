# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `hosts` opt-in allowlist (exact hosts and `*.` wildcard subdomains) — restricts interception to explicitly listed endpoints.

## [0.1.0] - 2026-06-10

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

[Unreleased]: https://github.com/GuglielmoCerri/khazad/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/GuglielmoCerri/khazad/releases/tag/v0.1.0
