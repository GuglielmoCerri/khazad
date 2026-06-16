<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo-dark.png">
  <img alt="Khazad Logo" src="docs/_static/logo-light.png" width="360px">
</picture>
</p>
<h2 align="center">Khazad — You shall not pass.</h2>

*Transparent, transport-layer semantic cache for LLM API calls powered by Redis Vector Sets.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://github.com/GuglielmoCerri/khazad)
[![Redis 8](https://img.shields.io/badge/Redis-8-red.svg)](https://redis.io/)

Khazad intercepts LLM HTTP traffic at the **transport layer** and serves semantically equivalent requests from a Redis vector cache — with zero changes to your application code.

## How it works

```
Your App (any LLM SDK built on httpx)
        │ HTTP
        ▼
Khazad transport patch
        │ parse once → host + model scope, conversation text, stream flag
        ▼
embed prompt ──► Redis Vector Set (VSIM)
        │
   similarity ≥ threshold? ── yes ──► replay cached response (JSON or SSE)
        │ no
        ▼
   real API call ──► cache response ──► return it unchanged
```

Key properties:

- **Model-aware** — each `(provider host, model)` pair gets its own vector set, so a `gpt-4o` answer is never served to a `gpt-4o-mini` call, no matter how similar the prompt. Set `cache_scope="host"` to scope by **provider host only**, letting every model or deployment on the same provider share one cache (different providers stay isolated — see [Configuration](#configuration)).
- **Conversation-aware** — the whole message list (system, user, assistant) is embedded, not just the last user turn. Two different conversations ending with the same follow-up question ("What about its population?") never collide.
- **Streaming both ways** — cache hits replay as real SSE streams (sync and async clients); cache misses that stream are captured chunk-by-chunk with no added latency and reassembled into a canonical JSON response, so a streamed answer can later serve a non-streamed request and vice versa. Aborted streams are never cached.

## Installation

**From PyPI** (once published):
```bash
uv add khazad
```

For the OpenAI embedding backend (optional):
```bash
uv add khazad[openai-embeddings]
```

**Local / development install:**
```bash
git clone https://github.com/GuglielmoCerri/khazad.git
cd khazad
uv sync --group dev
```

`uv sync` reads `pyproject.toml`, creates `.venv` if it doesn't exist, and installs the project itself in editable mode — no separate `pip install -e .` needed.

To use the local checkout from another project:
```bash
uv add --editable /path/to/khazad
```

## Two ways to use it

### 1. Functional singleton API

The simplest integration — two lines, zero refactoring:

```python
import khazad

khazad.init(redis_url="redis://localhost:6379", threshold=0.90)

# Your existing code runs unchanged — see provider sections below

print(khazad.get_stats())
# {'total_requests': 2, 'cache_hits': 1, 'cache_misses': 1,
#  'hit_rate': 0.5, 'avg_hit_similarity': 0.94}

khazad.stop()
```

Available functions: `init()`, `stop()`, `get_stats()`, `flush()`, `is_active()`

### 2. `Khazad` class (explicit lifecycle)

Use the `Khazad` class directly when you need explicit control over the instance — useful in long-running services, tests, or dependency injection:

```python
from khazad import Khazad

cache = Khazad(
    redis_url="redis://localhost:6379",
    threshold=0.90,
    ttl=3600,
    log_level="DEBUG",
)

print(cache.is_active())   # True
print(cache.get_stats())   # Stats(total_requests=0, ...)
cache.flush()              # clear all cached entries
cache.stop()               # restore original HTTP transports
```

The class exposes the same methods as the functional API: `stop()`, `get_stats()`, `flush()`, `is_active()`.

## Provider Examples

Khazad activates once and intercepts **every** LLM SDK that uses `httpx` underneath — no per-provider wiring needed. Pick the provider you use:

<details>
<summary><b>OpenAI</b> — official SDK against <code>api.openai.com</code></summary>

```python
import os
import time

from openai import OpenAI

from khazad import Khazad

cache = Khazad(redis_url="redis://localhost:6379", threshold=0.90, log_level="DEBUG")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

prompt = "What is the capital of Italy?"

for i in range(2):
    start = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[call {i + 1}] {elapsed:.1f}ms — {response.choices[0].message.content}")

print(cache.get_stats().to_dict())
cache.stop()
```

Matches `*/chat/completions` and `*/responses` paths. Streaming requests also cached.

</details>

<details>
<summary><b>Azure OpenAI</b> — Azure deployments with Entra ID auth via <code>AzureOpenAI</code> SDK</summary>

```python
import os
import time

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from khazad import CacheScope, Khazad

cache = Khazad(
    redis_url="redis://localhost:6379",
    threshold=0.90,
    cache_scope=CacheScope.HOST,
    namespace="azure_openai_example",
)

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
)

prompt = "What is the capital of Spain?"

for i in range(2):
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[call {i + 1}] {elapsed:.1f}ms — {response.choices[0].message.content}")

print(cache.get_stats().to_dict())
cache.stop()
```

Full example: [examples/azure_openai_entra.py](examples/azure_openai_entra.py). It authenticates with Microsoft Entra ID (`DefaultAzureCredential`) — no API key needed — and uses `cache_scope=CacheScope.HOST` so every deployment on the same Azure resource shares one cache. API-key auth works too: Khazad matches the request path (`/chat/completions`), not the auth method or host.

</details>

<details>
<summary><b>OpenAI-compatible proxies</b> — LiteLLM, vLLM, Ollama, Together, Groq, …</summary>

```python
import os

from openai import OpenAI

from khazad import Khazad

cache = Khazad(redis_url="redis://localhost:6379", threshold=0.90)

# Point base_url at any OpenAI-compatible server
client = OpenAI(
    base_url="http://localhost:4000/v1",   # LiteLLM proxy example
    api_key=os.environ.get("LITELLM_KEY", "sk-anything"),
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)

cache.stop()
```

Any host whose URL path ends with `/chat/completions` or `/responses` is cached. Covers vLLM (`http://host:8000/v1/...`), Ollama (`http://localhost:11434/v1/...`), Groq, Together, Mistral, etc.

</details>

<details>
<summary><b>Anthropic</b> — Claude via official SDK</summary>

```python
import os

from anthropic import Anthropic

from khazad import Khazad

cache = Khazad(redis_url="redis://localhost:6379", threshold=0.90)

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

message = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is the capital of Italy?"}],
)
print(message.content[0].text)

cache.stop()
```

Matches `api.anthropic.com/v1/messages`. Streaming responses replayed from cache as SSE.

</details>

<details>
<summary><b>Google Gemini</b> — <code>google-genai</code> SDK</summary>

```python
import os

from google import genai

from khazad import Khazad

cache = Khazad(redis_url="redis://localhost:6379", threshold=0.90)

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the capital of Italy?",
)
print(response.text)

cache.stop()
```

Matches `generativelanguage.googleapis.com/*/models/*:generateContent`. Gemini streaming (`:streamGenerateContent`) passes through uncached.

</details>

## Supported Providers

| Provider | URL pattern matched | Streaming |
|---|---|---|
| OpenAI Chat Completions | any host, path ending `/chat/completions` | cached + replayed |
| OpenAI Responses API | any host, path ending `/responses` | cached + replayed |
| Azure OpenAI | covered by chat/completions matcher | cached + replayed |
| OpenAI-compatible proxies | covered by chat/completions matcher | cached + replayed |
| Anthropic | `api.anthropic.com/v1/messages` | cached + replayed |
| Google Gemini | `generativelanguage.googleapis.com/*:generateContent` | pass-through |

## Limitations / When to use

Semantic caching trades exactness for cost and latency. Know the trade before turning it on.

**Good fit:**
- High-volume, repetitive traffic: FAQ bots, support assistants, RAG front-ends where many users ask near-identical questions
- Dev / test / CI environments — stop paying for the same prompt on every run
- Demos and load tests where deterministic, instant responses are a feature
- Cost ceilings on internal tools

**Bad fit — keep it off:**
- Answers that depend on *exact* wording: math, dates, quantities, code generation, legal/medical text. Two prompts at 0.91 similarity can require different answers; the cache will happily serve the wrong one.
- Workflows that rely on sampling variety (brainstorming, creative writing, "regenerate" buttons). A cache hit always returns the same response — the variety silently disappears.
- Agentic loops with tool calls — tool-call responses are not reconstructed from streams, and caching decisions mid-loop is rarely what you want.

**Operational caveats:**
- **Privacy**: prompts are embedded and responses are stored **in clear text in Redis**. If prompts may contain PII or secrets, set a `ttl`, enable Redis AUTH/TLS, and treat the Redis instance with the same care as your logs.
- **Process-wide patch**: Khazad wraps *every* `httpx.Client`/`AsyncClient` created after `init()` — non-LLM httpx traffic passes through untouched, but the patch is process-global. Call `stop()` on shutdown. Use `hosts=[...]` to restrict interception to the endpoints you actually want cached.
- **httpx-only**: SDKs built on `httpx` are covered (OpenAI, Anthropic, Gemini via `google-genai`, Mistral, and most proxies). SDKs using `requests`, `aiohttp`, or `boto3` (AWS Bedrock) are not intercepted.
- **Single process**: the patch lives in the Python process that called `init()`. Multiple workers share the Redis cache but each needs its own `init()`.
- **False-positive control**: start at `threshold=0.90` and *raise* it if you see wrong hits. Watch `avg_hit_similarity` in `get_stats()` — if it sits near your threshold, your traffic may be too diverse to cache safely.

## API Reference

The module exposes five functions as the singleton API. The `Khazad` class exposes the same surface as instance methods (no `init` — instantiation does that).

### `init(redis_url, threshold, ttl, namespace, embedder, embedding_model, log_level, hosts, cache_scope)`

Activate the global singleton. Builds the embedder, connects to Redis, installs the `httpx` transport patch. **Required call before any LLM traffic.** Calling twice without `stop()` in between is a no-op (warns).

```python
khazad.init(redis_url="redis://localhost:6379", threshold=0.90)
```

All parameters have defaults — see [Configuration](#configuration). Class equivalent: `Khazad(...)` constructor.

### `stop()`

Restore original `httpx` transports, close the Redis connection, and clear the singleton. Idempotent — safe to call when not active. Cached data in Redis stays — only the in-process patch is removed.

```python
khazad.stop()
```

Always call before process exit (or use a `try/finally`) to avoid leaking patched `httpx.Client.__init__` into other libraries that import after Khazad. Clients created while Khazad was active stop serving from the cache as soon as `stop()` is called. Class equivalent: `cache.stop()`.

### `get_stats() -> dict`

Snapshot of cache metrics as a plain dict. Thread-safe. Returns zero-stats if Khazad never initialized.

```python
khazad.get_stats()
# {'total_requests': 1000, 'cache_hits': 720, 'cache_misses': 280,
#  'hit_rate': 0.72, 'avg_hit_similarity': 0.943}
```

Use to track hit rate in production, expose as Prometheus gauge, or log periodically. Class equivalent: `cache.get_stats()` (returns a `Stats` dataclass — call `.to_dict()` for the same shape).

### `flush()`

Wipe **all** cached entries in the current namespace and reset stats counters to zero. Destructive — use for tests, dev resets, or after a prompt change that invalidates prior responses.

```python
khazad.flush()
```

Only deletes keys under the configured `namespace` prefix; other Redis data is untouched. No-op (warns) if Khazad never initialized. Class equivalent: `cache.flush()`.

### `is_active() -> bool`

Returns `True` if Khazad is currently running (initialized and not stopped). Useful for guarding code paths or asserting setup in tests.

```python
if not khazad.is_active():
    khazad.init(...)
```

Class equivalent: `cache.is_active()`.

## Configuration

All parameters are the same whether you use `khazad.init()` or `Khazad(...)`:

```python
Khazad(
    redis_url="redis://localhost:6379",  # Redis connection URL
    threshold=0.90,                      # Cosine similarity threshold (0.0–1.0)
    ttl=3600,                            # Cache TTL in seconds (None = no expiry)
    namespace="khazad",                  # Redis key prefix
    embedder="huggingface",              # "huggingface" (default, free) or "openai"
    embedding_model="redis/langcache-embed-v2",
    log_level="INFO",                    # DEBUG | INFO | WARNING | ERROR
    hosts=None,                          # Opt-in host allowlist (None = all hosts)
    cache_scope="model",                 # "model" (default) or "host" (one cache per provider, ignore model)
)
```

**`hosts` — opt-in allowlist.** By default Khazad considers traffic to any host that matches a provider URL pattern. Pass an explicit allowlist to restrict interception to the endpoints you intend to cache; everything else passes through untouched. Supports exact hosts and `*.` wildcard subdomains:

```python
khazad.init(hosts=["api.openai.com", "*.openai.azure.com"])
```

**`cache_scope` — share one cache across a provider's models.** Driven by the `CacheScope` enum (importable from `khazad`); the string values `"model"` and `"host"` are accepted too. By default (`CacheScope.MODEL`) each `(host, model)` pair gets its own vector set, so a `gpt-4o` answer never serves a `gpt-4o-mini` call. Set it to `CacheScope.HOST` to scope by **host only** — every model or deployment on the same provider then shares a single cache:

```python
from khazad import CacheScope

khazad.init(cache_scope=CacheScope.HOST)   # or cache_scope="host"
```

The host always stays part of the scope, so different providers never mix (an Azure OpenAI response is never replayed to a Gemini client). Use it only for format-compatible pools — e.g. multiple Azure OpenAI deployments, or treating `gpt-4o` and `gpt-4o-mini` as interchangeable. The trade-off is semantic: a smaller model may serve an answer originally produced by a larger one.

**Threshold guidance:**
- `0.95+` — strict, near-identical prompts only
- `0.90` — recommended default
- `0.85` — aggressive, higher hit rate

**TTL:** the response body expires in Redis after `ttl` seconds. Khazad prunes the matching vector automatically the next time it is found without a body, so expired entries clean themselves up.

## Embedding Backends

| Backend | Cost | Notes |
|---|---|---|
| `huggingface` (default) | Free | Downloads model on first use |
| `openai` | Paid | `uv add khazad[openai-embeddings]` |

## Architecture

Khazad follows **Hexagonal Architecture** (Ports & Adapters). The `Khazad` class is the single entry point — it owns the embedder, vector store, parsers, and cache logic directly, with no intermediate engine layer.

```
khazad/
├── khazad.py          # Khazad class — prepare / lookup / store, stats, lifecycle
├── _models.py         # ParsedRequest, CacheHit, Stats
├── _transport.py      # httpx transport patch + sync/async cached transports
├── ports/             # Abstract interfaces
│   ├── embedder.py    # Embedder
│   ├── parser.py      # ProviderParser (+ shared SSE helpers)
│   └── store.py       # VectorStore (scope-aware)
└── adapters/          # Concrete implementations
    ├── embedders/     # HuggingFace (free), OpenAI (paid)
    ├── parsers/       # OpenAI, OpenAI Responses, Anthropic, Gemini
    └── redis/         # Redis 8 Vector Sets
```

Each request is parsed **once** (`prepare`), producing the cache scope (`host/model`), the conversation text, and the stream flag. The embedding is computed once and reused between lookup and store. In Redis, each scope has its own vector set (`{namespace}:vset:{host}/{model}`); response bodies live under `{namespace}:resp:{key}`.

## Observability

```
[Khazad] CACHE HIT - Similarity: 0.94 - Latency: 4ms
[Khazad] CACHE MISS - Forwarding to API
```

```python
cache.get_stats().to_dict()
# {'total_requests': 1000, 'cache_hits': 720, 'cache_misses': 280,
#  'hit_rate': 0.72, 'avg_hit_similarity': 0.943}
```

## Requirements

- Python >= 3.10
- Redis 8 (Vector Sets support required)

```bash
docker run -d --name redis8 -p 6379:6379 redis:8
```

## Roadmap

- **Per-host / per-model thresholds** — different similarity bars for different traffic
- **Targeted invalidation** — delete cached entries matching a prompt, not just `flush()`
- **Prometheus metrics endpoint** — hit rate, latency, similarity distribution out of the box
- **Async embedder** — embed off the event loop natively instead of via thread executor
- **More transports** — `requests`/`aiohttp` interception, AWS Bedrock (`boto3`)

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Development

```bash
git clone https://github.com/GuglielmoCerri/khazad.git
cd khazad
uv sync --group dev

# Tests (no Redis or API keys needed — fakes and mock transports)
uv run python -m pytest tests/ -q

# Lint / format
uv run python -m ruff check . --fix
uv run python -m ruff format .

# Smoke test against a real endpoint (requires Redis 8 + credentials)
uv run python examples/azure_openai_entra.py
```

## License

MIT
