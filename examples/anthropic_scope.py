"""Share one cache across two Anthropic models with cache_scope=HOST.

By default Khazad scopes the cache to host+model, so claude-haiku and
claude-sonnet never share entries. With cache_scope=CacheScope.HOST the model
is dropped from the scope: the MISS stored against haiku is replayed as a HIT
for sonnet, because both deployments speak the same Anthropic wire format.

Call 1 (haiku) is a MISS; call 2 (sonnet) is a HIT served from the haiku entry.

Run from the repo root:
> uv run --group examples python -P examples/anthropic_scope.py
-P (safe path) stops the cwd from shadowing the installed `khazad` package.
"""

import os
import time

from anthropic import Anthropic

from khazad import CacheScope, Khazad

cache = Khazad(
    redis_url="redis://localhost:6379",
    threshold=0.90,
    cache_scope=CacheScope.HOST,
    namespace="anthropic_scope_example",
)
cache.flush()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
model = "claude-haiku-4-5-20251001"
prompt = "What is the capital of Japan?"

for i in range(2):
    start = time.perf_counter()
    message = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[call {i + 1}] {elapsed:.1f}ms — {message.content[0].text}")
    model = "claude-sonnet-4-5-20250929"

print(cache.get_stats().to_dict())
cache.stop()
