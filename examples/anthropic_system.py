"""Show that the cache key includes the system prompt, not just user turns.

Khazad embeds the full conversation -> system + all role:text lines, so a
different system prompt with the same user question is a different cache entry.
Calls 1 and 2 share the "concise" system prompt: MISS then HIT. Call 3 swaps in
a different system prompt, which embeds sufficiently far away to drop below the
threshold and is a MISS, so identical user text under different instructions
never cross-serves.

Run from the repo root:
> uv run --group examples python -P examples/anthropic_system.py
-P (safe path) stops the cwd from shadowing the installed `khazad` package.
"""

import os
import time

from anthropic import Anthropic

from khazad import Khazad

cache = Khazad(
    redis_url="redis://localhost:6379",
    threshold=0.95,
    namespace="anthropic_system_example",
)
cache.flush()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
model = "claude-haiku-4-5-20251001"
prompt = "What is the capital of Finland?"
systems = [
    "Answer concisely.",
    "Answer concisely.",
    "Answer explaining your reasoning step by step in great detail before giving the final answer. Also tell me a joke.",
]

for i, system in enumerate(systems):
    start = time.perf_counter()
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[call {i + 1}] {elapsed:.1f}ms — {message.content[0].text}")

print(cache.get_stats().to_dict())
cache.stop()
