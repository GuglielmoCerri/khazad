"""
Run from the repo root:
> uv run --group examples python -P examples/anthropic.py
-P (safe path) stops the cwd from shadowing the installed `khazad` package.
"""
import os
import time

from anthropic import Anthropic

from khazad import Khazad

cache = Khazad(redis_url="redis://localhost:6379", threshold=0.90, namespace="anthropic_example")

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
model = "claude-haiku-4-5-20251001"

prompt = "What is the capital of France?"

for i in range(2):
    start = time.perf_counter()
    message = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[call {i + 1}] {elapsed:.1f}ms — {message.content[0].text}")

print(cache.get_stats().to_dict())
cache.stop()
