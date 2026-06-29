"""
Run from the repo root:
> uv run --group examples python -P examples/openai.py
-P (safe path) stops the cwd from shadowing the installed `khazad` package.
"""
import os
import time

from openai import OpenAI

from khazad import Khazad

cache = Khazad(redis_url="redis://localhost:6379", threshold=0.90, namespace="openai_example")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
model = "gpt-4o-mini"

prompt = "What is the capital of Italy?"

for i in range(2):
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[call {i + 1}] {elapsed:.1f}ms — {response.choices[0].message.content}")

print(cache.get_stats().to_dict())
cache.stop()
