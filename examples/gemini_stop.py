"""Example to see how to use stopping Khazad.

Phase 1 runs with the cache active: the first call is a MISS (stored) and the
second is a HIT (replayed instantly).

Phase 2 calls ``cache.stop()``, which restores the original httpx transports,
so every subsequent request goes straight to the Gemini API: no cache hits, and
the stats stay unchanged from Phase 1 since nothing is cached while stopped. No log
are displayed for Phase 2 since the cache is stopped. In the final get_stats() there are 
just 2 total_requests.
"""

import os
import time

from google import genai

from khazad import Khazad

cache = Khazad(redis_url="redis://localhost:6379", threshold=0.90)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt = "What is the capital of Brazil?"

print("== caching active ==")
for i in range(2):
    start = time.perf_counter()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[call {i + 1}] {elapsed:.1f}ms — {response.text}...")

print(cache.get_stats().to_dict())


cache.stop()
print(f"\n== caching stopped (is_active={cache.is_active()}) ==")
for i in range(2):
    start = time.perf_counter()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[call {i + 1}] {elapsed:.1f}ms — {response.text}...")

print(cache.get_stats().to_dict())
