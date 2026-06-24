"""Example of excluding gemini api call using the ``hosts`` parameter.

Khazad only intercepts traffic to hosts in the ``hosts`` allowlist. Here the
allowlist contains only ``api.openai.com``, so Gemini's host
(``generativelanguage.googleapis.com``) is *not* covered: every request passes
straight through to the API untouched.

Because nothing is cached, the second identical call is just as slow as the
first and ``get_stats()`` stays at zero — unparseable/non-allowlisted requests
are never counted. Add ``generativelanguage.googleapis.com`` to the allowlist
(or drop the ``hosts`` argument entirely) to start caching Gemini.
"""

import os
import time

from google import genai

from khazad import Khazad

cache = Khazad(
    redis_url="redis://localhost:6379",
    threshold=0.90,
    hosts=["api.openai.com"],
)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt = "What is the capital of Brazil?"

for i in range(2):
    start = time.perf_counter()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[call {i + 1}] {elapsed:.1f}ms — {response.text}...")

print(cache.get_stats().to_dict())
cache.stop()
