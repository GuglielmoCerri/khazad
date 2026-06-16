import os
import time

from google import genai

from khazad import Khazad

cache = Khazad(redis_url="redis://localhost:6379", threshold=0.90)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for i in range(2):
    start = time.perf_counter()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="What is the capital of Italy?",
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[call {i + 1}] {elapsed:.1f}ms — {response.text}...")

print(cache.get_stats().to_dict())
cache.stop()
