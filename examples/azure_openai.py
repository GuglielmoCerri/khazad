import os
import time

from openai import OpenAI

from khazad import Khazad

cache = Khazad(
    redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
    threshold=0.90,
    log_level="DEBUG",
)

client = OpenAI(base_url=os.environ.get("ENDPOINT"), api_key=os.environ.get("OPENAI_API_KEY"))
deployment_name = os.environ.get("DEPLOYMENT_NAME") or "gpt-4.1"

prompt = "What is the capital of Italy?"

for i in range(2):
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"[call {i + 1}] {elapsed:.1f}ms — {response.choices[0].message.content}")

print(cache.get_stats().to_dict())
cache.stop()
