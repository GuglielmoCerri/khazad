import time

from openai import OpenAI

from khazad import Khazad

cache = Khazad(redis_url="redis://localhost:6379", threshold=0.90, namespace="ollama_example")

# Ollama exposes an OpenAI-compatible API at /v1/chat/completions.
# Start it with `ollama serve` and pull a model with e.g. `ollama pull llama3`.
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
model = "llama3"

prompt = "What is the capital of Spain?"

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
