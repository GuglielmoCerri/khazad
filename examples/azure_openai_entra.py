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
    deployment = "gpt-5.4"

print(cache.get_stats().to_dict())
cache.stop()
