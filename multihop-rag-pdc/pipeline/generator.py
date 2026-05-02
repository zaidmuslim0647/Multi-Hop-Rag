import time
from openai import OpenAI
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY


class GeminiGenerator:
    """Local Ollama generator (OpenAI-compatible). Name kept for import compatibility."""

    def __init__(self, api_key: str = None):
        self._client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=api_key or OLLAMA_API_KEY)
        self._call_count = 0

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        self._call_count += 1
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                msg = str(e)
                if "Connection" in msg or "connect" in msg.lower():
                    wait = 5 * (attempt + 1)
                    print(f"[ollama connection retry] attempt {attempt+1}/3, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("Ollama: max retries exceeded — is the server running?")

    def call_count(self) -> int:
        return self._call_count
